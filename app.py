import os
import glob
import time
from io import BytesIO

import torch
import torch.nn as nn  # noqa: F401 (kept for potential custom modules)
import numpy as np
from PIL import Image

import gradio as gr
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# -----------------------------------------------------------------------------
# Import your project modules (must be present in the repo)
# -----------------------------------------------------------------------------
from train_conv import ConvGeneratorModel, ConvConfig
from train_moe_conditional import MoEPixelTransformer, MoEPixelTransformerConfig
from train_conditional import PixelTransformer, PixelTransformerConfig
from vq_transformer import VQTransformer, VQTransformerConfig
from vq_vae import VQVAE

# -----------------------------------------------------------------------------
# Device selection — default to CPU (your note: MPS slower than CPU)
# -----------------------------------------------------------------------------
DEVICE = os.environ.get("DEVICE", "cpu")
assert DEVICE in {"cpu", "cuda", "mps"}, f"Unsupported DEVICE={DEVICE}"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

CKPT_CANDIDATES = (
    "pytorch_model.bin",
    "model.pt",
    "weights.pt",
    "checkpoint.pt",
    "state_dict.pt",
)

def _find_checkpoint_file(base_path: str) -> str:
    """Return a checkpoint file path inside a directory or accept a direct file.
    Raises FileNotFoundError if nothing is found.
    """
    if os.path.isdir(base_path):
        for name in CKPT_CANDIDATES:
            p = os.path.join(base_path, name)
            if os.path.exists(p):
                return p
        # also try to pick the first .pt/.bin inside the dir
        for ext in ("*.pt", "*.bin"):
            matches = glob.glob(os.path.join(base_path, ext))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"No checkpoint file found under: {base_path}")
    if os.path.exists(base_path):
        return base_path
    raise FileNotFoundError(f"Checkpoint path not found: {base_path}")


def _load_state_dict_cpu(base_or_file: str):
    """Always load a torch checkpoint onto CPU regardless of how it was saved."""
    ckpt = _find_checkpoint_file(base_or_file)
    return torch.load(ckpt, map_location="cpu")


def _to_pil_uint8(img_t: torch.Tensor, mode: str = "L") -> Image.Image:
    """Convert a tensor in [0,1] or [0,255] to a PIL image (grayscale by default)."""
    t = img_t.detach().cpu()
    if t.dim() == 3 and t.shape[0] in (1, 3):
        # CHW -> HWC
        t = t.permute(1, 2, 0)
    elif t.dim() == 2:
        pass
    else:
        # attempt to squeeze batch
        t = t.squeeze()
        if t.dim() == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
    np_img = t.numpy()
    if np_img.max() <= 1.0:
        np_img = (np_img * 255.0).clip(0, 255)
    np_img = np_img.astype(np.uint8)
    if mode == "L" and np_img.ndim == 3:
        # take single channel if present
        np_img = np_img[..., 0]
    return Image.fromarray(np_img, mode)


# -----------------------------------------------------------------------------
# Model registry / availability
# -----------------------------------------------------------------------------

available_models = {
    "pixel": False,      # PixelTransformer (autoregressive by pixel)
    "moe": False,        # MoEPixelTransformer (mixture of experts)
    "conv": False,       # ConvGenerator (direct generation)
    "vq": False,         # VQTransformer (autoregressive tokens)
    "vq-vae": False,     # VQ‑VAE (encode/decode)
    "diffusion": False,  # DDPM
}

conv_model = None
moe_model = None
pixel_model = None
vq_model = None
vq_transformer_model = None
diffusion_pipe = None

# -----------------------------------------------------------------------------
# Load ConvGenerator (safe, device-agnostic)
# -----------------------------------------------------------------------------
try:
    conv_config = ConvConfig.from_pretrained("my_conv")
    conv_model = ConvGeneratorModel(conv_config)
    sd = _load_state_dict_cpu("my_conv")
    conv_model.load_state_dict(sd, strict=False)
    conv_model.to(DEVICE).eval()
    available_models["conv"] = True
    print("✓ ConvGenerator model loaded successfully")
except Exception as e:
    conv_model = None
    print(f"✗ Error loading ConvGenerator: {e}")

# -----------------------------------------------------------------------------
# Load MoEPixelTransformer (safe map_location)
# -----------------------------------------------------------------------------
try:
    moe_config = MoEPixelTransformerConfig.from_pretrained("my_moe_model")
    moe_model = MoEPixelTransformer(moe_config)
    moe_sd = _load_state_dict_cpu("my_moe_model")
    moe_model.load_state_dict(moe_sd, strict=False)
    moe_model.to(DEVICE).eval()
    available_models["moe"] = True
    print("✓ MoEPixelTransformer model loaded successfully")
except Exception as e:
    moe_model = None
    print(f"✗ Error loading MoEPixelTransformer: {e}")

# -----------------------------------------------------------------------------
# Load PixelTransformer (safe map_location)
# -----------------------------------------------------------------------------
try:
    pixel_config = PixelTransformerConfig.from_pretrained("my_model")
    pixel_model = PixelTransformer(pixel_config)
    pixel_sd = _load_state_dict_cpu("my_model")
    pixel_model.load_state_dict(pixel_sd, strict=False)
    pixel_model.to(DEVICE).eval()
    available_models["pixel"] = True
    print("✓ PixelTransformer model loaded successfully")
except Exception as e:
    pixel_model = None
    print(f"✗ Error loading PixelTransformer: {e}")

# -----------------------------------------------------------------------------
# Load VQ‑VAE + VQ‑Transformer (safe map_location)
# -----------------------------------------------------------------------------
try:
    if os.path.exists("vq_vae_model.pt"):
        vq_model = VQVAE()
        vq_model.load_state_dict(torch.load("vq_vae_model.pt", map_location="cpu"))
        vq_model.to(DEVICE).eval()
        available_models["vq-vae"] = True
        print("✓ VQ‑VAE model loaded successfully")
        # VQ‑Transformer (optional)
        if os.path.exists("vq_transformer_model"):
            vq_trans_config = VQTransformerConfig.from_pretrained("vq_transformer_model")
            vq_transformer_model = VQTransformer(vq_trans_config)
            vq_t_sd = _load_state_dict_cpu("vq_transformer_model")
            vq_transformer_model.load_state_dict(vq_t_sd, strict=False)
            vq_transformer_model.to(DEVICE).eval()
            available_models["vq"] = True
            print("✓ VQ‑Transformer model loaded successfully")
except Exception as e:
    vq_transformer_model = None
    print(f"✗ Error loading VQ models: {e}")

# -----------------------------------------------------------------------------
# Load Diffusion pipeline if present
# -----------------------------------------------------------------------------
try:
    from diffusers import DDPMPipeline
    diffusion_model_dir = "my_diffusion_model"
    if os.path.exists(diffusion_model_dir):
        diffusion_pipe = DDPMPipeline.from_pretrained(diffusion_model_dir, torch_dtype=torch.float32)
        diffusion_pipe = diffusion_pipe.to(DEVICE)
        available_models["diffusion"] = True
        print("✓ Diffusion pipeline loaded successfully")
    else:
        print(f"✗ Diffusion model directory '{diffusion_model_dir}' not found, skipping diffusion.")
except Exception as e:
    diffusion_pipe = None
    print(f"✗ Error loading Diffusion pipeline: {e}")

# pick a default
selected_model = next((name for name, ok in available_models.items() if ok), "none")
print("Available models:", [n for n, ok in available_models.items() if ok])
print("Default selected model:", selected_model)

# -----------------------------------------------------------------------------
# Generation functions (return PIL.Image or an error string)
# -----------------------------------------------------------------------------

def gen_conv(digit: int):
    if conv_model is None:
        return None, "Conv model not loaded"
    try:
        with torch.no_grad():
            label = torch.tensor([digit], device=DEVICE).long()
            out = conv_model(label).squeeze()
            # Expect (28,28)
            if out.dim() == 3 and out.shape[0] == 1:
                out = out.squeeze(0)
            if out.dim() != 2:
                return None, f"Unexpected conv output shape: {tuple(out.shape)}"
            img = _to_pil_uint8(out, mode="L")
            return img, "ok"
    except Exception as e:
        return None, f"Conv error: {e}"


def _render_from_pixel_stream(generator):
    # Build a 28x28 frame by consuming 784 pixel values in [0..9]
    frame = np.zeros((28, 28), dtype=np.uint8)
    k = 0
    for v in generator:
        row, col = divmod(k, 28)
        val = int(v * 255 / 9)
        frame[row, col] = np.uint8(np.clip(val, 0, 255))
        k += 1
        if k >= 28 * 28:
            break
    return Image.fromarray(frame, mode="L")


def gen_pixel(digit: int):
    if pixel_model is None:
        return None, "Pixel model not loaded"
    try:
        with torch.no_grad():
            if hasattr(pixel_model, "generate_digit_stream"):
                img = _render_from_pixel_stream(pixel_model.generate_digit_stream(digit))
            elif hasattr(pixel_model, "generate"):
                out = pixel_model.generate(digit)  # project-specific API
                img = _to_pil_uint8(out, mode="L")
            else:
                return None, "Pixel model has no generation method"
            return img, "ok"
    except Exception as e:
        return None, f"Pixel error: {e}"


def gen_moe(digit: int):
    if moe_model is None:
        return None, "MoE model not loaded"
    try:
        with torch.no_grad():
            if hasattr(moe_model, "generate_digit_stream"):
                img = _render_from_pixel_stream(moe_model.generate_digit_stream(digit))
            elif hasattr(moe_model, "generate"):
                out = moe_model.generate(digit)
                img = _to_pil_uint8(out, mode="L")
            else:
                return None, "MoE model has no generation method"
            return img, "ok"
    except Exception as e:
        return None, f"MoE error: {e}"


def gen_vq(digit: int):
    if vq_model is None or vq_transformer_model is None:
        return None, "VQ models not loaded"
    try:
        with torch.no_grad():
            # Project-specific API, expected to return a tensor in [0,1]
            out = vq_transformer_model.generate(digit, vq_model, DEVICE)
            img = _to_pil_uint8(out, mode="L")
            return img, "ok"
    except Exception as e:
        return None, f"VQ error: {e}"


def gen_vq_vae_recon(digit: int):
    if vq_model is None:
        return None, "VQ‑VAE not loaded"
    try:
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        # find indices of the chosen digit
        indices = [i for i, (_, lbl) in enumerate(test_dataset) if lbl == digit]
        if not indices:
            return None, f"No examples of digit {digit} found"
        idx = indices[0]
        img, _ = test_dataset[idx]
        with torch.no_grad():
            img = img.unsqueeze(0).to(DEVICE)
            codes = vq_model.encode(img)
            rec = vq_model.decode(codes)  # expect [0,1]
        return _to_pil_uint8(rec.squeeze(), mode="L"), "ok"
    except Exception as e:
        return None, f"VQ‑VAE recon error: {e}"


def gen_diffusion(digit: int, steps: int):
    if diffusion_pipe is None:
        return None, "Diffusion not loaded"
    try:
        scheduler = diffusion_pipe.scheduler
        scheduler.set_timesteps(int(steps))
        samp_size = diffusion_pipe.unet.config.sample_size
        in_ch = diffusion_pipe.unet.config.in_channels
        img = torch.randn((1, in_ch, samp_size, samp_size), device=DEVICE, dtype=torch.float32)
        labels = torch.tensor([digit], device=DEVICE)
        supports_labels = getattr(diffusion_pipe.unet.config, "class_embed_type", None) is not None
        for t in scheduler.timesteps:
            with torch.no_grad():
                if supports_labels:
                    out = diffusion_pipe.unet(img, t, class_labels=labels).sample
                else:
                    out = diffusion_pipe.unet(img, t).sample
            img = scheduler.step(out, t, img).prev_sample
        img = (img / 2 + 0.5).clamp(0, 1)
        pil = _to_pil_uint8(img[0], mode="L")
        # Resize to 28x28 for visual parity with other models
        pil = pil.resize((28, 28), Image.NEAREST)
        return pil, "ok"
    except Exception as e:
        return None, f"Diffusion error: {e}"


# -----------------------------------------------------------------------------
# Router used by the single "Generate" button
# -----------------------------------------------------------------------------

def generate_router(digit: int, steps: int, model_name: str):
    if model_name == "conv":
        return gen_conv(digit)
    if model_name == "pixel":
        return gen_pixel(digit)
    if model_name == "moe":
        return gen_moe(digit)
    if model_name == "vq":
        return gen_vq(digit)
    if model_name == "vq-vae":
        return gen_vq_vae_recon(digit)
    if model_name == "diffusion":
        return gen_diffusion(digit, steps)
    return None, f"Model '{model_name}' not available"


# -----------------------------------------------------------------------------
# Gradio UI (Blocks)
# -----------------------------------------------------------------------------

choices = [name for name, ok in available_models.items() if ok]
if not choices:
    choices = ["none"]
    selected_model = "none"

with gr.Blocks(title="Digit Generators") as demo:
    gr.Markdown("""
    # Digit Generators
    Choose a model, pick a digit, hit **Generate**.
    """)

    with gr.Row():
        model_name = gr.Dropdown(
            label="Model",
            choices=choices,
            value=selected_model,
            info="Only loaded models are shown",
        )
        digit = gr.Slider(0, 9, step=1, value=0, label="Digit")
        steps = gr.Slider(10, 200, step=1, value=50, label="Diffusion steps (only used for diffusion)")

    with gr.Row():
        btn = gr.Button("Generate", variant="primary")
        out_img = gr.Image(label="Output", interactive=False)
        status = gr.Textbox(label="Status", interactive=False)

    # show availability snapshot
    gr.JSON(value=available_models, label="Model availability")

    def _toggle_steps_visibility(m):
        return gr.update(visible=(m == "diffusion"))

    model_name.change(_toggle_steps_visibility, model_name, steps)

    btn.click(generate_router, inputs=[digit, steps, model_name], outputs=[out_img, status])


# Expose demo for HF Spaces (Gradio) and local runs
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
