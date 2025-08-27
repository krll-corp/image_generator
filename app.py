import torch
import torch.nn as nn
from flask import Flask, request, Response, jsonify, render_template
from io import BytesIO
from PIL import Image
import numpy as np
import time
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from train_conv import ConvGeneratorModel, ConvConfig
from train_moe_conditional import MoEPixelTransformer, MoEPixelTransformerConfig
from train_conditional import PixelTransformer, PixelTransformerConfig
from vq_transformer import VQTransformer, VQTransformerConfig
from vq_vae import VQVAE

app = Flask(__name__, template_folder="templates")

#join the path where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.path.join(script_dir, '')
os.system("git lfs pull")

# Detect device — adjust if you want to force 'mps' or 'cuda'
device = 'cpu'  # don't know why but mps runs slower than cpu

# ------------------------------------------------------------------------------
# Model Status Tracking
# ------------------------------------------------------------------------------
available_models = {
    "pixel": False,     # PixelTransformer (autoregressive by pixel)
    "moe": False,       # MoEPixelTransformer (mixture of experts)
    "conv": False,      # ConvGenerator (direct generation)
    "vq": False,        # VQTransformer (autoregressive by token)
    "vq-vae": False,    # VQ-VAE only (encode/decode)
    "diffusion": False  # Diffusion model (DDPM)
}

# ------------------------------------------------------------------------------
# Load models
# ------------------------------------------------------------------------------
# 1. Load ConvGenerator
try:
    conv_config = ConvConfig.from_pretrained("my_conv")
    conv_model = ConvGeneratorModel.from_pretrained("my_conv", config=conv_config).to(device)
    conv_model.eval()
    available_models["conv"] = True
    print("✓ ConvGenerator model loaded successfully")
except Exception as e:
    conv_model = None
    print(f"✗ Error loading ConvGenerator: {str(e)}")

# 2. Load MoEPixelTransformer
try:
    moe_config = MoEPixelTransformerConfig.from_pretrained("my_moe_model")
    moe_model = MoEPixelTransformer.from_pretrained("my_moe_model", config=moe_config).to(device)
    moe_model.eval()
    available_models["moe"] = True
    print("✓ MoEPixelTransformer model loaded successfully")
except Exception as e:
    moe_model = None
    print(f"✗ Error loading MoEPixelTransformer: {str(e)}")

# 3. Load PixelTransformer
try:
    pixel_config = PixelTransformerConfig.from_pretrained("my_model")
    pixel_model = PixelTransformer.from_pretrained("my_model", config=pixel_config).to(device)
    pixel_model.eval()
    available_models["pixel"] = True
    print("✓ PixelTransformer model loaded successfully")
except Exception as e:
    pixel_model = None
    print(f"✗ Error loading PixelTransformer: {str(e)}")

# 4. Load VQ-VAE and VQ-Transformer models if available
vq_model = None
vq_transformer_model = None

try:
    # Load VQ-VAE
    if os.path.exists("vq_vae_model.pt"):
        vq_model = VQVAE()
        vq_model.load_state_dict(torch.load("vq_vae_model.pt", map_location=device))
        vq_model.to(device)
        vq_model.eval()
        available_models["vq-vae"] = True
        print("✓ VQ-VAE model loaded successfully")
        
        # Load VQ-Transformer
        if os.path.exists("vq_transformer_model/model.pt"):
            vq_trans_config = VQTransformerConfig.from_pretrained("vq_transformer_model")
            vq_transformer_model = VQTransformer.from_pretrained("vq_transformer_model", config=vq_trans_config).to(device)
            vq_transformer_model.eval()
            available_models["vq"] = True
            print("✓ VQ-Transformer model loaded successfully")
except Exception as e:
    print(f"✗ Error loading VQ models: {str(e)}")

# 5. Load Diffusion pipeline if available
diffusion_pipe = None
try:
    from diffusers import DDPMPipeline
    diffusion_model_dir = "my_diffusion_model"
    if os.path.exists(diffusion_model_dir):
        diffusion_pipe = DDPMPipeline.from_pretrained(
            diffusion_model_dir, torch_dtype=torch.float32
        ).to(device)
        available_models["diffusion"] = True
        print("✓ Diffusion pipeline loaded successfully")
    else:
        print(f"✗ Diffusion model directory '{diffusion_model_dir}' not found, skipping diffusion.")
except Exception as e:
    diffusion_pipe = None
    print(f"✗ Error loading Diffusion pipeline: {str(e)}")
# Select default model (use the first available one)
for model_name, is_available in available_models.items():
    if is_available:
        selected_model = model_name
        break
else:
    selected_model = "none"  # Fallback if no models are available

print(f"Available models: {[name for name, available in available_models.items() if available]}")
print(f"Default selected model: {selected_model}")

# ------------------------------------------------------------------------------
# FRONTEND
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the HTML page with UI."""
    return render_template("index.html", available_models=available_models)

# ------------------------------------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------------------------------------
@app.route("/select_model", methods=["POST"])
def select_model():
    """Updates the selected model."""
    global selected_model
    data = request.get_json()
    model_type = data.get("model_type", "pixel")
    
    if model_type in available_models and available_models[model_type]:
        selected_model = model_type
        return jsonify({"message": f"Selected model: {selected_model}"})
    else:
        return jsonify({"message": f"Model {model_type} not available"}), 400

# ------------------------------------------------------------------------------
# GENERATE DIGIT (Conv-based full 28x28 generation, no streaming)
# ------------------------------------------------------------------------------
@app.route("/generate_conv_digit", methods=["GET"])
def generate_conv_digit():
    """Handles full 28x28 generation for ConvGenerator (non-streamed)."""
    try:
        digit = int(request.args.get("digit", 0))
        print(f"Generating Conv image for digit {digit}...")

        # Convert digit to tensor
        label = torch.tensor([digit], device=device).long()

        with torch.no_grad():
            out = conv_model(label)  # Return shape (28, 28) after necessary squeezes
            out = out.squeeze()
            if out.dim() == 3:  
                out = out.squeeze(0)
            
            # Check shape
            if out.dim() != 2 or out.shape != (28, 28):
                raise ValueError(f"Expected tensor of shape (28, 28), got shape {tuple(out.shape)}")
            
            # Scale to [0, 255] (assuming model outputs are in [0,1])
            out = (out * 255.0).cpu().numpy().astype(np.uint8)

            # Convert to PIL Image
            out_img = Image.fromarray(out, mode="L")
            
            # Save as PNG in memory
            buf = BytesIO()
            out_img.save(buf, format="PNG")
            buf.seek(0)

            return Response(buf.getvalue(), mimetype="image/png")

    except Exception as e:
        print(f"Error generating conv image: {str(e)}")
        return Response(str(e), status=500)

# ------------------------------------------------------------------------------
# VQ MODEL GENERATION (token-based then decode with VQ-VAE)
# ------------------------------------------------------------------------------
@app.route("/generate_vq_digit", methods=["GET"])
def generate_vq_digit():
    """Generate image using VQ-Transformer and VQ-VAE."""
    try:
        if vq_model is None or vq_transformer_model is None:
            return Response("VQ models not loaded", status=500)
        
        digit = int(request.args.get("digit", 0))
        print(f"Generating VQ image for digit {digit}...")
        
        # Generate image using VQ-Transformer and VQ-VAE
        with torch.no_grad():
            generated_img = vq_transformer_model.generate(digit, vq_model, device)
            
            # Convert to numpy array
            img_array = generated_img.cpu().squeeze().numpy()
            
            # Scale to [0, 255]
            img_array = (img_array * 255).astype(np.uint8)
            
            # Convert to PIL Image
            out_img = Image.fromarray(img_array, mode="L")
            
            # Save as PNG in memory
            buf = BytesIO()
            out_img.save(buf, format="PNG")
            buf.seek(0)
            
            return Response(buf.getvalue(), mimetype="image/png")
    except Exception as e:
        print(f"Error generating VQ image: {str(e)}")
        return Response(str(e), status=500)

# ------------------------------------------------------------------------------
# VQ-VAE DIRECT RECONSTRUCTION (no autoregressive generation)
# ------------------------------------------------------------------------------
@app.route("/generate_vq_vae_digit", methods=["GET"])
def generate_vq_vae_digit():
    """Generate image directly using VQ-VAE (no transformer)."""
    try:
        if vq_model is None:
            return Response("VQ-VAE model not loaded", status=500)
        
        digit = int(request.args.get("digit", 0))
        print(f"Generating VQ-VAE reconstruction for digit {digit}...")
        
        # Create a test dataset to get a real MNIST digit
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        
        # Find examples of the requested digit
        digit_indices = [i for i, (_, label) in enumerate(test_dataset) if label == digit]
        
        if not digit_indices:
            return Response(f"No examples of digit {digit} found in test set", status=404)
        
        # Pick a random example of this digit
        import random
        idx = random.choice(digit_indices)
        img, _ = test_dataset[idx]
        
        # Process through VQ-VAE (encode and decode)
        with torch.no_grad():
            img = img.unsqueeze(0).to(device)  # Add batch dimension
            encoded_indices = vq_model.encode(img)
            reconstructed = vq_model.decode(encoded_indices)
            
            # Convert to numpy and scale to [0, 255]
            img_array = (reconstructed.cpu().squeeze().numpy() * 255).astype(np.uint8)
            
            # Create PIL image
            out_img = Image.fromarray(img_array, mode="L")
            
            # Save as PNG in memory
            buf = BytesIO()
            out_img.save(buf, format="PNG")
            buf.seek(0)
            
            return Response(buf.getvalue(), mimetype="image/png")
    except Exception as e:
        print(f"Error in VQ-VAE reconstruction: {str(e)}")
        return Response(str(e), status=500)


# ------------------------------------------------------------------------------
# DIFFUSION GENERATION (using DDPM pipeline)
# ------------------------------------------------------------------------------
@app.route("/generate_diffusion_digit", methods=["GET"])
def generate_diffusion_digit():
    """Generate image using diffusion model (DDPM)."""
    if diffusion_pipe is None:
        return Response("Diffusion model not loaded", status=500)
    try:
        digit = int(request.args.get("digit", 0))
        steps = int(request.args.get("steps", 50))
        print(f"Generating diffusion image for digit {digit} with {steps} steps...")
        num_steps = steps
        scheduler = diffusion_pipe.scheduler
        scheduler.set_timesteps(num_steps)

        img = torch.randn(
            (
                1,
                diffusion_pipe.unet.config.in_channels,
                diffusion_pipe.unet.config.sample_size,
                diffusion_pipe.unet.config.sample_size,
            ),
            device=device,
            dtype=torch.float32,
        )
        labels = torch.tensor([digit], device=device)

        for t in scheduler.timesteps:
            with torch.no_grad():
                model_output = diffusion_pipe.unet(img, t, class_labels=labels).sample
            img = scheduler.step(model_output, t, img).prev_sample

        img = (img / 2 + 0.5).clamp(0, 1)
        array = img.cpu().permute(0, 2, 3, 1).numpy()[0]
        array = (array * 255).astype(np.uint8)
        image = Image.fromarray(array.squeeze(), mode="L").resize((28, 28))
        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="image/png")
    except Exception as e:
        print(f"Error generating diffusion image: {str(e)}")
        return Response(str(e), status=500)

# ------------------------------------------------------------------------------
# STREAM DIGIT (Pixel-by-pixel generation or token-by-token for VQ)
# ------------------------------------------------------------------------------
@app.route("/stream_digit", methods=["GET"])
def stream_digit():
    """Streams generation for pixel models or VQ model."""
    try:
        digit = int(request.args.get("digit", 0))
        model_name = selected_model

        print(f"Streaming {model_name} image for digit {digit}...")
        
        if not available_models.get(model_name, False):
            return Response(f"data: Error: Model {model_name} is not available.\n\n", 
                           mimetype="text/event-stream")

        def pixel_stream():
            try:
                with torch.no_grad():
                    if model_name == "moe" and moe_model:
                        # MoE Pixel Transformer (pixel by pixel)
                        generator = moe_model.generate_digit_stream(digit)
                        for pixel in generator:
                            # Pixel is in [0..9], scale to [0..255]
                            pixel_value = int(pixel * 255 / 9)
                            yield f"data: {pixel_value}\n\n"
                            time.sleep(0.005)
                            
                    elif model_name == "pixel" and pixel_model:
                        # Standard Pixel Transformer
                        generator = pixel_model.generate_digit_stream(digit)
                        for pixel in generator:
                            # Pixel is in [0..9], scale to [0..255]
                            pixel_value = int(pixel * 255 / 9)
                            yield f"data: {pixel_value}\n\n"
                            time.sleep(0.005)
                            
                    elif model_name == "vq" and vq_transformer_model and vq_model:
                        # VQ-Transformer (token by token, with streaming decode)
                        generator = vq_transformer_model.generate_token_stream(digit, device)
                        tokens = []

                        # Stream token generation progress and partial image patches
                        for i, token in enumerate(generator):
                            tokens.append(token)
                            progress = int((i + 1) * 100 / 49)
                            yield f"data: token:{i+1}:{progress}\n\n"
                            time.sleep(0.01)

                            # Partial decode: pad remaining tokens with zero index
                            pad_tokens = tokens + [0] * (49 - len(tokens))
                            token_tensor = torch.tensor(pad_tokens, dtype=torch.long, device=device).reshape(1, 7, 7)
                            decoded_img = vq_model.decode(token_tensor)
                            img_array = (decoded_img.cpu().squeeze().numpy() * 255).astype(np.uint8)

                            # Stream full frame as CSV
                            flat_pixels = img_array.flatten().tolist()
                            yield f"data: frame:{','.join(str(int(p)) for p in flat_pixels)}\n\n"
                            time.sleep(0.001)
                    else:
                        yield "data: Error: Invalid model selected or model not available.\n\n"

            except Exception as e:
                print(f"Error in pixel stream: {str(e)}")
                yield f"data: Error: {str(e)}\n\n"

        return Response(pixel_stream(), mimetype="text/event-stream")

    except Exception as e:
        print(f"Error in stream_digit: {str(e)}")
        return Response(str(e), status=500)

# ------------------------------------------------------------------------------
# RUN THE APP
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=7860)
