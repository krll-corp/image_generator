#!/bin/bash

# Hugging Face Spaces deployment script

echo "Preparing for Hugging Face Spaces deployment..."

# Create app.py symlink for Spaces
cat > app_hf.py << 'EOF'
import gradio as gr
import torch
from PIL import Image
import numpy as np
import io
import base64

# Import your models
from app import *

def generate_digit_interface(digit, model_name, steps=50):
    """Gradio interface function"""
    try:
        if model_name == "conv" and conv_model:
            digit_tensor = torch.tensor([digit], dtype=torch.long).to(device)
            with torch.no_grad():
                generated = conv_model(digit_tensor)
                img_array = generated[0, 0].cpu().numpy()
                img_array = (img_array * 255).astype(np.uint8)
                return Image.fromarray(img_array, mode='L').resize((280, 280), Image.NEAREST)
        elif model_name == "diffusion" and diffusion_pipe:
            with torch.no_grad():
                image = diffusion_pipe(
                    class_labels=torch.tensor([digit]).to(device),
                    num_inference_steps=steps
                ).images[0]
                return image.resize((280, 280), Image.NEAREST)
        else:
            return Image.new('L', (280, 280), 0)
    except Exception as e:
        print(f"Error: {e}")
        return Image.new('L', (280, 280), 0)

# Create Gradio interface
with gr.Blocks(title="MNIST Digit Generator") as demo:
    gr.Markdown("# MNIST Digit Generator")
    gr.Markdown("Generate MNIST-style digits using various deep learning models.")
    
    with gr.Row():
        digit_input = gr.Number(label="Digit (0-9)", value=7, minimum=0, maximum=9, precision=0)
        model_select = gr.Dropdown(
            choices=[name for name, available in available_models.items() if available],
            label="Model",
            value=selected_model
        )
        steps_input = gr.Number(label="Steps (for diffusion)", value=50, minimum=1, maximum=1000)
    
    generate_btn = gr.Button("Generate")
    output_image = gr.Image(label="Generated Digit", type="pil")
    
    generate_btn.click(
        fn=generate_digit_interface,
        inputs=[digit_input, model_select, steps_input],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch()
EOF

echo "Hugging Face Spaces app created as app_hf.py"
echo "Upload to Hugging Face Spaces with Docker runtime"