# MNIST Digit Generator

An interactive web application that generates MNIST-style digits using various deep learning models. The application allows users to visualize digit generation in real-time through a web interface.

## Overview

This application demonstrates different approaches to image generation using neural networks:

- **PixelTransformer**: Autoregressive generation one pixel at a time
- **MoEPixelTransformer**: Mixture of Experts approach for pixel-by-pixel generation
- **ConvGenerator**: Direct generation of full images
- **VQTransformer**: Autoregressive generation by tokens, then decoding with VQ-VAE
- **VQ-VAE**: Direct encoding/decoding without autoregressive generation

## Requirements

- Python 3.7+
- PyTorch
- Flask
- Pillow
- NumPy
- TorchVision

## Installation

```bash
# Clone the repository
git clone https://github.com/krll-corp/image_generator.git
cd image_generator

# Install dependencies
pip install torch torchvision flask pillow numpy
```

## Usage

1. Firstly, you need to train all provided models (or ones you'd like to use) with provided scripts:
    ```bash
    # on Windows it's usually
    python train_model_youd_like_to_use
    # on Mac / Linux
    python3 train_model_youd_like_to_use
    ```

2. Start the Flask server:
   ```bash
   python app.py
   # or
   python3 a`app.py
   ```

3. Open your browser and navigate to `http://127.0.0.1:5000/`

4. Select a model using the dropdown menu and choose a digit to generate

5. Watch the generation process in real-time or view the final result

## Project Structure

- `app.py`: Main application file containing Flask routes and model loading
- `train_conv.py`: ConvGenerator model implementation
- `train_conditional.py`: PixelTransformer model implementation
- `train_moe_conditional.py`: MoEPixelTransformer model implementation
- `vq_transformer.py`: VQTransformer model implementation
- `vq_vae.py`: VQVAE model implementation
- `templates/`: HTML templates for the web interface

## Model Availability

The application checks for available models on startup. If a model is missing, the related functionality will be disabled in the UI. You'll need to train the models or obtain pre-trained weights to use all features.

## License

[MIT License](https://opensource.org/licenses/MIT)
