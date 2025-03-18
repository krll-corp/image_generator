import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset_vq import VQTokenizedMNISTDataset
from vq_transformer import VQTransformer, VQTransformerConfig
from vq_vae import VQVAE

import os
from tqdm import tqdm
import lion_pytorch

def train_vq_transformer():
    # Check for VQ-VAE model file, train if not exists
    vq_model_path = "vq_vae_model.pt"
    if not os.path.exists(vq_model_path):
        from train_vq_vae import train_vq_vae
        train_vq_vae()
    
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create dataset with VQ tokens
    train_dataset = VQTokenizedMNISTDataset(split="train", vq_model_path=vq_model_path, device=device)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize VQ-Transformer model
    config = VQTransformerConfig(
        vocab_size=512,  # VQ-VAE codebook size
        token_grid_size=7,  # 7x7 token grid
        n_layers=4, #8
        d_model=64, #256
        n_heads=8,
        dropout=0.1,
        max_position_embeddings=50,  # 1 label + 49 tokens
        label_offset=512,  # Labels are tokens 512-521
        batch_size=32,
        epochs=5,
        lr=1e-3,
        warmup_steps=500,
        device=device
    )
    
    model = VQTransformer(config).to(device)
    
    # Use Lion optimizer for faster training
    optimizer = lion_pytorch.Lion(model.parameters(), lr=config.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler with warmup
    total_steps = config.epochs * len(train_loader)
    warmup_steps = config.warmup_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    global_step = 0
    
    try:
        for epoch in range(config.epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
            for inputs, targets in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                logits = model(inputs)
                
                # Get loss
                loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                global_step += 1
                
    except KeyboardInterrupt:
        print("\nEmergency save triggered by keyboard interrupt...")
        model.save_pretrained("vq_transformer_model")
        print("Model saved to vq_transformer_model/")
        return model
    
    model.save_pretrained("vq_transformer_model")
    return model

if __name__ == "__main__":
    train_vq_transformer()
