import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from vq_vae import VQVAE
import matplotlib.pyplot as plt
import os

def train_vq_vae():
    """Train and save the VQ-VAE model."""
    # Use MPS if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Create model
    model = VQVAE(num_embeddings=512, embedding_dim=32).to(device)
    
    # Train
    print("Training VQ-VAE...")
    model.train_model(train_loader, epochs=5, lr=1e-3, device=device)
    
    # Save model
    output_path = "vq_vae_model.pt"
    torch.save(model.state_dict(), output_path)
    print(f"VQ-VAE model saved to {output_path}")
    
    # Visualize reconstruction results
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        reconstructed, _, indices = model(images)
        
        plt.figure(figsize=(12, 6))
        for i in range(8):
            # Original
            plt.subplot(2, 8, i+1)
            plt.imshow(images[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title("Original")
            
            # Reconstruction
            plt.subplot(2, 8, i+9)
            plt.imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title("Reconstructed")
        
        plt.tight_layout()
        plt.savefig("vq_vae_reconstruction.png")
        plt.close()
    
    return model

if __name__ == "__main__":
    train_vq_vae()
