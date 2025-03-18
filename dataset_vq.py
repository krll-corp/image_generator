import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from vq_vae import VQVAE

class VQTokenizedMNISTDataset(Dataset):
    """Dataset for MNIST where images are converted to discrete tokens using VQ-VAE."""
    
    def __init__(self, split="train", vq_model_path="vq_vae_model.pt", device="cpu", label_offset=512):
        super().__init__()
        self.label_offset = label_offset
        
        # Load MNIST
        transform = transforms.ToTensor()
        self.data = datasets.MNIST(root="./data", train=(split=="train"), 
                                  download=True, transform=transform)
        
        # Load VQ-VAE model
        self.vq_model = VQVAE()
        self.vq_model.load_state_dict(torch.load(vq_model_path, map_location=device))
        self.vq_model.to(device)
        self.vq_model.eval()
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = img.to(self.device).unsqueeze(0)  # Add batch dimension
        
        # Create label token by offsetting the label
        # Move the label token to the same device as img
        label_token = torch.tensor([self.label_offset + label], dtype=torch.long, device=self.device)
        
        # Encode image to tokens
        with torch.no_grad():
            tokens = self.vq_model.encode(img).squeeze(0)  # Shape: (7, 7) for 7x7 token grid
        
        # Flatten tokens for autoregressive modeling
        tokens_flat = tokens.flatten()  # Shape: (49,)
        
        # Prepare for autoregressive prediction: input = [label, tokens[:-1]], target = tokens
        input_tokens = torch.cat([label_token, tokens_flat[:-1]])
        target_tokens = tokens_flat
        
        # Return everything on CPU to avoid device confusion with DataLoader
        return input_tokens.cpu(), target_tokens.cpu()
