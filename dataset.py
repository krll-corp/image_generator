import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class ConditionalMNISTDataset(Dataset):
    """Dataset for conditional MNIST generation.
    For each MNIST example, we create:
    x = label token (offset by label_offset)
    y = actual image pixels
    """
    def __init__(self, split="train", label_offset=256):
        super().__init__()
        self.label_offset = label_offset
        
        # Load MNIST from torchvision
        transform = transforms.ToTensor()
        self.data = datasets.MNIST(root="./data", train=(split=="train"), 
                                 download=True, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        # Create label token by offsetting the label
        x = torch.tensor([self.label_offset + label], dtype=torch.long)
        # Keep image as is - will be reshaped in training loop
        y = (img * 255).byte()  # Convert to 0-255 range
        return x, y 