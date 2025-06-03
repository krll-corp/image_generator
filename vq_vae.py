import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()  # Straight-through estimator
        
        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=32):
        super().__init__()
        
        # Encoder for MNIST (28x28 -> 7x7)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(True),
            nn.Conv2d(32, embedding_dim, 1, stride=1)  # 7x7 -> 7x7xembedding_dim
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        
        # Decoder for MNIST (7x7 -> 28x28)
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()
        )
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def encode(self, x):
        z = self.encoder(x)
        _, _, indices = self.vq(z)
        return indices
    
    def decode(self, indices):
        # Convert indices to one-hot vectors
        batch_size, height, width = indices.shape
        one_hot = torch.zeros(batch_size, self.vq.num_embeddings, height, width, device=indices.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)
        
        # Retrieve embeddings and reshape
        quantized = torch.matmul(one_hot.permute(0, 2, 3, 1), self.vq.embedding.weight)
        quantized = quantized.permute(0, 3, 1, 2)
        
        # Decode
        reconstructed = self.decoder(quantized)
        return reconstructed
        
    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss, indices
    
    def train_model(self, train_loader, epochs=10, lr=1e-3, device='cpu'):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_vq_loss = 0
            
            for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(device)
                
                optimizer.zero_grad()
                
                reconstructed, vq_loss, _ = self(images)
                reconstruction_loss = F.mse_loss(reconstructed, images)
                
                loss = reconstruction_loss + vq_loss
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += reconstruction_loss.item()
                total_vq_loss += vq_loss.item()
            
            avg_loss = total_loss / len(train_loader)
            avg_recon_loss = total_recon_loss / len(train_loader)
            avg_vq_loss = total_vq_loss / len(train_loader)
            
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, VQ Loss: {avg_vq_loss:.4f}")
    
    @staticmethod
    def train_and_save(output_path="vq_vae_model.pt", device='cpu', batch_size=128, epochs=10):
        # Setup data
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create model
        model = VQVAE().to(device)
        
        # Train
        model.train_model(train_loader, epochs=epochs, device=device)
        
        # Save model
        torch.save(model.state_dict(), output_path)
        print(f"Model saved to {output_path}")
        
        return model
