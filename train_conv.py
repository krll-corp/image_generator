import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PretrainedConfig
from dataset import ConditionalMNISTDataset

############################
#      Config Class        #
############################
class ConvConfig(PretrainedConfig):
    model_type = "conv_generator"
    def __init__(self, latent_dim=100, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

############################
#      Model Class         #
############################
class ConvGeneratorModel(PreTrainedModel):
    config_class = ConvConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.label_emb = nn.Embedding(10, config.latent_dim)
        
        # Starting from 1x1, we need to get to 28x28
        # 1x1 -> 4x4 -> 8x8 -> 16x16 -> 28x28
        self.net = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(config.latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 16x16 -> 28x28
            nn.ConvTranspose2d(64, 1, 13, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.post_init()

    def forward(self, labels):
        emb = self.label_emb(labels)                 # (bsz, latent_dim)
        noise = torch.randn_like(emb)               # (bsz, latent_dim)
        z = noise + emb
        z = z.unsqueeze(-1).unsqueeze(-1)           # (bsz, latent_dim, 1, 1)
        out = self.net(z)                           # (bsz, 1, 28, 28)
        return out

def main():
    # Ensure MPS is available
    if not torch.backends.mps.is_available():
        print("MPS not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = "mps"
    print(f"Using device: {device}")

    dataset = ConditionalMNISTDataset("train")
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)  # Reduced workers for MPS

    config = ConvConfig(latent_dim=100)
    model = ConvGeneratorModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 1 # does perfectly
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for step, (x, y) in enumerate(loader, 1):
                # Move both inputs to device
                x = x.to(device)
                y = y.to(device)
                
                labels = (x[:,0] - 256).clamp(min=0, max=9)  # shape (bsz,)
                real_images = y.view(-1, 1, 28, 28).float() / 255.0     # Normalize to [0,1]

                optimizer.zero_grad()
                generated_images = model(labels)                      # (bsz, 1, 28, 28)

                # Use Mean Squared Error loss
                loss = F.mse_loss(generated_images, real_images)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
                print(f"\rEpoch {epoch+1}, Step [{step}/{len(loader)}], Loss: {loss.item():.4f}", end='')
            avg_loss = total_loss / len(loader)
            print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")
    except KeyboardInterrupt:
        model.save_pretrained("my_conv")
        config.save_pretrained("my_conv")
        print("\nEmergency save. Model saved to my_conv/")
        return

    model.save_pretrained("my_conv")
    config.save_pretrained("my_conv")
    print("Conv model saved to my_conv/")

if __name__ == "__main__":
    main()