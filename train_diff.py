import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from tqdm import tqdm

def train_diffusion():
    # Train and save a DDPM diffusion model on MNIST.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    # Conditional DDPM UNet for MNIST digits
    unet = UNet2DModel(
        sample_size=28,
        in_channels=1,
        out_channels=1,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=10,
    ).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler).to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-4) # changed from Adam

    epochs = 5
    print(f"Training DDPM for {epochs} epochs...")
    try:
        for epoch in range(1, epochs + 1):
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                noise = torch.randn_like(images)
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps, (images.shape[0],), device=device
                ).long()
                noisy = scheduler.add_noise(images, noise, timesteps)

                # Conditional noise prediction
                model_pred = unet(noisy, timesteps, class_labels=labels, return_dict=False)[0]
                loss = F.mse_loss(model_pred, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt, saving model...")
        output_dir = "my_diffusion_model"
        pipeline.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}/")
        return pipeline

    output_dir = "my_diffusion_model"
    pipeline.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}/")
    return pipeline

if __name__ == "__main__":
    train_diffusion()