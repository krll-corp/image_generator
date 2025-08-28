import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import ConditionalMNISTDataset
import os
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import lion_pytorch

# -------------------------------------------------------------------
# Config classes
# -------------------------------------------------------------------
@dataclass
class PixelTransformerConfig:
    vocab_size: int = 10  # for MNIST digits
    image_size: int = 28
    n_layers: int = 8
    d_model: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    max_position_embeddings: int = 28 * 28
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 10
    warmup_steps: int = 500
    device: str = field(default_factory=lambda: "mps")  # Default to CPU, set device explicitly later

    @classmethod
    def from_pretrained(cls, path: str):
        config_path = os.path.join(path, "config.pt")
        if not os.path.exists(config_path):
            raise ValueError(f"No config found at {config_path}")
        config_dict = torch.load(config_path, weights_only=False)
        return cls(**config_dict)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, "config.pt")
        torch.save(self.__dict__, config_path)


# -------------------------------------------------------------------
# Transformer building blocks
# -------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, config: PixelTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("mask", torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings))
                             .view(1,1, config.max_position_embeddings, config.max_position_embeddings))

    def forward(self, x):
        B, Seq, D = x.shape
        qkv = self.qkv(x)  # (B, Seq, 3*d_model)
        q, k, v = qkv.split(D, dim=-1)

        # reshape for multi-head
        q = q.view(B, Seq, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(B, Seq, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, Seq, self.n_heads, self.head_dim).transpose(1,2)

        # scaled dot-product
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(self.mask[:,:,:Seq,:Seq] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1,2).contiguous().view(B, Seq, D)
        out = self.o_proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: PixelTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = SelfAttention(config)
        self.dropout1 = nn.Dropout(config.dropout)

        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4*config.d_model),
            nn.GELU(),
            nn.Linear(4*config.d_model, config.d_model)
        )
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x):
        a = self.ln1(x)
        x = x + self.dropout1(self.attn(a))
        m = self.ln2(x)
        x = x + self.dropout2(self.mlp(m))
        return x


# -------------------------------------------------------------------
# Full PixelTransformer model
# -------------------------------------------------------------------
class PixelTransformer(nn.Module):
    def __init__(self, config: PixelTransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.d_model))

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, 10)  # 10 discrete bins

    def forward(self, x):
        B, Seq = x.shape
        token_emb = self.embedding(x)
        pos_emb = self.pos_embedding[:, :Seq, :]
        h = token_emb + pos_emb

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.output_head(h)  # (B, Seq, 10)
        return logits

    def generate_digit_stream(self, digit: int):
        """Generate a stream of pixels for a given digit."""
        self.eval()
        device = next(self.parameters()).device  # Get actual device from model parameters
        
        # Initialize sequence with digit
        seq = torch.tensor([digit], dtype=torch.long, device=device)
        
        for _ in range(1, self.config.image_size * self.config.image_size + 1):
            # Forward pass
            x_in = seq.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                logits = self.forward(x_in)
                
            # Get next token probabilities
            next_token_logits = logits[0, -1, :]  # Last position
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            seq = torch.cat([seq, next_token])
            
            # Yield the next pixel value
            yield next_token.cpu().item()

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[PixelTransformerConfig] = None,
        device: str = "cpu",
    ):
        """Load a pretrained model on a given device (default CPU).

        The original training configuration stores the device it was trained on
        (often ``mps`` when trained on a Mac).  Loading such checkpoints on a
        machine without MPS support would previously fail.  By always loading
        the state dictionary on CPU and explicitly moving the model to the
        requested device we make the checkpoint portable across devices.
        """
        if config is None:
            config = PixelTransformerConfig.from_pretrained(path)

        # Ensure the config reflects the actual runtime device
        config.device = device

        # Create model and load state dict on CPU
        model = cls(config)
        state_dict = torch.load(
            os.path.join(path, "model.pt"), map_location="cpu", weights_only=False
        )
        model.load_state_dict(state_dict)

        # Move model to the desired device
        model = model.to(device)
        return model

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save model to CPU first
        cpu_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(cpu_state_dict, os.path.join(path, "model.pt"))
        self.config.save_pretrained(path)


# -------------------------------------------------------------------
# Training Code
# -------------------------------------------------------------------
def train_pixel_transformer(config: PixelTransformerConfig):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = PixelTransformer(config).to(config.device)
    #optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    optimizer = lion_pytorch.Lion(model.parameters(), lr=config.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Simple linear warmup + decay
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
            for i, (imgs, labels) in enumerate(pbar):
                imgs = imgs.to(config.device)
                # Discretize into 10 bins
                imgs_discrete = torch.floor(imgs * 9).long().squeeze(1)
                B, H, W = imgs_discrete.shape
                imgs_discrete = imgs_discrete.view(B, H*W)

                logits = model(imgs_discrete[:, :-1])
                targets = imgs_discrete[:, 1:].contiguous()

                logits = logits.view(-1, 10)
                targets = targets.view(-1)

                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                global_step += 1
    except KeyboardInterrupt:
        print("\nEmergency save triggered by keyboard interrupt...")
        model.save_pretrained("my_model")
        print("Model saved to my_model/")
        return model

    model.save_pretrained("my_model")
    return model


if __name__ == "__main__":
    config = PixelTransformerConfig(
        epochs=1,
        n_layers=8,
        d_model=256,
        batch_size=4, #16 #64
        dropout=0.1,
        lr=1e-3,
        warmup_steps=500,
    )
    model = train_pixel_transformer(config)
    model.save_pretrained("my_model")
