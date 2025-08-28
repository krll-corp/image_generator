import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ConditionalMNISTDataset
import os
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from torchvision import transforms
import torchvision.datasets as datasets
import lion_pytorch

# -------------------------------------------------------------------
# Config classes
# -------------------------------------------------------------------
@dataclass
class MoEPixelTransformerConfig:
    vocab_size: int = 10  # for MNIST digits
    image_size: int = 28
    n_layers: int = 8
    d_model: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    gating_dropout: float = 0.1
    expert_count: int = 4
    expert_capacity: int = 4
    max_position_embeddings: int = 28 * 28
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 10
    warmup_steps: int = 500
    device: str = field(default="mps" if torch.backends.mps.is_available() else "cpu")

    @classmethod
    def from_pretrained(cls, path: str):
        # placeholder logic for loading config
        # adapt to your real scenario
        config_path = os.path.join(path, "config.pt")
        if not os.path.exists(config_path):
            raise ValueError(f"No config found at {config_path}")
        config_dict = torch.load(config_path)
        return cls(**config_dict)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, "config.pt")
        torch.save(self.__dict__, config_path)


# -------------------------------------------------------------------
# Mixture-of-Experts Block
# -------------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    def __init__(self, d_model, expert_count, gating_dropout=0.1):
        super().__init__()
        self.expert_count = expert_count
        self.linear = nn.Linear(d_model, expert_count)
        self.dropout = nn.Dropout(gating_dropout)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, d_model)
        # gating logits
        logits = self.linear(hidden_states)  # (batch, seq_len, expert_count)
        logits = self.dropout(logits)
        return logits


class MoEBlock(nn.Module):
    def __init__(self, config: MoEPixelTransformerConfig):
        super().__init__()
        self.experts = nn.ModuleList([Expert(config.d_model) for _ in range(config.expert_count)])
        self.gating_network = GatingNetwork(config.d_model, config.expert_count, gating_dropout=config.gating_dropout)
        self.expert_count = config.expert_count

        self.layernorm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        # hidden_states shape: (B, Seq, d_model)
        normed = self.layernorm(hidden_states)
        gating_logits = self.gating_network(normed)  # (B, Seq, expert_count)

        # Softmax over experts
        gates = torch.softmax(gating_logits, dim=-1)  # (B, Seq, expert_count)

        # Weighted sum of experts
        # For each token, compute a weighted combination of expert outputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Extract gating for expert i
            gate_i = gates[..., i].unsqueeze(-1)  # (B, Seq, 1)
            # Expert forward
            exp_out = expert(normed)  # (B, Seq, d_model)
            # Weighted output
            expert_outputs.append(exp_out * gate_i)

        combined = torch.stack(expert_outputs, dim=-1).sum(dim=-1)  # (B, Seq, d_model)
        hidden_states = hidden_states + self.dropout(combined)       # Residual
        return hidden_states


# -------------------------------------------------------------------
# Self-Attention Block
# -------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, config: MoEPixelTransformerConfig):
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
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, Seq, Seq)

        # causal mask
        attn_scores = attn_scores.masked_fill(self.mask[:,:,:Seq,:Seq] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, n_heads, Seq, head_dim)
        out = out.transpose(1,2).contiguous().view(B, Seq, D)
        out = self.o_proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: MoEPixelTransformerConfig):
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
        # Self-attention
        a = self.ln1(x)
        x = x + self.dropout1(self.attn(a))
        # FFN
        m = self.ln2(x)
        x = x + self.dropout2(self.mlp(m))
        return x


# -------------------------------------------------------------------
# Full MoEPixelTransformer model
# -------------------------------------------------------------------
class MoEPixelTransformer(nn.Module):
    def __init__(self, config: MoEPixelTransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.d_model))

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Insert a MoE block in the middle or after all blocks
        self.moe_block = MoEBlock(config)

        self.ln_f = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, 10)  # Predict next pixel distribution from 10 discrete bins

    def forward(self, x):
        # x: (B, Seq) indices from 0..9 for each pixel
        B, Seq = x.shape
        token_emb = self.embedding(x)  # (B, Seq, d_model)
        position_emb = self.pos_embedding[:, :Seq, :]
        hidden_states = token_emb + position_emb

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        # MoE block after transformer
        hidden_states = self.moe_block(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        logits = self.output_head(hidden_states)  # (B, Seq, 10)
        return logits

    def generate_digit_stream(self, digit: int):
        """
        Generator function: yields a pixel at a time.
        Suppose you have a decode method or iterative sample method.
        """
        self.eval()
        # Start generation with empty or some special tokens
        seq = [digit]  # or any label conditioning
        for pos in range(1, self.config.image_size * self.config.image_size + 1):
            x_in = torch.tensor([seq], dtype=torch.long, device="cpu")  # shape (1, len(seq))
            with torch.no_grad():
                logits = self.forward(x_in)
            # Get the last position's distribution
            last_logits = logits[0, -1, :]  # shape (10,)
            probs = torch.softmax(last_logits, dim=-1)
            # Sample or argmax
            next_token = torch.multinomial(probs, num_samples=1).item()
            seq.append(next_token)
            yield next_token

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: MoEPixelTransformerConfig = None,
        device: str = "cpu",
    ):
        """Load a saved model onto the specified device (default CPU).

        Checkpoints saved on macOS may reference the ``mps`` device.  Loading
        them on environments without MPS support would previously raise
        ``torch.UntypedStorage`` errors.  We load weights on CPU first and then
        move the model to the requested device to ensure compatibility."""
        if config is None:
            config = MoEPixelTransformerConfig.from_pretrained(path)

        # Update config to reflect the runtime device
        config.device = device

        model_path = os.path.join(path, "model.pt")
        model = cls(config)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        return model

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model.pt")
        torch.save(self.state_dict(), model_path)
        self.config.save_pretrained(path)


# -------------------------------------------------------------------
# Training Code
# -------------------------------------------------------------------
def train_moe_pixel_transformer(config: MoEPixelTransformerConfig):
    # Simple MNIST-based dataset, discretize pixel values into 10 bins.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = MoEPixelTransformer(config).to(config.device)
    #print(f"Model params: ")
    #optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    optimizer = lion_pytorch.Lion(model.parameters(), lr=config.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Simple linear warmup + decay scheduler
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
                # Convert images to discrete bins 0..9
                # Pixel values are in [0,1], multiply by 9
                imgs = imgs.to(config.device)
                imgs_discrete = torch.floor(imgs * 9).long().squeeze(1)  # (B, 28, 28)
                # Flatten images
                B, H, W = imgs_discrete.shape
                seq_length = H * W
                imgs_discrete = imgs_discrete.view(B, seq_length)

                # Forward
                logits = model(imgs_discrete[:, :-1])  # predict next pixel
                # Targets are shifted by 1
                targets = imgs_discrete[:, 1:].contiguous()

                # Flatten
                logits = logits.view(-1, 10)   # shape (B*(seq_length-1), 10)
                targets = targets.view(-1)     # shape (B*(seq_length-1))

                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                global_step += 1
    except KeyboardInterrupt:
        print("\nEmergency save triggered by keyboard interrupt...")
        model.save_pretrained("my_moe_model")
        print("Model saved to my_moe_model/")
        return model

    model.save_pretrained("my_moe_model")
    return model


if __name__ == "__main__":
    config = MoEPixelTransformerConfig(
        epochs=1,
        n_layers=8, #4
        expert_count=64,
        expert_capacity=4, ###
        d_model=256,
        batch_size=4, #16 #64
        dropout=0.1,
        gating_dropout=0.1,
        lr=1e-3,
        warmup_steps=500,
    )
    model = train_moe_pixel_transformer(config)
    model.save_pretrained("my_moe_model")