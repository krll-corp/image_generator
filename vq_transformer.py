import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from dataclasses import dataclass, field

@dataclass
class VQTransformerConfig:
    vocab_size: int = 512  # VQ-VAE codebook size
    token_grid_size: int = 7  # 7x7 grid of tokens
    n_layers: int = 8
    d_model: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    max_position_embeddings: int = 50  # 1 label token + 49 (7x7) image tokens
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 10
    warmup_steps: int = 500
    label_offset: int = 512  # Labels are tokens 512-521 (for digits 0-9)
    device: str = field(default="mps" if torch.backends.mps.is_available() else "cpu")
    
    @classmethod
    def from_pretrained(cls, path: str):
        config_path = os.path.join(path, "config.pt")
        if not os.path.exists(config_path):
            raise ValueError(f"No config found at {config_path}")
        config_dict = torch.load(config_path)
        return cls(**config_dict)
    
    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, "config.pt")
        torch.save(self.__dict__, config_path)


class SelfAttention(nn.Module):
    def __init__(self, config: VQTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.max_position_embeddings, 
                                                         config.max_position_embeddings))
                           .view(1, 1, config.max_position_embeddings, config.max_position_embeddings))

    def forward(self, x):
        B, Seq, D = x.shape
        qkv = self.qkv(x)  # (B, Seq, 3*d_model)
        q, k, v = qkv.split(D, dim=-1)

        # reshape for multi-head
        q = q.view(B, Seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Seq, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, Seq, Seq)

        # causal mask
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :Seq, :Seq] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, n_heads, Seq, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Seq, D)
        out = self.o_proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: VQTransformerConfig):
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


class VQTransformer(nn.Module):
    def __init__(self, config: VQTransformerConfig):
        super().__init__()
        self.config = config
        # Embed tokens from [0, 511] for image tokens plus [512, 521] for label tokens
        self.embedding = nn.Embedding(config.vocab_size + 10, config.d_model)  # 512 + 10 for labels
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.d_model))

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)  # Only predict image tokens

    def forward(self, x):
        B, Seq = x.shape
        token_emb = self.embedding(x)
        pos_emb = self.pos_embedding[:, :Seq, :]
        h = token_emb + pos_emb

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.output_head(h)  # (B, Seq, vocab_size)
        return logits

    def generate(self, label_digit, vq_model, device='cpu'):
        """Generate tokens conditioned on a digit, then decode to an image."""
        self.eval()
        
        # Start with label token
        label_token = self.config.label_offset + label_digit
        seq = torch.tensor([label_token], dtype=torch.long, device=device)
        
        # Generate 49 image tokens (7x7 grid)
        for _ in range(49):  # 7x7 = 49 tokens
            x_in = seq.unsqueeze(0)  # Add batch dim
            with torch.no_grad():
                logits = self.forward(x_in)
            
            # Sample next token
            next_token_logits = logits[0, -1, :]  # Last position 
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            seq = torch.cat([seq, next_token])
        
        # Reshape tokens to 7x7 grid and decode using VQ-VAE
        image_tokens = seq[1:].reshape(1, 7, 7)  # Skip label token
        with torch.no_grad():
            generated_img = vq_model.decode(image_tokens)
            
        return generated_img

    def generate_token_stream(self, digit, device='cpu'):
        """Generator function: yields a token at a time for UI streaming."""
        self.eval()
        
        # Start with label token
        label_token = self.config.label_offset + digit
        seq = torch.tensor([label_token], dtype=torch.long, device=device)
        
        for _ in range(49):  # Generate 7x7 = 49 tokens
            x_in = seq.unsqueeze(0)  # Add batch dim
            with torch.no_grad():
                logits = self.forward(x_in)
                
            # Get next token probabilities
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample token
            next_token = torch.multinomial(probs, num_samples=1)
            token_value = next_token.item()
            
            # Append to sequence and yield
            seq = torch.cat([seq, next_token])
            yield token_value
    
    @classmethod
    def from_pretrained(cls, path: str, config=None):
        if config is None:
            config = VQTransformerConfig.from_pretrained(path)
        model = cls(config)
        model_path = os.path.join(path, "model.pt")
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.to(config.device)
        return model

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model.pt")
        torch.save(self.state_dict(), model_path)
        self.config.save_pretrained(path)
