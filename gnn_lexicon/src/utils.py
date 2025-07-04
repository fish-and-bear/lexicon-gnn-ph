"""
Utility functions for Philippine Lexicon GNN.
Includes char-CNN embedding, config loading, normalization.
"""

from typing import List, Dict, Any
import torch
import torch.nn as nn
import yaml

class CharCNNEmbedder(nn.Module):
    """Character-level CNN embedder for string features."""
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(256, 32)
        self.conv = nn.Conv1d(32, out_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.conv.out_channels))
        
        max_len = max(len(t) for t in texts) if texts else 1
        max_len = max(max_len, 3)  # Ensure minimum length for conv
        
        ids = torch.zeros((len(texts), max_len), dtype=torch.long)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:max_len]):
                ids[i, j] = ord(c) % 256  # Ensure within embedding range
        
        x = self.embed(ids)  # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        x = self.conv(x)  # [batch, out_dim, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch, out_dim]
        return x

# Global embedder instance
_char_cnn = None

def get_char_cnn():
    """Get or create the global char CNN embedder."""
    global _char_cnn
    if _char_cnn is None:
        _char_cnn = CharCNNEmbedder()
    return _char_cnn

def char_cnn_embed(texts: List[str]) -> torch.Tensor:
    """Embeds a list of strings using a char-level CNN."""
    embedder = get_char_cnn()
    with torch.no_grad():
        return embedder(texts)

def normalize_numeric(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalizes a numeric value to [0,1]."""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (val - min_val) / (max_val - min_val)))

def load_config(path: str) -> Dict[str, Any]:
    """Loads a YAML config file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[utils] Config file {path} not found. Using defaults.")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Returns default configuration."""
    return {
        "in_dim": 64,
        "hidden_dim": 128,
        "out_dim": 64,
        "num_layers": 2,
        "heads": 4,
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 50,
        "grad_clip": 1.0,
        "early_stopping_patience": 10,
        "num_relations": 8,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "postgres": {
            "dbname": "fil_dict_db",
            "user": "postgres",
            "password": "postgres",
            "host": "localhost",
            "port": 5432
        },
        "model_path": "model.pt"
    }

def save_model(model: nn.Module, path: str) -> None:
    """Save model state dict."""
    torch.save(model.state_dict(), path)
    print(f"[utils] Model saved to {path}")

def load_model(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """Load model state dict."""
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"[utils] Model loaded from {path}")
    return model 