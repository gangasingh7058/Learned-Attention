from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelArgs:
    d_model: int = 256
    vocab_size: int = -1 # Set While Training
    max_Seq_len: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    ffn_dim_multiplier : int = 1
    multiple_of : int = 32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    scale_sigmoid_factor: float = 20.0