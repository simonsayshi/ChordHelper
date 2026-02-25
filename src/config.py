from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the model."""

    vocab_size: int = 1035  # 1035 unique characters in the dataset
    seq_len: int = 256  # maximum sequence length for input and output
    d_model: int = 512  # dimension of the model
    n_layers: int = 6  # number of transformer layers
    n_heads: int = 8  # number of attention heads
    dropout: float = 0.1
    # for distributed training
    flash_attention: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 32  # Batch size per GPU
    lr: float = 3e-4
    max_steps: int = 3000  # Total training steps
    log_every: int = 10  # How often to log metrics
    save_every: int = 100  # Checkpoint interval
    gradient_accumulation_steps: int = 1
    backend: str = "nccl"
