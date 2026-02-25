import torch
import torch.nn as nn
from config import ModelConfig, TrainingConfig
from model import ChordGPT


def estimate_memory():
    # 1. Load Configs
    model_config = ModelConfig()
    train_config = TrainingConfig()

    # 2. Precision Settings (Bytes per number)
    # Standard PyTorch default is FP32 (4 bytes)
    # Industrial training often uses Mixed Precision (Weights=4, Grads=4, Opt=8, Activations=2)
    # Let's assume standard FP32 for now as per your current code.
    BYTES_PER_PARAM = 4

    print(f"--- Memory Estimation for ChordGPT ---")
    print(
        f"Config: Layers={model_config.n_layers}, Dim={model_config.d_model}, "
        f"Heads={model_config.n_heads}, SeqLen={model_config.seq_len}"
    )
    print(f"Training: Batch={train_config.batch_size}, Optimizer=AdamW")

    # 3. Calculate Exact Parameter Count using 'Meta' Device
    # This creates the model structure without allocating real memory
    with torch.device("meta"):
        model = ChordGPT(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[1] Model Size")
    print(f"    Total Parameters: {total_params:,}")

    # 4. Calculate Static Memory (Weights + Gradients + Optimizer)
    # Weights: 4 bytes (FP32)
    # Gradients: 4 bytes (FP32)
    # AdamW States: 8 bytes (Momentum + Variance, FP32)
    # Total per param: 16 bytes

    static_mem_bytes = trainable_params * (4 + 4 + 8)
    static_mem_gb = static_mem_bytes / 1e9

    print(f"\n[2] Static Memory (Fixed)")
    print(f"    Weights + Grads + AdamW States: {static_mem_gb:.2f} GB")

    # 5. Estimate Activation Memory (Dynamic)
    # Formula approximation for Transformer Activations (standard attention):
    # Memory ≈ Batch * Seq * Hidden * Layers * (Multi-Head Attn + MLP Overhead)
    # A safe industrial rule of thumb for standard training is:
    # Bytes ≈ Batch * Seq * Hidden * Layers * 12 (roughly)

    # Note: Flash Attention reduces this significantly, but let's estimate worst-case.
    activation_factor = 12  # Empirical factor for Forward+Backward storage

    act_mem_bytes = (
        train_config.batch_size
        * model_config.seq_len
        * model_config.d_model
        * model_config.n_layers
        * activation_factor
    )

    act_mem_gb = act_mem_bytes / 1e9

    print(f"\n[3] Activation Memory (Dynamic)")
    print(f"    Batch Size {train_config.batch_size}: ~{act_mem_gb:.2f} GB")

    # 6. CUDA Context Overhead
    # PyTorch kernels take up ~500MB - 1GB just by turning on.
    overhead_gb = 0.8

    total_gb = static_mem_gb + act_mem_gb + overhead_gb

    print(f"\n[4] Total Estimated VRAM")
    print(f"    ---------------------------")
    print(f"    Total: ~{total_gb:.2f} GB")
    print(f"    ---------------------------")

    # Recommendation
    if total_gb > 8.0:
        print("    ⚠️  WARNING: Exceeds 8GB (RTX 2080 Super). Reduce Batch Size!")
    elif total_gb > 24.0:
        print("    ⚠️  WARNING: Exceeds 24GB (RTX 3090/4090). Use DDP or TP!")
    else:
        print("    ✅  Fits comfortably.")


if __name__ == "__main__":
    estimate_memory()
