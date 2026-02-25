import os
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import logging

from config import ModelConfig, TrainingConfig
from model import ChordGPT
from dataset import ChordDataset
from monitor import GPUMonitor
from utils import cycle

# Import the new wrapper
from distributed_utils import AsyncDDP

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend=TrainingConfig.backend)
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        print(f"Distributed Init: Rank {rank}/{world_size}, Local Rank {local_rank}")
        return rank, local_rank, world_size, True
    else:
        logger.info("Not using DDP. Running on single GPU.")
        return 0, 0, 1, False


def get_dataloaders(
    data_path, batch_size, seq_len, rank, world_size, is_ddp, val_ratio=0.01
):
    full_ds = ChordDataset(data_path, seq_len)

    n = len(full_ds)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    g = torch.Generator().manual_seed(1337)  # stable split
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    if is_ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, train_sampler


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total = 0.0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(1, n)


def train():
    # 1. Setup
    rank, local_rank, world_size, is_ddp = setup_distributed()
    master_process = rank == 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    monitor = GPUMonitor()
    if master_process:
        os.makedirs("runs", exist_ok=True)
        os.makedirs("checkPoints", exist_ok=True)
        writer = SummaryWriter(log_dir=f"runs/chord_exp_v1")

    config = ModelConfig()
    train_config = TrainingConfig()

    # 2. Create Model
    model = ChordGPT(config).to(device)

    # 3. Wrap in AsyncDDP
    if is_ddp:
        model = AsyncDDP(model, device_id=local_rank)
        raw_model = model.module
    else:
        raw_model = model
        # Dummy sync method for single GPU so code doesn't break
        model.synchronize = lambda: None

    # 4. Optimizer (on raw_model parameters)
    param_dict = {pn: p for pn, p in raw_model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr)

    # 5. Data Iterator (Infinite)
    train_loader, val_loader, sampler = get_dataloaders(
        "../Data/tokenized_data_clean.txt",
        train_config.batch_size,
        config.seq_len,
        rank,
        world_size,
        is_ddp,
        val_ratio=0.01,
    )
    train_iter = cycle(train_loader)

    if master_process:
        logger.info(f"Starting training on {device}...")

    model.train()
    step = 0
    t0 = time.time()

    # --- TRAINING LOOP ---
    while step < train_config.max_steps:
        # Get Batch
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # Shuffle logic for DDP
        if is_ddp and (step % len(train_loader) == 0):
            sampler.set_epoch(step)

        # Forward
        logits, loss = model(x, y)

        # Backward
        optimizer.zero_grad()
        # Compute gradients, hooks are fired to sync asynchronously in the background while we do other CPU work.
        loss.backward()

        # --- CRITICAL: WAIT FOR SYNC ---
        # We explicitly wait for all async handles to finish before stepping.
        model.synchronize()

        # Clip & Update
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % train_config.log_every == 0 and master_process:
            # Evaluate on validation set
            val_loss = evaluate(raw_model, val_loader, device)
            writer.add_scalar("Val/Loss", val_loss, step)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            tokens_processed = train_config.batch_size * config.seq_len * world_size
            throughput = tokens_processed / dt
            gpu_stats = monitor.get_stats()

            print(
                f"Step {step} | Loss: {loss.item():.4f} | Speed: {throughput:.2f} tok/s"
            )

            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/Throughput", throughput, step)
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], step)

            for g in gpu_stats:
                idx = g["id"]
                writer.add_scalar(f"System/GPU_{idx}_Util", g["util_pct"], step)
                writer.add_scalar(f"System/GPU_{idx}_Temp", g["temp_c"], step)
                writer.add_scalar(f"System/GPU_{idx}_MemUsedGB", g["mem_used_gb"], step)
                writer.add_scalar(
                    f"System/GPU_{idx}_MemInTotal", g["mem_total_gb"], step
                )

        # Checkpointing
        if step > 0 and step % train_config.save_every == 0 and master_process:
            checkpoint_path = f"checkPoints/checkpoint_step_{step}.pt"
            torch.save(raw_model.state_dict(), checkpoint_path)

        step += 1

    if master_process:
        torch.save(raw_model.state_dict(), "final_model.pt")
        writer.close()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
