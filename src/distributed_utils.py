from time import time

import torch
import torch.nn as nn
import torch.distributed as dist


class AsyncDDP(nn.Module):
    def __init__(self, module, device_id):
        super().__init__()
        self.module = module
        self.device_id = device_id

        # We store handles to the async communication tasks here
        self.async_handles = []

        # ---- stats for TensorBoard ----
        self._allreduce_calls = 0
        self._allreduce_bytes = 0
        self._last_sync_wait_ms = 0.0

        # 1. Broadcast parameters to ensure all GPUs start with the exact same weights
        self._broadcast_params()

        # 2. Register the Async Hooks
        self._register_hooks()

    def _broadcast_params(self):
        """
        Broadcasts the model parameters from rank 0 to all other processes.
        """
        if not dist.is_initialized():
            return

        # print(f"[AsyncDDP] Broadcasting initial parameters from Rank 0...")
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def _register_hooks(self):
        """
        Registers a post_accumulate_grad_hook on every parameter.
        """
        for p in self.module.parameters():
            if p.requires_grad:
                # This hook fires exactly when the gradient is calculated and accumulated.
                p.register_post_accumulate_grad_hook(self._make_async_hook())

    def _make_async_hook(self):
        def hook(param):
            if not dist.is_initialized():
                return

            if param.grad is None:
                return

            # --- metrics ---
            self._allreduce_calls += 1
            self._allreduce_bytes += param.grad.numel() * param.grad.element_size()
            # --- THE PICOTRON / INDUSTRIAL METHOD ---
            # 1. Fire the All-Reduce asynchronously.
            #    The CPU sends the command and IMMEDIATELY continues.
            #    It does NOT wait for the GPU to finish.
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)

            # 2. Store the handle and the param so we can finish it later
            self.async_handles.append((handle, param))

        return hook

    def synchronize(self):
        """
        BLOCKS the CPU until all async gradients are synced.
        Must be called after loss.backward() and before optimizer.step().
        """
        if not dist.is_initialized():
            return
        t0 = time.perf_counter()
        world_size = dist.get_world_size()

        # Wait for every single async handle to finish
        for handle, param in self.async_handles:
            handle.wait()  # <--- The CPU blocks here!

            # Now that we know the sum is ready, we average it
            param.grad /= world_size

        # Clear handles for the next step
        self.async_handles = []
        t1 = time.perf_counter()
        self._last_sync_wait_ms = (t1 - t0) * 1000.0

    def get_and_reset_comm_stats(self):
        """
        Return per-step comm stats, then reset counters.
        Call once per training step (after synchronize()).
        """
        stats = {
            "allreduce_calls": int(self._allreduce_calls),
            "allreduce_bytes": int(self._allreduce_bytes),
            "sync_wait_ms": float(self._last_sync_wait_ms),
        }
        self._allreduce_calls = 0
        self._allreduce_bytes = 0
        self._last_sync_wait_ms = 0.0
        return stats

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class BucketDDP(nn.Module):
    """
    - Pack grads into flat buffers (buckets)
    - Launch async all_reduce per bucket when all grads in that bucket are ready
    - synchronize(): wait bucket handles, average, then unpack back to param.grad

    This reduces all_reduce call count drastically vs AsyncDDP.
    """
    def __init__(self, module, device_id, bucket_mb=25):
        super().__init__()
        self.module = module
        self.device_id = device_id
        self.bucket_bytes = int(bucket_mb * 1024 * 1024)

        # comm stats (per step)
        self._allreduce_calls = 0
        self._allreduce_bytes = 0
        self._last_sync_wait_ms = 0.0

        self._broadcast_params()

        # Build buckets once based on parameter sizes
        self._build_buckets()
        self._register_hooks()

    def _broadcast_params(self):
        if not dist.is_initialized():
            return
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def _build_buckets(self):
        # Each bucket stores:
        # - buffer: flat tensor
        # - entries: [(param, offset, numel), ...]
        # - ready_count, total_count
        # - handle (async all_reduce)
        params = [p for p in self.module.parameters() if p.requires_grad]

        self.buckets = []
        cur_entries = []
        cur_bytes = 0
        cur_numel = 0
        dtype = None
        device = None

        def flush_bucket():
            nonlocal cur_entries, cur_bytes, cur_numel, dtype, device
            if not cur_entries:
                return
            buf = torch.empty(cur_numel, device=device, dtype=dtype)
            b = {
                "buffer": buf,
                "entries": cur_entries,
                "ready_count": 0,
                "total_count": len(cur_entries),
                "handle": None,
            }
            self.buckets.append(b)
            cur_entries = []
            cur_bytes = 0
            cur_numel = 0
            dtype = None
            device = None

        for p in params:
            # assume grads will have same dtype/device as param
            p_bytes = p.numel() * p.element_size()

            # Start a new bucket if:
            # - empty -> initialize dtype/device
            # - dtype/device mismatch
            # - would exceed bucket size
            if not cur_entries:
                dtype = p.dtype
                device = p.device

            if (p.dtype != dtype) or (p.device != device) or (cur_bytes + p_bytes > self.bucket_bytes):
                flush_bucket()
                dtype = p.dtype
                device = p.device

            offset = cur_numel
            numel = p.numel()
            cur_entries.append((p, offset, numel))
            cur_numel += numel
            cur_bytes += p_bytes

        flush_bucket()

        # map param -> (bucket_id, offset, numel)
        self.param_to_bucket = {}
        for bi, b in enumerate(self.buckets):
            for (p, off, n) in b["entries"]:
                self.param_to_bucket[p] = (bi, off, n)

    def _register_hooks(self):
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_bucket_hook(p))

    def _make_bucket_hook(self, p):
        def hook(param):
            if (not dist.is_initialized()) or (param.grad is None):
                return

            bi, off, n = self.param_to_bucket[param]
            b = self.buckets[bi]

            # Pack grad into bucket buffer
            b["buffer"][off : off + n].copy_(param.grad.view(-1))

            # Mark ready; if bucket complete, launch one async all_reduce
            b["ready_count"] += 1
            if b["ready_count"] == b["total_count"]:
                self._allreduce_calls += 1
                self._allreduce_bytes += b["buffer"].numel() * b["buffer"].element_size()
                b["handle"] = dist.all_reduce(b["buffer"], op=dist.ReduceOp.SUM, async_op=True)

        return hook

    def synchronize(self):
        if not dist.is_initialized():
            return
        t0 = time.perf_counter()
        world = dist.get_world_size()

        # Wait each bucket reduce and unpack back to param.grad
        for b in self.buckets:
            h = b["handle"]
            if h is None:
                # If you ever skip backward on some params, you might see this.
                # For your current model you should normally not.
                continue
            h.wait()
            b["buffer"] /= world

            # Unpack back into param.grad
            for (p, off, n) in b["entries"]:
                if p.grad is None:
                    continue
                p.grad.view(-1).copy_(b["buffer"][off : off + n])

            # Reset per-step state
            b["handle"] = None
            b["ready_count"] = 0

        self._last_sync_wait_ms = (time.perf_counter() - t0) * 1000.0

    def get_and_reset_comm_stats(self):
        out = {
            "allreduce_calls": int(self._allreduce_calls),
            "allreduce_bytes": int(self._allreduce_bytes),
            "sync_wait_ms": float(self._last_sync_wait_ms),
        }
        self._allreduce_calls = 0
        self._allreduce_bytes = 0
        self._last_sync_wait_ms = 0.0
        return out

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)