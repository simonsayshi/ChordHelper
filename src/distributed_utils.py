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

        world_size = dist.get_world_size()

        # Wait for every single async handle to finish
        for handle, param in self.async_handles:
            handle.wait()  # <--- The CPU blocks here!
            
            # Now that we know the sum is ready, we average it
            param.grad /= world_size
            
        # Clear handles for the next step
        self.async_handles = []

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)