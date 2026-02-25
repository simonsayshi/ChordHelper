import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Get World Size
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # 2. Split Output Dimension
        assert out_features % world_size == 0, "Output features must be divisible by world size"
        self.output_per_partition = out_features // world_size
        
        # 3. Create Partial Weights
        # Shape: [Out_Partition, In]
        self.weight = nn.Parameter(torch.Tensor(self.output_per_partition, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_per_partition))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Standard init, but careful to scale correctly if needed
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        # Input: [Batch, Seq, In]
        # Weight: [Out_Part, In]
        # Output: [Batch, Seq, Out_Part]
        
        # Local Matmul (No communication needed yet!)
        output = F.linear(input, self.weight, self.bias)
        
        # The output is "Sharded". GPU 0 has the first half of the vector, GPU 1 has the second.
        # We do NOT gather here. We let the next layer handle it.
        return output

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # 1. Split Input Dimension
        assert in_features % world_size == 0, "Input features must be divisible by world size"
        self.input_per_partition = in_features // world_size
        
        # 2. Create Partial Weights
        # Shape: [Out, In_Partition]
        self.weight = nn.Parameter(torch.Tensor(out_features, self.input_per_partition))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        # Input is Sharded: [Batch, Seq, In_Part]
        # Weight: [Out, In_Part]
        
        # Local Matmul
        # Output: [Batch, Seq, Out] (Partial Sum)
        output = F.linear(input, self.weight)
        
        # ALL-REDUCE (Sum across GPUs)
        if dist.is_initialized():
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            
        # Add bias (Bias is replicated, so we add it AFTER reduction)
        if self.bias is not None:
            output = output + self.bias
            
        return output