import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
from config import ModelConfig
import logging

logger = logging.getLogger(__name__)


class CasualSelfAttention(nn.Module):
    ## QKV
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = nn.Linear(config.d_model, 3 * config.d_model)
        self.project_out = nn.Linear(config.d_model, config.d_model)
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attention
        )
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.seq_len, config.seq_len)).view(
                    1, 1, config.seq_len, config.seq_len
                ),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.attention(x).split(C, dim=2)

        # Reshape for multi-head attention.  n_head * d_head = d_model
        q = q.view(B, T, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (B, n_heads, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (B, n_heads, T, d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (B, n_heads, T, d_head)

        if self.flash:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            atten = (q @ k.transpose(-2, -1)) / math.sqrt(
                self.d_head
            )  # (B, n_heads, T, T)
            atten = atten.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            atten = F.softmax(atten, dim=-1)  # (B, n_heads, T, T)
            y = atten @ v  # (B, n_heads, T, d_head)
        y = attn.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)
        return self.project_out(y)  # (B, T, d_model)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = CasualSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ChordGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.position_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model),
                wpe=nn.Embedding(config.seq_len, config.d_model),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layers)]
                ),
                drop=nn.Dropout(config.dropout),
                ln_f=nn.LayerNorm(config.d_model),
            )
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # weight tying -> perfomance improment
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if p.dim() > 1 and pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        device = x.device
        B, T = x.size()

        assert (
            T <= self.config.seq_len
        ), f"Sequence length {t} exceeds block size {self.config.seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)  # shape (t)
        # Token embedding + pos embedding
        token_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(token_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # Training: Calculate Loss
            logits = self.lm_head(x)

            # Use CrossEntropyLoss (handles class imbalances and ignored padding)
            # Flatten: (B*T, Vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            return logits, loss
        else:
            # Inference: Return logits only for the last step usually, but here we return all
            logits = self.lm_head(x)
            return logits, None


def test_model():
    print("--- Starting Local Sanity Check ---")

    # 1. Config
    config = ModelConfig(vocab_size=1035, seq_len=256)
    print(f"Model Config: {config}")

    # 2. Model
    try:
        model = ChordGPT(config)
        summary(model, input_size=(4, 256), dtypes=[torch.long])
        print(
            f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
        )
    except Exception as e:
        print(f"Failed to create model: {e}")
        raise e

    # 3. Data (Dummy batch for speed)
    # We simulate a batch of data (Batch Size=4, Seq Len=256)
    x = torch.randint(0, 1035, (4, 256))
    y = torch.randint(0, 1035, (4, 256))

    # 4. Forward Pass
    if torch.cuda.is_available():
        print("Moving to GPU...")
        model = model.cuda()
        x = x.cuda()
        y = y.cuda()

    logits, loss = model(x, y)
    print(f"Output Logits Shape: {logits.shape}")  # Should be [4, 256, 1035]
    print(f"Initial Loss: {loss.item()}")

    print("--- Sanity Check Passed ✅ ---")


if __name__ == "__main__":
    test_model()
