import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint



class SpatialAttn(nn.Module):
    """Softmax self-attention over spatial positions (for stages with N <= ~4k)."""
    def __init__(self, ch, heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(ch//4, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.heads = heads
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = [t.view(B, self.heads, C // self.heads, H * W).transpose(-1, -2) for t in (q, k, v)]
        o = F.scaled_dot_product_attention(q, k, v)          # (B, h, N, C/h)
        o = o.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(o)

class ChannelAttn(nn.Module):
    """Restormer-style transposed attention: C x C affinity, O(N*C^2). For mid-res stages."""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(ch//4, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.dw = nn.Conv2d(3 * ch, 3 * ch, 3, padding=1, groups=3 * ch)  # local mixing
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.heads = heads
        self.temp = nn.Parameter(torch.ones(heads, 1, 1))
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.dw(self.qkv(self.norm(x))).chunk(3, dim=1)
        q, k, v = [t.view(B, self.heads, C // self.heads, H * W) for t in (q, k, v)]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        a = (q @ k.transpose(-1, -2)) * self.temp                 # (B, h, C/h, C/h)
        o = (a.softmax(-1) @ v).reshape(B, C, H, W)
        return x + self.proj(o)

class Ckpt(nn.Module):
    """Wrap a block so its activations are recomputed in backward (saves memory)."""
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, x):
        if self.training and x.requires_grad:
            return checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)

class FFN(nn.Module):
    def __init__(self, ch, mult=2):
        super().__init__()
        self.norm = nn.GroupNorm(ch//4, ch)
        self.net = nn.Sequential(nn.Conv2d(ch, mult * ch * 2, 1), nn.GLU(dim=1),
                                 nn.Conv2d(mult * ch, mult * ch, 3, padding=1, groups=mult * ch),
                                 nn.SiLU(), nn.Conv2d(mult * ch, ch, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x):
        return x + self.net(self.norm(x))

def attn_block(ch, tokens):
    """Pick the attention type by token count of the stage; None for high-res stages."""
    if tokens <= 256:
        return Ckpt(nn.Sequential(SpatialAttn(ch), FFN(ch)))
    if tokens <= 1024:
        return Ckpt(nn.Sequential(ChannelAttn(ch), FFN(ch)))
    return nn.Identity()
