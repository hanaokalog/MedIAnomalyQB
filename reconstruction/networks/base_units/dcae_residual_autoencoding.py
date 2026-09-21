"""
Minimal PyTorch codelet reproducing Residual Autoencoding from DC-AE
(Chen et al., "Deep Compression Autoencoder for Efficient High-Resolution
Diffusion Models", ICLR 2025).

Key idea:
  * Downsample stage: learned path   = Conv -> PixelUnshuffle
                      shortcut path  = PixelUnshuffle -> channel averaging (parameter-free)
  * Upsample stage:   learned path   = Conv -> PixelShuffle
                      shortcut path  = channel duplication -> PixelShuffle (parameter-free)
  Output of each stage = learned path + shortcut path.
  The shortcut is a pure space<->channel rearrangement that already provides a
  near-identity mapping, so the learned path only has to model the residual.
  This makes optimisation much easier under deep spatial compression and is
  the main reason DC-AE stays sharp at f32-f128.

Reference implementation: mit-han-lab/efficientvit, applications/dc_ae
  (nn/ops.py: ConvPixelUnshuffleDownSampleLayer,
              PixelUnshuffleChannelAveragingDownSampleLayer,
              ConvPixelShuffleUpSampleLayer,
              ChannelDuplicatingPixelUnshuffleUpSampleLayer)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Downsampling (encoder side)
# ---------------------------------------------------------------------------
class ConvPixelUnshuffleDown(nn.Module):
    """Learned path: Conv(in -> out/f^2) -> PixelUnshuffle(f)  => (out, H/f, W/f)"""

    def __init__(self, in_ch: int, out_ch: int, factor: int, kernel_size: int = 3):
        super().__init__()
        assert out_ch % (factor**2) == 0
        self.factor = factor
        self.conv = nn.Conv2d(in_ch, out_ch // factor**2, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(self.conv(x), self.factor)


class PixelUnshuffleChannelAvgDown(nn.Module):
    """Shortcut path (parameter-free): PixelUnshuffle(f) -> group-average channels down to out_ch"""

    def __init__(self, in_ch: int, out_ch: int, factor: int):
        super().__init__()
        assert (in_ch * factor**2) % out_ch == 0
        self.factor = factor
        self.group = in_ch * factor**2 // out_ch
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)             # (B, in*f^2, H/f, W/f)
        B, C, H, W = x.shape
        return x.view(B, self.out_ch, self.group, H, W).mean(dim=2)


class ResidualDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, factor: int = 2):
        super().__init__()
        self.main = ConvPixelUnshuffleDown(in_ch, out_ch, factor)
        self.shortcut = PixelUnshuffleChannelAvgDown(in_ch, out_ch, factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Upsampling (decoder side)
# ---------------------------------------------------------------------------
class ConvPixelShuffleUp(nn.Module):
    """Learned path: Conv(in -> out*f^2) -> PixelShuffle(f)  => (out, H*f, W*f)"""

    def __init__(self, in_ch: int, out_ch: int, factor: int, kernel_size: int = 3):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv2d(in_ch, out_ch * factor**2, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.conv(x), self.factor)


class ChannelDupPixelShuffleUp(nn.Module):
    """Shortcut path (parameter-free): duplicate channels up to out*f^2, then PixelShuffle(f)"""

    def __init__(self, in_ch: int, out_ch: int, factor: int):
        super().__init__()
        assert (out_ch * factor**2) % in_ch == 0
        self.factor = factor
        self.repeats = out_ch * factor**2 // in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)      # (B, out*f^2, H, W)
        return F.pixel_shuffle(x, self.factor)


class ResidualUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, factor: int = 2):
        super().__init__()
        self.main = ConvPixelShuffleUp(in_ch, out_ch, factor)
        self.shortcut = ChannelDupPixelShuffleUp(in_ch, out_ch, factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Per-stage body (swap freely: ResBlock / EfficientViT / attention blocks)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 32):
        super().__init__()
        g = min(groups, ch)
        self.body = nn.Sequential(
            nn.GroupNorm(g, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(g, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)  # start as identity so the shortcut dominates early on
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


# ---------------------------------------------------------------------------
# Minimal DC-AE-style encoder / decoder
# ---------------------------------------------------------------------------
class DCAEEncoder(nn.Module):
    """
    widths    : channel width per stage (input side first). #stages = len(widths),
                spatial compression = 2**(len(widths)-1)
    latent_ch : number of latent channels
    """

    def __init__(self, in_ch: int = 1, widths=(64, 128, 256, 512, 512, 1024),
                 depths=(1, 2, 2, 2, 2, 2), latent_ch: int = 32):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, widths[0], 3, padding=1)
        stages = []
        for i, (w, d) in enumerate(zip(widths, depths)):
            blocks = [ResBlock(w) for _ in range(d)]
            if i < len(widths) - 1:
                blocks.append(ResidualDown(w, widths[i + 1], factor=2))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        # Projection to the latent also gets a shortcut (the paper's "residual" project_out)
        self.proj = nn.Conv2d(widths[-1], latent_ch, 3, padding=1)
        self.proj_res = PixelUnshuffleChannelAvgDown(widths[-1], latent_ch, factor=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.stem(x))
        return self.proj(h) + self.proj_res(h)


class DCAEDecoder(nn.Module):
    def __init__(self, out_ch: int = 1, widths=(1024, 512, 512, 256, 128, 64),
                 depths=(2, 2, 2, 2, 2, 1), latent_ch: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(latent_ch, widths[0], 3, padding=1)
        self.proj_res = ChannelDupPixelShuffleUp(latent_ch, widths[0], factor=1)
        stages = []
        for i, (w, d) in enumerate(zip(widths, depths)):
            blocks = [ResBlock(w) for _ in range(d)]
            if i < len(widths) - 1:
                blocks.append(ResidualUp(w, widths[i + 1], factor=2))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.GroupNorm(32, widths[-1]), nn.SiLU(),
                                  nn.Conv2d(widths[-1], out_ch, 3, padding=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z) + self.proj_res(z)
        return self.head(self.stages(h))


class DCAE(nn.Module):
    def __init__(self, in_ch: int = 1, latent_ch: int = 32):
        super().__init__()
        self.encoder = DCAEEncoder(in_ch=in_ch, latent_ch=latent_ch)
        self.decoder = DCAEDecoder(out_ch=in_ch, latent_ch=latent_ch)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DCAE(in_ch=1, latent_ch=32)
    x = torch.randn(2, 1, 256, 256)
    y, z = model(x)
    print("input :", tuple(x.shape))
    print("latent:", tuple(z.shape), "  (spatial f32, 32 ch)")
    print("output:", tuple(y.shape))
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n:.1f}M")

    # Sanity check: the parameter-free shortcuts alone form an exact round-trip
    down = PixelUnshuffleChannelAvgDown(4, 16, 2)
    up = ChannelDupPixelShuffleUp(16, 4, 2)
    t = torch.randn(1, 4, 8, 8)
    print("shortcut round-trip err:", (up(down(t)) - t).abs().max().item())
