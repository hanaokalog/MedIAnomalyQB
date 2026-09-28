"""
Random-shift wrapper for perceptual losses (LPIPS / VGG / DINO feature distance).

Why: patch- or stride-based feature extractors only penalise what they see inside
their fixed grid, so a decoder can learn grid-aligned block artifacts that the
loss never "notices". Shifting BOTH images by the same random offset in
[0, patch) each step randomises where the grid falls, so seams get penalised
on average and the artifacts disappear.

Usage:
    loss_fn = ShiftedPerceptualLoss(perceptual_net, patch=14)   # DINOv2: 14, DINOv3: 16, VGG/LPIPS: 8-16
    loss = loss_fn(recon, target)

`perceptual_net` must be a callable (x, y) -> scalar (e.g. lpips.LPIPS, or your own
DINO feature-distance module). Only the wrapper is defined here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_shift_pair(x: torch.Tensor, y: torch.Tensor, max_shift: int,
                      pad_mode: str = "reflect", per_sample: bool = True):
    """
    Shift x and y by the same random integer offset (dx, dy) in [0, max_shift).
    Implemented as reflect-pad by max_shift then crop back to the original size,
    so no zero borders are introduced and the output shape is unchanged.

    per_sample=True draws an independent offset for every element in the batch
    (more decorrelation), False uses one offset for the whole batch (cheaper).
    """
    if max_shift <= 0:
        return x, y
    B, C, H, W = x.shape
    xp = F.pad(x, (max_shift, max_shift, max_shift, max_shift), mode=pad_mode)
    yp = F.pad(y, (max_shift, max_shift, max_shift, max_shift), mode=pad_mode)

    if not per_sample:
        dy, dx = torch.randint(0, 2 * max_shift + 1, (2,)).tolist()
        return xp[..., dy:dy + H, dx:dx + W], yp[..., dy:dy + H, dx:dx + W]

    # per-sample crop via unfold-free indexing: build a gather grid once per batch
    dy = torch.randint(0, 2 * max_shift + 1, (B,), device=x.device)
    dx = torch.randint(0, 2 * max_shift + 1, (B,), device=x.device)
    rows = (dy[:, None] + torch.arange(H, device=x.device)[None, :])          # (B, H)
    cols = (dx[:, None] + torch.arange(W, device=x.device)[None, :])          # (B, W)
    bidx = torch.arange(B, device=x.device)[:, None, None]
    xs = xp[bidx, :, rows[:, :, None], cols[:, None, :]]                       # (B, H, W, C)
    ys = yp[bidx, :, rows[:, :, None], cols[:, None, :]]
    return xs.permute(0, 3, 1, 2).contiguous(), ys.permute(0, 3, 1, 2).contiguous()


class ShiftedPerceptualLoss(nn.Module):
    """
    Wraps any pairwise perceptual loss with a shared random shift (and optional
    random flips, which also help break axis-aligned artifacts).
    """

    def __init__(self, perceptual: nn.Module, patch: int = 14, flips: bool = True,
                 per_sample: bool = True, resize_to: int | None = None):
        super().__init__()
        self.perceptual = perceptual
        self.patch = patch
        self.flips = flips
        self.per_sample = per_sample
        self.resize_to = resize_to   # e.g. 224 for DINO on 128^2 inputs (shift happens BEFORE resize)

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.training:
            # shift at input resolution; the grid of the feature net then lands on a random phase
            shift = self.patch if self.resize_to is None else max(1, round(self.patch * recon.shape[-1] / self.resize_to))
            recon, target = random_shift_pair(recon, target, shift, per_sample=self.per_sample)
            if self.flips:
                if torch.rand(()) < 0.5:
                    recon, target = recon.flip(-1), target.flip(-1)
                if torch.rand(()) < 0.5:
                    recon, target = recon.flip(-2), target.flip(-2)
        if self.resize_to is not None:
            recon = F.interpolate(recon, size=self.resize_to, mode="bilinear", align_corners=False, antialias=True)
            target = F.interpolate(target, size=self.resize_to, mode="bilinear", align_corners=False, antialias=True)
        return self.perceptual(recon, target)


if __name__ == "__main__":
    torch.manual_seed(0)

    class DummyPerceptual(nn.Module):
        """stand-in for LPIPS/DINO: 16x16 patch-mean distance (a loss that *is* blind to seams)"""
        def forward(self, a, b):
            return (F.avg_pool2d(a, 16) - F.avg_pool2d(b, 16)).abs().mean()

    x = torch.rand(4, 1, 128, 128)
    y = x + 0.1 * torch.randn_like(x)

    # shape / alignment check: identical inputs must give exactly zero after the shared shift
    xs, ys = random_shift_pair(x, x.clone(), max_shift=14)
    print("shapes:", tuple(xs.shape), tuple(ys.shape), " aligned:", torch.equal(xs, ys))

    loss_fn = ShiftedPerceptualLoss(DummyPerceptual(), patch=16, resize_to=None).train()
    print("loss (shifted):  ", float(loss_fn(y, x)))
    loss_fn.eval()
    print("loss (unshifted):", float(loss_fn(y, x)))
