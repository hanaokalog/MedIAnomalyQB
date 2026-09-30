"""
GPU (PyTorch) replacement for `make_noise_like` in utils/aeu_qb_worker.py.

Same generative model as the original scipy version:
  z1 ~ N(0,1) per channel, Gaussian-blurred with radius rad1 ~ U(1, 17), normalized to unit std
  z2 ~ N(0,1) single channel, Gaussian-blurred with radius rad2 ~ U(1, 17), normalized to unit std
  z2 <- relu(z2 - U(0,2)) ** U(0.01, 2.01)            (blob mask)
  noise = z1 * z2 * std(x_i) * sigma

Differences from the scipy version (statistically negligible):
  * Gaussian blur is done in the Fourier domain on a zero-mean white-noise field generated on a
    (pad_factor*H)^2 canvas and cropped, instead of scipy's reflect boundary with 4-sigma truncation.
    With pad_factor=2 and rad <= 17, wrap-around is > 7 sigma away and has no visible effect.
  * Uses torch's RNG (pass a torch.Generator on the same device for reproducibility).
Everything is batched: no Python loop over images.
"""
import math
import torch


def _gaussian_blur_fft(z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Per-sample Gaussian blur with periodic boundary. z: (B,C,S,S), sigma: (B,)"""
    S = z.shape[-1]
    fy = torch.fft.fftfreq(S, device=z.device)
    fx = torch.fft.rfftfreq(S, device=z.device)
    f2 = fy[:, None] ** 2 + fx[None, :] ** 2                           # (S, S//2+1)
    Hf = torch.exp(-2.0 * math.pi ** 2 * sigma.view(-1, 1, 1, 1) ** 2 * f2)
    return torch.fft.irfft2(torch.fft.rfft2(z) * Hf, s=(S, S))


@torch.no_grad()
def make_noise_like_gpu(x: torch.Tensor, sigma: float = 1.0,
                        generator: torch.Generator = None, pad_factor: int = 2) -> torch.Tensor:
    """x: (B,C,H,W) on any device. Returns noise with the same shape/device (float32)."""
    B, C, Hh, Ww = x.shape
    assert Hh == Ww
    dev = x.device
    S = Hh * pad_factor
    kw = dict(device=dev, generator=generator)

    mother_std = x.detach().float().flatten(1).std(dim=1, unbiased=False)       # numpy .std() = ddof 0

    rad1 = torch.rand(B, **kw) * 16 + 1
    rad2 = torch.rand(B, **kw) * 16 + 1
    shift = torch.rand(B, **kw) * 2
    power = torch.rand(B, **kw) * 2.0 + 0.01

    z1 = _gaussian_blur_fft(torch.randn(B, C, S, S, **kw), rad1)[..., :Hh, :Ww]
    z2 = _gaussian_blur_fft(torch.randn(B, 1, S, S, **kw), rad2)[..., :Hh, :Ww]

    z1 = z1 / z1.flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
    z2 = z2 / z2.flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)

    z2 = (z2 - shift.view(-1, 1, 1, 1)).clamp_min(0) ** power.view(-1, 1, 1, 1)

    return z1 * z2 * (mother_std * sigma).view(-1, 1, 1, 1)                  # z2 broadcasts over C


# ---------------------------------------------------------------------------
# Drop-in change in AEU_QBWorker.train_epoch:
#
#     img = data_batch['img'].cuda(non_blocking=True)
#     if 0 < noise_level:
#         img_noised = img + make_noise_like_gpu(img, noise_level, generator=self.noise_gen)
#     else:
#         img_noised = img
#
# (create once, e.g. in __init__/set_seed:  self.noise_gen = torch.Generator(device='cuda').manual_seed(seed))
# Call it outside torch.autocast so the FFT runs in float32.
# ---------------------------------------------------------------------------
