"""
Lossless compression of a binary vector with a position-dependent Bernoulli model
(arithmetic / range coding).

Each element X_i is independent with P(X_i = 1) = p_i, where p_i is known and may
differ per position. As long as the encoder and decoder share the same p sequence,
there is no need to transmit p itself, and the achieved code length is essentially
the ideal -log2 p(x).

    pip install constriction numpy
"""
import numpy as np
import constriction
from PIL import Image
import io


def encode(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compress binary vector x under per-position priors p; return a uint32 array."""
    x = np.asarray(x, dtype=np.int32)            # symbols must be int32
    p = np.asarray(p, dtype=np.float64)          # P(X_i = 1)
    model = constriction.stream.model.Bernoulli(perfect=False)
    enc = constriction.stream.queue.RangeEncoder()
    enc.encode(x, model, p)                       # use p_i for each x_i
    return enc.get_compressed()


def decode(compressed: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Reconstruct the original binary vector from compressed data using the same p."""
    p = np.asarray(p, dtype=np.float64)
    model = constriction.stream.model.Bernoulli(perfect=False)
    dec = constriction.stream.queue.RangeDecoder(compressed)
    return dec.decode(model, p)                   # decode len(p) elements


def encoded_length_residual(x: np.ndarray) -> int:
    if len(x.shape)==2:
        mode = 'L'
    else:
        mode = 'RGB'
    with Image.fromarray(x, mode) as img:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.tell()



if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 100_000

    # Known per-position priors, and one instance drawn from them
    p = rng.uniform(0.001, 0.999, size=N)
    x = (rng.random(N) < p).astype(np.int32)

    compressed = encode(x, p)
    x_rec = decode(compressed, p)
    assert np.array_equal(x, x_rec), "lossless round-trip failed"

    bits = compressed.size * 32
    ideal = -(x * np.log2(p) + (1 - x) * np.log2(1 - p)).sum()
    print(f"naive storage     : {N} bits")
    print(f"ideal -log2 p(x)  : {ideal:8.1f} bits")
    print(f"actual code length: {bits:8d} bits ({bits/8:.0f} bytes)")
    print(f"overhead          : {bits - ideal:.1f} bits (whole stream, ~constant)")
    print("lossless round-trip OK")
