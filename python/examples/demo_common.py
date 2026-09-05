"""What the examples share that needs nothing but numpy and threepp.

`warp_common.py` is the films' shared module and it imports Warp, so anything
put there is a Warp dependency for whoever uses it. The hello-world demos and
the pure-renderer showcases must keep running on `pip install threepp` alone,
so the pieces they share live here instead, and `warp_common` re-exports them
for the films.
"""
import numpy as np

# --- Radiance .hdr -------------------------------------------------------------


def encode_rgbe(rgb):
    """Vectorised linear-RGB float -> Radiance RGBE bytes, shape (H, W, 4)."""
    rgb = np.maximum(np.asarray(rgb, np.float64), 0.0)
    m = rgb.max(axis=2)
    mask = m >= 1e-32
    safe = np.where(mask, m, 1.0)
    mant, exp = np.frexp(safe)             # m = mant * 2**exp,  mant in [0.5, 1)
    scale = np.where(mask, mant * 256.0 / safe, 0.0)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    for c in range(3):
        out[..., c] = np.clip(rgb[..., c] * scale, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(mask, np.clip(exp + 128, 0, 255), 0).astype(np.uint8)
    return out


def write_radiance_hdr(path, rgb):
    """Write an (H, W, 3) linear float array as an uncompressed Radiance .hdr.

    Four demos build their own equirect sky -- each a hand-tuned look, so the
    pixels stay where they are -- and then all four end the same way: encode to
    RGBE, dodge the RLE signature, write the two-line header. That tail is what
    lives here. Returns `path`, to hand straight to RGBELoader.
    """
    rgbe = encode_rgbe(rgb)
    # An RGBE row starting (2, 2, <128) reads as "adaptive RLE" to a .hdr
    # reader; nudge the one pixel that could fake that signature, so stb takes
    # the scanlines uncompressed.
    if rgbe[0, 0, 0] == 2 and rgbe[0, 0, 1] == 2 and rgbe[0, 0, 2] < 128:
        rgbe[0, 0, 0] = 3
    h, w = rgbe.shape[:2]
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (h, w))
        f.write(rgbe.tobytes())
    return path
