# Turn the CC0 drone photo of sea foam into a tileable, rank-equalised 512^2
# foam-density tile (R8) and a C++ header that embeds it.
#   python -I make_foam_tile.py <photo.jpg> <out_dir>
# Source: "Ocean foam drone view" by Tommy Lisbin, Huntington Beach, 2017,
# CC0 1.0 (Unsplash before 5 June 2017, via Wikimedia Commons).
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

src, out_dir = sys.argv[1], sys.argv[2]
N = 512          # output tile
OVER = 0.30      # overlap fraction blended across the wrap

img = Image.open(src).convert("RGB")
a = np.asarray(img, np.float32) / 255.0
a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)   # to linear
lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]

# The foam over open water: away from the sand (bottom), the sunlit backwash
# (left) and the dark water (top). Full-res pixels of the 4000 x 3000 frame.
x0, y0, side = 1500, 760, 1700
crop = lum[y0:y0 + side, x0:x0 + side]

# Foam density, not brightness: take out the large-scale illumination and water
# colour (a blur many foam cells wide), normalise the local contrast.
big = gaussian_filter(crop, 120)
hp = crop - big
sd = np.sqrt(gaussian_filter(hp * hp, 160)) + 1e-4
z = hp / sd

# Resample so that the tile plus its overlap fits the crop.
M = int(round(N * (1 + OVER)))
zi = np.asarray(Image.fromarray(z).resize((M, M), Image.LANCZOS), np.float32)

# Make it wrap: cross-fade the overlap, first in x then in y. The values are
# zero-mean, so dividing by sqrt(w^2 + (1 - w)^2) keeps the contrast in the seam
# (a plain cross-fade of two unrelated regions halves it).
def wrap_axis(v, axis):
    v = np.moveaxis(v, axis, 0)
    o = M - N
    t = (np.arange(o) + 0.5) / o
    w = 0.5 - 0.5 * np.cos(np.pi * t)             # 0 -> 1
    w = w.reshape(-1, *([1] * (v.ndim - 1)))
    head, tail = v[:o], v[N:N + o]
    blend = (w * head + (1 - w) * tail) / np.sqrt(w * w + (1 - w) * (1 - w))
    out = np.concatenate([blend, v[o:N]], 0)
    return np.moveaxis(out, 0, axis)

t = wrap_axis(wrap_axis(zi, 1), 0)
assert t.shape == (N, N), t.shape

# Rank-equalise: the fraction of the tile above any level IS that level.
order = np.argsort(t.ravel(), kind="stable")
eq = np.empty(t.size, np.float32)
eq[order] = (np.arange(t.size) + 0.5) / t.size
eq = eq.reshape(N, N)
u8 = np.clip(np.round(eq * 255.0), 0, 255).astype(np.uint8)

Image.fromarray(u8).save(f"{out_dir}/ocean_foam_tile.png")
# 2 x 2 tiling, to look for the seam and the period.
Image.fromarray(np.tile(u8, (2, 2))).save(f"{out_dir}/ocean_foam_tile_2x2.png")
# What it does to a foam level: the covered set at F = 0.8, 0.5, 0.25, 0.1.
rows = [np.clip((eq - (1 - F)) / 0.06 + 0.5, 0, 1) for F in (0.8, 0.5, 0.25, 0.1)]
strip = np.concatenate(rows, 1)
sea = np.array([18, 52, 78], np.float32)
rgb = sea[None, None, :] * (1 - strip[..., None]) + 235 * strip[..., None]
Image.fromarray(rgb.astype(np.uint8)).save(f"{out_dir}/ocean_foam_tile_levels.png")

# The header: 512 * 512 bytes packed into 64-bit words (little-endian).
words = np.frombuffer(u8.tobytes(), dtype="<u8")
with open(f"{out_dir}/OceanFoamTile.inc", "w", newline="\n") as f:
    for i in range(0, words.size, 6):
        f.write(",".join(f"0x{int(w):016x}ull" for w in words[i:i + 6]) + ",\n")
print("ok", u8.shape, words.size, "words")
