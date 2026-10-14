"""Bake the far-tree impostors of vulkan_fjord from its card trees.

    python -I bake_tree_impostors.py <threepp_data>/models/fjord/trees

Past a few hundred metres a tree is 20-30 pixels tall, and the renderer pays per
INSTANCE on the CPU, so the forest cannot keep drawing lite trees out to the
horizon (it used to thin them 1 in 2 and then stop: the trees visibly vanished).
The far tier is instead one merged mesh per forest cell of crossed cards, each
showing a picture of the tree it stands in for. This script draws those
pictures from the trees themselves: every species (pine, birch, spruce, the
order of CanopyForest's slots) and every lite variant, seen from the front
(along -z) and from the side (along -x), orthographic, with a small z-buffered
rasteriser: leaf cards sampled from their own atlas (cut out at 0.4, times the
kit's baked canopy occlusion), trunks from their bark.

Out (next to the trees):
  impostor_albedo.webp  RGBA, sRGB colour, alpha = coverage (lossless)
  impostor_normal.webp  RGB, the leaves' own normals in the card's frame
                        (x right, y up, z toward the viewer), 0.5-biased

Layout: a grid of CELL_W x CELL_H cells, one row per species, columns
variant*2 + view (0 front, 1 side). In each cell the tree is scaled uniformly
to fit, centred, its lowest point on the cell's bottom edge: scale =
min(CELL_W / (2 * halfW), CELL_H / (ymax - ymin)), halfW = max |x|, |z| over
the variant's lite meshes. examples/vulkan/vulkan_fjord.cpp computes the same
numbers from the same meshes to size the cards.
"""
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SPECIES = ("pine", "birch", "spruce")
BARK_REPEAT = {"pine": (1.2, 0.1), "birch": (1.0, 1.0), "spruce": (1.2, 0.1)}
CELL_W, CELL_H = 128, 256
SS = 4  # supersampling per axis
ALPHA_CUT = 0.4


def read_tree(path):
    b = Path(path).read_bytes()
    assert b[:4] == b"TRE1"
    (count,) = struct.unpack_from("<I", b, 4)
    o, out = 8, {}
    for _ in range(count):
        (ln,) = struct.unpack_from("<B", b, o)
        o += 1
        name = b[o:o + ln].decode("ascii")
        o += ln
        n, t, flags = struct.unpack_from("<IIB", b, o)
        o += 9
        f = np.frombuffer(b, "<f4", 10, o).astype(np.float64)
        o += 40
        pmin, pmax, umin, umax = f[0:3], f[3:6], f[6:8], f[8:10]
        pos = pmin + np.frombuffer(b, "<u2", n * 3, o).reshape(n, 3) / 65535.0 * (pmax - pmin)
        o += n * 6
        nrm = np.maximum(np.frombuffer(b, "<i1", n * 4, o).reshape(n, 4)[:, :3] / 127.0, -1.0)
        o += n * 4
        uv = umin + np.frombuffer(b, "<u2", n * 2, o).reshape(n, 2) / 65535.0 * (umax - umin)
        o += n * 4
        col = np.ones((n, 3))
        if flags & 1:
            col = np.frombuffer(b, "<u1", n * 4, o).reshape(n, 4)[:, :3] / 255.0
            o += n * 4
        if flags & 2:
            idx = np.frombuffer(b, "<u4", t * 3, o).reshape(t, 3)
            o += t * 12
        else:
            idx = np.frombuffer(b, "<u2", t * 3, o).reshape(t, 3)
            o += t * 6
        nl = np.linalg.norm(nrm, axis=1, keepdims=True)
        out[name] = dict(pos=pos, nrm=nrm / np.maximum(nl, 1e-6), uv=uv, col=col, idx=idx.astype(np.int64))
    return out


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def sample(img, u, v, wrap):
    """Bilinear, img float HxWxC, u/v in texture space with v running DOWN the image."""
    h, w = img.shape[:2]
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]

    def at(xx, yy):
        if wrap:
            xx, yy = xx % w, yy % h
        else:
            xx, yy = np.clip(xx, 0, w - 1), np.clip(yy, 0, h - 1)
        return img[yy, xx]

    a = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
    c = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
    return a * (1 - fy) + c * fy


def raster(meshes, view, leaf_tex, bark_tex, bark_rep):
    """One cell at SS x resolution: (rgb linear, alpha, normal) arrays."""
    W, H = CELL_W * SS, CELL_H * SS
    allpos = np.concatenate([m["pos"] for m in meshes.values()])
    half_w = max(np.abs(allpos[:, 0]).max(), np.abs(allpos[:, 2]).max())
    ymin, ymax = allpos[:, 1].min(), allpos[:, 1].max()
    scale = min(CELL_W / (2 * half_w), CELL_H / (ymax - ymin)) * SS
    # view 0: looking along -z (right = +x, toward viewer = +z); view 1: along -x (right = -z, toward = +x)
    right = np.array([1.0, 0, 0]) if view == 0 else np.array([0, 0, -1.0])
    toward = np.array([0, 0, 1.0]) if view == 0 else np.array([1.0, 0, 0])

    rgb = np.zeros((H, W, 3))
    alpha = np.zeros((H, W))
    nrm = np.zeros((H, W, 3))
    zbuf = np.full((H, W), -1e9)
    for name, m in meshes.items():
        leaf = name.endswith("_leaf")
        p = m["pos"]
        sx = W / 2 + (p @ right) * scale
        sy = H - (p[:, 1] - ymin) * scale  # image rows grow downward
        sz = p @ toward
        for tri in m["idx"]:
            xs, ys = sx[tri], sy[tri]
            x0, x1 = int(max(np.floor(xs.min()), 0)), int(min(np.ceil(xs.max()), W - 1))
            y0, y1 = int(max(np.floor(ys.min()), 0)), int(min(np.ceil(ys.max()), H - 1))
            if x1 < x0 or y1 < y0:
                continue
            det = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(det) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / det
            l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / det
            l2 = 1 - l0 - l1
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            lam = np.stack([l0, l1, l2], -1)[inside]
            z = lam @ sz[tri]
            uv = lam @ m["uv"][tri]
            if leaf:
                tex = sample(leaf_tex, uv[:, 0], uv[:, 1], False)
                keep = tex[:, 3] >= ALPHA_CUT
                col = srgb_to_linear(tex[:, :3]) * (lam @ m["col"][tri])
            else:
                u = uv[:, 0] * bark_rep[0]
                v = uv[:, 1] * bark_rep[1] + (1 - bark_rep[1])
                tex = sample(bark_tex, u, v, True)
                keep = np.ones(len(lam), bool)
                col = srgb_to_linear(tex[:, :3]) * 0.85
            n = lam @ m["nrm"][tri]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
            yy, xx = np.nonzero(inside)
            yy, xx = yy + y0, xx + x0
            front = keep & (z > zbuf[yy, xx])
            yy, xx = yy[front], xx[front]
            zbuf[yy, xx] = z[front]
            rgb[yy, xx] = col[front]
            alpha[yy, xx] = 1.0
            nv = np.stack([n[front] @ right, n[front][:, 1], n[front] @ toward], -1)
            nrm[yy, xx] = nv
    return rgb, alpha, nrm


def downsample(img):
    h, w = img.shape[:2]
    return img.reshape(h // SS, SS, w // SS, SS, *img.shape[2:]).mean(axis=(1, 3))


def dilate(rgb, alpha, passes=24):
    """Push colour into empty texels so mips do not pull black into the edges."""
    rgb = rgb.copy()
    known = alpha > 0.0
    for _ in range(passes):
        if known.all():
            break
        acc = np.zeros_like(rgb)
        cnt = np.zeros(alpha.shape)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            sh = np.roll(np.roll(known, dy, 0), dx, 1)
            acc += np.where(sh[..., None], np.roll(np.roll(rgb, dy, 0), dx, 1), 0)
            cnt += sh
        grow = (~known) & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow][:, None]
        known = known | grow
    return rgb


def main(tree_dir):
    tree_dir = Path(tree_dir)
    atlas_rgb = np.zeros((CELL_H * len(SPECIES), CELL_W * 6, 3))
    atlas_a = np.zeros((CELL_H * len(SPECIES), CELL_W * 6))
    atlas_n = np.zeros((CELL_H * len(SPECIES), CELL_W * 6, 3))
    for row, sp in enumerate(SPECIES):
        meshes = read_tree(tree_dir / f"{sp}.tree")
        leaf_tex = np.asarray(Image.open(tree_dir / f"{sp}_lite.webp").convert("RGBA"), np.float64) / 255.0
        bark_tex = np.asarray(Image.open(tree_dir / f"{sp}_bark.webp").convert("RGBA"), np.float64) / 255.0
        for var in range(3):
            parts = {k: v for k, v in meshes.items() if k.startswith(f"lite{var}_")}
            for view in range(2):
                rgb, a, n = raster(parts, view, leaf_tex, bark_tex, BARK_REPEAT[sp])
                a_ds = downsample(a)
                rgb_ds = downsample(rgb * a[..., None]) / np.maximum(a_ds[..., None], 1e-6)
                n_ds = downsample(n * a[..., None])
                n_ds /= np.maximum(np.linalg.norm(n_ds, axis=2, keepdims=True), 1e-6)
                col = var * 2 + view
                ys, xs = slice(row * CELL_H, (row + 1) * CELL_H), slice(col * CELL_W, (col + 1) * CELL_W)
                atlas_rgb[ys, xs] = dilate(rgb_ds, a_ds)
                atlas_n[ys, xs] = dilate(np.where(a_ds[..., None] > 0, n_ds, 0), a_ds)
                atlas_a[ys, xs] = a_ds
                print(f"{sp} lite{var} view {view}: coverage {a_ds.mean():.3f}", flush=True)
    albedo = np.dstack([linear_to_srgb(atlas_rgb), atlas_a])
    Image.fromarray(np.round(albedo * 255).astype(np.uint8), "RGBA").save(
        tree_dir / "impostor_albedo.webp", quality=85, method=6, exact=True, alpha_quality=100)
    normal = np.clip(atlas_n * 0.5 + 0.5, 0, 1)
    Image.fromarray(np.round(normal * 255).astype(np.uint8), "RGB").save(
        tree_dir / "impostor_normal.webp", quality=90, method=6)
    for f in ("impostor_albedo.webp", "impostor_normal.webp"):
        print(f, (tree_dir / f).stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
