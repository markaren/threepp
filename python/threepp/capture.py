"""Headless pictures, image comparison and a text frame report, in one place.

Every visual question about a threepp scene reduces to the same three steps: render it
headless for a few frames from a known camera, look at the pixels (or compare them with an
earlier picture), and read what the renderer spent its time on. This module is that glue,
so a check is a short script instead of a one-off harness each time:

    import threepp as tp
    from threepp import capture

    frame = capture.render(scene, camera, backend="vulkan", size=(640, 360), frames=60)
    frame.save("after.png")
    print(frame.report())                       # pass timings as text (Vulkan)

    d = capture.compare("before.png", frame)    # PSNR / max delta / hot pixels
    print(d)
    assert d.passes()                           # the golden test's default gate

``render`` builds a headless canvas and renderer for the backend named, or drives a
renderer you pass in (``backend=my_renderer``), so a running script can take a still of
its own scene. ``compare`` uses the same metric as the C++ golden test
(``examples/libs/capture_util.hpp``: imageDiff), with the same defaults for the gate:
PSNR >= 40 dB and a per-channel max delta <= 32, measured against a same-GPU
run-to-run floor of about 61 dB and a max delta of 1 on a converged static frame.

Reproducibility, measured on an RTX 4060 laptop GPU with a box, a mirror ball and a sun:
a FRESH Vulkan renderer rendering the same scene for the same number of frames gave a
bit-identical picture (PSNR 99 dB, max delta 0). The same renderer rendering on carries its
temporal state (TAA, denoiser) into the next picture, so frame 400 differs from frame 200
by a little (55 dB, max delta about 20 to 35 on the ball). For a strict before/after check
make a new renderer per picture and keep ``frames`` the same; the default gate is for the
other case and for different hardware.

PNGs are written and read with the standard library and numpy only (8-bit RGB and RGBA,
non-interlaced), so the module needs nothing beyond what ``import threepp`` already needs.

Sizes: a headless canvas on Windows falls back to a hidden window whose width the window
manager clamps to about 232 px, so a narrower request comes back wider. ``render`` checks
the returned image against the requested size and raises rather than returning a surprise.

Timings: the Vulkan renderer exposes per-frame and per-pass GPU timings
(``frame_timings``, ``pass_timings``); the GL renderer has none, so ``Frame.report()``
says so for GL instead of inventing numbers. There are no per-frame counters (device
waits, acceleration-structure rebuilds, upload bytes) yet; that needs renderer-side work.
"""

from __future__ import annotations

import math
import os
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np

__all__ = [
    "Frame", "Diff", "render", "make_renderer", "compare", "report",
    "read_png", "write_png", "DEFAULT_MIN_PSNR", "DEFAULT_MAX_DELTA", "DEFAULT_HOT_THRESH",
]

# The golden test's gate (tests/renderers/VulkanGolden_test.cpp): PSNR well above the
# measured same-GPU floor, a max delta that catches a firefly PSNR would average away.
DEFAULT_MIN_PSNR = 40.0
DEFAULT_MAX_DELTA = 32
DEFAULT_HOT_THRESH = 4

# Below this width a Windows headless canvas comes back wider than asked (see module doc).
MIN_SAFE_WIDTH = 232


# --------------------------------------------------------------------------- #
# PNG, standard library only
# --------------------------------------------------------------------------- #

def write_png(path, rgb):
    """Write an (H, W, 3) or (H, W, 4) uint8 array as a PNG. Returns the path."""
    rgb = np.ascontiguousarray(rgb, np.uint8)
    if rgb.ndim == 2:
        rgb = rgb[:, :, None].repeat(3, axis=2)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError(f"write_png: expected (H, W, 3|4) uint8, got {rgb.shape}")
    h, w, c = rgb.shape
    color_type = 2 if c == 3 else 6
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))  # filter 0 per row

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))
    return path


def _unfilter(filtered, h, stride, bpp):
    """Undo PNG's per-row filters (types 0..4). Returns the raw (H, stride) bytes."""
    out = np.zeros((h, stride), np.uint8)
    pos = 0
    prev = np.zeros(stride, np.int32)
    for y in range(h):
        ftype = filtered[pos]
        row = np.frombuffer(filtered, np.uint8, stride, pos + 1).astype(np.int32)
        pos += 1 + stride
        if ftype == 0:
            cur = row
        elif ftype == 2:                       # Up
            cur = (row + prev) & 0xFF
        elif ftype in (1, 3, 4):               # Sub, Average, Paeth: a left dependency, so per byte
            cur = np.zeros(stride, np.int32)
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                if ftype == 1:
                    pred = a
                elif ftype == 3:
                    pred = (a + b) >> 1
                else:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (row[i] + pred) & 0xFF
        else:
            raise ValueError(f"read_png: unknown filter type {ftype}")
        out[y] = cur
        prev = cur
    return out


def read_png(path):
    """Read an 8-bit RGB or RGBA non-interlaced PNG as an (H, W, 3|4) uint8 array.

    Grey and grey+alpha are expanded to RGB / RGBA. Palette, 16-bit and interlaced
    files raise; Pillow reads those if you need them (``np.asarray(Image.open(p))``).
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"read_png: {path} is not a PNG")
    pos, idat, ihdr = 8, [], None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    if ihdr is None:
        raise ValueError(f"read_png: {path} has no IHDR")
    w, h, depth, color_type, _, _, interlace = ihdr
    if depth != 8 or interlace != 0:
        raise ValueError(f"read_png: {path}: only 8-bit non-interlaced PNGs are read here (depth={depth}, interlace={interlace})")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"read_png: {path}: palette PNGs are not read here (color type {color_type})")
    raw = _unfilter(zlib.decompress(b"".join(idat)), h, w * channels, channels)
    img = raw.reshape(h, w, channels)
    if channels == 1:
        img = img.repeat(3, axis=2)
    elif channels == 2:
        img = np.concatenate([img[:, :, :1].repeat(3, axis=2), img[:, :, 1:]], axis=2)
    return np.ascontiguousarray(img)


def _as_image(x) -> np.ndarray:
    """A Frame, an array, or a path -> an (H, W, C) uint8 array."""
    if isinstance(x, Frame):
        return x.image
    if isinstance(x, (str, os.PathLike)):
        return read_png(x)
    a = np.asarray(x)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = a[:, :, None]
    return a


# --------------------------------------------------------------------------- #
# Image comparison
# --------------------------------------------------------------------------- #

@dataclass
class Diff:
    """The result of ``compare``: the C++ imageDiff metric, with a gate.

    ``psnr`` is 99 dB when the images are identical (the C++ convention). ``hot_pct`` is
    the percentage of channel samples differing by more than ``thresh``. ``shape`` is the
    common image shape, or None when the shapes differed (then nothing else is meaningful
    and ``passes()`` is False).
    """
    mse: float = 0.0
    psnr: float = 0.0
    max_delta: int = 0
    hot_pct: float = 0.0
    thresh: int = DEFAULT_HOT_THRESH
    shape: Optional[tuple] = None
    shapes: Optional[tuple] = None   # (a.shape, b.shape) when they differ

    @property
    def identical(self) -> bool:
        return self.shape is not None and self.max_delta == 0

    def passes(self, min_psnr: float = DEFAULT_MIN_PSNR, max_delta: int = DEFAULT_MAX_DELTA) -> bool:
        """The golden test's gate: PSNR at least ``min_psnr`` and every channel within ``max_delta``."""
        if self.shape is None:
            return False
        return self.psnr >= min_psnr and self.max_delta <= max_delta

    def __str__(self):
        if self.shape is None:
            return f"shape mismatch {self.shapes[0]} vs {self.shapes[1]}"
        return (f"PSNR={self.psnr:5.1f} dB  maxD={self.max_delta:3d}  hot(>{self.thresh})={self.hot_pct:6.3f}%"
                f"  mse={self.mse:.3f}")


def compare(a, b, thresh: int = DEFAULT_HOT_THRESH) -> Diff:
    """Compare two pictures: Frames, uint8 arrays, or PNG paths, in any mix.

    An RGBA image is compared with an RGB one on its first three channels. Float arrays
    in 0..1 are quantised to uint8 first.
    """
    ia, ib = _as_image(a), _as_image(b)
    c = min(ia.shape[2], ib.shape[2])
    ia, ib = ia[:, :, :c], ib[:, :, :c]
    if ia.shape != ib.shape:
        return Diff(thresh=thresh, shape=None, shapes=(ia.shape, ib.shape))
    d = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
    mse = float(np.mean(d.astype(np.float64) ** 2))
    psnr = 10.0 * math.log10(255.0 * 255.0 / mse) if mse > 0.0 else 99.0
    return Diff(mse=mse, psnr=psnr, max_delta=int(d.max()) if d.size else 0,
                hot_pct=100.0 * float(np.count_nonzero(d > thresh)) / float(d.size) if d.size else 0.0,
                thresh=thresh, shape=ia.shape)


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

def _tp():
    import threepp as tp
    return tp


def _backend_name(renderer) -> str:
    return "vulkan" if _tp().is_vulkan(renderer) else "gl"


def make_renderer(backend: str = "vulkan", size: Sequence[int] = (640, 360), *, title: str = "capture",
                  antialiasing: int = 4):
    """A headless renderer of the named backend (``"vulkan"`` or ``"gl"``).

    The renderer keeps its canvas alive (the bindings' keep_alive), so nothing else needs
    holding. Raises on a GL-only build when Vulkan is asked for. Mind the one-GL-context-
    per-process rule: make one GL renderer per process and reuse it.
    """
    tp = _tp()
    w, h = int(size[0]), int(size[1])
    if w < MIN_SAFE_WIDTH:
        raise ValueError(f"make_renderer: width {w} is below the {MIN_SAFE_WIDTH} px a headless canvas can "
                         f"guarantee on Windows; ask for at least that and shrink the picture afterwards")
    backend = backend.lower()
    if backend == "vulkan":
        if not getattr(tp, "HAS_VULKAN", False):
            raise RuntimeError("make_renderer: this threepp build has no Vulkan backend")
        canvas = tp.Canvas(title, width=w, height=h, headless=True, vsync=False)
        return tp.VulkanRenderer(canvas)
    if backend == "gl":
        canvas = tp.Canvas(title, width=w, height=h, antialiasing=antialiasing, headless=True, vsync=False)
        return tp.GLRenderer(canvas)
    raise ValueError(f"make_renderer: backend must be 'vulkan' or 'gl', got {backend!r}")


def _size(renderer) -> tuple:
    """(width, height) of a renderer, whichever shape its binding returns."""
    s = renderer.size()
    if hasattr(s, "width"):
        return int(s.width()), int(s.height())
    return int(s[0]), int(s[1])


def _read_rgb(renderer) -> np.ndarray:
    img = renderer.read_pixels()
    if img is None:
        raise RuntimeError("capture.render: read_pixels returned nothing")
    img = np.asarray(img)
    return np.ascontiguousarray(img[..., :3])


def _timings(renderer) -> tuple:
    """(frame_timings dict, pass_timings list), empty on a renderer without them."""
    ft = getattr(renderer, "frame_timings", None)
    pt = getattr(renderer, "pass_timings", None)
    return (dict(ft) if ft is not None else {}), (list(pt) if pt is not None else [])


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #

@dataclass
class Frame:
    """A rendered picture with where it came from and what it cost.

    ``image`` is (H, W, 3) uint8, top row first. ``frame_timings`` and ``pass_timings``
    are the renderer's for the LAST frame rendered (Vulkan only; empty on GL).
    ``per_frame`` holds the ``frame_timings`` of every frame when ``render`` was asked to
    keep them (``keep_timings=True``), for a convergence or warm-up plot.
    """
    image: np.ndarray
    backend: str
    frames: int
    size: tuple
    frame_timings: dict = field(default_factory=dict)
    pass_timings: list = field(default_factory=list)
    per_frame: list = field(default_factory=list)
    path: Optional[str] = None

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def save(self, path) -> str:
        """Write the picture as a PNG and remember the path."""
        write_png(path, self.image)
        self.path = os.fspath(path)
        return self.path

    def stats(self) -> dict:
        """Cheap picture statistics: mean and max per channel, the fraction of near-black
        and saturated pixels. Enough to tell "black frame", "blown out" and "flat" apart
        without opening the image."""
        img = self.image
        lum = img.astype(np.float32).mean(axis=2)
        return {
            "mean": [float(x) for x in img.reshape(-1, 3).mean(axis=0)],
            "max": [int(x) for x in img.reshape(-1, 3).max(axis=0)],
            "min": [int(x) for x in img.reshape(-1, 3).min(axis=0)],
            "black_pct": 100.0 * float(np.mean(lum < 2.0)),
            "saturated_pct": 100.0 * float(np.mean(lum > 253.0)),
            "flat": bool(img.max() == img.min()),
        }

    def report(self, top: int = 12) -> str:
        """The frame as text: size, backend, picture stats, and the GPU time by pass."""
        return report(self, top=top)

    def __str__(self):
        return f"Frame({self.backend}, {self.width}x{self.height}, {self.frames} frames)"


def render(scene, camera, *, backend: Union[str, Any] = "vulkan", size: Sequence[int] = (640, 360),
           frames: int = 1, path=None, keep_timings: bool = False, before_frame=None,
           strict_size: bool = True) -> Frame:
    """Render ``scene`` from ``camera`` headless for ``frames`` frames and return the last one.

    backend: ``"vulkan"``, ``"gl"``, or a renderer you already have (its canvas size wins
        over ``size``). A new renderer is made per call for the string forms, so for many
        captures in one process make one with ``make_renderer`` and pass it in; a GL
        process can hold only one context anyway.
    frames: how many frames to draw before reading back. The Vulkan path's TAA, denoiser
        and probe GI converge over a static camera; the golden test uses 200, a quick look
        needs 30 to 60, a GL frame needs 1.
    path: write the picture there as a PNG when given.
    keep_timings: keep every frame's ``frame_timings`` in ``Frame.per_frame``.
    before_frame: ``before_frame(i)`` is called before frame i, for a scene that must step
        (physics, animation) between frames.
    strict_size: raise when the picture does not match the requested size (see the module
        doc on narrow headless canvases).
    """
    renderer = make_renderer(backend, size) if isinstance(backend, str) else backend
    name = _backend_name(renderer)
    frames = max(1, int(frames))
    w, h = _size(renderer)
    if hasattr(camera, "aspect") and w > 0 and h > 0 and abs(float(camera.aspect) - w / h) > 1e-3:
        camera.aspect = w / h
        camera.update_projection_matrix()
    per_frame = []
    for i in range(frames):
        if before_frame is not None:
            before_frame(i)
        renderer.render(scene, camera)
        if keep_timings and name == "vulkan":
            per_frame.append(dict(renderer.frame_timings))
    img = _read_rgb(renderer)
    ft, pt = _timings(renderer)
    frame = Frame(image=img, backend=name, frames=frames, size=(int(img.shape[1]), int(img.shape[0])),
                  frame_timings=ft, pass_timings=pt, per_frame=per_frame)
    if isinstance(backend, str) and strict_size and frame.size != (int(size[0]), int(size[1])):
        raise RuntimeError(f"capture.render: asked for {tuple(int(x) for x in size)} but the {name} canvas gave "
                           f"{frame.size}; widths below {MIN_SAFE_WIDTH} px are clamped on Windows")
    if path is not None:
        frame.save(path)
    return frame


# --------------------------------------------------------------------------- #
# Text report
# --------------------------------------------------------------------------- #

def _fmt_ms(x) -> str:
    return f"{float(x):7.3f}"


def report(source, top: int = 12) -> str:
    """Frame timings as text, from a Frame or a renderer.

    For the Vulkan backend: the frame totals (GPU total, pass sum, CPU record and frame),
    the ``top`` costliest passes of the render graph with their view, and the non-zero
    stage timings from ``frame_timings``. For GL: a line saying there are none.
    """
    if isinstance(source, Frame):
        head = [f"{source}  "]
        st = source.stats()
        head.append(f"mean rgb=({st['mean'][0]:.1f}, {st['mean'][1]:.1f}, {st['mean'][2]:.1f})  "
                    f"black={st['black_pct']:.1f}%  saturated={st['saturated_pct']:.1f}%" + ("  FLAT" if st["flat"] else ""))
        ft, pt, backend = source.frame_timings, source.pass_timings, source.backend
    else:
        head = []
        ft, pt = _timings(source)
        backend = _backend_name(source)
    lines = list(head)
    if backend != "vulkan" or not ft:
        lines.append("no GPU timings: the GL backend has none" if backend != "vulkan" else "no GPU timings on this device")
        return "\n".join(lines)
    lines.append(f"gpu_total={_fmt_ms(ft.get('gpu_total_ms', 0))} ms  graph={_fmt_ms(ft.get('gpu_pass_sum_ms', 0))} ms  "
                 f"cpu_record={_fmt_ms(ft.get('cpu_record_ms', 0))} ms  cpu_frame={_fmt_ms(ft.get('cpu_frame_ms', 0))} ms  "
                 f"cpu_ensure_scene={_fmt_ms(ft.get('cpu_ensure_scene_ms', 0))} ms")
    if pt:
        rows = sorted(pt, key=lambda r: -float(r[2]))[:top]
        total = sum(float(r[2]) for r in pt) or 1.0
        width = max(len(str(r[0])) for r in rows)
        lines.append(f"passes ({len(pt)} rows, {top} costliest):")
        for name, view, ms in rows:
            lines.append(f"  {str(name):<{width}}  view {view:<3} {_fmt_ms(ms)} ms  {100.0 * float(ms) / total:5.1f}%")
    stages = [(k, v) for k, v in ft.items()
              if k.endswith("_ms") and not k.startswith(("gpu_", "cpu_")) and float(v) > 0.0]
    if stages:
        stages.sort(key=lambda kv: -float(kv[1]))
        lines.append("stages: " + "  ".join(f"{k[:-3]}={float(v):.3f}" for k, v in stages))
    return "\n".join(lines)
