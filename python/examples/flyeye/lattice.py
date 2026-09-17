"""The 721-column hex eye: lattice coordinates, frame -> luminance, and BoxEye.

Everything here reproduces flyvis ``datasets/rendering/eye.py`` (BoxEye(extent=15,
kernel_size=13)) and ``datasets/sintel_utils.py`` (sample_lum) from the exported npz,
without importing flyvis. Device and dtype follow the input tensor.

Pixel geometry: column j samples the 13x13 box mean centred at
    (row, col) = (H//2 + receptor_centers[j, 0], W//2 + receptor_centers[j, 1])
with receptor_centers = (trunc(13*(u + v/2)), 13*v). Image +row is DOWN. Neighbouring
columns are 13 or 14.5 px apart (a slightly squashed hexagon, pointy top and bottom).
Column order is flyvis's (u outer, v inner) loop, identical to every 721-node layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import MODEL_NPZ


class HexLattice:
    """Column coordinates and the BoxEye sampling taken from flyeye_model.npz."""

    def __init__(self, npz_path: str | Path = MODEL_NPZ):
        m = np.load(npz_path)
        self.extent = int(m["boxeye_extent"])  # 15
        self.kernel_size = int(m["boxeye_kernel_size"])  # 13
        self.min_frame_size = tuple(int(s) for s in m["boxeye_min_frame_size"])  # (391, 391)
        self.pad = tuple(int(p) for p in m["boxeye_pad"])  # (6, 6, 6, 6) left, right, top, bottom
        uv = m["boxeye_column_uv"].astype(np.int64)
        self.u = torch.from_numpy(uv[:, 0])  # (721,)
        self.v = torch.from_numpy(uv[:, 1])
        self.centers = torch.from_numpy(m["boxeye_receptor_centers"].astype(np.int64))  # (721, 2) row, col
        self.n_columns = len(self.u)
        self.central_column = int(torch.nonzero((self.u == 0) & (self.v == 0))[0, 0])

    def box_eye(self, frames: torch.Tensor) -> torch.Tensor:
        """(H, W) -> (721,) or (T, H, W) -> (T, 721): 13x13 box mean sampled at the columns.

        Same ops as flyvis BoxEye 'mean': frames smaller than 391 px are resized
        (bilinear, antialias, like torchvision ttf.resize on tensors), zero-padded by 6,
        convolved with a 13x13 kernel of ones, divided by 169, then indexed.
        """
        single = frames.dim() == 2
        x = frames[None] if single else frames  # (T, H, W)
        if not x.is_floating_point():
            raise TypeError("box_eye expects luminance in [0, 1] as a float tensor")
        H, W = x.shape[-2:]
        mh, mw = self.min_frame_size
        if H < mh or W < mw:
            x = F.interpolate(x[:, None], size=(mh, mw), mode="bilinear", align_corners=False, antialias=True)[:, 0]
            H, W = mh, mw
        k = torch.ones(1, 1, self.kernel_size, self.kernel_size, dtype=x.dtype, device=x.device)
        c = self.centers.to(x.device)
        rows, cols = c[:, 0] + H // 2, c[:, 1] + W // 2
        # chunks of 8 frames: CPU float64 conv2d uses im2col, ~220 MB per 403x403 frame
        out = torch.cat([
            (F.conv2d(F.pad(x[i : i + 8], self.pad)[:, None], k)[:, 0] / self.kernel_size**2)[:, rows, cols]
            for i in range(0, len(x), 8)
        ])  # (T, 721)
        return out[0] if single else out


# PIL "L" conversion: L = (R*19595 + G*38470 + B*7471 + 0x8000) >> 16 (ITU-R 601-2 luma).
def luma_u8(rgb: torch.Tensor) -> torch.Tensor:
    """uint8 (..., 3 or 4) RGB[A] -> uint8 (...) luma, bit-identical to PIL convert('L')."""
    c = rgb[..., :3].to(torch.int32)
    return ((c[..., 0] * 19595 + c[..., 1] * 38470 + c[..., 2] * 7471 + 0x8000) >> 16).to(torch.uint8)


def luminance(rgb: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Display-encoded sRGB uint8 (..., 3|4) -> flyvis input luminance L/255 in [0, 1].

    This is what the pretrained models saw (Sintel PNGs through sample_lum): 8-bit luma of
    the display-encoded image, not linear light, not log, no mean subtraction.
    For BGRA frames (threepp FrameTensors color), flip the channels first.
    """
    return luma_u8(rgb).to(dtype) / 255


def load_png_luminance(path: str | Path, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """PNG -> (H, W) luminance exactly as flyvis_reference.py loads it."""
    from PIL import Image

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return luminance(torch.from_numpy(rgb.copy()), dtype)
