"""Ground truth per frame from the eye views' own AOVs, on the 721-column lattice.

Image space per column: x = right (+col), y = up (-row), as readouts.py. Flow is in rad/s:
the time derivative of the unit ray to a scene point, projected on the column's unit
image-right and image-up tangents (lattice.column_tangents), the model the templates use.

- AovFlow: motion AOV (prevNDC - currNDC, GL NDC y up) -> (2, 721) flow. mode "centre"
  samples the column-centre pixel (the Phase 1 sanity check); mode "box" converts every
  pixel of the column's 13x13 BoxEye footprint to flow and averages, projected on the
  centre tangents (the same box kernel the receptors use).
- Footprint: per column, the luminance (luma / 255) Michelson contrast (max - min) /
  (max + min) over the 13x13 footprint, and the fraction of footprint pixels with no
  geometry (reversed-Z depth 0 = sky).
- ForwardRange: distance to the scene ahead along the vehicle forward axis, from the
  reversed-Z depth AOV at the pixels within cone_deg of that axis; smooth_tau turns a
  range series into time to contact tau = z / (-dz/dt).

Motion at sky pixels: the AOV reads ZERO there, also under pure rotation (scenarios.py
--scout: 384 sky columns under a 90 deg/s yaw read 0 against 1.44 rad/s analytic), so
AovFlow takes depth and the camera's angular velocity and puts the flow of a point at
infinity, -w x ray, on sky pixels.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .lattice import HexLattice


def pixel_ndc(rows: torch.Tensor, cols: torch.Tensor, size: int) -> torch.Tensor:
    """(..., ) pixel rows and cols -> (..., 2) NDC (x right, y up), pixel centres."""
    return torch.stack([(cols + 0.5) / size * 2 - 1, 1 - (rows + 0.5) / size * 2], dim=-1)


def linear_depth(depth: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """Reversed-Z NDC in [0, 1] (1 = near, 0 = far) -> distance along the view axis, m.

    threepp's Vulkan projection is the GL one with z' = 0.5 w - 0.5 z, so
    ndc = n (f / D - 1) / (f - n) and D = n f / (ndc (f - n) + n)."""
    d = depth.double()
    return near * far / (d * (far - near) + near)


class AovFlow:
    """Motion AOV of one size x size, fov_deg eye -> (2, 721) angular flow, rad/s."""

    def __init__(self, lattice: HexLattice, dt: float, size: int = 403, fov_deg: float = 90.0, mode: str = "centre"):
        if mode not in ("centre", "box"):
            raise ValueError(f"mode {mode!r}: 'centre' or 'box'")
        rc = lattice.pixel_rc(size)
        k = torch.arange(lattice.kernel_size) - lattice.kernel_size // 2
        if mode == "centre":
            self.rows, self.cols = rc[:, 0, None], rc[:, 1, None]  # (721, 1)
        else:
            self.rows = (rc[:, 0, None, None] + k[None, :, None]).expand(-1, len(k), len(k)).reshape(len(rc), -1)
            self.cols = (rc[:, 1, None, None] + k[None, None, :]).expand(-1, len(k), len(k)).reshape(len(rc), -1)
        self.rows, self.cols = self.rows.cuda(), self.cols.cuda()  # (721, P)
        self.ndc = pixel_ndc(self.rows.double(), self.cols.double(), size)  # (721, P, 2)
        self.tan = math.tan(math.radians(fov_deg) / 2)
        self.tangents = lattice.column_tangents(size, fov_deg, torch.float64).cuda()  # (721, 2, 3)
        self.dt, self.mode = dt, mode

    def ray(self, ndc):
        p = torch.cat([ndc * self.tan, -torch.ones_like(ndc[..., :1])], dim=-1)
        return p / p.norm(dim=-1, keepdim=True)

    def __call__(self, motion: torch.Tensor, depth: torch.Tensor | None = None, w_cam=None) -> torch.Tensor:
        """motion (H, W, 4) -> (2, 721). With depth (H, W) and w_cam (3,) (the camera's angular
        velocity in its own frame, rad/s), sky pixels (depth 0), where the AOV reads zero,
        take the flow of a point at infinity, -w_cam x ray."""
        m = motion[self.rows, self.cols, :2].double()  # (721, P, 2)
        ray = self.ray(self.ndc)
        flow3 = (ray - self.ray(self.ndc + m)) / self.dt  # (721, P, 3)
        if depth is not None and w_cam is not None:
            w = torch.as_tensor(np.asarray(w_cam), dtype=torch.float64, device=ray.device).expand_as(ray)
            sky = (depth[self.rows, self.cols] == 0)[..., None]
            flow3 = torch.where(sky, -torch.linalg.cross(w, ray, dim=-1), flow3)
        return torch.einsum("ni,nki->kn", flow3.mean(1), self.tangents)


class Footprint:
    """Per-column Michelson contrast of luma / 255 and sky fraction over the 13x13 footprint."""

    def __init__(self, lattice: HexLattice, size: int = 403):
        rc = lattice.pixel_rc(size).cuda()
        self.rows, self.cols = rc[:, 0], rc[:, 1]
        self.k = lattice.kernel_size

    def _windows(self, img: torch.Tensor) -> torch.Tensor:
        """(H, W) -> (721, 169) values of each column's footprint."""
        h = self.k // 2
        pad = torch.nn.functional.pad(img[None, None], (h, h, h, h), mode="replicate")[0, 0]
        u = pad.unfold(0, self.k, 1).unfold(1, self.k, 1)  # (H, W, k, k), window centred at (r, c)
        return u[self.rows, self.cols].reshape(len(self.rows), -1)

    def contrast(self, lum: torch.Tensor) -> torch.Tensor:
        w = self._windows(lum)
        hi, lo = w.max(1).values, w.min(1).values
        return (hi - lo) / (hi + lo).clamp_min(1e-6)

    def sky_fraction(self, depth: torch.Tensor) -> torch.Tensor:
        return (self._windows(depth) == 0).float().mean(1)


class ForwardRange:
    """Range to the scene along the vehicle forward axis (-Z) from the eyes' depth AOVs.

    Uses every pixel whose ray is within cone_deg of the forward axis (on the +-45 rig a
    half disc at the inner edge of each eye, about 60 px per eye at 1.5 deg). Per pixel:
    view-axis depth D -> point D / (-ray_z) * ray in the camera frame -> vehicle frame ->
    forward component. __call__ returns (median, min) over those pixels of both eyes in m,
    sky pixels excluded (NaN if none). nearest(depths) is the smallest view-axis depth in
    any pixel of any eye, m (a clearance check: under ~2 m the vehicle is inside geometry)."""

    def __init__(self, eye_rotations, near: float, far: float, size: int = 403, fov_deg: float = 90.0,
                 cone_deg: float = 1.5):
        r = torch.arange(size, dtype=torch.float64)
        ndc = pixel_ndc(r[:, None].expand(size, size), r[None, :].expand(size, size), size)  # (H, W, 2)
        t = math.tan(math.radians(fov_deg) / 2)
        d = torch.cat([ndc * t, -torch.ones(size, size, 1, dtype=torch.float64)], dim=-1)
        d = d / d.norm(dim=-1, keepdim=True)
        R = torch.as_tensor(np.asarray(eye_rotations), dtype=torch.float64)  # (B, 3, 3)
        dv = torch.einsum("bij,hwj->bhwi", R, d)  # vehicle frame
        self.select = (-dv[..., 2] >= math.cos(math.radians(cone_deg))).cuda()  # (B, H, W)
        self.scale = (-dv[..., 2] / -d[None, ..., 2]).cuda()  # forward component per unit view-axis depth
        self.near, self.far = near, far
        self.count = [int(x) for x in self.select.sum((1, 2))]

    def __call__(self, depths) -> tuple[float, float]:
        z = []
        for b, dep in enumerate(depths):
            ok = self.select[b] & (dep > 0)
            z.append(linear_depth(dep[ok], self.near, self.far) * self.scale[b][ok])
        z = torch.cat(z)
        if z.numel() == 0:
            return float("nan"), float("nan")
        return float(z.median()), float(z.min())

    def nearest(self, depths) -> float:
        return float(linear_depth(torch.stack([d.max() for d in depths]).max(), self.near, self.far))


def smooth_tau(z: np.ndarray, dt: float, half: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """tau = z / (-dz/dt) with dz/dt from a local linear fit over +-half frames. Returns
    (tau, zdot); tau is inf where the range is not closing (zdot >= -1e-3 m/s)."""
    n = len(z)
    zdot = np.full(n, np.nan)
    for k in range(n):
        a, b = max(0, k - half), min(n, k + half + 1)
        y = z[a:b]
        ok = np.isfinite(y)
        if ok.sum() >= 3:
            x = (np.arange(a, b) - k)[ok] * dt
            zdot[k] = np.polyfit(x, y[ok], 1)[0]
    tau = np.where(zdot < -1e-3, z / np.maximum(-zdot, 1e-9), np.inf)
    return tau, zdot


def rotation_flow(w_body, eye_rotations, lattice: HexLattice, size: int = 403, fov_deg: float = 90.0) -> torch.Tensor:
    """(B, 2, 721) analytic flow of points at infinity under body rotation w (rad/s), float64."""
    from .readouts import eye_geometry, project_flow

    d, t = eye_geometry(size, fov_deg, eye_rotations, lattice)
    w = torch.as_tensor(np.asarray(w_body), dtype=torch.float64).expand_as(d)
    return project_flow(-torch.linalg.cross(w, d, dim=-1), t)
