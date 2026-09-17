"""Sensor readouts on the lobula plate inputs: local motion field, wide-field rotation, looming.

Image space per column: x = right (+col), y = up (-row). Camera frame (threepp): looks down
-Z, +X right, +Y up. Vehicle frame = threepp object frame: -Z forward, +Y up, +X right.
eye_rotations (B, 3, 3) map each eye's camera frame into the vehicle frame.

A static world point at unit direction d moves at -w x d under vehicle rotation w. Its
image-space flow at a column is that 3D flow projected onto the column's unit image-right
and image-up tangents (lattice.column_tangents). That projection, not the full pixel
Jacobian, is the model the templates use; Phase 2 calibrates units and any residual.
"""

from __future__ import annotations

import numpy as np
import torch

from .lattice import HexLattice, default_lattice

MOTION_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")

# Preferred-direction peak of the central column above its pre-roll state, from the Phase 0
# edge gate (threepp-rendered ON/OFF edges at 13 columns/s, dt 1/100, README table).
PHASE0_PEAKS = dict(T4a=0.857, T4b=0.729, T4c=0.671, T4d=0.111, T5a=0.474, T5b=0.336, T5c=0.480, T5d=0.982)


class MotionField:
    """T4a-d/T5a-d rates -> (..., 2, 721) opponent flow estimate per column (x right, y up).

    raw(v) is relu(v) of the eight types minus their rest rate at uniform grey 0.5, in
    MOTION_TYPES order, without gains. Preferred directions (image space): a left, b right,
    c up, d down; x = (b - a), y = (c - d), each with ON (T4) and OFF (T5) summed.

    gains: None = all 1 (the default); "phase0" = 1 / PHASE0_PEAKS; or 8 numbers.
    Default None because on the Phase 0 movies the per-column vector in the 20 frames after
    the edge arrives is within 45 deg of truth for 75 % of inner columns with unit gains and
    66 % with "phase0" (a peak ratio at one speed does not carry over: T4d x9 flips off_up).
    """

    def __init__(self, lobe, gains=None, dt: float = 0.01):
        self.types = MOTION_TYPES
        self.index = torch.stack([lobe.type_index(n) for n in self.types])  # (8, 721)
        saved = lobe.v.clone()
        grey = torch.full((self.index.shape[1],), 0.5, dtype=lobe.dtype, device=lobe.device)
        v = lobe.fade_in(grey, dt)  # rest state; one more second of grey changes rates by < 3.3e-3
        self.rest = torch.relu(v[self.index]).clone()  # (8, 721)
        lobe.v = saved
        if gains is None:
            g = [1.0] * 8
        elif isinstance(gains, str) and gains == "phase0":
            g = [1 / PHASE0_PEAKS[n] for n in self.types]
        else:
            g = [float(x) for x in gains]
        self.gains = torch.tensor(g, dtype=lobe.dtype, device=lobe.device)

    def raw(self, v: torch.Tensor) -> torch.Tensor:
        """v (N,) or (B, N) -> (..., 8, 721) rest-subtracted rates, no gains."""
        return torch.relu(v[..., self.index]) - self.rest

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        r = self.raw(v) * self.gains[:, None]
        x = r[..., 1, :] - r[..., 0, :] + r[..., 5, :] - r[..., 4, :]
        y = r[..., 2, :] - r[..., 3, :] + r[..., 6, :] - r[..., 7, :]
        return torch.stack([x, y], dim=-2)


def eye_geometry(size: int, fov_deg: float, eye_rotations, lattice: HexLattice | None = None):
    """Column rays (B, 721, 3) and image tangents (B, 721, 2, 3) in the vehicle frame, float64."""
    lat = lattice or default_lattice()
    R = torch.as_tensor(np.asarray(eye_rotations), dtype=torch.float64).reshape(-1, 3, 3)
    d = lat.column_rays(size, fov_deg, torch.float64)
    t = lat.column_tangents(size, fov_deg, torch.float64)
    return torch.einsum("bij,nj->bni", R, d), torch.einsum("bij,nkj->bnki", R, t)


def project_flow(flow3: torch.Tensor, tangents: torch.Tensor) -> torch.Tensor:
    """3D flow (..., B, 721, 3) on the tangents (B, 721, 2, 3) -> image-space field (..., B, 2, 721)."""
    return torch.einsum("...bni,bnki->...bkn", flow3, tangents)


class _Cached:
    """Keeps float64 filters and hands out a copy on the field's device and dtype."""

    def _on(self, name: str, like: torch.Tensor) -> torch.Tensor:
        key = (name, like.device, like.dtype)
        if key not in self._cache:
            self._cache[key] = getattr(self, name).to(device=like.device, dtype=like.dtype)
        return self._cache[key]


class RotationReadout(_Cached):
    """HS/VS-style wide-field rotation: field (..., B, 2, 721) -> (..., 3) estimate of (wx, wy, wz).

    templates[k] (3, B, 2, 721) is the field of a unit rotation about vehicle axis k.
    method "matched" (Franz-Krapp matched filters): <template_k, field> / <template_k, template_k>,
    so a pure unit rotation about k reads 1 on k plus the templates' overlap on the others.
    method "lstsq": least-squares w from the same templates (exact on a pure rotation field).
    """

    def __init__(self, size: int, fov_deg: float, eye_rotations, method: str = "matched", lattice=None):
        d, t = eye_geometry(size, fov_deg, eye_rotations, lattice)
        e = torch.eye(3, dtype=torch.float64)[:, None, None, :].expand(3, *d.shape)
        self.templates = project_flow(-torch.linalg.cross(e, d.expand(3, *d.shape), dim=-1), t)
        A = self.templates.reshape(3, -1)
        if method == "matched":
            f = A / (A * A).sum(1, keepdim=True)
        elif method == "lstsq":
            f = torch.linalg.pinv(A.T)  # (3, M)
        else:
            raise ValueError(f"method {method!r}: 'matched' or 'lstsq'")
        self.method = method
        self.filters = f.reshape(self.templates.shape)
        self._cache = {}

    def __call__(self, field: torch.Tensor) -> torch.Tensor:
        return torch.einsum("kbcn,...bcn->...k", self._on("filters", field), field)


def default_centres(step_deg: float = 10.0, max_yaw: float = 90.0, max_pitch: float = 60.0) -> torch.Tensor:
    """(C, 3) grid of vehicle-frame directions: yaw about +Y (positive = left), pitch up."""
    yaw = np.radians(np.arange(-max_yaw, max_yaw + 1e-9, step_deg))
    pitch = np.radians(np.arange(-max_pitch, max_pitch + 1e-9, step_deg))
    Y, P = np.meshgrid(yaw, pitch, indexing="ij")
    c = np.stack([-np.sin(Y) * np.cos(P), np.sin(P), -np.cos(Y) * np.cos(P)], axis=-1).reshape(-1, 3)
    return torch.from_numpy(c)


class Looming(_Cached):
    """LPLC2-style looming: field (B, 2, 721) -> dict(value, centre, values).

    For each centre c and each column within rf_deg of it, the flow is projected on the
    outward radial direction from c (the image-space version of the great-circle tangent
    pointing away from c). Columns are split into four cardinal quadrants around c (right,
    up, left, down in the vehicle frame), each quadrant is averaged and half-wave rectified,
    and the quadrants are combined with min: all four arms must see outward motion, so
    one-sided lateral flow gives about zero. value = max over centres, centre is the unit
    vehicle-frame direction that won, values is (C,).

    opponent (default True) averages the signed radial component before rectifying, so
    inward motion cancels outward motion in the same quadrant (LPLC2 gets this inhibition
    through LPi interneurons). opponent=False rectifies per column first. On a live
    textured scene the circuit's T4/T5 field is never zero, and per-column rectification
    turns that noise into a floor that rotation raises as much as an approach does
    (README, Phase 1 sanity check); on exact synthetic flow the two agree.

    Centres default to a 10 deg grid, kept only where the eyes cover all four quadrants
    (smallest quadrant has at least half the columns of the largest).
    """

    def __init__(self, size: int, fov_deg: float, eye_rotations, centres=None, rf_deg: float = 30.0, lattice=None,
                 opponent: bool = True):
        self.opponent = opponent
        d, t = eye_geometry(size, fov_deg, eye_rotations, lattice)
        B = d.shape[0]
        d, t = d.reshape(-1, 3), t.reshape(-1, 2, 3)  # (M, 3), (M, 2, 3)
        c = default_centres() if centres is None else torch.as_tensor(np.asarray(centres), dtype=torch.float64)
        c = c / c.norm(dim=1, keepdim=True)
        cos = c @ d.T  # (C, M)
        away = cos[..., None] * d[None] - c[:, None]  # tangent at d pointing away from c, |away| = sin
        radial = torch.einsum("cmi,mki->cmk", away, t)
        radial = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (C, M, 2)
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64).expand_as(c)
        e2 = up - (up * c).sum(1, keepdim=True) * c
        e2 = torch.where(e2.norm(dim=1, keepdim=True) > 1e-6, e2, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64))
        e2 = e2 / e2.norm(dim=1, keepdim=True)
        e1 = torch.linalg.cross(c, e2, dim=1)  # right of c (c forward, e2 up)
        ang = torch.atan2(d @ e2.T, d @ e1.T).T  # (C, M)
        quad = torch.remainder(torch.round(ang / (np.pi / 2)), 4).long()  # 0 right, 1 up, 2 left, 3 down
        inside = (cos >= np.cos(np.radians(rf_deg))) & (away.norm(dim=-1) > np.sin(np.radians(0.5)))
        mask = torch.stack([inside & (quad == q) for q in range(4)], dim=1).to(torch.float64)  # (C, 4, M)
        count = mask.sum(2)  # (C, 4)
        if centres is None:
            keep = (count.min(1).values > 0) & (count.min(1).values >= 0.5 * count.max(1).values)
            c, radial, mask, count = c[keep], radial[keep], mask[keep], count[keep]
        self.centres = c  # (C, 3)
        self.counts = count
        self.radial = radial.reshape(len(c), B, 721, 2).permute(0, 1, 3, 2).contiguous()  # (C, B, 2, 721)
        self.quadrants = (mask / count.clamp_min(1)[..., None]).reshape(len(c), 4, B, 721)  # mean weights
        self._cache = {}

    def __call__(self, field: torch.Tensor) -> dict:
        out = torch.einsum("cbkn,bkn->cbn", self._on("radial", field), field)
        if self.opponent:
            q = torch.relu(torch.einsum("cqbn,cbn->cq", self._on("quadrants", field), out))  # (C, 4)
        else:
            q = torch.einsum("cqbn,cbn->cq", self._on("quadrants", field), torch.relu(out))
        values = q.min(dim=1).values
        i = torch.argmax(values)
        return dict(value=values[i], centre=self._on("centres", field)[i], values=values)
