"""The readouts the training loss is computed through.

Two decoders, both implemented, selected with `--decoder`:

`LinearDecoder` ("linear")
    The repair round's readout without its adaptation stage: one shared 8 -> 2 matrix from the
    T4a-d/T5a-d rates (rest-subtracted, exactly as `readouts.MotionField` does) to the image
    field, trained jointly with the lobe. No per-column parameters and no temporal filter, so
    anything that suppresses a static pattern has to happen in the circuit, which is the
    hypothesis under test. Initialised from the fitted map of "V3c adapt tau=2.0 + V2" in
    `C:\\dev\\_flyeye\\repair\\variants_c.json` when that file is present, else from the unit
    opponent map (`readouts.MotionField`'s default).

`FlyvisDecoder` ("flyvis")
    flyvis's own flow decoder, reproduced in our torch from flyvis 1.2.0's source. Exact
    sources (wheel `flyvis-1.2.0-py3-none-any.whl`):

    - `flyvis/config/task/task.yaml:30-41` - `decoder.flow`: `type: DecoderGAVP`,
      `shape: [8, 2]`, `kernel_size: 5`, `const_weight: 0.001`, `n_out_features: null`,
      `p_dropout: 0.5`; `loss.flow: l2norm`.
    - `flyvis/task/decoder.py:190-332` - `DecoderGAVP`. `base` = one
      `Conv2dHexSpace(in=34, out=8, k=5, padding=2)` + `BatchNorm2d(8)` + `Softplus` +
      `Dropout(0.5)` (the default `batch_norm=True`, `activation="Softplus"`); `decoder` = one
      `Conv2dHexSpace(8, out_channels + 1 = 3, k=5, padding=2)`; `normalize_last=True` divides
      the 2 output channels by `softplus(extra channel) + 1`. Input is `relu` of the
      connectome's `output_cell_types` activity (`decoder.py:295-297`), stored in a
      31 x 31 hexagonal map (`decoder.py:300-310`).
    - `flyvis/task/decoder.py:112-187` - `Conv2dHexSpace`: the 5 x 5 kernel is masked to the
      19 positions of a hex of radius 2, and re-masked before every forward.
    - `flyvis/connectome/fib25-fib19_v2.2.json:24607` - `output_units`, the 34 cell types
      (`model.OUTPUT_TYPES`).
    - `flyvis/task/objectives.py:10-22` - `l2norm`: the mean over samples of
      `sqrt(sum over (frames, ndim, hexals) of squared error)`.
    - `flyvis/config/scheduler/scheduler.yaml` - Adam, lr 5e-5 for both the network and the
      decoder, stepwise down to 5e-6 over the run; `flyvis/config/penalizer/penalizer.yaml` +
      `flyvis/solver.py:849-886` - an activity penalty on `bias` only (SGD, separate step):
      `0.1 * mean(asymmetric_weighting(5.0 - mean central activity, 1.0, 0.1) ** 2)`,
      with `asymmetric_weighting(x, g, d) = g relu(x) - d relu(-x)`
      (`flyvis/utils/tensor_utils.py:443-461`).

    Reproduced here: the architecture, the input (34 relu'd output types on the hex map), the
    `l2norm` loss and the activity penalty (`activity_penalty` in `train.py`). NOT reproduced:
    flyvis's Sintel data pipeline, its augmentation, its dt of 0.02, its 250k iterations and
    its stepwise lr schedule (this is a 30-minute smoke on rendered flight at dt 0.01).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..readouts import MOTION_TYPES

VARIANTS_C = Path(r"C:\dev\_flyeye\repair\variants_c.json")
V2_KEY = "V3c adapt tau=2.0 + V2"
UNIT_MAP = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]] * 2, dtype=np.float32)  # MotionField unit gains


def hex_coords(radius: int):
    """Axial (u, v) of a hexagon of the given radius, flyvis `utils.hex_utils.get_hex_coords`."""
    u, v = [], []
    for a in range(-radius, radius + 1):
        for b in range(max(-radius, -a - radius), min(radius, -a + radius) + 1):
            u.append(a)
            v.append(b)
    return np.array(u), np.array(v)


def initial_map(path: Path = VARIANTS_C, key: str = V2_KEY) -> tuple[np.ndarray, str]:
    """(8, 2) initial 8 -> 2 map and where it came from."""
    if path.exists():
        d = json.loads(path.read_text())
        if key in d and d[key].get("fitted", {}).get("W"):
            return np.array(d[key]["fitted"]["W"], np.float32), f"{path.name} [{key}]"
    return UNIT_MAP.copy(), "unit opponent map (MotionField default)"


class LinearDecoder(nn.Module):
    """(..., 8, 721) T4/T5 rates -> (..., 2, 721) image field, one shared 8 -> 2 matrix."""

    types = MOTION_TYPES

    def __init__(self, rest: torch.Tensor, init: np.ndarray | None = None, source="given"):
        super().__init__()
        W, src = (init, source) if init is not None else initial_map()
        self.W = nn.Parameter(torch.tensor(np.asarray(W, np.float32)))
        self.register_buffer("rest", rest.clone())
        self.init_source = src

    def set_rest(self, rest: torch.Tensor):
        self.rest.copy_(rest)

    def rescale(self, factor: float):
        with torch.no_grad():
            self.W.mul_(float(factor))

    def forward(self, rates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...kn,kc->...cn", rates - self.rest, self.W)

    def describe(self) -> dict:
        W = self.W.detach().cpu().numpy()
        return dict(kind="linear", init_source=self.init_source, W=W.round(5).tolist(),
                    angles_deg=np.degrees(np.arctan2(W[:, 1], W[:, 0])).round(1).tolist(),
                    norms=np.hypot(W[:, 0], W[:, 1]).round(4).tolist(),
                    n_params=sum(p.numel() for p in self.parameters()))


class Conv2dHexSpace(nn.Conv2d):
    """flyvis `task/decoder.py:112-187`: a Conv2d whose odd kernel is masked to a hex."""

    def __init__(self, in_ch, out_ch, kernel_size, const_weight=1e-3, padding=0):
        super().__init__(in_ch, out_ch, kernel_size, padding=padding)
        if kernel_size % 2 == 0:
            raise ValueError(f"{kernel_size} is even; must be odd")
        if const_weight is not None:
            self.weight.data.fill_(const_weight)
            self.bias.data.fill_(const_weight)
        if kernel_size > 1:
            u, v = hex_coords(kernel_size // 2)
            mask = np.zeros(tuple(self.weight.shape), np.float32)
            mask[:, :, u - u.min(), v - v.min()] = 1
            self.register_buffer("mask", torch.from_numpy(mask))
            self.weight.data.mul_(self.mask)
            self._hex = True
        else:
            self._hex = False

    def forward(self, x):
        w = self.weight * self.mask if self._hex else self.weight
        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding)


class FlyvisDecoder(nn.Module):
    """flyvis DecoderGAVP(shape=[8, 2], kernel_size=5) on the 34 output cell types."""

    def __init__(self, u: np.ndarray, v: np.ndarray, in_ch=34, shape=(8, 2), kernel_size=5,
                 const_weight=1e-3, p_dropout=0.5, batch_norm=True, normalize_last=True):
        super().__init__()
        p = (kernel_size - 1) // 2
        self.register_buffer("mu", torch.from_numpy((u - u.min()).astype(np.int64)))
        self.register_buffer("mv", torch.from_numpy((v - v.min()).astype(np.int64)))
        self.H, self.W_map = int(u.max() - u.min() + 1), int(v.max() - v.min() + 1)
        self.out_channels = shape[-1]
        self.normalize_last = normalize_last
        base = []
        c_in = in_ch
        for c in shape[:-1]:
            base += [Conv2dHexSpace(c_in, c, kernel_size, const_weight, padding=p)]
            if batch_norm:
                base += [nn.BatchNorm2d(c)]
            base += [nn.Softplus()]
            if p_dropout:
                base += [nn.Dropout(p_dropout)]
            c_in = c
        self.base = nn.Sequential(*base)
        self.decoder = nn.Sequential(Conv2dHexSpace(c_in, self.out_channels + (1 if normalize_last else 0),
                                                    kernel_size, const_weight, padding=p))

    def forward(self, rates: torch.Tensor) -> torch.Tensor:
        """rates (T, B, 34, 721) relu(v), not rest-subtracted -> (T, B, 2, 721)."""
        T, B, C, N = rates.shape
        x = torch.relu(rates).reshape(T * B, C, N)
        m = x.new_zeros(T * B, C, self.H, self.W_map)
        m[:, :, self.mu, self.mv] = x
        out = self.decoder(self.base(m))
        if self.normalize_last:
            out = out[:, :self.out_channels] / (nn.functional.softplus(out[:, self.out_channels:]) + 1)
        return out[:, :, self.mu, self.mv].reshape(T, B, self.out_channels, N)

    def describe(self) -> dict:
        return dict(kind="flyvis", shape=[8, 2], kernel_size=5, const_weight=1e-3, p_dropout=0.5,
                    batch_norm=True, normalize_last=True, in_channels=34,
                    map=[self.H, self.W_map], n_params=sum(p.numel() for p in self.parameters()))


def build(kind: str, lobe, dt=0.01):
    """(decoder, gather index (K, 721), the cell types it reads)."""
    from .model import OUTPUT_TYPES

    if kind == "linear":
        idx = lobe.gather_index(MOTION_TYPES)
        return LinearDecoder(lobe.rest_rates(idx, dt)).to(lobe.device), idx, list(MOTION_TYPES)
    if kind == "flyvis":
        idx = lobe.gather_index(OUTPUT_TYPES)
        u = lobe.u[idx[0]].cpu().numpy()
        v = lobe.v_coord[idx[0]].cpu().numpy()
        return FlyvisDecoder(u, v, in_ch=len(OUTPUT_TYPES)).to(lobe.device), idx, list(OUTPUT_TYPES)
    raise ValueError(f"decoder {kind!r}: 'linear' or 'flyvis'")
