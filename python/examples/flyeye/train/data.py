"""The Phase 2 recordings as training windows.

Reads `C:\\dev\\_flyeye\\phase2\\<scenario>_<light>.npz` (format in `plans/flyeye-handoff.md`
section 3). Per recording: `receptors` (T, B, 721) f16 (PIL luma/255 of the BoxEye frame, the
convention the pretrained models were trained on), `flow_box` (T, B, 2, 721) rad/s on the
column-centre unit image tangents, `sky_frac` (T, B, 721), `seg`/`labels`, `w_body` (T, 3).

Target: `flow_box` mapped to the column centre's pixel velocity through `scoring.pixel_map`
(the exact pinhole Jacobian, which is what `scoring.py` scores against) and divided by 13 px,
so the units are columns/s, x right and y up.

    `target="columns"` (default)  the pixel velocity itself
    `target="unit"`               its unit direction where the true speed is at least
                                  `scoring.MIN_SPEED` (0.25 columns/s), zero below that

Masks: sky columns (`sky_frac >= 0.5`) are dropped from the loss. Rest frames are KEPT with
their target, which the recorder measured as exactly zero at rest - that is the hypothesis:
the training task now penalises a T4/T5 response to a still textured scene.

Lag: circuit frame k is trained against the truth at frame k - `lag` (default 2 frames,
20 ms), the lag `scoring.py`'s gate uses.

Windows: `Streams` keeps `n_streams` independent (recording, eye) streams. A stream starts at
a uniformly random frame of a uniformly random recording, is faded in there exactly as the
recorder did (`fade_in`, 100 steps at dt 0.01) and burnt in for `warmup` further frames under
`no_grad`, then yields `chunks` consecutive windows of `window` frames with the state carried
detached between them, before it is resampled. So every window is a real, causal 0.5 s of
flight and the fade-in cost is amortised over `chunks * window` trained frames.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import scoring as sc
from . import TEST_SCEN, TRAIN_SCEN

DT = 0.01
LAG = 2  # frames, the Phase 2 gate lag


class Recording:
    """One recording, on the GPU: receptors, target field, valid mask."""

    def __init__(self, key, M: np.ndarray, device="cuda", target="columns", lag=LAG, data=sc.DATA):
        self.key = key
        z, self.meta = sc.load(*key, data)
        rec = z["receptors"].astype(np.float32)
        vel = np.einsum("nkl,tbln->tbkn", M, z["flow_box"].astype(np.float64)) / sc.PX_PER_COLUMN
        speed = np.hypot(vel[:, :, 0], vel[:, :, 1])
        if target == "unit":
            g = np.where(speed >= sc.MIN_SPEED, 1.0 / np.maximum(speed, 1e-12), 0.0)
            vel = vel * g[:, :, None]
        elif target != "columns":
            raise ValueError(f"target {target!r}: 'columns' or 'unit'")
        valid = z["sky_frac"].astype(np.float32) < 0.5
        t = lambda a, d: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=d)
        self.receptors = t(rec, torch.float32)  # (T, B, 721)
        self.target = t(vel.astype(np.float32), torch.float32)  # (T, B, 2, 721)
        self.valid = t(valid, torch.bool)  # (T, B, 721)
        self.true_speed = t(speed.astype(np.float32), torch.float32)
        self.T, self.B = self.receptors.shape[:2]
        self.lag = lag
        self.labels = [(str(s).split("|")[0], int(str(s).split("|")[1]), int(str(s).split("|")[2]))
                       for s in z["labels"]]
        self.is_rest = torch.zeros(self.T, dtype=torch.bool, device=device)
        for name, a, b in self.labels:
            if name == "rest":
                self.is_rest[a:b] = True

    def frames(self, k: torch.Tensor, eye: int):
        """(receptors (K, 721), target (K, 2, 721), valid (K, 721)) at circuit frames k, lagged truth."""
        kt = (k - self.lag).clamp_min(0)
        return self.receptors[k, eye], self.target[kt, eye], self.valid[kt, eye]


class Data:
    """Recordings for a split, loaded on demand."""

    def __init__(self, scenarios=TRAIN_SCEN, lights=sc.LIGHTS, device="cuda", target="columns",
                 lag=LAG, data=sc.DATA):
        self.M = sc.pixel_map()
        self.scenarios, self.lights = tuple(scenarios), tuple(lights)
        self.device, self.target, self.lag, self.root = device, target, lag, data
        self._r = {}

    def keys(self):
        return [(s, l) for l in self.lights for s in self.scenarios]

    def __getitem__(self, key) -> Recording:
        if key not in self._r:
            self._r[key] = Recording(key, self.M, self.device, self.target, self.lag, self.root)
        return self._r[key]

    def load_all(self):
        for k in self.keys():
            self[k]
        return self

    @property
    def n_frames(self):
        return sum(self[k].T for k in self.keys())


class Stream:
    """One (recording, eye) cursor with its carried state."""

    __slots__ = ("run", "eye", "pos", "left", "v")

    def __init__(self):
        self.run, self.eye, self.pos, self.left, self.v = None, 0, 0, 0, None


class Streams:
    """A batch of `n` streams over a Data split; `batch()` returns one training window."""

    def __init__(self, data: Data, lobe, n=8, window=50, warmup=100, chunks=4, dt=DT, seed=0):
        self.data, self.lobe = data.load_all(), lobe
        self.n, self.window, self.warmup, self.chunks, self.dt = n, window, warmup, chunks, dt
        self.rng = np.random.default_rng(seed)
        self.keys = data.keys()
        self.streams = [Stream() for _ in range(n)]
        self.fade_steps = 0

    @torch.no_grad()
    def _assign(self, todo: list[Stream]):
        """Give every stream in `todo` a fresh (recording, eye, start frame), fade it in there
        and burn in `warmup` frames. Batched over the streams: one fade-in and one warm-up for
        all of them, not one each."""
        need = self.warmup + self.chunks * self.window
        for s in todo:
            run = self.data[self.keys[self.rng.integers(len(self.keys))]]
            s.run, s.eye = run, int(self.rng.integers(run.B))
            s.pos = int(self.rng.integers(0, max(1, run.T - need)))
            s.left = self.chunks
        x0 = torch.stack([s.run.receptors[s.pos, s.eye] for s in todo])
        v = self.lobe.fade_in(x0, self.dt)
        self.fade_steps += int(1.0 / self.dt) * len(todo)
        A = self.lobe.matrix()
        inv_tau, bias_n = self.lobe.node_params(self.dt)
        for j in range(self.warmup):
            x = torch.stack([s.run.receptors[min(s.pos + j, s.run.T - 1), s.eye] for s in todo])
            v = self.lobe.step(v, x, self.dt, A, inv_tau, bias_n)
        for i, s in enumerate(todo):
            s.v = v[i]
            s.pos = min(s.pos + self.warmup, s.run.T - self.window)

    def batch(self):
        """(receptors (W, n, 721), target (W, n, 2, 721), valid (W, n, 721), v0 (n, N), frames).

        A stream whose state is not finite, or which has used up its chunks, is resampled.
        """
        todo = [s for s in self.streams
                if s.run is None or s.left <= 0 or s.pos + self.window > s.run.T
                or not torch.isfinite(s.v).all()]
        if todo:
            self._assign(todo)
        W = self.window
        xs, ys, ms = [], [], []
        for s in self.streams:
            k = torch.arange(s.pos, s.pos + W, device=s.run.receptors.device)
            x, y, m = s.run.frames(k, s.eye)
            xs.append(x)
            ys.append(y)
            ms.append(m)
        x = torch.stack(xs, 1)
        y = torch.stack(ys, 1)
        m = torch.stack(ms, 1)
        v0 = torch.stack([s.v for s in self.streams]).detach()
        return x, y, m, v0

    def advance(self, v_end: torch.Tensor):
        for i, s in enumerate(self.streams):
            s.v = v_end[i].detach()
            s.pos += self.window
            s.left -= 1


def rest_probe(data: Data, key=("yaw", "bright"), frames=sc.REST, eye=0):
    """A fixed textured-rest window (receptors (K, 1, 721), valid (K, 721)) for the static-response log."""
    run = data[key]
    k = torch.arange(frames.start, frames.stop, device=run.receptors.device)
    return run.receptors[k, eye][:, None], run.valid[k, eye]


def split_note():
    return dict(train_scenarios=list(TRAIN_SCEN), test_scenarios=list(TEST_SCEN),
                lights=list(sc.LIGHTS), note="all three lightings in both splits; trajectories held out")
