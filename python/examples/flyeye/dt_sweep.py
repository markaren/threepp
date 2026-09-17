"""Phase 0: Euler dt stability of the own runtime, and OpticLobe.step timing.

    py -3.14 python/examples/flyeye/dt_sweep.py

A procedural edge movie with the rendered stimuli's geometry (403x403 px, grey 128, ON
204 / OFF 53, one-pixel area-coverage edge, 13 columns/s = 169 px/s, the same 0.19 s
pre-roll) is sampled at 60, 100 and 200 Hz and run at dt = 1/fps, followed by 3 s holding
the last frame. Reports whether each run stays finite and bounded, and the max deviation
of the central T4a-d / T5a-d traces from the dt = 1/200 run on the 1/60 time grid.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message=".*[Ss]parse.*")

from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.optic_lobe import OpticLobe  # noqa: E402

SIZE, GREY, ON, OFF, MARGIN = 403, 128, 204, 53, 2.0
PX_PER_S = 13 * 13.0  # 13 columns/s, 13 px per column
T_START = 0.19  # render_edges: edge at -margin until frame preroll-1 = 19 at 100 Hz
T_MOVIE = 2.61  # 261 frames at 100 Hz
T_HOLD = 3.0
TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def edge_frame(t, direction, level, device):
    """Luminance frame at time t: region of `level` entering from the border opposite to the motion."""
    e = -MARGIN + max(0.0, t - T_START) * PX_PER_S  # edge distance from the entering border, px
    idx = torch.arange(SIZE, device=device, dtype=torch.float64)
    cov = (e - idx).clamp(0, 1)  # area coverage of pixel [i, i+1) by [0, e)
    if direction in ("left", "up"):
        cov = cov.flip(0)
    lum = (GREY + cov * (level - GREY)) / 255
    if direction in ("right", "left"):
        return lum[None, :].expand(SIZE, SIZE).float()
    return lum[:, None].expand(SIZE, SIZE).float()


def run(lobe, lat, fps, direction, level):
    dt = 1.0 / fps
    n_movie = int(round(T_MOVIE * fps))
    n_hold = int(round(T_HOLD * fps))
    box = torch.stack([lat.box_eye(edge_frame(k * dt, direction, level, lobe.device)) for k in range(n_movie)])
    lobe.fade_in(box[0], dt)
    trace = np.empty((n_movie, len(TYPES)))
    vmax = 0.0
    finite = True
    for k in range(n_movie + n_hold):
        v = lobe.step(box[min(k, n_movie - 1)], dt)
        if k < n_movie:
            trace[k] = [lobe.central(t, v).item() for t in TYPES]
        if k % 10 == 0 or k == n_movie + n_hold - 1:
            vmax = max(vmax, v.abs().max().item())
            finite &= bool(torch.isfinite(v).all())
    t_axis = (np.arange(n_movie) + 1) * dt  # response k is the state after frame k
    return t_axis, trace, vmax, finite, v.abs().max().item()


def timing(device, n_warm=50, n=200):
    lobe = OpticLobe(device=device)
    x = torch.full((721,), 0.5, device=device)
    lobe.reset()
    ts = []
    for k in range(n_warm + n):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        lobe.step(x, 0.01)
        if device == "cuda":
            torch.cuda.synchronize()
        if k >= n_warm:
            ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)), 1e3 * float(np.percentile(ts, 90))


def main():
    only_timing, no_timing = "--timing-only" in sys.argv, "--no-timing" in sys.argv
    torch.set_num_threads(6)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lat = HexLattice()
    lobe = OpticLobe(device=device)
    if only_timing:
        return _timing()
    print(f"device {device}; min time_const {lobe.time_const.min().item() * 1e3:.2f} ms "
          f"(clamped at dt=1/60: {(lobe.time_const < 1 / 60).sum().item()} nodes)")
    res = {}
    for fps in (200, 100, 60):
        worst = 0.0
        for pol, level in (("on", ON), ("off", OFF)):
            for d in ("right", "left", "up", "down"):
                t_axis, tr, vmax, finite, vend = run(lobe, lat, fps, d, level)
                res[(fps, pol, d)] = (t_axis, tr)
                worst = max(worst, vmax)
                if not finite:
                    print(f"  fps {fps} {pol}_{d}: NON-FINITE")
        print(f"dt 1/{fps}: finite {all(np.isfinite(res[(fps, p, d)][1]).all() for p in ('on', 'off') for d in ('right', 'left', 'up', 'down'))}, "
              f"max |v| over all nodes and steps (incl. 3 s hold) {worst:.3f}")
    grid = res[(60, "on", "right")][0]
    grid = grid[grid <= T_MOVIE]
    for fps in (100, 60):
        dev_abs, amp = 0.0, 0.0
        per_type = np.zeros(len(TYPES))
        for pol in ("on", "off"):
            for d in ("right", "left", "up", "down"):
                t_ref, tr_ref = res[(200, pol, d)]
                t_x, tr_x = res[(fps, pol, d)]
                for j in range(len(TYPES)):
                    a = np.interp(grid, t_ref, tr_ref[:, j])
                    b = np.interp(grid, t_x, tr_x[:, j])
                    dd = np.abs(a - b).max()
                    per_type[j] = max(per_type[j], dd)
                    dev_abs = max(dev_abs, dd)
                    amp = max(amp, tr_ref[:, j].max() - tr_ref[0, j])
        print(f"dt 1/{fps} vs 1/200: max |dv| of central T4/T5 traces {dev_abs:.4f} "
              f"(largest T4/T5 response swing at 1/200: {amp:.3f}, ratio {dev_abs / amp:.3f}); per type "
              + " ".join(f"{t}:{x:.3f}" for t, x in zip(TYPES, per_type)))
    opp = dict(right="left", left="right", up="down", down="up")
    for fps in (200, 100, 60):
        cells = []
        for j, t in enumerate(TYPES):
            pol = "on" if t.startswith("T4") else "off"
            b = int(np.searchsorted(res[(fps, pol, "right")][0], T_START))  # first response after motion starts
            pk = {d: res[(fps, pol, d)][1][b:, j].max() - res[(fps, pol, d)][1][b - 1, j] for d in opp}
            pref = max(pk, key=pk.get)
            p, n = max(pk[pref], 0), max(pk[opp[pref]], 0)
            cells.append(f"{t} {pref} {(p - n) / (p + n):.2f}")
        print(f"dt 1/{fps} preferred direction and DSI: " + ", ".join(cells))
    if not no_timing:
        _timing()


def _timing():
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        med, p90 = timing(dev)
        print(f"OpticLobe.step {dev} float32: median {med:.3f} ms, p90 {p90:.3f} ms (200 steps after 50 warm-up)")


if __name__ == "__main__":
    main()
