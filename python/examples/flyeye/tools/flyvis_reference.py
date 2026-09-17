"""flyvis reference responses for a directory of PNG frames (Phase 0 ground truth).

Runs in the throwaway flyvis venv, never in the threepp interpreter:

    set FLYVIS_ROOT_DIR=C:\\dev\\_flyvis\\data
    C:\\dev\\_flyvis\\venv\\Scripts\\python.exe python/examples/flyeye/tools/flyvis_reference.py ^
        C:\\dev\\_flyeye\\synthetic\\frames 0.01 C:\\dev\\_flyeye\\synthetic\\ref_float32.npz

Pipeline, identical to flyvis tutorial 07 (custom stimuli) with the pretrained
flow/0000/000 network:

1. Luminance. Each PNG (sorted by file name) is converted exactly the way flyvis's own
   datasets load Sintel frames (flyvis/datasets/sintel_utils.py:57, sample_lum):
       lum = np.float32(PIL.Image.open(png).convert("L")) / 255
   'L' is PIL's ITU-R 601-2 luma, L = R*299/1000 + G*587/1000 + B*114/1000 on the
   8-bit display-encoded (sRGB) values, alpha ignored. No linearisation, no log, no mean
   subtraction; range [0, 1], neutral grey 0.5. In --dtype float64 the same 8-bit L
   values are divided in float64 (np.float64(L) / 255).
2. BoxEye(extent=15, kernel_size=13) on the (1, T, H, W) movie gives (1, T, 1, 721).
   A 403x403 frame needs no resize (min frame size 391).
3. state = network.fade_in_state(1.0, dt, movie[[0]]): v starts at the bias, then
   int(1.0/dt) Euler steps with input linspace(0,1,n)[k]*(frame0-0.5)+0.5.
4. responses = network.simulate(movie[None], dt, initial_state=state): one Euler step
   per frame; responses[i] is the state after frame i.

Saved (np.savez_compressed): movie (T,H,W), boxeye (T,721), initial_state (45669,),
responses (T,45669) in the run dtype, plus dt, dtype, n_fade_steps, frame names, timing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("FLYVIS_ROOT_DIR", r"C:\dev\_flyvis\data")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def _patch_datamate_windows() -> None:
    """Same Windows fix as export_flyvis.py (datamate 1.0.0 unlinks an open h5 file)."""
    import datamate.directory as dd
    import datamate.io as dio
    import h5py

    def _write_h5(path, val):
        val = np.asarray(val)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            path.rmdir()
        elif path.exists():
            path.unlink()
        with h5py.File(path, libver="latest", mode="w") as f:
            f["data"] = val
            f.swmr_mode = True

    dd._write_h5 = _write_h5
    dio._write_h5 = _write_h5


MODEL = "flow/0000/000"


def load_movie(frame_dir: Path, dtype) -> tuple[np.ndarray, list[str]]:
    files = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() == ".png")
    if not files:
        raise SystemExit(f"no PNG frames in {frame_dir}")
    frames = []
    for p in files:
        with Image.open(p) as im:
            lum8 = np.asarray(im.convert("L"))
        if dtype == np.float32:
            frames.append(np.float32(lum8) / 255)  # == flyvis sample_lum, bit for bit
        else:
            frames.append(np.float64(lum8) / 255)
    return np.stack(frames).astype(dtype), [p.name for p in files]


def run(frame_dir: Path, dt: float, out: Path, dtype_name: str, t_fade_in: float = 1.0) -> dict:
    dtype = np.float64 if dtype_name == "float64" else np.float32
    torch_dtype = torch.float64 if dtype_name == "float64" else torch.float32
    _patch_datamate_windows()
    import flyvis
    from flyvis.datasets.rendering import BoxEye

    # flyvis allocates its stimulus buffer, target_sum zeros and fade-in linspace with the
    # default dtype, so float64 needs the default switched before anything is built.
    torch.set_default_dtype(torch_dtype)

    movie_np, names = load_movie(frame_dir, dtype)
    T, H, W = movie_np.shape

    t0 = time.perf_counter()
    network = flyvis.NetworkView(flyvis.results_dir / MODEL).init_network()
    network = network.to(torch_dtype)
    t_init = time.perf_counter() - t0

    eye = BoxEye(extent=15, kernel_size=13)
    eye.conv = eye.conv.to(torch_dtype)
    with torch.no_grad():
        t0 = time.perf_counter()
        # (1, T, H, W) -> (1, T, 1, 721) -> (T, 1, 721), as in tutorial 07
        if dtype_name == "float64":
            # float64 conv2d on CPU falls back to an im2col path that tries to allocate
            # ~57 GB for 261 403x403 frames; the same op in chunks of frames is identical.
            movie = torch.cat([eye(torch.from_numpy(movie_np[i : i + 8])[None])[0] for i in range(0, T, 8)], dim=0)
        else:
            movie = eye(torch.from_numpy(movie_np)[None])[0]
        t_box = time.perf_counter() - t0

        t0 = time.perf_counter()
        state = network.fade_in_state(t_fade_in, dt, movie[[0]])
        t_fade = time.perf_counter() - t0

        t0 = time.perf_counter()
        responses = network.simulate(movie[None], dt, initial_state=state)  # (1, T, 45669)
        t_sim = time.perf_counter() - t0

    initial_state = state.nodes.activity[0]
    result = dict(
        movie=movie_np,
        boxeye=movie[:, 0, :].cpu().numpy().astype(dtype),
        initial_state=initial_state.cpu().numpy().astype(dtype),
        responses=responses[0].cpu().numpy().astype(dtype),
    )
    for k in ("boxeye", "initial_state", "responses"):
        assert result[k].dtype == dtype, (k, result[k].dtype)
    assert responses.dtype == torch_dtype, responses.dtype

    timing = dict(
        init_network_s=t_init,
        boxeye_s=t_box,
        fade_in_s=t_fade,
        fade_in_steps=int(t_fade_in / dt),
        simulate_s=t_sim,
        simulate_s_per_frame=t_sim / T,
        torch_threads=torch.get_num_threads(),
        device=str(flyvis.device),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **result,
        dt=np.float64(dt),
        dtype=np.array(dtype_name),
        t_fade_in=np.float64(t_fade_in),
        n_fade_steps=np.int32(int(t_fade_in / dt)),
        frame_names=np.array(names),
        model_path=np.array(MODEL),
        flyvis_version=np.array(flyvis.__version__),
        timing=np.array(json.dumps(timing)),
    )
    print(f"{out}: T={T} HxW={H}x{W} dtype={dtype_name} " + json.dumps(timing))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, help="directory of PNG frames, sorted by name")
    ap.add_argument("dt", type=float, help="integration step = seconds per frame")
    ap.add_argument("out", type=Path, help="output .npz")
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    ap.add_argument("--t-fade-in", type=float, default=1.0)
    args = ap.parse_args(argv)
    run(args.frames, args.dt, args.out, args.dtype, args.t_fade_in)


if __name__ == "__main__":
    sys.exit(main())
