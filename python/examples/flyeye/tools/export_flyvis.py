"""Export the pretrained flyvis optic-lobe model to a flyvis-free npz.

This is the ONLY file in the flyeye package that imports flyvis. It runs once, offline,
in a throwaway venv (Python 3.12, CPU torch, ``pip install flyvis``, pretrained models
downloaded), never in the threepp interpreter:

    set FLYVIS_ROOT_DIR=C:\\dev\\_flyvis\\data
    C:\\dev\\_flyvis\\venv\\Scripts\\python.exe python/examples/flyeye/tools/export_flyvis.py

Output: python/examples/flyeye/data/flyeye_model.npz (np.savez_compressed).

Other ensemble members (flow/0000/000..049; the directory index IS the rank by minimum
validation loss, 000 best: validation/loss.h5 is strictly increasing with the index):

    ... export_flyvis.py --member 003 --out C:\\dev\\_flyeye\\ensemble\\models\\flow_0000_003.npz
    ... export_flyvis.py --member 000,001,002 --out-dir C:\\dev\\_flyeye\\ensemble\\models
    ... export_flyvis.py --model flow/0000/017 --out some.npz

--out-dir writes <dir>/flow_0000_NNN.npz (the model path with "/" -> "_"). A model other
than the default flow/0000/000 needs --out or --out-dir, so the committed model is never
overwritten by accident. With no arguments the behaviour is unchanged.

Everything the runtime needs to reproduce flyvis 1.2.0 exactly is in the npz:

  Dynamics (flyvis/network/dynamics.py:207-218 + network.py:404-411), per node i,
      tau_eff_i = max(time_const_i, float32(dt))
      r         = relu(v)
      v_next    = v + dt * (1/tau_eff) * (-v + bias + scatter_add(post, weight * r[pre]) + x)
  with x non-zero only on the 8 receptor types R1..R8, which all receive the SAME BoxEye
  column value (flyvis/network/stimulus.py:128-131, 206), no scaling or offset.

  weight = sign * syn_count * syn_strength exactly as PPNeuronIGRSynapses.
  write_derived_params computes it (dynamics.py:165-167), taken from the model's own
  parameter API so the float32 bits are the ones flyvis uses.

  BoxEye (flyvis/datasets/rendering/eye.py:47-171): 13x13 box mean with zero padding
  (pad 6 on every side), then sampled at 721 hex centres (y, x) + (H//2, W//2), in the
  node (u, v) order of the receptor layers.

A self-check at the end reloads the npz with numpy only and compares one dynamics step
and one BoxEye frame against flyvis. Pass --no-check to skip.
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


def _patch_datamate_windows() -> None:
    """datamate 1.0.0 _write_h5 unlinks a file it still holds open on its error path,
    which raises WinError 32 on Windows the first time flyvis builds its connectome
    cache. Replace it with a writer that never holds the handle across the unlink."""
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


_patch_datamate_windows()

import flyvis  # noqa: E402
from flyvis.datasets.rendering import BoxEye  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "data" / "flyeye_model.npz"
MODEL = "flow/0000/000"


def build_input_convention(eye: BoxEye, dt_trained: float) -> dict:
    return {
        "luminance": (
            "Grayscale in [0, 1]: PIL Image.open(png).convert('L') -> float32 / 255 "
            "(flyvis/datasets/sintel_utils.py:57 sample_lum). ITU-R 601-2 luma of the "
            "display-encoded sRGB image; NOT linearised, NOT log-transformed, no mean "
            "subtraction. 0.5 is the neutral grey the network rests at."
        ),
        "training_data": (
            "Sintel 'final' pass PNGs (training/final), 24 fps. Per sequence: center "
            "crop 0.7, 3 overlapping vertical splits of width 391+2*13=417 px; the "
            "cropped height (~305 px) is below BoxEye.min_frame_size 391, so BoxEye "
            "resizes each split to 391x391 with torchvision.transforms.functional.resize "
            "(bilinear, antialias) before the 13x13 box mean (sintel.py:102-122, "
            "eye.py:125-127). Augmentation on the 721 hexals, in this order "
            "(sintel.py:578-587): random temporal crop of 19 frames; gaussian pixel "
            "noise std 0.08 then clamp(min=0) (augmentation/hex.py:310-322); contrast c = "
            "exp(N(0, 0.2)) and brightness b = N(0, 0.1): c*(x-0.5)+0.5+c*b then "
            "clamp(min=0) (hex.py:245-260); hex flip p=0.5 over axes 0-3; hex rotation "
            "by n*60 deg p=0.5; piecewise-constant (nearest-exact) resampling from 24 fps "
            "to 1/dt = 50 fps (sintel.py:400-406)."
        ),
        "receptor_nodes": (
            "Every one of R1, R2, R3, R4, R5, R6, R7, R8 receives the same BoxEye value "
            "for its column: x[receptor_node_index[k, j]] = boxeye[j] for k in 0..7. "
            "All other nodes get x = 0. No gain, no offset (stimulus.py:128-131, 206)."
        ),
        "boxeye": {
            "extent": int(eye.extent),
            "kernel_size": int(eye.kernel_size),
            "ftype": "mean",
            "pad_left_right_top_bottom": [int(p) for p in eye.pad],
            "padding_value": 0.0,
            "division": "sum over the zero-padded 13x13 window / 169 (also at borders)",
            "min_frame_size_hw": [int(s) for s in eye.min_frame_size.tolist()],
            "smaller_frames": "resized to min_frame_size with ttf.resize before the box",
            "sample": (
                "hexal j = filtered[H//2 + receptor_centers[j,0], W//2 + "
                "receptor_centers[j,1]] (row, col). receptor_centers[j] = (long(13*(u+v/2)), "
                "13*v): python float -> torch.long truncates toward zero, so the half-"
                "pixel rows of odd v round toward the frame centre row."
            ),
            "order": (
                "u in -15..15 outer, v in max(-15,-15-u)..min(15,15-u) inner; this is "
                "exactly the node order of every 721-node layer (checked on export)."
            ),
            "for_403x403": (
                "centre (201, 201); rows and cols of the centres span 6..396, so every "
                "13x13 window is fully inside the frame: no padding pixels contribute."
            ),
        },
        "fade_in": (
            "network.fade_in_state(t_fade_in, dt, movie[[0]]) (network.py:588-626): "
            "state starts at v = bias (dynamics.py:183); n = int(t_fade_in / dt) steps; "
            "frame k input = linspace(0, 1, n)[k] * (frame0 - 0.5) + 0.5 (torch.linspace in the "
            "default dtype, float32 normally); returns the state after the last of the n steps."
        ),
        "simulate": (
            "network.simulate(movie[None], dt, initial_state) (network.py:628-696, "
            "535-546): one Euler step per movie frame, frame i held for exactly one dt, "
            "no interpolation or resampling; response[i] is the state AFTER consuming "
            "frame i (the initial state is not in the output)."
        ),
        "dt_trained": dt_trained,
        "dtype": "float32 everywhere in flyvis (params, stimulus buffer, target_sum zeros)",
    }


def export(out: Path, check: bool = True, model: str = MODEL) -> None:
    t0 = time.time()
    model_dir = flyvis.results_dir / model
    nv = flyvis.NetworkView(model_dir)
    net = nv.init_network()
    chkpt = nv.get_checkpoint("best")
    net.clamp()  # exactly what Network.forward does first (network.py:527)
    net.eval()
    conn = net.connectome

    with torch.no_grad():
        params = net._param_api()
        tau = params.nodes.time_const.detach().cpu().numpy().astype(np.float32)
        bias = params.nodes.bias.detach().cpu().numpy().astype(np.float32)
        weight = params.edges.weight.detach().cpu().numpy().astype(np.float32)

    type_names = conn.unique_cell_types[:].astype(str)
    node_types = conn.nodes.type[:].astype(str)
    name_to_idx = {n: i for i, n in enumerate(type_names)}
    cell_type = np.array([name_to_idx[t] for t in node_types], dtype=np.int16)
    u = conn.nodes.u[:]
    v = conn.nodes.v[:]
    assert u.min() >= -128 and u.max() <= 127 and v.min() >= -128 and v.max() <= 127
    input_types = conn.input_cell_types[:].astype(str)
    is_receptor = np.isin(node_types, input_types)

    pre = conn.edges.source_index[:].astype(np.int64)
    post = conn.edges.target_index[:].astype(np.int64)
    assert pre.max() < 2**31 and post.max() < 2**31

    # Parameter sharing, for exact float64 reproduction and for inspection:
    # sign and syn_strength are per edge TYPE (source_type, target_type), syn_count is per
    # (source_type, target_type, du, dv) and stored as log(mean count) with
    # syn_count = exp(raw) (initialization.py:199-202, 451). weight above is the float32
    # product flyvis uses; a float64 run recomputes sign * exp(float64(raw)) * strength.
    ep = net.edge_params
    edge_type_index = ep["sign"].indices.cpu().numpy()
    assert np.array_equal(edge_type_index, ep["syn_strength"].indices.cpu().numpy())
    edge_count_index = ep["syn_count"].indices.cpu().numpy()
    assert edge_type_index.max() < 2**15 and edge_count_index.max() < 2**15
    edge_type_sign = net.edges_sign.detach().cpu().numpy().astype(np.float32)
    edge_type_syn_strength = net.edges_syn_strength.detach().cpu().numpy().astype(np.float32)
    edge_count_log = net.edges_syn_count.detach().cpu().numpy().astype(np.float32)
    with torch.no_grad():
        w32 = (
            torch.from_numpy(edge_type_sign)[edge_type_index]
            * torch.from_numpy(edge_count_log).exp()[edge_count_index]
            * torch.from_numpy(edge_type_syn_strength)[edge_type_index]
        ).numpy()
    assert np.array_equal(w32, weight), "per-type expansion != model weight"
    print("per-edge-type expansion (torch float32) == model weight: exact")

    receptor_node_index = np.asarray(net.stimulus.input_index, dtype=np.int32)  # (8, 721)
    assert np.array_equal(np.sort(receptor_node_index.ravel()), np.nonzero(is_receptor)[0])

    eye = BoxEye(extent=15, kernel_size=13)
    rc = eye.receptor_centers.cpu().numpy().astype(np.int32)  # (721, 2) (y, x)
    col_uv = []
    for uu in range(-eye.extent, eye.extent + 1):
        for vv in range(max(-eye.extent, -eye.extent - uu), min(eye.extent, eye.extent - uu) + 1):
            col_uv.append((uu, vv))
    col_uv = np.array(col_uv, dtype=np.int32)
    for k in range(receptor_node_index.shape[0]):
        idx = receptor_node_index[k]
        assert np.array_equal(np.stack([u[idx], v[idx]], 1), col_uv), input_types[k]

    dt_trained = float(nv.dir.config.task.dataset.dt)
    convention = build_input_convention(eye, dt_trained)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        # nodes, flyvis node index order
        cell_type=cell_type,
        u=u.astype(np.int8),
        v=v.astype(np.int8),
        time_const=tau,
        bias=bias,
        is_receptor=is_receptor,
        tau_clamp_rule=np.array("tau_eff = max(time_const, float32(dt))"),
        # edges, flyvis edge order
        pre=pre.astype(np.int32),
        post=post.astype(np.int32),
        weight=weight,
        # shared edge parameters (weight == sign[ti] * exp(count_log[ci]) * strength[ti])
        edge_type_index=edge_type_index.astype(np.int16),
        edge_count_index=edge_count_index.astype(np.int16),
        edge_type_sign=edge_type_sign,
        edge_type_syn_strength=edge_type_syn_strength,
        edge_count_log=edge_count_log,
        # names and provenance
        type_names=type_names.astype("U16"),
        input_type_names=input_types.astype("U16"),
        dt_trained=np.float32(dt_trained),
        model_path=np.array(model),
        checkpoint=np.array(str(Path(chkpt).relative_to(flyvis.results_dir)).replace("\\", "/")),
        flyvis_version=np.array(flyvis.__version__),
        connectome_file=np.array(str(conn.config.file)),
        input_convention=np.array(json.dumps(convention)),
        # stimulus path
        receptor_node_index=receptor_node_index,
        boxeye_extent=np.int32(eye.extent),
        boxeye_kernel_size=np.int32(eye.kernel_size),
        boxeye_receptor_centers=rc,
        boxeye_column_uv=col_uv.astype(np.int8),
        boxeye_min_frame_size=np.asarray(eye.min_frame_size.tolist(), dtype=np.int32),
        boxeye_pad=np.asarray(eye.pad, dtype=np.int32),
        boxeye_pixel_rc_403=(rc + 403 // 2).astype(np.int32),
    )
    size = out.stat().st_size
    print(f"wrote {out} ({size} bytes, {size / 1e6:.2f} MB) in {time.time() - t0:.1f} s")
    print(f"nodes {len(cell_type)} edges {len(pre)} types {len(type_names)} dt_trained {dt_trained}")
    print(
        "params: time_const", tuple(net.nodes_time_const.shape),
        "bias", tuple(net.nodes_bias.shape),
        "sign", tuple(net.edges_sign.shape),
        "syn_count", tuple(net.edges_syn_count.shape),
        "syn_strength", tuple(net.edges_syn_strength.shape),
    )

    if check:
        self_check(net, eye, out)


def _numpy_step(m, v, x, dt, dtype):
    n = len(m["cell_type"])
    tau = np.maximum(m["time_const"].astype(dtype), dtype(np.float32(dt)))
    r = np.maximum(v, 0)
    syn = np.bincount(m["post"], weights=(m["weight"].astype(dtype) * r[m["pre"]]), minlength=n)
    return v + dt * (1.0 / tau) * (-v + m["bias"].astype(dtype) + syn.astype(dtype) + x)


def self_check(net, eye, path: Path) -> None:
    """Single dynamics step and one BoxEye frame, flyvis vs numpy-from-npz."""
    from flyvis.utils.tensor_utils import AutoDeref

    m = dict(np.load(path))
    n = len(m["cell_type"])
    rng = np.random.default_rng(0)
    for dt in (0.02, 0.01, 1 / 60):
        v0 = rng.normal(0.0, 1.0, n).astype(np.float32)
        x_col = rng.uniform(0.0, 1.0, 721).astype(np.float32)
        x = np.zeros(n, np.float32)
        x[m["receptor_node_index"]] = x_col[None, :]
        with torch.no_grad():
            params = net._param_api()
            state = net._state_api(
                AutoDeref(nodes=AutoDeref(activity=torch.from_numpy(v0)[None]), edges=AutoDeref())
            )
            nxt = net._next_state(params, state, torch.from_numpy(x)[None], dt)
            ref = nxt.nodes.activity[0].cpu().numpy()
        own32 = _numpy_step(m, v0, x, dt, np.float32).astype(np.float32)
        own64 = _numpy_step(m, v0.astype(np.float64), x.astype(np.float64), dt, np.float64)
        d32 = float(np.abs(own32 - ref).max())
        d64 = float(np.abs(own64 - ref.astype(np.float64)).max())
        print(
            f"single step dt={dt:.5f}: max|own_float32 - flyvis| = {d32:.3e}, "
            f"max|own_float64 - flyvis| = {d64:.3e}, max|dv| = {np.abs(ref - v0).max():.3e}"
        )

    frame = rng.uniform(0.0, 1.0, (403, 403)).astype(np.float32)
    ref = eye(torch.from_numpy(frame)[None, None]).reshape(-1).cpu().numpy()
    k = int(m["boxeye_kernel_size"]) // 2
    csum = np.zeros((404, 404), np.float64)
    csum[1:, 1:] = frame.astype(np.float64).cumsum(0).cumsum(1)
    rr, cc = m["boxeye_pixel_rc_403"][:, 0], m["boxeye_pixel_rc_403"][:, 1]
    box = (
        csum[rr + k + 1, cc + k + 1] - csum[rr - k, cc + k + 1]
        - csum[rr + k + 1, cc - k] + csum[rr - k, cc - k]
    ) / 169.0
    print(f"BoxEye 403x403 random frame: max|own - flyvis| = {np.abs(box - ref).max():.3e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help=f"output npz (default {DEFAULT_OUT} for {MODEL} only)")
    ap.add_argument("--out-dir", type=Path, default=None, help="directory for <model path with _>.npz, one per model")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--member", default=None, help="flow/0000 member index NNN, or a comma list")
    sel.add_argument("--model", default=None, help="model path relative to flyvis.results_dir, e.g. flow/0000/017")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)
    if args.member:
        models = [f"flow/0000/{int(m):03d}" for m in args.member.split(",") if m.strip()]
    else:
        models = [args.model or MODEL]
    if args.out is not None and args.out_dir is not None:
        ap.error("--out and --out-dir are exclusive")
    if args.out is not None and len(models) > 1:
        ap.error("several models need --out-dir")
    for model in models:
        if args.out_dir is not None:
            out = args.out_dir / (model.replace("/", "_") + ".npz")
        elif args.out is not None:
            out = args.out
        elif model == MODEL:
            out = DEFAULT_OUT
        else:
            ap.error(f"{model} is not the default model: pass --out or --out-dir")
        print(f"== export {model} -> {out}", flush=True)
        export(out, check=not args.no_check, model=model)


if __name__ == "__main__":
    sys.exit(main())
