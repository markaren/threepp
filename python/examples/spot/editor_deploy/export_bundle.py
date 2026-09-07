"""Export a trained checkpoint as a POLICY BUNDLE: weights + a spec that declares its contract.

    python export_bundle.py                       # spot_steps.pt -> ./bundle_spot_steps/
    python export_bundle.py --model other.pt --out ./bundle_other

The bundle is torch-free at run time (numpy forward pass), which is what lets the editor's
embedded interpreter run a policy without importing torch into the app. Two files:

    policy.npz   actor weights + the observation normalizer's mean/var
    spec.json    the contract: obs terms IN ORDER, action decoding, control rate, PD gains

Why a spec and not just weights: a policy is useless without the exact observation it was
trained on. The training env is the only place that knows that layout, so IT writes the
declaration and the runner interprets it. Everything a deployment needs to reproduce the
observation is named here rather than reimplemented (and drifting) per player.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPOT = os.path.dirname(_HERE)
_EXAMPLES = os.path.dirname(_SPOT)
_PYROOT = os.path.dirname(_EXAMPLES)
sys.path.insert(0, _PYROOT)
sys.path.insert(0, _SPOT)
sys.path.insert(0, os.path.join(_SPOT, "scratch_distillation"))

from threepp.rl import load_policy                                     # noqa: E402
from spot_deploy import (DEFAULT, ISAAC, ACTION_SCALE, STIFF_GAINS,    # noqa: E402
                         GAIT_PERIOD, LEGS, Z0)
from spot_terrain_env import (SCAN_GX, SCAN_GY, N_SCAN, CONTROL_HZ,    # noqa: E402
                             DT, OBS_DIM, ACT_DIM)

# The policy speaks ISAAC joint order and names joints "fl_hx"; the URDF (and therefore any
# articulation built from it) names them "fl.hx". One translation, stated once, exported.
URDF_JOINT = {L + "_" + j: L + "." + j for L in LEGS for j in ("hx", "hy", "kn")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_SPOT, "spot_steps.pt"))
    ap.add_argument("--out", default=os.path.join(_HERE, "bundle_spot_steps"))
    args = ap.parse_args()

    ac, norm, meta = load_policy(args.model, device="cpu")
    sd = ac.state_dict()
    os.makedirs(args.out, exist_ok=True)

    # actor.N.weight for the Linear layers only (ELU sits at the odd indices and has no params)
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    arrays, layers = {}, []
    for n, i in enumerate(idx):
        W = sd[f"actor.{i}.weight"].numpy().astype(np.float32)
        b = sd[f"actor.{i}.bias"].numpy().astype(np.float32)
        arrays[f"W{n}"], arrays[f"b{n}"] = W, b
        layers.append([int(W.shape[1]), int(W.shape[0])])
    if norm is not None:
        arrays["norm_mean"] = norm.mean.numpy().astype(np.float32)
        arrays["norm_var"] = norm.var.numpy().astype(np.float32)
    np.savez(os.path.join(args.out, "policy.npz"), **arrays)

    obs_dim = int(meta.get("obs_dim", OBS_DIM))
    assert layers[0][0] == obs_dim, f"actor input {layers[0][0]} != obs_dim {obs_dim}"
    assert layers[-1][1] == ACT_DIM, f"actor output {layers[-1][1]} != act_dim {ACT_DIM}"

    spec = {
        "format": "threepp-policy-bundle/1",
        "source": {"checkpoint": os.path.basename(args.model), "trainer": "threepp.rl PPO"},
        "policy": {
            "file": "policy.npz",
            "layers": layers,                 # [[in, out], ...] applied in order
            "activation": "elu",              # between layers, not after the last
            "deterministic": True,            # deploy uses the mean action, not a sample
        },
        # (x - mean) / sqrt(var + eps), clamped. Absent => no normalization.
        "normalizer": ({"kind": "running_mean_var", "eps": 1e-8, "clip": float(norm.clip)}
                       if norm is not None else None),
        "control": {"hz": CONTROL_HZ, "dt": DT},
        "action": {
            "kind": "joint_position_offset",  # target = default_q + scale * action
            "scale": ACTION_SCALE,
            "clip": None,
        },
        # The joint contract. `joints` is the policy's own order; every per-joint vector in
        # this spec (default_q, and the action output) is in THAT order. A runner maps it onto
        # whatever order the simulator reports by NAME — never by position.
        "joints": [URDF_JOINT[n] for n in ISAAC],
        "default_q": [float(DEFAULT[n]) for n in ISAAC],
        "limits": {URDF_JOINT[n]: None for n in ISAAC},   # taken from the URDF at load time
        # What the robot needs from whoever builds it. A deployment that cannot honour these
        # is not running the plant this policy was trained on — the runner says so out loud.
        "plant": {
            "gains": {"stiffness": STIFF_GAINS["hx"][0], "damping": STIFF_GAINS["hx"][1],
                      "max_force_by_joint": {URDF_JOINT[L + "_" + j]: STIFF_GAINS[j][2]
                                             for L in LEGS for j in ("hx", "hy", "kn")}},
            "fixed_base": False,
            "self_collision": False,
            "physics_dt": 0.005,              # the training substep (tgs_pcm)
            "solver_position_iterations": 12,
            "spawn_height": Z0,
            "up_axis": "z",                   # the URDF's own up; the runner works in ANY world up
        },
        # THE OBSERVATION, IN ORDER. Terms are computed by the runner; dims must sum to obs_dim.
        "obs": [
            {"term": "base_lin_vel_body", "dim": 3},
            {"term": "base_ang_vel_body", "dim": 3},
            {"term": "projected_gravity", "dim": 3},
            {"term": "command", "dim": 3, "fields": ["vx", "vy", "wz"]},
            {"term": "joint_pos_rel_default", "dim": ACT_DIM},
            {"term": "joint_vel", "dim": ACT_DIM},
            {"term": "last_action", "dim": ACT_DIM},
            {"term": "gait_clock", "dim": 2, "period": GAIT_PERIOD,
             "encoding": "sin_cos", "advance": "after_step"},
            {"term": "base_height_above_ground", "dim": 1},
            {"term": "height_scan", "dim": N_SCAN, "grid_x": list(SCAN_GX), "grid_y": list(SCAN_GY),
             "relative_to": "height_under_base", "clip": [-1.0, 1.0], "frame": "heading"},
        ],
        "obs_dim": obs_dim,
        "act_dim": ACT_DIM,
        "command_envelope": {"vx": [-1.0, 1.5], "vy": 0.8, "wz": 1.2},
    }
    assert sum(t["dim"] for t in spec["obs"]) == obs_dim, "obs terms do not sum to obs_dim"

    with open(os.path.join(args.out, "spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print(f"[bundle] {args.out}")
    print(f"  policy.npz  layers={layers} activation=elu  norm={'yes' if norm is not None else 'no'}")
    print(f"  spec.json   obs_dim={obs_dim} act_dim={ACT_DIM} joints={len(spec['joints'])} "
          f"control={CONTROL_HZ} Hz")
    print(f"  joints (policy order): {', '.join(spec['joints'])}")


if __name__ == "__main__":
    main()
