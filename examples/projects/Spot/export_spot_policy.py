"""Export a trained Spot policy to a dependency-free .tpnn flat binary for native
C++ deployment (see SpotPolicy.hpp). Strips everything deployment doesn't need —
the critic, log_std, optimizer state — keeping only the actor MLP's Linear
weights/biases and (when present) the observation normalizer, so the C++ side
never links torch.

Handles two checkpoint flavours:
  * TorchScript  (Isaac Lab's downloaded spot_policy.pt: torch.jit.load) — the
    actor is a scripted nn.Sequential; we pull actor.<i>.weight/bias in order.
    A non-identity scripted normalizer is rejected (we can't read its stats).
  * state_dict   (our own save_policy from threepp.rl.ppo) — actor.<i>.weight/bias
    live under ckpt["model"]; the obs RunningNorm (ckpt["norm"], when the policy
    was trained with normalize_obs=True) is exported as a leading NORM BLOCK
    (the .tpnn TPN2 format), which the C++ SpotPolicy applies as
    clamp((x-mean)/sqrt(var+1e-8), +-clip) before layer 0.

Also writes a golden reference (spot_policy_ref.bin: N RAW-obs->action pairs run
through norm+MLP) so the C++ self-test can verify parity without torch.

    python export_spot_policy.py [policy.pt] [-o out_dir] [--ref-n 64]

Default input: the clean-lineage clock base gait
python/examples/spot/scratch_distillation/scratch_flat_best.pt (50-d: 48 proprio
+ 2 clock, normalize_obs=True). Pass an Isaac TorchScript .pt to export that instead.
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

ELU, NONE = 1, 0

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))   # examples/projects/Spot -> repo


def _linear_layers_from_sequential(seq):
    """Walk an nn.Sequential-like module, returning [(weight[out,in], bias[out], act)]
    where act is the activation that FOLLOWS each Linear (ELU=1, none=0)."""
    # Ordered child modules. For a scripted module, _modules preserves "0".."6".
    children = list(seq._modules.items()) if hasattr(seq, "_modules") else list(seq.named_children())
    layers = []
    pending = None  # (weight, bias) of a Linear awaiting its trailing activation
    for name, mod in children:
        kind = getattr(mod, "original_name", type(mod).__name__)
        has_w = hasattr(mod, "weight") and getattr(mod, "weight", None) is not None
        if kind == "Linear" or (has_w and getattr(mod.weight, "ndim", 0) == 2):
            if pending is not None:
                layers.append((*pending, NONE))  # two Linears in a row: no activation between
            w = mod.weight.detach().cpu().float().numpy()
            b = (mod.bias.detach().cpu().float().numpy()
                 if getattr(mod, "bias", None) is not None else np.zeros(w.shape[0], np.float32))
            pending = (w, b)
        elif kind == "ELU":
            if pending is None:
                raise RuntimeError("activation before any Linear — unexpected actor structure")
            layers.append((*pending, ELU))
            pending = None
        elif kind in ("ReLU", "Tanh", "GELU", "SiLU"):
            raise RuntimeError(f"activation '{kind}' not supported by SpotPolicy yet (only ELU); "
                               f"add it to SpotPolicy::act and the Act enum, then here.")
        # else: Dropout/Identity/etc. — skip (no-ops at eval)
    if pending is not None:
        layers.append((*pending, NONE))  # final Linear, no trailing activation
    return layers


def _extract_actor(ckpt_path):
    """Return (actor_module_or_none, layers, meta, norm). `layers` is the resolved
    [(w,b,act)] chain; `norm` is {mean,var,clip} (np arrays + float) or None; `meta`
    carries obs/act dims for a sanity print."""
    # Try TorchScript first (Isaac), then a plain state_dict checkpoint (ours). A non-TorchScript
    # file makes torch.jit.load raise (often RuntimeError) -> fall through to the state_dict path;
    # only ONCE jit.load has SUCCEEDED do we apply the normalizer guard (whose RuntimeError is real).
    m = None
    try:
        m = torch.jit.load(ckpt_path, map_location="cpu").eval()
    except Exception:
        m = None
    if m is not None:
        # Guard: a non-identity scripted normalizer would shift obs out from under layer 0
        # and we can't read its stats here -> refuse rather than silently break parity.
        norm_mod = getattr(m, "normalizer", None)
        if norm_mod is not None:
            bufs = list(norm_mod.named_buffers()) if hasattr(norm_mod, "named_buffers") else []
            pars = list(norm_mod.named_parameters()) if hasattr(norm_mod, "named_parameters") else []
            if bufs or pars:
                raise RuntimeError("TorchScript policy has a non-identity normalizer with "
                                   f"{len(bufs)} buffers / {len(pars)} params — not supported "
                                   "(export from the state_dict checkpoint, which carries the RunningNorm).")
        layers = _linear_layers_from_sequential(m.actor)
        return m, layers, {"src": "torchscript"}, None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt)
    meta = ckpt.get("meta", {})
    # obs RunningNorm (normalize_obs=True) -> leading norm block (TPN2). The C++ side applies
    # clamp((x-mean)/sqrt(var+1e-8), +-clip) before the MLP, matching threepp.rl.RunningNorm.norm.
    norm = None
    nrm = ckpt.get("norm")
    if nrm is not None:
        norm = {"mean": nrm["mean"].detach().cpu().float().numpy(),
                "var": nrm["var"].detach().cpu().float().numpy(),
                "clip": float(nrm["clip"])}
    # Collect actor.<i>.weight / actor.<i>.bias in ascending i.
    idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for j, i in enumerate(idxs):
        w = sd[f"actor.{i}.weight"].detach().cpu().float().numpy()
        b = sd[f"actor.{i}.bias"].detach().cpu().float().numpy()
        act = ELU if j < len(idxs) - 1 else NONE  # our _mlp uses ELU between layers, none on output
        layers.append((w, b, act))
    return None, layers, {"src": "state_dict", **meta}, norm


def _apply_norm(obs, norm):
    """RAW obs -> normalized obs, matching threepp.rl.RunningNorm.norm (numpy)."""
    if norm is None:
        return obs
    return np.clip((obs - norm["mean"]) / np.sqrt(norm["var"] + 1e-8), -norm["clip"], norm["clip"])


def write_tpnn(path, layers, norm=None):
    """Flat binary. TPN1 = MLP only; TPN2 = leading norm block + MLP.
       TPN2: magic 'TPN2', u32 normDim, f32[normDim] mean, f32[normDim] var, f32 clip, <then as TPN1>."""
    with open(path, "wb") as f:
        if norm is None:
            f.write(b"TPN1")
        else:
            f.write(b"TPN2")
            nd = int(norm["mean"].shape[0])
            assert norm["var"].shape[0] == nd
            f.write(struct.pack("<I", nd))
            f.write(np.ascontiguousarray(norm["mean"], "<f4").tobytes())
            f.write(np.ascontiguousarray(norm["var"], "<f4").tobytes())
            f.write(struct.pack("<f", float(norm["clip"])))
        f.write(struct.pack("<I", len(layers)))
        for w, b, act in layers:
            out, inp = w.shape
            assert b.shape[0] == out
            f.write(struct.pack("<III", inp, out, act))
            f.write(np.ascontiguousarray(w, "<f4").tobytes())   # [out, in] row-major
            f.write(np.ascontiguousarray(b, "<f4").tobytes())


def write_ref(path, module, layers, in_dim, out_dim, n, norm=None, seed=0):
    """Golden RAW-obs->action pairs for the C++ self-test. The C++ feeds RAW obs and applies the
    norm internally, so the action here is forward(norm(raw_obs)). Uses the torch module when we
    have one (TorchScript, norm=None); otherwise runs norm+layers in numpy."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, in_dim, generator=g).cpu().float().numpy()   # randn exercises both ELU branches
    normed = _apply_norm(obs, norm)
    if module is not None:
        with torch.no_grad():
            act = module(torch.from_numpy(normed)).cpu().float().numpy()
    else:
        x = normed
        for w, b, a in layers:
            x = x @ w.T + b
            if a == ELU:
                x = np.where(x > 0, x, np.expm1(x))
        act = x
    with open(path, "wb") as f:
        f.write(b"TPR1")
        f.write(struct.pack("<III", n, in_dim, out_dim))
        for i in range(n):
            f.write(np.ascontiguousarray(obs[i], "<f4").tobytes())     # RAW obs (C++ normalizes internally)
            f.write(np.ascontiguousarray(act[i], "<f4").tobytes())


def main():
    default_pt = os.path.join(_REPO, "python", "examples", "spot", "scratch_distillation",
                              "scratch_flat_best.pt")
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", nargs="?", default=default_pt,
                    help="input .pt (TorchScript or save_policy checkpoint; default scratch_flat_best.pt)")
    ap.add_argument("-o", "--out-dir", default=_HERE)
    ap.add_argument("--ref-n", type=int, default=64, help="number of golden obs/action pairs")
    args = ap.parse_args()
    if not os.path.exists(args.policy):
        sys.exit(f"no policy at {args.policy}")

    module, layers, meta, norm = _extract_actor(args.policy)
    in_dim, out_dim = int(layers[0][0].shape[1]), int(layers[-1][0].shape[0])
    if norm is not None and norm["mean"].shape[0] != in_dim:
        sys.exit(f"norm dim {norm['mean'].shape[0]} != MLP input dim {in_dim}")
    print(f"[export] {args.policy}  ({meta['src']})")
    chain = "  ".join(f"{w.shape[1]}->{w.shape[0]}{'+ELU' if a else ''}" for w, b, a in layers)
    print(f"[export] {chain}")
    print(f"[export] obs_dim {in_dim}  act_dim {out_dim}  layers {len(layers)}  "
          f"norm {'yes (TPN2)' if norm is not None else 'no (TPN1)'}")

    os.makedirs(args.out_dir, exist_ok=True)
    tpnn = os.path.join(args.out_dir, "spot_policy.tpnn")
    ref = os.path.join(args.out_dir, "spot_policy_ref.bin")
    write_tpnn(tpnn, layers, norm)
    write_ref(ref, module, layers, in_dim, out_dim, args.ref_n, norm)
    print(f"[export] wrote {tpnn}  ({os.path.getsize(tpnn)} bytes)")
    print(f"[export] wrote {ref}  ({args.ref_n} obs/action pairs)")


if __name__ == "__main__":
    main()
