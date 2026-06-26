"""Export a trained Spot policy to a dependency-free .tpnn flat binary for native
C++ deployment (see SpotPolicy.hpp). Strips everything deployment doesn't need —
the critic, log_std, optimizer state — keeping only the actor MLP's Linear
weights/biases and their activations, so the C++ side never links torch.

Handles two checkpoint flavours:
  * TorchScript  (Isaac Lab's downloaded spot_policy.pt: torch.jit.load) — the
    actor is a scripted nn.Sequential; we pull actor.<i>.weight/bias in order.
  * state_dict   (our own save_policy from threepp.rl.ppo) — actor.<i>.weight/bias
    live under ckpt["model"]; obs normalization (if any) is a separate concern,
    folded in as a leading affine layer when present (TODO; identity for now).

Also writes a golden reference (spot_policy_ref.bin: N obs->action pairs from the
torch forward) so the C++ self-test can verify parity without torch.

    python export_spot_policy.py [policy.pt] [-o out_dir] [--ref-n 64]

Default input: ~/.cache/threepp/spot/spot_policy.pt (the Isaac flat-ground policy).
"""
import argparse
import os
import pathlib
import struct
import sys

import numpy as np
import torch

ELU, NONE = 1, 0


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
    """Return (actor_module_or_none, layers, meta). `layers` is the resolved
    [(w,b,act)] chain; `meta` carries obs/act dims for a sanity print."""
    # Try TorchScript first (Isaac), then a plain state_dict checkpoint (ours).
    try:
        m = torch.jit.load(ckpt_path, map_location="cpu").eval()
        # Guard: a non-identity normalizer would shift obs out from under layer 0.
        norm = getattr(m, "normalizer", None)
        if norm is not None:
            bufs = list(norm.named_buffers()) if hasattr(norm, "named_buffers") else []
            pars = list(norm.named_parameters()) if hasattr(norm, "named_parameters") else []
            if bufs or pars:
                raise RuntimeError("policy has a non-identity normalizer with "
                                   f"{len(bufs)} buffers / {len(pars)} params — fold it into the "
                                   "export as a leading affine layer before trusting parity.")
        layers = _linear_layers_from_sequential(m.actor)
        return m, layers, {"src": "torchscript"}
    except RuntimeError:
        raise
    except Exception:
        pass

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt)
    meta = ckpt.get("meta", {})
    if ckpt.get("norm") is not None:
        raise RuntimeError("checkpoint carries an obs RunningNorm — fold mean/var into the export "
                           "as a leading affine layer before trusting parity (not yet implemented).")
    # Collect actor.<i>.weight / actor.<i>.bias in ascending i.
    idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for j, i in enumerate(idxs):
        w = sd[f"actor.{i}.weight"].detach().cpu().float().numpy()
        b = sd[f"actor.{i}.bias"].detach().cpu().float().numpy()
        act = ELU if j < len(idxs) - 1 else NONE  # our _mlp uses ELU between layers, none on output
        layers.append((w, b, act))
    return None, layers, {"src": "state_dict", **meta}


def write_tpnn(path, layers):
    with open(path, "wb") as f:
        f.write(b"TPN1")
        f.write(struct.pack("<I", len(layers)))
        for w, b, act in layers:
            out, inp = w.shape
            assert b.shape[0] == out
            f.write(struct.pack("<III", inp, out, act))
            f.write(np.ascontiguousarray(w, "<f4").tobytes())   # [out, in] row-major
            f.write(np.ascontiguousarray(b, "<f4").tobytes())


def write_ref(path, module, layers, in_dim, out_dim, n, seed=0):
    """Golden obs->action pairs for the C++ self-test. Uses the torch module when
    we have one (TorchScript); otherwise runs the resolved layers in numpy."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, in_dim, generator=g)  # randn exercises both ELU branches
    if module is not None:
        with torch.no_grad():
            act = module(obs).cpu().float().numpy()
    else:
        x = obs.cpu().float().numpy()
        for w, b, a in layers:
            x = x @ w.T + b
            if a == ELU:
                x = np.where(x > 0, x, np.expm1(x))
        act = x
    obs = obs.cpu().float().numpy()
    with open(path, "wb") as f:
        f.write(b"TPR1")
        f.write(struct.pack("<III", n, in_dim, out_dim))
        for i in range(n):
            f.write(np.ascontiguousarray(obs[i], "<f4").tobytes())
            f.write(np.ascontiguousarray(act[i], "<f4").tobytes())


def main():
    default_pt = pathlib.Path.home() / ".cache" / "threepp" / "spot" / "spot_policy.pt"
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", nargs="?", default=str(default_pt),
                    help="input .pt (TorchScript or save_policy checkpoint)")
    ap.add_argument("-o", "--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--ref-n", type=int, default=64, help="number of golden obs/action pairs")
    args = ap.parse_args()
    if not os.path.exists(args.policy):
        sys.exit(f"no policy at {args.policy}")

    module, layers, meta = _extract_actor(args.policy)
    in_dim, out_dim = int(layers[0][0].shape[1]), int(layers[-1][0].shape[0])
    print(f"[export] {args.policy}  ({meta['src']})")
    chain = "  ".join(f"{w.shape[1]}->{w.shape[0]}{'+ELU' if a else ''}" for w, b, a in layers)
    print(f"[export] {chain}")
    print(f"[export] obs_dim {in_dim}  act_dim {out_dim}  layers {len(layers)}")

    os.makedirs(args.out_dir, exist_ok=True)
    tpnn = os.path.join(args.out_dir, "spot_policy.tpnn")
    ref = os.path.join(args.out_dir, "spot_policy_ref.bin")
    write_tpnn(tpnn, layers)
    write_ref(ref, module, layers, in_dim, out_dim, args.ref_n)
    print(f"[export] wrote {tpnn}  ({os.path.getsize(tpnn)} bytes)")
    print(f"[export] wrote {ref}  ({args.ref_n} obs/action pairs)")


if __name__ == "__main__":
    main()
