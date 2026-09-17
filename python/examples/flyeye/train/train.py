"""Fit the flyvis free parameters on the Phase 2 rendered flights.

    py -3.14 python/examples/flyeye/train/train.py --dry                       # 50 steps, gradient check
    py -3.14 python/examples/flyeye/train/train.py --run linear --minutes 13
    py -3.14 python/examples/flyeye/train/train.py --run flyvis --decoder flyvis --loss l2norm --minutes 13

Training split: `straight_3`, `straight_10`, `yaw`, `roll` in all three lightings (12
recordings, 8220 frames x 2 eyes). `straight_30`, `pitch` and `approach` are never seen; they
are the held-out flights `evaluate.py` scores on.

Adam, two parameter groups (`--lr-lobe` 3e-4 for tau/bias/syn_strength, `--lr-dec` 1e-3 for
the decoder), gradient clipping, and `syn_strength` clamped at 0 after every step, which is
flyvis's `Network.clamp()`. With `--decoder flyvis` the defaults follow flyvis instead
(`--loss l2norm`, `--activity-penalty`; lr stays ours, see decoder.py's docstring for what of
flyvis is and is not reproduced).

Logged every `--log-every` steps: loss, the direction accuracy of the decoded field on the
window (the gate's measure, in sample), per-type tau and bias drift, `syn_strength` drift
(L2 from the start), the rest-frame |field| p50 on a fixed textured-rest probe window, and
steps/s. Checkpoints go to `C:\\dev\\_flyeye\\retrain\\<run>\\`: `model_<step>.npz` in
`data/flyeye_model.npz`'s format plus `decoder_<step>.pt` and `log.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flyeye import scoring as sc  # noqa: E402
from flyeye.train import RETRAIN_ROOT, TRAIN_SCEN  # noqa: E402
from flyeye.train import data as D  # noqa: E402
from flyeye.train import decoder as dec  # noqa: E402
from flyeye.train.model import TrainableLobe, unit_field  # noqa: E402

DT = 0.01
ACTIVITY = dict(baseline=5.0, weight=0.1, below=1.0, above=0.1)  # flyvis penalizer.yaml


def masked_mse(pred, target, mask):
    """Mean squared error per valid column-frame over both field components."""
    m = mask[:, :, None, :]
    n = m.sum() * pred.shape[2]
    return (((pred - target) ** 2) * m).sum() / n.clamp_min(1)


def l2norm(pred, target, mask):
    """flyvis `objectives.l2norm`: mean over samples of sqrt(sum over frames, ndim, hexals)."""
    e = ((pred - target) * mask[:, :, None, :]) ** 2
    return e.sum(dim=(0, 2, 3)).sqrt().mean()


def activity_penalty(central_v):
    """flyvis `solver.activity_penalty_step`, as a loss term on `bias`.

    central_v (T, B, 65) is v at the u = v = 0 node of each type. The temporal mean is taken
    over the last three quarters of the window, then
    0.1 * mean((1.0 relu(5 - a) - 0.1 relu(a - 5)) ** 2).
    """
    a = central_v[central_v.shape[0] // 4:].mean(0)
    d = ACTIVITY["baseline"] - a
    w = ACTIVITY["below"] * torch.relu(d) - ACTIVITY["above"] * torch.relu(-d)
    return ACTIVITY["weight"] * (w ** 2).mean()


def direction_accuracy(pred, target, mask, min_speed=sc.MIN_SPEED):
    """Fraction of valid column-frames whose decoded angle is within 45 deg of the truth,
    counting only frames whose true speed is at least min_speed (in the target's units)."""
    sp = torch.hypot(target[:, :, 0], target[:, :, 1])
    ok = mask & (sp >= min_speed)
    if not ok.any():
        return float("nan")
    a = torch.atan2(pred[:, :, 1], pred[:, :, 0]) - torch.atan2(target[:, :, 1], target[:, :, 0])
    err = (a + math.pi) % (2 * math.pi) - math.pi
    return float((err.abs()[ok] < math.radians(45)).float().mean())


@torch.no_grad()
def rest_response(lobe, decoder, index, probe, dt=DT):
    """(|field| p50, p90) of the unit readout and of the decoder on a textured-rest window."""
    x, valid = probe
    rates, _, _ = lobe.run_window(x, lobe.fade_in(x[0], dt), dt, index)
    out = {}
    pick = lambda f: torch.hypot(f[:, :, 0], f[:, :, 1])[:, 0][valid].flatten().float()
    if rates.shape[2] == 8:
        mag = pick(unit_field(rates, lobe.rest_rates(index, dt)))
        out["unit_p50"] = float(mag.median())
        out["unit_p90"] = float(torch.quantile(mag, 0.9))
    decoder.eval()
    mag = pick(decoder(rates))
    decoder.train()
    out["dec_p50"] = float(mag.median())
    out["dec_p90"] = float(torch.quantile(mag, 0.9))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="linear", help="subdirectory of C:/dev/_flyeye/retrain")
    ap.add_argument("--decoder", default="linear", choices=["linear", "flyvis"])
    ap.add_argument("--loss", default=None, choices=["mse", "l2norm"], help="default mse, l2norm for --decoder flyvis")
    ap.add_argument("--target", default="columns", choices=["columns", "unit"])
    ap.add_argument("--minutes", type=float, default=13.0, help="wall-clock cap on the training loop")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--window", type=int, default=50, help="truncated-BPTT window, frames")
    ap.add_argument("--streams", type=int, default=8, help="independent (recording, eye) streams = batch")
    ap.add_argument("--warmup", type=int, default=100, help="no-grad frames after the fade-in")
    ap.add_argument("--chunks", type=int, default=4, help="windows per stream before it is resampled")
    ap.add_argument("--lr-lobe", type=float, default=3e-4)
    ap.add_argument("--lr-dec", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lag", type=int, default=D.LAG)
    ap.add_argument("--activity-penalty", action="store_true", help="flyvis's penalty on bias")
    ap.add_argument("--no-scale-init", action="store_true", help="skip the data-driven rescale of the linear map")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--rest-every", type=int, default=25, help="refresh the grey-0.5 rest offset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="50 steps, report the gradient check, write nothing")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    loss_name = args.loss or ("l2norm" if args.decoder == "flyvis" else "mse")
    if args.decoder == "flyvis" and args.loss is None:
        args.activity_penalty = True
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or (RETRAIN_ROOT / args.run)

    t0 = time.time()
    lobe = TrainableLobe(device=dev)
    data = D.Data(TRAIN_SCEN, target=args.target, lag=args.lag, device=dev)
    decoder, index, types = dec.build(args.decoder, lobe, DT)
    streams = D.Streams(data, lobe, n=args.streams, window=args.window, warmup=args.warmup,
                        chunks=args.chunks, dt=DT, seed=args.seed)
    probe = D.rest_probe(data)
    print(f"setup {time.time() - t0:.1f} s | {data.n_frames} train frames in {len(data.keys())} recordings | "
          f"decoder {args.decoder} reads {len(types)} types, {sum(p.numel() for p in decoder.parameters())} params | "
          f"lobe params tau {lobe.log_tau.numel()} bias {lobe.bias.numel()} strength {lobe.strength.numel()} | "
          f"loss {loss_name} target {args.target}", flush=True)

    central = lobe.central_index() if args.activity_penalty else None
    opt = torch.optim.Adam([
        dict(params=[lobe.log_tau, lobe.bias, lobe.strength], lr=args.lr_lobe),
        dict(params=list(decoder.parameters()), lr=args.lr_dec),
    ])
    loss_fn = dict(mse=masked_mse, l2norm=l2norm)[loss_name]

    log = dict(args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               loss=loss_name, split=D.split_note(), decoder=decoder.describe(), types=types,
               device=dev, steps=[], grad_check=None)
    hist = []
    best = float("inf")
    n_steps = 50 if args.dry else args.steps
    cap = 2.0 * 60 if args.dry else args.minutes * 60
    t_start = time.time()
    rest0 = None

    for step in range(1, n_steps + 1):
        if step % args.rest_every == 1 and hasattr(decoder, "set_rest"):
            decoder.set_rest(lobe.rest_rates(index, DT))
        x, y, m, v0 = streams.batch()
        rates, extra, v_end = lobe.run_window(x, v0, DT, index, extra_index=central)
        pred = decoder(rates)
        loss = loss_fn(pred, y, m)
        total = loss + (activity_penalty(extra) if central is not None else 0.0)
        opt.zero_grad(set_to_none=True)
        total.backward()

        if step == 1:
            g = {}
            for name, p in (("log_tau", lobe.log_tau), ("bias", lobe.bias), ("strength", lobe.strength)):
                gr = p.grad
                g[name] = dict(present=gr is not None,
                               finite=bool(torch.isfinite(gr).all()) if gr is not None else False,
                               norm=float(gr.norm()) if gr is not None else 0.0,
                               nonzero=int((gr != 0).sum()) if gr is not None else 0, n=p.numel())
            gd = torch.cat([p.grad.flatten() for p in decoder.parameters() if p.grad is not None])
            g["decoder"] = dict(present=True, finite=bool(torch.isfinite(gd).all()),
                                norm=float(gd.norm()), nonzero=int((gd != 0).sum()), n=gd.numel())
            log["grad_check"] = g
            print("gradient check: " + " | ".join(f"{k} norm {d['norm']:.3e} nonzero {d['nonzero']}/{d['n']} "
                                                  f"finite {d['finite']}" for k, d in g.items()), flush=True)
            if args.decoder == "linear" and not args.no_scale_init:
                with torch.no_grad():
                    p, t = pred * m[:, :, None, :], y * m[:, :, None, :]
                    s = float((p * t).sum() / (p * p).sum().clamp_min(1e-20))
                    decoder.rescale(s)
                    log["scale_init"] = s
                    print(f"linear map rescaled by {s:.3f} to match the target's scale "
                          f"(init {decoder.init_source})", flush=True)
                opt.zero_grad(set_to_none=True)
                streams.advance(v_end)
                continue

        gn = torch.nn.utils.clip_grad_norm_([p for gp in opt.param_groups for p in gp["params"]], args.clip)
        opt.step()
        with torch.no_grad():
            lobe.strength.clamp_(min=0)  # flyvis Network.clamp(), 'non_negative'
        streams.advance(v_end)

        acc = direction_accuracy(pred.detach(), y, m)
        loss = float(loss.detach())
        hist.append(loss)
        if step % args.log_every == 0 or step == 1 or step == n_steps:
            el = time.time() - t_start
            row = dict(step=step, loss=round(loss, 6), acc45=round(acc, 4),
                       grad_norm=round(float(gn), 4), seconds=round(el, 1),
                       steps_per_s=round(step / max(el, 1e-9), 3), **lobe.drift(),
                       **{f"rest_{k}": round(v, 5) for k, v in
                          rest_response(lobe, decoder, index, probe).items()})
            log["steps"].append(row)
            print(f"{step:6d} loss {row['loss']:.5f} acc45 {row['acc45']:.3f} |g| {row['grad_norm']:.3f} "
                  f"rest p50 {row.get('rest_unit_p50', row.get('rest_dec_p50')):.4f} "
                  f"tau {row['tau_l2']:.4f} bias {row['bias_l2']:.4f} syn {row['strength_rel_l2']:.4f} "
                  f"zero {row['strength_zero']} | {row['steps_per_s']:.2f} steps/s {el / 60:.1f} min", flush=True)
        if not args.dry and (step % args.ckpt_every == 0):
            lobe.export_npz(out / f"model_{step:06d}.npz")
            torch.save(decoder.state_dict(), out / f"decoder_{step:06d}.pt")
        best = min(best, loss)
        if time.time() - t_start > cap:
            print(f"time cap {cap / 60:.1f} min reached at step {step}", flush=True)
            break

    el = time.time() - t_start
    k = max(1, len(hist) // 20)
    log["summary"] = dict(steps=len(hist), seconds=round(el, 1), steps_per_s=round(len(hist) / max(el, 1e-9), 3),
                          loss_start=round(float(np.mean(hist[:k])), 6), loss_end=round(float(np.mean(hist[-k:])), 6),
                          loss_best=round(best, 6), fade_in_steps=streams.fade_steps,
                          trained_frames=len(hist) * args.window * args.streams)
    log["drift_table"] = lobe.drift_table(top=10)
    print(json.dumps(log["summary"], indent=1), flush=True)
    if args.dry:
        d = log["grad_check"]
        okg = all(v["finite"] and v["nonzero"] > 0 for v in d.values())
        down = log["summary"]["loss_end"] < log["summary"]["loss_start"]
        print(f"DRY RUN: gradients on every group {okg}; loss decreased {down} "
              f"({log['summary']['loss_start']:.5f} -> {log['summary']['loss_end']:.5f})")
        (RETRAIN_ROOT / f"dry_{args.run}.json").parent.mkdir(parents=True, exist_ok=True)
        (RETRAIN_ROOT / f"dry_{args.run}.json").write_text(json.dumps(log, indent=1, default=float))
        return 0 if (okg and down) else 1
    out.mkdir(parents=True, exist_ok=True)
    lobe.export_npz(out / "model_final.npz")
    torch.save(decoder.state_dict(), out / "decoder_final.pt")
    (out / "log.json").write_text(json.dumps(log, indent=1, default=float))
    print(f"wrote {out / 'model_final.npz'}, {out / 'decoder_final.pt'}, {out / 'log.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
