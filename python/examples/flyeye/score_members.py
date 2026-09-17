"""Readout repair round, scoring: every ensemble member and the ensemble mean x readout variants.

    py -3.14 python/examples/flyeye/score_members.py score          # all members + ensemble (~30 min, CUDA)
    py -3.14 python/examples/flyeye/score_members.py score --members 000 --no-ensemble
    py -3.14 python/examples/flyeye/score_members.py table          # markdown table from scores.json
    py -3.14 python/examples/flyeye/score_members.py figure         # figures/repair_round.png
    py -3.14 python/examples/flyeye/score_members.py video --member 000 --readout "rec adapt2+V2"

No rendering, no threepp import. Inputs: the Phase 2 recordings C:/dev/_flyeye/phase2 (truth),
the member replays C:/dev/_flyeye/ensemble/raw/<NNN>/<run>.npz (t4t5_raw, members.py) and the
member models C:/dev/_flyeye/ensemble/models/flow_0000_<NNN>.npz (gratings). Writes
C:/dev/_flyeye/repair/scores.json (every number), figures/repair_round.png and
C:/dev/_flyeye/renders/repair/*.mp4.

Readouts, all through repair.score_variant (lag 20 ms, own motion-blind baseline = the readout on the
run's own rest frames 30-100 ping-pong repeated, 10 s settle dropped; anything fitted to truth is
fitted on bright only, dim and dark are the test split):
  P2 unit             MotionField unit gains, x = b - a, y = c - d, T4 + T5 (the Phase 2 field); looming L0
  rec adapt2+V2       causal EMA adaptation tau 2 s, then an 8 -> 2 least-squares map fitted on this member's bright runs
  run adapt1+V2       the same with tau 1 s (runner-up)
  tf adapt1+V1b       truth-free: tau 1 s, this member's grating directions, gain 1 / first-harmonic amplitude
                      (amplitude floored at 5 % of the member's largest, so a non-selective type cannot blow up)
  loom adapt0.5 unit  the recommended looming readout: tau 0.5 s, unit map, read as L2rot
  loom adapt0.5 V1a   the same with this member's grating unit vectors (sign-aligns reversed members), L2rot
Ensemble: "mean raw" = the readouts applied to the mean of the 10 members' rest-subtracted t4t5_raw
(V2 refitted on it); "mean field" = the mean of the members' own readout fields (and baselines), for
the readouts that are not linear in raw with one shared map (rec, tf, loom V1a). P2 unit and
loom adapt0.5 unit are linear with a shared map, so their mean field equals their mean-raw field.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyeye import repair as rp  # noqa: E402
from flyeye import scoring as sc  # noqa: E402
from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.readouts import MOTION_TYPES  # noqa: E402

HERE = Path(__file__).resolve().parent
ENS = Path(r"C:\dev\_flyeye\ensemble")
OUT = Path(r"C:\dev\_flyeye\repair")
SCORES = OUT / "scores.json"
RENDERS = Path(r"C:\dev\_flyeye\renders\repair")
MEMBERS = [f"{i:03d}" for i in range(10)]
AMP_FLOOR = 0.05
MEAN_FIELD = {"rec adapt2+V2": False, "tf adapt1+V1b": False, "loom adapt0.5 V1a": True}


# ---------------------------------------------------------------------------------------------- data
def load_member_raws(member, runs):
    """{key: float32 (T, B, 8, 721)} of one member's replay."""
    out = {}
    for key in runs.keys():
        z = np.load(ENS / "raw" / member / f"{key[0]}_{key[1]}.npz")
        out[key] = z["t4t5_raw"].astype(np.float32)
    return out


def member_gratings(member):
    """This member's subtype directions from procedural drifting gratings (repair.grating_rates, 8 columns, 4 Hz, c 0.5)."""
    from flyeye.optic_lobe import OpticLobe
    from flyeye.readouts import MotionField

    lat = HexLattice()
    lat.centers = lat.centers.to("cuda")
    lobe = OpticLobe(ENS / "models" / f"flow_0000_{member}.npz", device="cuda", dtype=torch.float32)
    mf = MotionField(lobe)
    u, v = lat.u.numpy(), lat.v.numpy()
    inner = (np.abs(u) + np.abs(v) + np.abs(u + v)) // 2 <= rp.INNER_RADIUS
    dirs = np.arange(rp.N_DIRS) * 360.0 / rp.N_DIRS
    r = rp.grating_rates(lat, lobe, mf.rest, dirs, rp.GRATING["tf_hz"], rp.GRATING["lambda_columns"], rp.GRATING["contrast"])
    curves = r[:, :, inner].mean(2)
    per = {}
    for j, name in enumerate(MOTION_TYPES):
        ang, amp, dsi, peak, peak_ang = rp.vector_mean(dirs, curves[:, j])
        per[name] = dict(angle_deg=round(ang, 2), amplitude=round(amp, 5), dsi=round(dsi, 3))
    del lobe, mf
    torch.cuda.empty_cache()
    return per, curves


def grating_map(per):
    ang = np.radians([per[n]["angle_deg"] for n in MOTION_TYPES])
    U = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    amp = np.array([per[n]["amplitude"] for n in MOTION_TYPES])
    amp = np.maximum(amp, AMP_FLOOR * amp.max())
    return U, (1.0 / amp).astype(np.float32)


# ---------------------------------------------------------------------------------------------- readouts
def readouts(runs, raws, grat=None):
    """name -> (factory, looming?) for one member (or the mean raw when grat is None)."""
    ad = lambda t: (lambda r: rp.adapt(r, t))

    def fitted(name, tau):
        def make():
            W, n = rp.fit_linear(runs, raws, pre=ad(tau))
            v = rp.Variant(name, U=W, pre=ad(tau), note=f"adaptation tau {tau} s; 8->2 map fitted on bright (N {n})")
            v.fitted = dict(W=W.tolist(), n=n)
            return v
        return make

    R = {"P2 unit": (lambda: rp.Variant("P2 unit", note="MotionField unit gains (Phase 2 field)"), True),
         "rec adapt2+V2": (fitted("rec adapt2+V2", 2.0), False),
         "run adapt1+V2": (fitted("run adapt1+V2", 1.0), False)}
    if grat is not None:
        U, g = grating_map(grat)
        R["tf adapt1+V1b"] = (lambda: rp.Variant("tf adapt1+V1b", U=U, gains=g, pre=ad(1.0),
                                                 note="adaptation tau 1 s, member grating vectors, gain 1/amp"), False)
    R["loom adapt0.5 unit"] = (lambda: rp.Variant("loom adapt0.5 unit", pre=ad(0.5), note="adaptation tau 0.5 s, unit map"), True)
    if grat is not None:
        R["loom adapt0.5 V1a"] = (lambda: rp.Variant("loom adapt0.5 V1a", U=U, pre=ad(0.5),
                                                     note="adaptation tau 0.5 s, member grating unit vectors"), True)
    return R


class Tap(rp.Variant):
    """Wraps a variant; adds every field and baseline it produces to running sums (for the mean-field ensemble)."""

    def __init__(self, v, acc, keyof):
        super().__init__(v.name, v.U, v.gains, v.pre, v.note)
        self.v, self.acc, self.keyof = v, acc, keyof
        self.fitted = v.fitted

    def _add(self, what, raw, f):
        k = (self.keyof[id(raw)], what)
        if k in self.acc:
            self.acc[k] += f
        else:
            self.acc[k] = f.copy()

    def field(self, raw):
        f = self.v.field(raw)
        self._add("field", raw, f)
        return f

    def baseline(self, raw):
        f = self.v.baseline(raw)
        self._add("base", raw, f)
        return f


class Fixed(rp.Variant):
    """A readout whose fields and baselines are given: raws passed to score_variant are the run keys."""

    def __init__(self, name, acc, n, note):
        super().__init__(name, note=note)
        self.acc, self.n = acc, n

    def field(self, key):
        return self.acc[(key, "field")] / self.n

    def baseline(self, key):
        return self.acc[(key, "base")] / self.n


def fmt(x, f="{:.2f}"):
    return "n/a" if x is None else f.format(x)


def save(scores):
    SCORES.write_text(json.dumps(scores, indent=1, default=float))


def run_one(scores, member, name, v, loom, runs, raws):
    t0 = time.time()
    res = rp.score_variant(v, runs, raws, looming=loom)
    row = rp.summarise(res)
    row["fitted"] = v.fitted
    row["loom_eps"] = res.get("loom_eps")
    row["rotation_offsets"] = {l: {s: res["rotation"][l][s]["offset"] for s in sc.AXIS} for l in res["rotation"]}
    scores.setdefault("rows", {}).setdefault(member, {})[name] = row
    save(scores)
    loomtxt = ""
    if loom:
        ln = "L0" if name == "P2 unit" else "L2rot"
        loomtxt = f" | {ln} margin " + "/".join(fmt(row['looming'][ln][l]['margin_peak']) for l in sc.LIGHTS)
    r2 = row["r2_seg"]["bright"]
    print(f"{member:10s} {name:20s} gate {row['gate']:.4f} ({row['bin']}) base {row['baseline']:.4f} lift {row['lift']:+.4f} "
          f"dim {row['test_dim']:.4f} dark {row['test_dark']:.4f} p19 {row['pooled_19']:.4f} "
          f"R2b {r2['yaw']:.2f}/{r2['pitch']:.2f}/{r2['roll']:.2f} signs {row['signs']}{loomtxt}  [{time.time() - t0:.0f} s]", flush=True)


def score(members, ensemble=True, only=None, resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    scores = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    runs = sc.Runs()
    acc, n_acc = {}, 0
    raw_sum = {}
    for m in members:
        t0 = time.time()
        grat, curves = member_gratings(m)
        scores.setdefault("gratings", {})[m] = dict(types=grat, curves=np.asarray(curves).round(5).tolist())
        raws = load_member_raws(m, runs)
        keyof = {id(a): k for k, a in raws.items()}
        print(f"member {m}: gratings + load {time.time() - t0:.0f} s; angles "
              + " ".join(f"{n}{grat[n]['angle_deg']:+.0f}/{grat[n]['amplitude']:.3f}" for n in MOTION_TYPES), flush=True)
        for name, (make, loom) in readouts(runs, raws, grat).items():
            if only and name not in only:
                continue
            v = make()
            if ensemble and name in MEAN_FIELD:
                v = Tap(v, acc.setdefault(name, {}), keyof)
            if resume and name in scores.get("rows", {}).get(m, {}):
                if isinstance(v, Tap):
                    for k, a in raws.items():
                        v.field(a)
                        v.baseline(a)
                print(f"{m:10s} {name:20s} kept from scores.json" + (" (re-accumulated)" if isinstance(v, Tap) else ""), flush=True)
                continue
            run_one(scores, m, name, v, loom, runs, raws)
        if ensemble:
            for k, a in raws.items():
                if k in raw_sum:
                    raw_sum[k] += a
                else:
                    raw_sum[k] = a.copy()
            n_acc += 1
        del raws, keyof
    if ensemble and n_acc:
        scores["ensemble_members"] = members
        mean = {k: a / n_acc for k, a in raw_sum.items()}
        del raw_sum
        for name, (make, loom) in readouts(runs, mean, None).items():
            if only and name not in only:
                continue
            run_one(scores, "mean raw", name, make(), loom, runs, mean)
        del mean
        keys = {k: k for k in runs.keys()}
        for name, loom in MEAN_FIELD.items():
            if name not in acc:
                continue
            run_one(scores, "mean field", name, Fixed(name, acc[name], n_acc, f"mean of the members' own {name} fields"),
                    loom, runs, keys)
            del acc[name]
    return scores



# ---------------------------------------------------------------------------------------------- table
READOUTS = ["P2 unit", "rec adapt2+V2", "run adapt1+V2", "tf adapt1+V1b", "loom adapt0.5 unit", "loom adapt0.5 V1a"]
GROUPS = MEMBERS + ["mean raw", "mean field"]


def loom_name(readout):
    return "L0" if readout == "P2 unit" else "L2rot"


def signed_r2(row, light, axis):
    return row["r2_seg"][light][axis] * (1 if row["gain_seg"][light][axis] >= 0 else -1)


def table(scores=None, groups=None, readouts=None):
    scores = scores or json.loads(SCORES.read_text())
    rows = scores["rows"]
    head = ("| member | readout | gate (bin) | own base | gate - base | pooled-19 | test gate dim / dark | "
            "2-4 col/s gate / base | R^2 bright yaw / pitch / roll | signs b / d / k (of 18) | loom margin b / d / k |")
    out = [head, "|" + "---|" * 11]
    for g in groups or GROUPS:
        for name in readouts or READOUTS:
            r = rows.get(g, {}).get(name)
            if r is None:
                continue
            lm = "-"
            if "looming" in r:
                ln = loom_name(name)
                lm = f"{ln} " + " / ".join(fmt(r['looming'][ln][l]['margin_peak']) for l in sc.LIGHTS)
            r2 = " / ".join(f"{signed_r2(r, 'bright', a):+.2f}" for a in ("yaw", "pitch", "roll"))
            out.append(f"| {g} | {name} | {r['gate']:.3f} ({r['bin'].replace(' columns/s', '')}) | {r['baseline']:.3f} | {r['lift']:+.3f} | "
                       f"{r['pooled_19']:.3f} | {r['test_dim']:.3f} / {r['test_dark']:.3f} | {r['by_speed'][3]:.3f} / {r['baseline_by_speed'][3]:.3f} | {r2} | "
                       f"{' / '.join(str(r['signs'][l]) for l in sc.LIGHTS)} | {lm} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- figure
RCOL = {"P2 unit": "#999999", "rec adapt2+V2": "#d55e00", "run adapt1+V2": "#e69f00", "tf adapt1+V1b": "#009e73",
        "loom adapt0.5 unit": "#0072b2", "loom adapt0.5 V1a": "#56b4e9"}


def figure(path=HERE / "figures" / "repair_round.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = json.loads(SCORES.read_text())
    rows = scores["rows"]
    groups = [g for g in GROUPS if g in rows]
    fig = plt.figure(figsize=(26, 26))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5], hspace=0.32, wspace=0.18)
    x = np.arange(len(groups))
    w = 0.13

    def shade(ax):
        for i, g in enumerate(groups):
            if g.startswith("mean"):
                ax.axvspan(i - 0.48, i + 0.48, color="#fff1c9", zorder=0)

    ax = fig.add_subplot(gs[0, :])
    shade(ax)
    for j, name in enumerate(READOUTS):
        off = (j - (len(READOUTS) - 1) / 2) * w
        gv = [rows[g].get(name, {}).get("gate", np.nan) for g in groups]
        bv = [rows[g].get(name, {}).get("baseline", np.nan) for g in groups]
        ax.bar(x + off, gv, w * 0.92, color=RCOL[name], label=name, zorder=2)
        ax.plot(x + off, bv, "_", color="k", ms=11, mew=2.2, zorder=3, label="own motion-blind baseline" if j == 0 else None)
    for yv, txt, ls in ((0.5, "target 0.5", "-"), (0.362, "Phase 2 gate 0.362", "--"), (0.319, "Phase 2 baseline 0.319", "-."),
                        (0.25, "chance 0.25", ":")):
        ax.axhline(yv, color="#b00000" if yv == 0.5 else "k", lw=1, ls=ls, zorder=1)
        ax.text(len(groups) - 0.45, yv + 0.004, txt, ha="right", fontsize=10)
    ax.set_xticks(x, groups, fontsize=12)
    for t in ax.get_xticklabels():
        if t.get_text().startswith("mean"):
            t.set_fontweight("bold")
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0.15, 0.68)
    ax.set_ylabel("gate: fraction within 45 deg, best speed bin (all lightings)", fontsize=11)
    ax.set_title("Per-column direction gate (bars) and each readout's own motion-blind baseline (black ticks); "
                 "shaded = ensemble mean of members 000-009", fontsize=13)
    ax.legend(ncol=7, fontsize=10, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    shade(ax)
    for j, name in enumerate(READOUTS):
        off = (j - (len(READOUTS) - 1) / 2) * w
        ax.bar(x + off, [rows[g].get(name, {}).get("lift", np.nan) for g in groups], w * 0.92, color=RCOL[name], zorder=2)
    ax.axhline(0.043, color="k", lw=1, ls="--")
    ax.text(len(groups) - 0.45, 0.045, "Phase 2 lift 0.043", ha="right", fontsize=10)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, groups, fontsize=10, rotation=30)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("gate minus own baseline")
    ax.set_title("Motion lift (gate minus own baseline), colours as above", fontsize=13)

    ax = fig.add_subplot(gs[1, 1])
    lrows = [(g, n) for g in groups for n in ("P2 unit", "loom adapt0.5 unit", "loom adapt0.5 V1a") if n in rows[g]]
    cols = [(n, l) for n in ("L0", "L2rot") for l in sc.LIGHTS]
    M = np.full((len(groups), 9), np.nan)
    for i, g in enumerate(groups):
        for j, (name, ln) in enumerate((("P2 unit", "L0"), ("loom adapt0.5 unit", "L2rot"), ("loom adapt0.5 V1a", "L2rot"))):
            if name in rows[g] and "looming" in rows[g][name]:
                for k, l in enumerate(sc.LIGHTS):
                    mp = rows[g][name]["looming"][ln][l]["margin_peak"]
                    M[i, j * 3 + k] = -1 if mp is None else mp
    im = ax.imshow(np.log2(np.clip(M, 0.05, None)), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, "n/a" if M[i, j] < 0 else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        fontweight="bold" if M[i, j] > 1 else "normal")
    ax.set_xticks(range(9), [f"{a}\n{l}" for a in ("P2 unit\nL0", "adapt0.5 unit\nL2rot", "adapt0.5 V1a\nL2rot") for l in sc.LIGHTS],
                  fontsize=8)
    ax.set_yticks(range(len(groups)), groups, fontsize=10)
    for j in (3, 6):
        ax.axvline(j - 0.5, color="k", lw=1.2)
    ax.axhline(len(MEMBERS) - 0.5, color="k", lw=1.2)
    ax.set_title("Looming margin = approach peak / max rotation false alarm (red > 1, bold; n/a = readout is zero on every run)",
                 fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.03, label="log2 margin")

    ax = fig.add_subplot(gs[2, :])
    rnames = ["P2 unit", "rec adapt2+V2", "run adapt1+V2", "tf adapt1+V1b"]
    labels, R = [], []
    for g in groups:
        for n in rnames:
            if n in rows[g]:
                labels.append(f"{g} | {n}")
                R.append([signed_r2(rows[g][n], l, a) for a in ("yaw", "pitch", "roll") for l in sc.LIGHTS])
    R = np.array(R).T  # (9, rows)
    im = ax.imshow(R, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=6.5, rotation=90,
                    color="white" if abs(R[i, j]) > 0.75 else "k")
    ax.set_yticks(range(9), [f"{a} {l}" for a in ("yaw", "pitch", "roll") for l in sc.LIGHTS], fontsize=10)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    for i, lab in enumerate(ax.get_xticklabels()):
        if lab.get_text().startswith("mean"):
            lab.set_fontweight("bold")
    for j in range(len(rnames), len(labels), len(rnames)):
        ax.axvline(j - 0.5, color="k", lw=0.8)
    for i in (3, 6):
        ax.axhline(i - 0.5, color="k", lw=1.2)
    ax.set_title("Rotation: R^2 of the six steady-segment means vs rate, signed by the gain (blue = right sign; "
                 "bright = train for V2 maps, dim/dark = test)", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.02, label="signed R^2")
    fig.suptitle("flyeye readout repair round: 10 flyvis members + ensemble mean x readouts, Phase 2 recordings (21 runs, Geiranger)",
                 fontsize=16)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=75, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------------------------- video
def repaired_raw(run_name, member):
    """float32 t4t5_raw of one run: a member, or the 10-member mean ('mean raw')."""
    if member in MEMBERS:
        return {member: np.load(ENS / "raw" / member / f"{run_name}.npz")["t4t5_raw"].astype(np.float32)}
    return {m: np.load(ENS / "raw" / m / f"{run_name}.npz")["t4t5_raw"].astype(np.float32) for m in MEMBERS}


def video(run_name, member="000", readout="rec adapt2+V2", loom_readout=None):
    """Data-only mp4 of one recording: per eye, receptor columns, the Phase 2 field, the repaired field, true flow."""
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    from flyeye.video import VideoOut

    scores = json.loads(SCORES.read_text())["rows"]
    scen, light = run_name.rsplit("_", 1)
    runs = sc.Runs()
    run = runs[(scen, light)]
    z, meta = run.z, run.meta
    T = run.T
    loom_readout = loom_readout or ("loom adapt0.5 V1a" if member == "mean field" else "loom adapt0.5 unit")
    tau = {"rec adapt2+V2": 2.0, "run adapt1+V2": 1.0}[readout]
    raws = repaired_raw(run_name, member)
    src = MEMBERS if member == "mean field" else [member]
    fld = np.zeros((T, 2, 2, 721), np.float32)
    lfld = np.zeros((T, 2, 2, 721), np.float32)
    for m in (src if member == "mean field" else [None]):
        raw = raws[m] if m else (raws[member] if member in raws else np.mean([raws[k] for k in MEMBERS], 0))
        W = np.array(scores[m or member][readout]["fitted"]["W"], np.float32)
        fld += rp.map_field(rp.adapt(raw, tau), W)
        if loom_readout == "loom adapt0.5 V1a":
            U, _ = grating_map(json.loads(SCORES.read_text())["gratings"][m or member]["types"])
        else:
            U = rp.UNIT
        lfld += rp.map_field(rp.adapt(raw, 0.5), U)
    fld /= len(src) if member == "mean field" else 1
    lfld /= len(src) if member == "mean field" else 1
    p2 = z["field"].astype(np.float32)
    truth = z["flow_box"].astype(np.float32)
    rec = z["receptors"].astype(np.float32)

    # readout series, calibrated with the bright rotation gains of that row, offset = this run's rest mean
    def rot_dps(f, row):
        r = sc.field_rotation(f, run)
        r = r - r[sc.REST].mean(0)
        g = np.array([row["gain_seg"]["bright"][a] for a in ("pitch", "yaw", "roll")])
        return np.degrees(r / g)

    p2row, rrow, lrow = scores["000"]["P2 unit"], scores[member][readout], scores[member][loom_readout]
    rot_p2, rot_rep = rot_dps(p2, p2row), rot_dps(fld, rrow)
    loom_p2 = sc.field_looming(p2, run) / p2row["looming"]["L0"][light]["threshold"]
    _, frot = rp.rot_projection(lfld, run)
    l0 = sc.field_looming(lfld, run)
    loom_rep = l0 / (np.sqrt((frot ** 2).mean((1, 2, 3))) + lrow["loom_eps"]["rotation_rms"]) / lrow["looming"]["L2rot"][light]["threshold"]
    w_true = np.degrees(z["w_body"])
    inv_tau = z["loom_truth"].astype(np.float64)
    seg = z["seg"]
    labels_ = [str(s).split("|")[0] for s in z["labels"]]

    S, GAP, TEXT = meta["size"], 4, 70
    lat = HexLattice()
    rc = lat.pixel_rc(S).float().cuda()
    yy, xx = torch.meshgrid(torch.arange(S, device="cuda"), torch.arange(S, device="cuda"), indexing="ij")
    pix = torch.stack([yy.flatten(), xx.flatten()], 1).float()
    d, i = zip(*(torch.cdist(c, rc).min(1) for c in pix.split(16384)))
    index = torch.cat(i).view(S, S)
    inside = (torch.cat(d) <= 8.5).view(S, S)
    W_, H_ = 2 * S + GAP, TEXT + 4 * S + 3 * GAP
    W_, H_ = W_ + W_ % 2, H_ + H_ % 2
    full_p2 = float(meta.get("video_field_full", 0.3))
    full_rep = float(np.percentile(np.linalg.norm(fld[sc.SKIP:], axis=2), 99))
    full_truth = float(np.percentile(np.linalg.norm(truth[sc.SKIP:], axis=2), 99))
    RENDERS.mkdir(parents=True, exist_ok=True)
    tag = member.replace(" ", "_")
    path = RENDERS / f"{run_name}_{tag}_{readout.split()[0]}.mp4"
    proc = subprocess_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe(), W_, H_, path)
    moving = [(a, b) for name, a, b in run.segments if name != "rest"]
    if len(moving) == 1:
        a, b = moving[0]
        still_idx = {int(a + f * (b - a)) for f in (0.5, 0.85, 0.98)}
    else:
        still_idx = {int(a + 0.75 * (b - a)) for a, b in (moving[i] for i in (0, 2, 3) if i < len(moving))}
    names = ["receptor columns (BoxEye luma)", f"Phase 2 field (member 000, unit gains), full {full_p2:g}",
             f"repaired: {member} {readout}, full {full_rep:.3g}", f"true flow (box), full {full_truth:.3g} rad/s"]
    gpu = lambda a: torch.as_tensor(a, device="cuda")
    for k in range(T):
        img = torch.zeros(H_, W_, 3, dtype=torch.uint8, device="cuda")
        for e in range(2):
            x0, y = e * (S + GAP), TEXT
            g = (gpu(rec[k, e]).clamp(0, 1) * 255).to(torch.uint8)[index] * inside
            img[y:y + S, x0:x0 + S] = g[..., None]
            for row, (f, full) in enumerate(((p2, full_p2), (fld, full_rep), (truth, full_truth))):
                y = TEXT + (row + 1) * (S + GAP)
                img[y:y + S, x0:x0 + S] = VideoOut.field_rgb(gpu(f[k, e]), full)[index] * inside[..., None]
        im = Image.fromarray(img.cpu().numpy())
        dr = ImageDraw.Draw(im)
        wt, rp2, rrp = w_true[k], rot_p2[k], rot_rep[k]
        dr.text((6, 3), f"{run_name}  t {k * 0.01:5.2f} s  segment '{labels_[seg[k]]}'   data only: recorded receptors, "
                        f"t4t5_raw and renderer truth", fill=(255, 255, 255))
        dr.text((6, 19), f"deg/s pitch/yaw/roll  true {wt[0]:+6.1f} {wt[1]:+6.1f} {wt[2]:+6.1f} | P2 est {rp2[0]:+6.1f} {rp2[1]:+6.1f} "
                         f"{rp2[2]:+6.1f} | repaired est {rrp[0]:+6.1f} {rrp[1]:+6.1f} {rrp[2]:+6.1f}", fill=(255, 255, 255))
        dr.text((6, 35), f"looming / its rotation threshold ({light}):  P2 L0 {loom_p2[k]:5.2f}   repaired {loom_readout} L2rot "
                         f"{loom_rep[k]:5.2f}   (> 1 = above every rotation false alarm)", fill=(255, 255, 255))
        dr.text((6, 51), f"true 1/tau {inv_tau[k]:5.3f} 1/s   eye 0 = yaw +45 (left), eye 1 = yaw -45 (right); "
                         f"hue = direction (red right, green-yellow up, cyan left, violet down)", fill=(255, 255, 255))
        for e, yaw in enumerate((45, -45)):
            for row, lab in enumerate(names):
                lab = f"eye {yaw:+d}: {lab}" if row == 0 else lab
                xx0, yy0 = e * (S + GAP) + 4, TEXT + row * (S + GAP) + 4
                dr.rectangle((xx0 - 2, yy0 - 2, xx0 + 6 * len(lab) + 2, yy0 + 12), fill=(0, 0, 0))
                dr.text((xx0, yy0), lab, fill=(255, 255, 255))
        proc.stdin.write(im.tobytes())
        if k in still_idx:
            im.save(RENDERS / f"{run_name}_{tag}_{readout.split()[0]}_f{k:04d}.png")
    proc.stdin.close()
    proc.wait()
    print(f"wrote {path}")


def subprocess_ffmpeg(exe, W, H, path):
    import subprocess

    return subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "100",
                             "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["score", "table", "figure", "video"])
    ap.add_argument("--members", default=",".join(MEMBERS))
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep rows already in scores.json (mean-field sums are rebuilt)")
    ap.add_argument("--only", nargs="*", help="readout names to score")
    ap.add_argument("--member", default="000", help="video: member, 'mean raw' or 'mean field'")
    ap.add_argument("--readout", default="rec adapt2+V2", help="video: the repaired readout")
    ap.add_argument("--runs", default="yaw_bright,approach_bright")
    args = ap.parse_args(argv)
    if args.what == "score":
        score(args.members.split(","), ensemble=not args.no_ensemble, only=args.only, resume=args.resume)
    elif args.what == "table":
        print(table())
    elif args.what == "figure":
        figure()
    elif args.what == "video":
        for r in args.runs.split(","):
            video(r, args.member, args.readout)


if __name__ == "__main__":
    main()
