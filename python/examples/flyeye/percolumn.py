"""Per-column and magnitude-dependent readout maps: does a fitted map that varies with column
position or with response magnitude close the residual direction gap?

    py -3.14 python/examples/flyeye/percolumn.py run --group "mean field" 008 009 000
    py -3.14 python/examples/flyeye/percolumn.py figure

Offline on the Phase 2 recordings (C:/dev/_flyeye/phase2, truth) and the member replays
(C:/dev/_flyeye/ensemble/raw/<NNN>, t4t5_raw). No rendering, no threepp import, scoring.py's
rules untouched, so every number is comparable to the repair round in
C:/dev/_flyeye/repair/scores.json.

Every readout is causal EMA adaptation (tau 2 s, repair.adapt) over t4t5_raw followed by a
linear 8 -> 2 map; only the map varies:

  SHARED   one 8 -> 2 map for all columns and speeds (the repair round's "rec adapt2+V2",
           refitted here with the same stride 2, so it reproduces that row = pipeline check)
  PC       a separate ridge 8 -> 2 map per column per eye (lam = alpha * trace(XtX)/8 per column)
  PCr      the same with a stronger ridge
  PCS      PC averaged with each column's 6 hex neighbours (7-column mean of the fitted maps)
  MAG      one shared map per tercile of |adapted rate| per column-frame (causal, usable)
  ORACLE   one shared map per TRUE angular-speed bin -- DIAGNOSTIC UPPER BOUND, not a sensor

Fitted on the BRIGHT runs only; bright is train, dim and dark are test. Every readout is scored
against its OWN motion-blind baseline (the same readout on the run's own rest frames 30-100,
ping-pong repeated with 10 s of settle dropped), exactly as the repair round did.

Groups: a single member ("000", "008", "009") or "mean field" = the mean of the 10 members' own
readout fields (each member gets its own fitted maps), which is what scored 0.493 with SHARED.

Writes C:/dev/_flyeye/percolumn/percolumn.json (all numbers) and residual.npz (per-column
circular means for the figure), plus figures/percolumn.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyeye import repair as rp  # noqa: E402
from flyeye import scoring as sc  # noqa: E402
from flyeye.lattice import HexLattice  # noqa: E402

HERE = Path(__file__).resolve().parent
ENS = Path(r"C:\dev\_flyeye\ensemble")
OUT = Path(r"C:\dev\_flyeye\percolumn")
JSON = OUT / "percolumn.json"
RESID = OUT / "residual.npz"
MEMBERS = [f"{i:03d}" for i in range(10)]
GROUPS = {"000": ["000"], "008": ["008"], "009": ["009"], "mean field": MEMBERS}
TAU, LAG = 2.0, 2
ALPHA, ALPHA_R = 0.01, 0.3
Q_MAG = 3
MIN_BIN_N = 2000  # samples a speed bin needs for its own ORACLE map, else the shared map
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = 721
VARIANTS = ["SHARED", "PC", "PCr", "PCS", "MAG", "ORACLE"]


# ---------------------------------------------------------------------------------------------- data
def load_raw(member: str, key) -> np.ndarray:
    z = np.load(ENS / "raw" / member / f"{key[0]}_{key[1]}.npz")
    return z["t4t5_raw"].astype(np.float32)


def adapted(member, key, tau=TAU):
    """(A, AB): the adapted raw (T, B, 8, N) and the adapted motion-blind input, cuda float32."""
    raw = load_raw(member, key)
    A = torch.as_tensor(rp.adapt(raw, tau), device=DEV)
    AB = torch.as_tensor(rp.adapt(rp.rest_input(raw), tau)[rp.SETTLE:], device=DEV)
    return A, AB


def selection(run: sc.Run, lag=LAG):
    """Fitting samples: (k, sel (K,B,N) bool, Y (K,B,2,N) unit true direction, sbin (K,B,N))."""
    T = run.T
    k = np.nonzero((run.into >= sc.SKIP) & (np.arange(T) >= lag))[0]
    kt = k - lag
    ang, sp, sbin = run.truth(False)
    sel = (sp[kt] >= sc.MIN_SPEED) & run.scene[kt]
    a = ang[kt]
    return k, sel, np.stack([np.cos(a), np.sin(a)], 2).astype(np.float32), sbin[kt]


def truth_sbin_all(run: sc.Run, lag=LAG) -> np.ndarray:
    """(T, B, N) the true speed bin of the truth frame the circuit frame is compared with."""
    _, _, sbin = run.truth(False)
    kt = np.clip(np.arange(run.T) - lag, 0, run.T - 1)
    return sbin[kt]


# ---------------------------------------------------------------------------------------------- fitting
class Fit:
    """Per-member accumulators for every map family (bright runs only)."""

    def __init__(self):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=DEV)
        self.pc_xx, self.pc_xy = z(2, 8, 8, N), z(2, 8, 2, N)
        self.sh_xx, self.sh_xy = z(8, 8), z(8, 2)
        self.mg_xx, self.mg_xy = z(Q_MAG, 8, 8), z(Q_MAG, 8, 2)
        self.sp_xx, self.sp_xy = z(sc.N_SPEED, 8, 8), z(sc.N_SPEED, 8, 2)
        self.n_pc = torch.zeros(2, N, dtype=torch.float64, device=DEV)
        self.n_mg, self.n_sp = np.zeros(Q_MAG), np.zeros(sc.N_SPEED)
        self.n_sh = 0
        self.mags = []

    def collect_mag(self, A, k, sel):
        m = A[k].norm(dim=2)  # (K, B, N)
        self.mags.append(m[torch.as_tensor(sel, device=DEV)][::37].float().cpu().numpy())

    def edges(self):
        v = np.concatenate(self.mags)
        return np.quantile(v, np.arange(1, Q_MAG) / Q_MAG)

    def add(self, A, k, sel, Y, sbin, mag_edges):
        X = A[k]  # (K, B, 8, N)
        m = torch.as_tensor(sel, device=DEV)[:, :, None].to(X.dtype)  # (K, B, 1, N)
        Xm, Ym = X * m, torch.as_tensor(Y, device=DEV) * m
        self.pc_xx += torch.einsum("kbin,kbjn->bijn", Xm, X).double()
        self.pc_xy += torch.einsum("kbin,kbcn->bicn", Xm, Ym).double()
        self.n_pc += m.sum((0, 2)).double()
        s = slice(None, None, 2)  # the repair round's stride 2, same base frames
        self.sh_xx += torch.einsum("kbin,kbjn->ij", Xm[s], X[s]).double()
        self.sh_xy += torch.einsum("kbin,kbcn->ic", Xm[s], Ym[s]).double()
        self.n_sh += int(m[s].sum().item())
        mb = np.digitize(X.norm(dim=2).float().cpu().numpy(), mag_edges)  # (K, B, N) in [0, Q)
        for q in range(Q_MAG):
            mq = m * torch.as_tensor(mb == q, device=DEV)[:, :, None].to(X.dtype)
            self.mg_xx[q] += torch.einsum("kbin,kbjn->ij", X * mq, X).double()
            self.mg_xy[q] += torch.einsum("kbin,kbcn->ic", X * mq, Ym).double()
            self.n_mg[q] += float(mq.sum().item())
        for q in np.unique(sbin):
            mq = m * torch.as_tensor(sbin == q, device=DEV)[:, :, None].to(X.dtype)
            self.sp_xx[q] += torch.einsum("kbin,kbjn->ij", X * mq, X).double()
            self.sp_xy[q] += torch.einsum("kbin,kbcn->ic", X * mq, Ym).double()
            self.n_sp[q] += float(mq.sum().item())

    def solve(self, mag_edges):
        """{name: map data} for this member."""
        out = {}
        Wsh = torch.linalg.solve(self.sh_xx, self.sh_xy)  # (8, 2)
        out["SHARED"] = Wsh.float()
        xx = self.pc_xx.permute(0, 3, 1, 2).contiguous()  # (2, N, 8, 8)
        xy = self.pc_xy.permute(0, 3, 1, 2).contiguous()  # (2, N, 8, 2)
        tr = torch.diagonal(xx, dim1=2, dim2=3).sum(-1) / 8  # (2, N)
        eye = torch.eye(8, dtype=torch.float64, device=DEV)
        for name, alpha in (("PC", ALPHA), ("PCr", ALPHA_R)):
            lam = (alpha * tr).clamp(min=1e-12)
            W = torch.linalg.solve(xx + lam[..., None, None] * eye, xy)  # (2, N, 8, 2)
            out[name] = W.float()
            out[f"lam_{name}"] = float(lam.median().item())
        P1 = torch.as_tensor(sc.hex_pool(radius=1), dtype=torch.float32, device=DEV)
        out["PCS"] = torch.einsum("mn,bnic->bmic", P1, out["PC"])
        mg = []
        for q in range(Q_MAG):
            mg.append(torch.linalg.solve(self.mg_xx[q], self.mg_xy[q]) if self.n_mg[q] > MIN_BIN_N else Wsh)
        out["MAG"] = torch.stack(mg).float()
        out["mag_edges"] = mag_edges
        sp = []
        for q in range(sc.N_SPEED):
            sp.append(torch.linalg.solve(self.sp_xx[q], self.sp_xy[q]) if self.n_sp[q] > MIN_BIN_N else Wsh)
        out["ORACLE"] = torch.stack(sp).float()
        out["n_pc"] = self.n_pc.float().cpu().numpy()
        out["n_sh"], out["n_mg"], out["n_sp"] = self.n_sh, self.n_mg.copy(), self.n_sp.copy()
        return out


def apply_map(name, maps, A, sel_idx=None):
    """(T, B, 8, N) -> (T, B, 2, N) with the named map family."""
    W = maps[name]
    if name == "SHARED":
        return torch.einsum("tbin,ic->tbcn", A, W)
    if name in ("PC", "PCr", "PCS"):
        return torch.einsum("tbin,bnic->tbcn", A, W)
    Wg = W[torch.as_tensor(sel_idx, device=DEV)]  # (T, B, N, 8, 2)
    return torch.einsum("tbin,tbnic->tbcn", A, Wg)


def selector(name, maps, A, run, oracle_bins):
    if name == "MAG":
        return np.digitize(A.norm(dim=2).float().cpu().numpy(), maps["mag_edges"])
    if name == "ORACLE":
        return oracle_bins
    return None


# ---------------------------------------------------------------------------------------------- residual bookkeeping
class Residual:
    """Circular sums of the direction residual (circuit minus truth) per column, speed bin, scenario."""

    def __init__(self):
        self.col = {}
        self.speed = {}
        self.scen = {}

    @staticmethod
    def _acc(d, key, z, n):
        if key in d:
            d[key][0] += z
            d[key][1] += n
        else:
            d[key] = [z, n]

    def add(self, light, scen, err, ok, scene, sbin):
        m = ok & scene
        e = np.exp(1j * np.radians(err))
        z = np.where(m, e, 0)
        self._acc(self.col, light, z.sum(0), m.sum(0))  # (B, N)
        self._acc(self.scen, (light, scen), z.sum(), m.sum())
        zs = np.zeros(sc.N_SPEED, complex)
        ns = np.zeros(sc.N_SPEED)
        f = m.ravel()
        np.add.at(zs, sbin.ravel()[f], z.ravel()[f])
        np.add.at(ns, sbin.ravel()[f], 1)
        self._acc(self.speed, light, zs, ns)

    def report(self):
        out = {}
        for light, (z, n) in self.col.items():
            mean = z / np.maximum(n, 1)
            ang, conc = np.degrees(np.angle(mean)), np.abs(mean)
            good = n.sum(0) > 200
            out[light] = dict(
                col_angle_median=round(float(np.median(np.degrees(np.angle(z.sum(0)[good] / np.maximum(n.sum(0)[good], 1))))), 2),
                col_angle_abs_median=round(float(np.median(np.abs(ang[n > 200]))), 2),
                col_angle_p10_p90=[round(float(v), 2) for v in np.percentile(ang[n > 200], [10, 90])],
                col_angle_circular_sd_over_columns=round(float(np.degrees(np.sqrt(max(-2 * np.log(max(
                    abs(np.exp(1j * np.radians(ang[n > 200])).mean()), 1e-9)), 0)))), 2),
                conc_median=round(float(np.median(conc[n > 200])), 3),
                per_eye_angle=[round(float(np.degrees(np.angle(z[b].sum() / max(n[b].sum(), 1)))), 2) for b in range(2)],
                per_eye_conc=[round(float(abs(z[b].sum() / max(n[b].sum(), 1))), 3) for b in range(2)],
                pooled_angle=round(float(np.degrees(np.angle(z.sum()))), 2), pooled_conc=round(float(abs(z.sum() / n.sum())), 3))
            zs, ns = self.speed[light]
            mn = zs / np.maximum(ns, 1)
            out[light]["by_speed_angle"] = [round(float(np.degrees(np.angle(v))), 1) if c > 200 else None for v, c in zip(mn, ns)]
            out[light]["by_speed_conc"] = [round(float(abs(v)), 3) if c > 200 else None for v, c in zip(mn, ns)]
            out[light]["by_speed_n"] = ns.astype(int).tolist()
            out[light]["by_scenario_angle"] = {s: round(float(np.degrees(np.angle(self.scen[(light, s)][0]))), 1)
                                               for s in sc.SCENARIOS if (light, s) in self.scen}
            out[light]["by_scenario_conc"] = {s: round(float(abs(self.scen[(light, s)][0] / max(self.scen[(light, s)][1], 1))), 3)
                                              for s in sc.SCENARIOS if (light, s) in self.scen}
        return out


# ---------------------------------------------------------------------------------------------- one group
def run_group(group, runs, variants=VARIANTS, save_maps=True):
    members = GROUPS[group]
    t0 = time.time()
    # ---- fit on bright, caching the adapted bright runs (float16 on the CPU) for the second pass
    fits, cache = {}, {}
    for m in members:
        F = Fit()
        for key in runs.keys(("bright",)):
            A, _ = adapted(m, key)
            cache[(m, key)] = A.half().cpu()
            k, sel, Y, sbin = selection(runs[key])
            F.collect_mag(A, k, sel)
            del A
        edges = F.edges()
        for key in runs.keys(("bright",)):
            A = cache[(m, key)].to(DEV).float()
            k, sel, Y, sbin = selection(runs[key])
            F.add(A, k, sel, Y, sbin, edges)
            del A
        fits[m] = F.solve(edges)
        print(f"  {group} member {m}: fitted, per-column N median {np.median(fits[m]['n_pc']):.0f}, "
              f"shared N {fits[m]['n_sh']}, mag edges {np.round(edges, 4).tolist()}, "
              f"speed-bin N {fits[m]['n_sp'].astype(int).tolist()} [{time.time() - t0:.0f} s]", flush=True)
    cache.clear()

    # ---- score every variant in one pass over the 21 runs
    P = torch.as_tensor(runs.P, dtype=torch.float32, device=DEV)
    C = {v: {} for v in variants}
    Cp = {v: {} for v in variants}
    Cb = {v: {} for v in variants}
    rot = {v: {} for v in variants}
    res = {v: Residual() for v in variants}
    for key in runs.keys():
        run = runs[key]
        oracle = truth_sbin_all(run)
        As = {}
        for m in members:
            As[m] = adapted(m, key)
        for v in variants:
            f = torch.zeros(run.T, 2, 2, N, device=DEV)
            fb = torch.zeros_like(f)
            for m in members:
                A, AB = As[m]
                f += apply_map(v, fits[m], A, selector(v, fits[m], A, run, oracle))
                fb += apply_map(v, fits[m], AB, selector(v, fits[m], AB, run, oracle))
            f /= len(members)
            fb /= len(members)
            fn = f.cpu().numpy()
            C[v][key] = sc.direction_counts(fn, run, [LAG])
            Cp[v][key] = sc.direction_counts(torch.einsum("mn,tbkn->tbkm", P, f).cpu().numpy(), run, [LAG], pooled=True)
            Cb[v][key] = sc.direction_counts(fb.cpu().numpy(), run, [LAG])
            if key[0] in sc.AXIS:
                rot[v][key] = sc.field_rotation(fn, run, device=DEV)
            _, kt, err, _, ok, _ = sc.frame_hits(fn, run, LAG)
            res[v].add(key[1], key[0], err, ok, run.scene[kt], run.truth(False)[2][kt])
            del f, fb, fn
        del As
        run.drop()
        print(f"  {group}: {key[0]}_{key[1]} done [{time.time() - t0:.0f} s]", flush=True)

    out = {}
    for v in variants:
        g = sc.gate_from_counts(C[v])
        b = sc.gate_from_counts(Cb[v], lag_index=0, bin_index=g["bin_index"])
        p = sc.gate_from_counts(Cp[v], lag_index=0)
        R = sc.score_rotation(rot[v], runs)
        row = dict(name=v, gate=g["value"], bin=g["bin"], n=g["n"], baseline=b["value"],
                   lift=round(g["value"] - b["value"], 4), by_light=g["by_light"], baseline_by_light=b["by_light"],
                   fixed_bin_gate=g["by_speed"][3], fixed_bin_base=b["by_speed"][3], pooled_19=p["value"],
                   pooled_bin=p["bin"], overall=g["overall"], by_speed=g["by_speed"], baseline_by_speed=b["by_speed"],
                   n_by_speed=g["n_by_speed"],
                   r2_seg={l: R[l]["r2_seg"] for l in R}, gain_sign={l: {s: int(np.sign(R[l][s]["gain_seg"])) for s in sc.AXIS}
                                                                     for l in R},
                   signs={l: R[l]["signs_right"] for l in R}, residual=res[v].report())
        if v in ("PC", "PCr"):
            row["ridge_lam_median"] = float(np.mean([fits[m][f"lam_{v}"] for m in members]))
            row["ridge_alpha"] = ALPHA if v == "PC" else ALPHA_R
        if v == "MAG":
            row["mag_edges"] = [float(x) for x in np.mean([fits[m]["mag_edges"] for m in members], 0)]
            row["mag_n"] = np.mean([fits[m]["n_mg"] for m in members], 0).astype(int).tolist()
        if v == "ORACLE":
            row["speed_bin_n"] = np.mean([fits[m]["n_sp"] for m in members], 0).astype(int).tolist()
        row["per_column_n_median"] = float(np.median([np.median(fits[m]["n_pc"]) for m in members]))
        out[v] = row
        print(f"{group:12s} {v:8s} gate {row['gate']:.4f} ({row['bin']}) base {row['baseline']:.4f} lift {row['lift']:+.4f} | "
              f"bright {g['by_light']['bright']:.4f} dim {g['by_light']['dim']:.4f} dark {g['by_light']['dark']:.4f} | "
              f"2-4 {row['fixed_bin_gate']:.4f}/{row['fixed_bin_base']:.4f} | p19 {row['pooled_19']:.4f} | "
              f"resid bright {row['residual']['bright']['pooled_angle']:+.1f} deg conc "
              f"{row['residual']['bright']['pooled_conc']:.3f}", flush=True)
    cols = {f"{group}|{v}|{l}": (res[v].col[l][0] / np.maximum(res[v].col[l][1], 1)) for v in variants for l in res[v].col}
    return out, cols


def run(groups, variants=VARIANTS):
    OUT.mkdir(parents=True, exist_ok=True)
    scores = json.loads(JSON.read_text()) if JSON.exists() else {}
    cols = dict(np.load(RESID)) if RESID.exists() else {}
    runs = sc.Runs()
    for g in groups:
        out, c = run_group(g, runs, variants)
        scores.setdefault(g, {}).update(out)
        scores["meta"] = dict(tau=TAU, lag_frames=LAG, alpha=ALPHA, alpha_strong=ALPHA_R, q_mag=Q_MAG,
                              min_bin_n=MIN_BIN_N, note="fitted on bright; dim and dark are test")
        JSON.write_text(json.dumps(scores, indent=1, default=float))
        cols.update(c)
        np.savez(RESID, **cols)
    print(f"wrote {JSON} and {RESID}")
    return scores


# ---------------------------------------------------------------------------------------------- held-out trajectories
FIT_SCEN = ("straight_3", "straight_10", "yaw", "roll")
TEST_SCEN = ("straight_30", "pitch", "approach")


def holdout(group, fit_scen=FIT_SCEN, test_scen=TEST_SCEN, variants=("SHARED", "PC", "PCS")):
    """Fit on the BRIGHT runs of fit_scen only, score on test_scen (all lightings).

    Bright, dim and dark share the scene and the trajectories, so they test lighting only. A
    per-column map can encode the scene's typical flow direction at each column, which would score
    on any lighting of the same flight. Holding trajectories out is the cheapest test of that.
    """
    runs = sc.Runs()
    members = GROUPS[group]
    fits = {}
    for m in members:
        F = Fit()
        cache = {}
        for s in fit_scen:
            A, _ = adapted(m, (s, "bright"))
            cache[s] = A.half().cpu()
            k, sel, Y, sbin = selection(runs[(s, "bright")])
            F.collect_mag(A, k, sel)
            del A
        edges = F.edges()
        for s in fit_scen:
            A = cache[s].to(DEV).float()
            k, sel, Y, sbin = selection(runs[(s, "bright")])
            F.add(A, k, sel, Y, sbin, edges)
            del A
        fits[m] = F.solve(edges)
    keys = [(s, l) for l in sc.LIGHTS for s in test_scen]
    C = {v: {} for v in variants}
    Cb = {v: {} for v in variants}
    for key in keys:
        run = runs[key]
        oracle = truth_sbin_all(run)
        As = {m: adapted(m, key) for m in members}
        for v in variants:
            f = torch.zeros(run.T, 2, 2, N, device=DEV)
            fb = torch.zeros_like(f)
            for m in members:
                A, AB = As[m]
                f += apply_map(v, fits[m], A, selector(v, fits[m], A, run, oracle))
                fb += apply_map(v, fits[m], AB, selector(v, fits[m], AB, run, oracle))
            C[v][key] = sc.direction_counts((f / len(members)).cpu().numpy(), run, [LAG])
            Cb[v][key] = sc.direction_counts((fb / len(members)).cpu().numpy(), run, [LAG])
            del f, fb
        del As
        run.drop()
    out = {}
    for v in variants:
        g = sc.gate_from_counts(C[v], scenarios=test_scen)
        b = sc.gate_from_counts(Cb[v], lag_index=0, bin_index=g["bin_index"], scenarios=test_scen)
        out[v] = dict(gate=g["value"], bin=g["bin"], n=g["n"], baseline=b["value"], lift=round(g["value"] - b["value"], 4),
                      by_light=g["by_light"], baseline_by_light=b["by_light"], fixed_bin_gate=g["by_speed"][3],
                      fixed_bin_base=b["by_speed"][3], by_scenario=g["by_scenario"])
        print(f"{group:12s} holdout {v:8s} gate {out[v]['gate']:.4f} ({out[v]['bin']}) base {out[v]['baseline']:.4f} "
              f"lift {out[v]['lift']:+.4f} | bright {g['by_light']['bright']:.4f} dim {g['by_light']['dim']:.4f} "
              f"dark {g['by_light']['dark']:.4f} | 2-4 {out[v]['fixed_bin_gate']:.4f}/{out[v]['fixed_bin_base']:.4f}", flush=True)
    scores = json.loads(JSON.read_text()) if JSON.exists() else {}
    scores.setdefault(group, {})["holdout"] = dict(fit_scenarios=list(fit_scen), test_scenarios=list(test_scen), rows=out)
    JSON.write_text(json.dumps(scores, indent=1, default=float))
    return out


# ---------------------------------------------------------------------------------------------- table
def table(scores=None, groups=None):
    scores = scores or json.loads(JSON.read_text())
    groups = groups or [g for g in GROUPS if g in scores]
    head = ("| group | readout | gate (bin) | own base | lift | 2-4 col/s gate / base | pooled-19 | dim / dark test | "
            "R^2 bright y/p/r | R^2 dark y/p/r | signs b/d/k | median per-column residual (bright / dark) |")
    out = [head, "|" + "---|" * 12]
    for g in groups:
        for v in VARIANTS:
            r = scores[g].get(v)
            if r is None:
                continue
            sr = lambda l: " / ".join(f"{r['r2_seg'][l][a] * r['gain_sign'][l][a]:+.2f}" for a in ("yaw", "pitch", "roll"))
            rb, rk = r["residual"]["bright"], r["residual"]["dark"]
            out.append(f"| {g} | {v} | {r['gate']:.3f} ({r['bin'].replace(' columns/s', '')}) | {r['baseline']:.3f} | "
                       f"{r['lift']:+.3f} | {r['fixed_bin_gate']:.3f} / {r['fixed_bin_base']:.3f} | {r['pooled_19']:.3f} | "
                       f"{r['by_light']['dim']:.3f} / {r['by_light']['dark']:.3f} | {sr('bright')} | {sr('dark')} | "
                       f"{'/'.join(str(r['signs'][l]) for l in sc.LIGHTS)} | "
                       f"{rb['col_angle_median']:+.0f} deg (|{rb['col_angle_abs_median']:.0f}|) / "
                       f"{rk['col_angle_median']:+.0f} deg |")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- figure
def figure(path=HERE / "figures" / "percolumn.png", group="mean field"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = json.loads(JSON.read_text())
    cols = dict(np.load(RESID))
    lat = HexLattice()
    c = lat.centers.numpy()
    X, Y = c[:, 1].astype(float), -c[:, 0].astype(float)
    groups = [g for g in GROUPS if g in scores]
    fig = plt.figure(figsize=(22, 24))
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1.1, 1.0, 0.5], hspace=0.35, wspace=0.3)

    # rows 0-1: per-column residual angle maps, before (SHARED) and after (PC), per eye, bright and dark
    for row, light in enumerate(("bright", "dark")):
        for j, (v, b) in enumerate([(v, b) for v in ("SHARED", "PC") for b in (0, 1)]):
            ax = fig.add_subplot(gs[row, j])
            key = f"{group}|{v}|{light}"
            m = cols[key][b]
            ang = np.degrees(np.angle(m))
            sca = ax.scatter(X, Y, c=ang, s=42, cmap="twilight_shifted", vmin=-180, vmax=180)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            r = scores[group][v]["residual"][light]
            ax.set_title(f"{v}, eye {b} ({'+45 L' if b == 0 else '-45 R'}), {light}: per-column mean\n"
                         f"residual {r['per_eye_angle'][b]:+.0f} deg, conc {r['per_eye_conc'][b]:.2f}", fontsize=9)
            if j == 3:
                fig.colorbar(sca, ax=ax, fraction=0.045, label="residual angle (circuit - truth), deg")

    # row 2: gate and own baseline per readout and group
    ax = fig.add_subplot(gs[2, :])
    w = 0.8 / len(VARIANTS)
    x = np.arange(len(groups))
    colr = dict(SHARED="#999999", PC="#d55e00", PCr="#e69f00", PCS="#009e73", MAG="#0072b2", ORACLE="#cc79a7")
    for j, v in enumerate(VARIANTS):
        off = (j - (len(VARIANTS) - 1) / 2) * w
        gv = [scores[g].get(v, {}).get("gate", np.nan) for g in groups]
        bv = [scores[g].get(v, {}).get("baseline", np.nan) for g in groups]
        ax.bar(x + off, gv, w * 0.9, color=colr[v], label=v + (" (oracle)" if v == "ORACLE" else ""), zorder=2,
               hatch="//" if v == "ORACLE" else None, edgecolor="white")
        ax.plot(x + off, bv, "_", color="k", ms=12, mew=2.2, zorder=3, label="own motion-blind baseline" if j == 0 else None)
        for xi, (gvv, bvv) in enumerate(zip(gv, bv)):
            if np.isfinite(gvv):
                ax.text(xi + off, gvv + 0.004, f"{gvv:.3f}", ha="center", fontsize=7, rotation=90)
    for yv, txt, ls, col in ((0.5, "target 0.5", "-", "#b00000"), (0.493, "repair round SHARED mean field 0.493", "--", "k"),
                             (0.320, "its motion-blind baseline 0.320", "-.", "k"), (0.25, "chance 0.25", ":", "k")):
        ax.axhline(yv, color=col, lw=1, ls=ls, zorder=1)
        ax.text(len(groups) - 0.45, yv + 0.004, txt, ha="right", fontsize=9)
    ax.set_xticks(x, groups, fontsize=12)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0.15, 1.02)
    ax.set_ylabel("gate: fraction within 45 deg, best speed bin, all lightings")
    ax.set_title("Direction gate per readout (bars) and its own motion-blind baseline (black ticks); maps fitted on the "
                 "BRIGHT runs of the SAME flights that are scored (dim and dark share the trajectories)", fontsize=12)
    ax.legend(ncol=8, fontsize=9, loc="upper left")

    # row 3: residual angle and concentration vs true speed, per readout
    edges = sc.SPEED_EDGES
    ctr = np.sqrt(edges[:-1] * edges[1:])
    for j, light in enumerate(("bright", "dark")):
        ax = fig.add_subplot(gs[3, j])
        for v in VARIANTS:
            r = scores[group][v]["residual"][light]
            y = [np.nan if a is None else a for a in r["by_speed_angle"]]
            ax.plot(ctr, y, "o-", color=colr[v], label=v, ms=4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("true angular speed (columns/s)")
        ax.set_ylabel("circular mean residual angle (deg)")
        ax.set_title(f"{group}: residual angle vs true speed, {light}", fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[3, 2])
    for v in VARIANTS:
        r = scores[group][v]["residual"]["bright"]
        ax.plot(ctr, [np.nan if a is None else a for a in r["by_speed_conc"]], "o-", color=colr[v], label=v, ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("true angular speed (columns/s)")
    ax.set_ylabel("concentration |mean e^i err|")
    ax.set_title(f"{group}: residual concentration vs speed, bright", fontsize=11)
    ax.grid(alpha=0.3)
    # held-out trajectories: the decisive test
    ax = fig.add_subplot(gs[3, 3])
    hg = [g for g in groups if "holdout" in scores.get(g, {})]
    if hg:
        hv = list(scores[hg[0]]["holdout"]["rows"])
        xh = np.arange(len(hg))
        wh = 0.8 / len(hv)
        for j, v in enumerate(hv):
            off = (j - (len(hv) - 1) / 2) * wh
            gv = [scores[g]["holdout"]["rows"][v]["gate"] for g in hg]
            bv = [scores[g]["holdout"]["rows"][v]["baseline"] for g in hg]
            ax.bar(xh + off, gv, wh * 0.9, color=colr[v], label=v, zorder=2)
            ax.plot(xh + off, bv, "_", color="k", ms=14, mew=2.2, zorder=3)
            for xi, gvv in zip(xh, gv):
                ax.text(xi + off, gvv + 0.006, f"{gvv:.3f}", ha="center", fontsize=8)
        ax.axhline(0.5, color="#b00000", lw=1)
        ax.set_xticks(xh, hg)
        ax.set_ylim(0.15, 0.75)
        ax.set_ylabel("gate on held-out flights")
        ax.legend(fontsize=8)
        h = scores[hg[0]]["holdout"]
        ax.set_title("HELD-OUT TRAJECTORIES: fitted on " + "+".join(s.replace("straight_", "s") for s in h["fit_scenarios"])
                     + "\nscored on " + "+".join(s.replace("straight_", "s") for s in h["test_scenarios"])
                     + " (black ticks = own motion-blind baseline)", fontsize=10)
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    lines = [f"{'group':12s} {'readout':8s} {'gate':>6s} {'base':>6s} {'lift':>6s} {'bright':>7s} {'dim':>6s} {'dark':>6s} "
             f"{'p19':>6s} {'2-4 gate/base':>14s} {'|resid| per column':>19s} | held-out flights: gate / base / lift"]
    for g in groups:
        for v in VARIANTS:
            r = scores[g].get(v)
            if not r:
                continue
            h = scores[g].get("holdout", {}).get("rows", {}).get(v)
            ho = f"{h['gate']:.3f} / {h['baseline']:.3f} / {h['lift']:+.3f}" if h else "-"
            lines.append(f"{g:12s} {v:8s} {r['gate']:6.3f} {r['baseline']:6.3f} {r['lift']:+6.3f} "
                         f"{r['by_light']['bright']:7.3f} {r['by_light']['dim']:6.3f} {r['by_light']['dark']:6.3f} "
                         f"{r['pooled_19']:6.3f} {r['fixed_bin_gate']:6.3f} / {r['fixed_bin_base']:.3f}  "
                         f"{r['residual']['bright']['col_angle_abs_median']:6.1f} deg (bright)      {ho}")
    lines += ["", "gate = fraction of scene column-frames within 45 deg of the renderer's truth, best speed bin (N >= 20k), lag 20 ms;",
              "base = the same readout on the run's own rest frames, repeated (motion-blind); ORACLE picks its map by the TRUE speed."]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.suptitle("flyeye: per-column and magnitude-dependent readout maps on the Phase 2 recordings "
                 "(adaptation tau 2 s, fitted on bright, lag 20 ms)", fontsize=15)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "holdout", "table", "figure"])
    ap.add_argument("--group", nargs="*", default=["mean field"])
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    args = ap.parse_args(argv)
    if args.what == "run":
        run(args.group, args.variants)
    elif args.what == "holdout":
        for g in args.group:
            holdout(g)
    elif args.what == "table":
        print(table())
    else:
        figure()


if __name__ == "__main__":
    main()
