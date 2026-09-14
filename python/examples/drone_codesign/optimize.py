"""Gates and the pilot run for the sea-drone propeller co-design.

    python python/examples/drone_codesign/optimize.py --selftest

The claim this file has to earn is small and specific: that a gradient taken
through the Warp chain in `prop.py` walks to the SAME propeller a brute-force
search over a quarter of a million designs finds. Until that is true, nothing
downstream -- a hull in the same graph, a Michell wave-making term, a
sculpted blade -- is worth writing, because a wrong adjoint does not crash and
does not look wrong; it quietly optimises something else.

Four gates, in the order they are allowed to be believed:

  --check-tables  the copied polynomials ARE the demo's, parsed out of its
                  source with `ast` (never imported: it opens a window), and
                  the kernel evaluates them to 1e-6 of a numpy reference.
  --fd            central finite differences against the tape, 8 designs x 3
                  coordinates. This is the gate that decides everything.
  --grid          240 x 240 x 8 brute force, same kernels, no tape.
  --opt           300 Adam steps from the brief's initial guess.
  --selftest      all four, and a verdict comparing the last two.
"""

import argparse
import ast
import json
import math
import os
import sys
import time

import numpy as np
import warp as wp
import warp.optim

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PY = os.path.dirname(os.path.dirname(_HERE))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import prop  # noqa: E402

DEMO = os.path.join(os.path.dirname(_HERE), "warp_prop_vortex.py")
W_PEAK = 0.30           # the demo's own wake-peak default, carried in the JSON


# ── reporting helpers ──────────────────────────────────────────────────────

def row_of(rows, i):
    return {k: float(v[i]) for k, v in rows.items()}


def brief_header(brief):
    h = brief["hull"]
    return (
        f"  hull       L {h['L']} m, S_wet {h['S_wet']} m2, "
        f"1+k {1.0 + h['k_form']}, C_R {h['C_R']} (residuary placeholder)\n"
        f"  speeds     V_survey {brief['V_survey']} m/s, "
        f"V_sprint {brief['V_sprint']} m/s   "
        f"R_T {prop.resistance(brief['V_survey'], brief):.1f} / "
        f"{prop.resistance(brief['V_sprint'], brief):.1f} N\n"
        f"  factors    w {brief['w']}, t {brief['t']}\n"
        f"  limits     E_batt {brief['E_batt']} kWh, n_max {brief['n_max']} "
        f"rps, D_max {brief['D_max']} m, depth "
        f"[{brief['depth_min']}, {brief['depth_max']}] m\n"
        f"  penalties  lam_cav {brief['lam_cav']}, lam_n {brief['lam_n']}, "
        f"lam_box {brief['lam_box']}\n"
        f"  screw      B{prop.BLADES}-{prop.EAR:.2f}, box P/D "
        f"[{prop.PD_LO}, {prop.PD_HI}], J [{prop.J_LO}, {prop.J_HI}]")


def design_line(r, brief):
    return (f"D {r['D']:.4f} m  P/D {r['PD']:.4f}  depth {r['depth']:.3f} m  "
            f"n {r['n_survey'] * 60.0:7.1f} rpm  J {r['J_survey']:.4f}  "
            f"eta_0 {r['eta_survey']:.4f}  P_D {r['P_survey']:7.2f} W  "
            f"range {r['range_km']:8.3f} km  Burrill(sprint) "
            f"{r['ratio_sprint']:.4f}  [{', '.join(prop.binding(r, brief))}]")


# ── gate 1: the tables ─────────────────────────────────────────────────────

def parse_demo_tables(path=DEMO):
    """_KT_TERMS / _KQ_TERMS and a few constants, out of the demo's SOURCE.

    `warp_prop_vortex.py` opens a window at import, so it is never imported:
    the file is read as text and walked as a syntax tree, which is also the
    only way to read a value that is guarded behind a CLI default."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name):
            try:
                out[tgt.id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
        elif isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
            for nm, val in zip(tgt.elts, node.value.elts):
                if isinstance(nm, ast.Name):
                    try:
                        out[nm.id] = ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        pass
    return out


def check_tables(device, verbose=True):
    ok = True
    src = parse_demo_tables()
    for name, mine in (("_KT_TERMS", prop.KT_TERMS), ("_KQ_TERMS",
                                                      prop.KQ_TERMS)):
        theirs = src.get(name)
        if theirs is None:
            print(f"  FAIL  {name} not found in {os.path.basename(DEMO)}")
            ok = False
            continue
        same = tuple(tuple(t) for t in theirs) == tuple(tuple(t)
                                                        for t in mine)
        if verbose:
            print(f"  {'ok  ' if same else 'FAIL'}  {name}: {len(mine)} terms,"
                  f" identical to the demo's source: {same}")
        ok = ok and same
    for name, mine in (("EAR", prop.EAR), ("RHO", prop.RHO),
                       ("P_ATM", prop.P_ATM), ("P_VAP", prop.P_VAP),
                       ("PD_LO", prop.PD_LO), ("PD_HI", prop.PD_HI),
                       ("J_LO", prop.J_LO), ("J_HI", prop.J_HI),
                       ("BETA_0", prop.BETA_0),
                       ("BURRILL_5PCT", prop.BURRILL_5PCT),
                       ("BURRILL_P", prop.BURRILL_P)):
        theirs = src.get(name)
        good = theirs is not None and float(theirs) == float(mine)
        ok = ok and good
        if not good and verbose:
            print(f"  FAIL  {name}: demo {theirs!r} vs ours {mine!r}")
    if verbose:
        print(f"  ok    11 scalar constants (EAR, RHO, the box, Burrill's "
              f"line) match the demo's source")

    rng = np.random.default_rng(7)
    j = rng.uniform(prop.J_LO, prop.J_HI, 64)
    pd = rng.uniform(0.2, 1.6, 64)          # deliberately outside the box too
    kt_g, kq_g = prop.poly_gpu(j, pd, device=device)
    kt_n, kq_n = prop.open_water_np(j, pd)
    ekt = float(np.max(np.abs(kt_g - kt_n)))
    ekq = float(np.max(np.abs(kq_g - kq_n)))
    good = ekt < 1.0e-6 and ekq < 1.0e-6
    ok = ok and good
    if verbose:
        print(f"  {'ok  ' if good else 'FAIL'}  k_poly vs numpy at 64 random "
              f"(J, P/D), clamp and fade included: max |dK_T| {ekt:.3e}, "
              f"max |dK_Q| {ekq:.3e}")
    return ok


# ── gate 2: the finite-difference check ────────────────────────────────────

def fd_designs(brief, n=8, n_cav=2, seed=20260914):
    """Designs in the box, none of them sitting ON the cavitation kink -- a
    central difference across a hinge measures the kink, not the gradient.

    Two of the eight are forced to be CAVITATING (a small screw at high
    pitch), because otherwise the hinge terms contribute nothing, the depth
    column of the table is all zeros, and the check would say nothing about
    the half of the objective the constraints live in."""
    rng = np.random.default_rng(seed)
    cav, plain = [], []
    buf = None
    while len(cav) < n_cav or len(plain) < n - n_cav:
        want_cav = len(cav) < n_cav
        cand = np.array([[rng.uniform(0.150, 0.170) if want_cav
                          else rng.uniform(0.20, 0.39),
                          rng.uniform(1.10, 1.35) if want_cav
                          else rng.uniform(0.55, 1.35),
                          rng.uniform(0.21, 0.44)]])
        buf = prop.chain(cand, brief, buf=buf, device=_DEV)
        ratio = float(prop.read_rows(buf)["ratio_sprint"][0])
        if abs(ratio - 1.0) <= 2.0e-3:
            continue                    # too close to the kink to difference
        if ratio > 1.0 and len(cav) < n_cav:
            cav.append(cand[0])
        elif ratio < 1.0 and len(plain) < n - n_cav:
            plain.append(cand[0])
    return np.array(cav + plain)


def fd_check(brief, device, verbose=True):
    designs = fd_designs(brief)
    n = designs.shape[0]
    buf = prop.Buffers(n, device=device, requires_grad=True)

    # range_ref is the objective's own normaliser and is HELD FIXED here, as
    # it is during an optimisation: it is a constant of the problem, not a
    # function of the design, and a finite difference that moved it would be
    # differencing a different objective at each step.
    prop.chain(designs, brief, buf=buf, device=device)
    base = prop.read_rows(buf)
    buf.range_ref.assign(np.ascontiguousarray(base["range_km"]))

    tape = wp.Tape()
    prop.chain(designs, brief, buf=buf, device=device, tape=tape)
    tape.backward(loss=buf.loss)
    adj = buf.design.grad.numpy().copy()

    fd = np.zeros_like(adj)
    for c in range(3):
        h = 1.0e-4 * np.abs(designs[:, c])
        plus, minus = designs.copy(), designs.copy()
        plus[:, c] += h
        minus[:, c] -= h
        prop.chain(plus, brief, buf=buf, device=device)
        lp = prop.read_rows(buf)["loss"].copy()
        prop.chain(minus, brief, buf=buf, device=device)
        lm = prop.read_rows(buf)["loss"].copy()
        # Per-ROW losses, not the reduced scalar: the rows are independent, so
        # perturbing a coordinate in all of them at once still gives each row
        # its own one-dimensional difference, in two launches instead of 16.
        fd[:, c] = (lp - lm) / (2.0 * h)

    names = ("dJ/dD", "dJ/d(P/D)", "dJ/ddepth")
    worst, worst_at = 0.0, ""
    ok = True
    if verbose:
        print("   row      D     P/D   depth  cav   coord           "
              "finite-diff          adjoint      rel")
    for i in range(n):
        for c in range(3):
            a, f = adj[i, c], fd[i, c]
            absd = abs(a - f)
            rel = absd / max(abs(f), 1.0e-300)
            good = rel < 0.01 or absd < 1.0e-6
            ok = ok and good
            if abs(f) > 1.0e-9 and rel > worst:
                worst, worst_at = rel, f"row {i} {names[c]}"
            if verbose:
                print(f"  {i:4d} {designs[i, 0]:6.3f} {designs[i, 1]:7.3f} "
                      f"{designs[i, 2]:7.3f} "
                      f"{base['ratio_sprint'][i]:5.2f}  {names[c]:<11s}"
                      f"{f:+18.10e} {a:+16.10e} {rel:8.1e}"
                      f"  {'ok' if good else 'FAIL'}")
    if verbose:
        print(f"  worst component {worst:.2e} relative"
              f"{' at ' + worst_at if worst_at else ''}"
              f"  (pass < 1e-2, or < 1e-6 absolute where the gradient is 0)")
        print(f"  Newton residual |T(1-t) - R_T|/R_T after "
              f"{prop.NEWTON_ITERS} iterations: max "
              f"{max(base['res_survey'].max(), base['res_sprint'].max()):.2e}")
    return ok, worst, worst_at


# ── gate 3: the brute force ────────────────────────────────────────────────

def grid_search(brief, device, nd=240, npd=240, nh=8, chunk=65536,
                verbose=True):
    ds = np.linspace(prop.D_MIN, brief["D_max"], nd)
    pds = np.linspace(prop.PD_LO, prop.PD_HI, npd)
    hs = np.linspace(brief["depth_min"], brief["depth_max"], nh)
    gd, gp, gh = np.meshgrid(ds, pds, hs, indexing="ij")
    grid = np.stack([gd.ravel(), gp.ravel(), gh.ravel()], axis=1)
    total = grid.shape[0]
    if verbose:
        print(f"  {nd} x {npd} x {nh} = {total} designs, same kernels, no tape")

    buf = prop.Buffers(min(chunk, total), device=device, requires_grad=False)
    best = None
    t0 = time.perf_counter()
    for s in range(0, total, chunk):
        blk = grid[s:s + chunk]
        if blk.shape[0] != buf.n_rows:
            buf = prop.Buffers(blk.shape[0], device=device,
                               requires_grad=False)
        prop.chain(blk, brief, buf=buf, device=device)
        r = prop.read_rows(buf)
        feas = (r["ratio_sprint"] <= 1.0) & (r["n_survey"] <= brief["n_max"])
        if not feas.any():
            continue
        idx = np.where(feas)[0]
        k = idx[int(np.argmax(r["range_km"][idx]))]
        if best is None or r["range_km"][k] > best["range_km"]:
            best = row_of(r, int(k))
    dt = time.perf_counter() - t0
    if verbose:
        print(f"  swept in {dt:.2f} s")
        print(f"  feasible optimum (ratio_sprint <= 1, n_survey <= n_max):")
        print(f"    {design_line(best, brief)}")
        print(f"  binding: {', '.join(prop.binding(best, brief))}")
        print(f"  grid pitch: dD {ds[1] - ds[0]:.5f} m, "
              f"d(P/D) {pds[1] - pds[0]:.5f}, ddepth {hs[1] - hs[0]:.5f} m")
    return best


# ── gate 4: the optimisation ───────────────────────────────────────────────

def design_json(r, brief, iteration):
    return dict(
        D=r["D"], PD=r["PD"],
        n_survey=r["n_survey"], n_sprint=r["n_sprint"],
        V_survey=brief["V_survey"], V_sprint=brief["V_sprint"],
        w_mean=brief["w"], w_peak=W_PEAK, t=brief["t"],
        depth=r["depth"], blades=prop.BLADES, EAR=prop.EAR,
        eta0=r["eta_survey"], P_D=r["P_survey"], range_km=r["range_km"],
        burrill_ratio_survey=r["ratio_survey"],
        burrill_ratio_sprint=r["ratio_sprint"],
        iteration=iteration, brief=brief,
    )


def optimise(brief, device, steps=300, lr=0.01, out_dir=None, verbose=True,
             snapshots=(0, 50, 200)):
    d_max, h_max = brief["D_max"], brief["depth_max"]
    u0 = np.array([0.20 / d_max, 0.80,
                   0.5 * (brief["depth_min"] + brief["depth_max"]) / h_max])
    lo = np.array([prop.D_MIN / d_max, prop.PD_LO, brief["depth_min"] / h_max])
    hi = np.array([1.0, prop.PD_HI, 1.0])

    u = u0.copy()
    u_wp = wp.array(u.astype(np.float32), dtype=wp.float32, device=device)
    g_wp = wp.zeros(3, dtype=wp.float32, device=device)
    adam = wp.optim.Adam([u_wp], lr=lr)
    buf = prop.Buffers(1, device=device, requires_grad=True)
    tape = wp.Tape()
    scale = np.array([d_max, 1.0, h_max])

    def unpack(uu):
        return (np.clip(uu, lo, hi) * scale).reshape(1, 3)

    # range_ref is the STEP-0 range, so the objective's first term starts at
    # exactly -1 and the penalty weights mean the same thing at every step.
    prop.chain(unpack(u), brief, buf=buf, device=device)
    buf.range_ref.assign(np.ascontiguousarray(
        prop.read_rows(buf)["range_km"]))
    range_ref = float(buf.range_ref.numpy()[0])

    saved, first, last = {}, None, None
    t0 = time.perf_counter()
    for it in range(steps + 1):
        tape.reset()
        prop.chain(unpack(u), brief, buf=buf, device=device, tape=tape)
        r = row_of(prop.read_rows(buf), 0)
        if it == 0:
            first = r
        last = r
        if it in snapshots or it == steps:
            saved[it] = design_json(r, brief, it)
        if it == steps:
            break
        tape.backward(loss=buf.loss)
        g = buf.design.grad.numpy()[0]
        g_wp.assign((g * scale).astype(np.float32))
        adam.step([g_wp])
        u = np.clip(u_wp.numpy().astype(np.float64), lo, hi)
        u_wp.assign(u.astype(np.float32))
    wall = time.perf_counter() - t0

    written = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for it, blob in sorted(saved.items()):
            name = ("design_final.json" if it == steps
                    else f"design_iter{it:04d}.json")
            with open(os.path.join(out_dir, name), "w",
                      encoding="utf-8") as fh:
                json.dump(blob, fh, indent=2, sort_keys=True)
            written.append(name)
    if verbose:
        print(f"  {steps} Adam steps, lr {lr}, normalised "
              f"(D/D_max, P/D, depth/depth_max), projected to the box after "
              f"every step")
        print(f"  range_ref (step-0 range) {range_ref:.3f} km")
        print(f"  step   0  {design_line(first, brief)}")
        print(f"  step {steps:3d}  {design_line(last, brief)}")
        print(f"  wall time {wall:.2f} s "
              f"({1000.0 * wall / (steps + 1):.1f} ms per forward+backward)")
        if written:
            print(f"  wrote {', '.join(written)}")
            print(f"     to {out_dir}")
    return first, last, wall


# ── the verdict ────────────────────────────────────────────────────────────

def verdict(best, last, brief):
    """PASS if Adam landed on the brute-force optimum: within 1 % in D and
    P/D and 0.5 % in range, or -- for a coordinate the box is holding -- the
    same binding constraint with that coordinate equal."""
    ed = abs(last["D"] - best["D"]) / max(best["D"], 1e-12)
    ep = abs(last["PD"] - best["PD"]) / max(best["PD"], 1e-12)
    er = abs(last["range_km"] - best["range_km"]) / max(best["range_km"], 1e-12)
    bg, bo = prop.binding(best, brief), prop.binding(last, brief)
    tight = ed < 0.01 and ep < 0.01 and er < 0.005
    same_bind = set(bg) == set(bo) and bg != ["interior"] and ed < 0.01 \
        and ep < 0.01
    lines = [f"  D      grid {best['D']:.5f}  adam {last['D']:.5f}   "
             f"rel {ed:.2e}  (pass < 1e-2)",
             f"  P/D    grid {best['PD']:.5f}  adam {last['PD']:.5f}   "
             f"rel {ep:.2e}  (pass < 1e-2)",
             f"  range  grid {best['range_km']:.4f}  adam "
             f"{last['range_km']:.4f}  rel {er:.2e}  (pass < 5e-3)",
             f"  binding  grid [{', '.join(bg)}]  adam [{', '.join(bo)}]"]
    return (tight or same_bind), lines


# ── CLI ────────────────────────────────────────────────────────────────────

_DEV = "cuda:0"


def main():
    global _DEV
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-tables", action="store_true")
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--opt", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default=os.path.join(_HERE, "out"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    wp.init()
    _DEV = args.device or ("cuda:0" if wp.get_cuda_device_count() else "cpu")
    brief = json.loads(json.dumps(prop.DEFAULT_BRIEF))    # a private copy

    any_gate = any((args.check_tables, args.fd, args.grid, args.opt,
                    args.selftest))
    if not any_gate:
        args.selftest = True

    print("=" * 92)
    print("sea-drone propeller co-design, pilot (slice A) -- device", _DEV)
    print(brief_header(brief))
    print("=" * 92)

    ok = True
    if args.check_tables or args.selftest:
        print("\n[1/4] --check-tables: the polynomials are the demo's")
        ok &= check_tables(_DEV)
    if args.fd or args.selftest:
        print("\n[2/4] --fd: central finite differences vs the tape")
        good, _, _ = fd_check(brief, _DEV)
        ok &= good
    best = None
    if args.grid or args.selftest:
        print("\n[3/4] --grid: brute force")
        best = grid_search(brief, _DEV)
    last = None
    if args.opt or args.selftest:
        print("\n[4/4] --opt: Adam")
        _, last, _ = optimise(brief, _DEV, steps=args.steps, lr=args.lr,
                              out_dir=args.out)

    if args.selftest:
        agree, lines = verdict(best, last, brief)
        print("\n--- verdict, the brief's own case ------------------------")
        for ln in lines:
            print(ln)
        ok &= agree

        # THE BRIEF'S OPTIMUM IS A CORNER: D and P/D both stop at the box, and
        # a corner is reached by any descent direction whose SIGNS are right,
        # so agreeing there says little about the magnitudes. The magnitudes
        # are what --fd measures -- and this second case measures them again
        # end to end: lift the tip-clearance limit to D_max 1.0 m and the
        # diameter's optimum becomes INTERIOR, a stationary point of the
        # efficiency the polynomials define, which a wrong gradient walks past.
        print("\n--- verdict, D_max 1.0 m (the diameter optimum interior) -")
        brief2 = json.loads(json.dumps(prop.DEFAULT_BRIEF))
        brief2["D_max"] = 1.00
        best2 = grid_search(brief2, _DEV, verbose=False)
        _, last2, _ = optimise(brief2, _DEV, steps=args.steps, lr=args.lr,
                               out_dir=None, verbose=False)
        agree2, lines2 = verdict(best2, last2, brief2)
        print(f"  grid  {design_line(best2, brief2)}")
        print(f"  adam  {design_line(last2, brief2)}")
        for ln in lines2:
            print(ln)
        ok &= agree2

        both = agree and agree2
        print(f"\n{'PASS' if ok else 'FAIL'}: the tape's optimum "
              f"{'reproduces' if both else 'does NOT reproduce'} the "
              f"brute-force optimum"
              f"{'' if ok else ' (or an earlier gate failed)'}")
        sys.exit(0 if ok else 2)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
