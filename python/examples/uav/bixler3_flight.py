"""The HobbyKing Bixler 3 flying a route on its identified flight model: a straight leg north, a
right turn, a leg east, over a pond, with line-of-sight guidance and a wind you set.

The aircraft is ../rigs/bixler3_rig.py, which any scene can import: Bixler3 (the 6-DOF model of
B. M. Simmons, "System Identification of a Nonlinear Flight Dynamics Model for a Small,
Fixed-Wing UAV", Master's thesis, Virginia Tech, 2018, http://hdl.handle.net/10919/95324, with
its numbers from bixler3_spec.json, and a throttle that is ours: the thesis has none), Autopilot
and LOS (ours), and Visual (bixler3.glb posed from the model). This file is the scene, the
cameras, the telemetry and the checks.

    python bixler3_flight.py                              # a window; C cycles the camera
    python bixler3_flight.py --wind 0                     # calm
    python bixler3_flight.py --wind 5 --from 315          # 5 m/s from the north-west
    python bixler3_flight.py --turbulence 1               # the wind with Dryden gusts on it
    python bixler3_flight.py --hold heading               # hold the heading, not the track: it drifts
    python bixler3_flight.py --shot 20 --out bixler3.png  # headless still at t = 20 s
    python bixler3_flight.py --stills 8,28,38 --out-dir out
    python bixler3_flight.py --telemetry                  # no renderer: a line a second and a summary
    python bixler3_flight.py --telemetry --csv b3.csv     # and every step to a file
    python bixler3_flight.py --checks                     # the ten checks, each line numbered
    python bixler3_flight.py --checks 4,10                # some of them

Options: --wind m/s (default 3), --from deg (where it blows from: 270, the default, is west, a
crosswind on the first leg and a tailwind on the second), --turbulence k (the wind becomes an
m350_rig.Wind: --wind is then its speed 10 m up, a logarithmic profile above and below, and
Dryden gusts at k times MIL-HDBK-1797's intensities; --seed picks the gusts), --airspeed 12,
--height 60, --lookahead 40 (m), --hold course|heading, --seconds (default: to the end of the
route), --size 1280x720, --cam chase|high|ground (the stills' camera).

What is drawn: the route at the commanded height (white), the poles at its waypoints, the aim
point the guidance steers at (orange) and the line of sight to it, the flown track (cyan), and
the wind as an arrow on the ground at the start (1 m per m/s). Needs bixler3.glb (build it
once: see build_bixler3_blender.py). The telemetry and the checks need only numpy.
"""
import argparse
import copy
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # python/ (threepp)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rigs"))   # examples/rigs/ (bixler3_rig)

from bixler3_rig import (LOS, Autopilot, Bixler3, Visual, Wind, cross_track, load_spec, model_path,  # noqa: E402
                         ned_to_world, rot_nb)

ROUTE = [(0.0, 0.0), (300.0, 0.0), (300.0, 250.0)]          # NED (north, east), m
START = (-60.0, 0.0)                                         # trimmed here, on the first leg's course
DEG = 180.0 / math.pi
LON, LAT = (3, 5, 10, 7), (4, 9, 11, 6)                      # (u, w, q, theta) and (v, p, r, phi) in the rigid state


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wind", type=float, default=3.0, help="wind speed, m/s")
    ap.add_argument("--from", dest="from_deg", type=float, default=270.0, help="the direction the wind blows from, deg (0 north, 90 east)")
    ap.add_argument("--turbulence", type=float, default=None, help="Dryden gusts on the wind, as a multiple of the standard's intensities")
    ap.add_argument("--seed", type=int, default=3, help="the gusts' seed")
    ap.add_argument("--airspeed", type=float, default=12.0)
    ap.add_argument("--height", type=float, default=60.0)
    ap.add_argument("--lookahead", type=float, default=None)
    ap.add_argument("--hold", choices=("course", "heading"), default="course")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--checks", nargs="?", const="all", default=None, help="all, or a comma-separated list of 1..10")
    ap.add_argument("--shot", type=float, default=None)
    ap.add_argument("--stills", default=None, help="comma-separated times, s")
    ap.add_argument("--out", default="bixler3_flight.png")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--cam", choices=("chase", "high", "ground"), default="chase")
    return ap.parse_args(argv)


def wind_ned(speed, from_deg):
    """The air's velocity (NED) for a wind of `speed` m/s blowing from `from_deg`."""
    a = math.radians(from_deg)
    return (-speed * math.cos(a), -speed * math.sin(a), 0.0)


# --------------------------------------------------------------------------- #
#  The flight
# --------------------------------------------------------------------------- #
class Flight:
    """The Bixler 3, its autopilot and its guidance on ROUTE."""

    def __init__(self, a, spec=None):
        self.a = a
        self.spec = load_spec() if spec is None else spec
        self.reset()

    def reset(self):
        a = self.a
        if a.turbulence is None:
            wind = wind_ned(a.wind, a.from_deg)              # the same wind everywhere
        else:
            wind = Wind(a.wind, from_deg=a.from_deg, turbulence=a.turbulence, seed=a.seed)   # a profile, and gusts
        self.b = Bixler3(self.spec, wind=wind, h0=0.0)
        # trimmed on the first leg's course, crabbed into the wind
        self.b.trim(a.airspeed, n=START[0], e=START[1], h=a.height, course=math.atan2(ROUTE[1][1] - ROUTE[0][1], ROUTE[1][0] - ROUTE[0][0]))
        self.ap = Autopilot(self.b, airspeed=a.airspeed, hold=a.hold)
        self.los = LOS(ROUTE, lookahead=a.lookahead, spec=self.spec)
        self.rows = []

    def step(self):
        b, a = self.b, self.a
        vg = b.ground_velocity()
        chi = self.los.update(b.n, b.e, math.hypot(vg[0], vg[1]))
        if a.hold == "course":
            self.ap.command(course=chi, altitude=a.height, airspeed=a.airspeed)
        else:
            self.ap.command(heading=chi, altitude=a.height, airspeed=a.airspeed)
        self.ap.update()
        o = b.out
        va = b.air_velocity_ned()
        wn = b.wind_ned()
        self.rows.append((b.t, b.n, b.e, b.h, o["Va"], math.hypot(vg[0], vg[1]), math.atan2(vg[1], vg[0]),
                          b.psi, b.phi, b.theta, o["alpha"], o["beta"], math.atan2(va[1], va[0]),
                          self.los.k, self.los.e, cross_track(ROUTE, b.n, b.e), o["throttle"], b.omega,
                          o["T"], o["de"], o["da"], o["dr"], wn[0], wn[1], wn[2]))
        b.step()

    def advance(self, seconds):
        for _ in range(int(round(seconds / self.b.dt))):
            if self.finished():
                return
            self.step()

    def finished(self):
        return self.los.done or (self.a.seconds is not None and self.b.t >= self.a.seconds - 1e-9)


COLS = ("t", "n", "e", "h", "Va", "Vg", "chi", "psi", "phi", "theta", "alpha", "beta", "chi_air", "leg", "e_los",
        "cross_track", "throttle", "omega", "T", "d_e", "d_a", "d_r", "wind_n", "wind_e", "wind_d")
C = {k: i for i, k in enumerate(COLS)}


def leg_figures(f, k):
    """The settled middle of leg k (from 80 m after its start to 60 m before its end): the
    cross-track error, the crab angle against the wind triangle, the ground speed."""
    a = np.array(f.rows)
    leg = a[a[:, C["leg"]] == k]
    (an, ae), (bn, be) = ROUTE[k], ROUTE[k + 1]
    L = math.hypot(bn - an, be - ae)
    s = ((leg[:, C["n"]] - an) * (bn - an) + (leg[:, C["e"]] - ae) * (be - ae)) / L
    mid = leg[(s > 80.0) & (s < L - 60.0)]
    if len(mid) == 0:
        return None
    chi = math.atan2(be - ae, bn - an)
    t_hat = np.array([math.cos(chi), math.sin(chi), 0.0])
    W = np.array([mid[:, C["wind_n"]].mean(), mid[:, C["wind_e"]].mean(), 0.0])     # the mean wind it met there
    w_cross = float(np.cross(t_hat, W)[2])                           # + from the left of the track
    rem = lambda x: np.mean([math.remainder(v, 2.0 * math.pi) for v in x])   # noqa: E731
    return {"rms": float(np.sqrt(np.mean(mid[:, C["e_los"]] ** 2))), "max": float(np.abs(mid[:, C["e_los"]]).max()),
            "mean": float(mid[:, C["e_los"]].mean()), "w_cross": w_cross,
            "triangle": DEG * math.asin(max(-1.0, min(1.0, -w_cross / f.a.airspeed))),
            "crab_air": DEG * rem(mid[:, C["chi_air"]] - mid[:, C["chi"]]),
            "crab_nose": DEG * rem(mid[:, C["psi"]] - mid[:, C["chi"]]), "vg": float(mid[:, C["Vg"]].mean()),
            "bank": (DEG * mid[:, C["phi"]].min(), DEG * mid[:, C["phi"]].max()),
            "bank_sd": DEG * float(mid[:, C["phi"]].std()), "pitch_sd": DEG * float(mid[:, C["theta"]].std()),
            "h": (mid[:, C["h"]].min(), mid[:, C["h"]].max()), "Va": (mid[:, C["Va"]].min(), mid[:, C["Va"]].max())}


def summary(f):
    """The figures the wind is about: per leg the crab angle against the wind triangle, the
    track's error, and what the aircraft did."""
    a = np.array(f.rows)
    gust = "" if f.a.turbulence is None else f", gusts x {f.a.turbulence:g} (the speed is the wind's 10 m up)"
    print(f"[bixler3] {a[-1, C['t']]:.1f} s, wind {f.a.wind:.1f} m/s from {f.a.from_deg:.0f} deg{gust}, {f.a.hold} hold, "
          f"look-ahead {f.los.delta:.0f} m")
    for k, name in ((0, "leg 1 (north)"), (1, "leg 2 (east)")):
        g = leg_figures(f, k)
        if g is None:
            continue
        print(f"   {name}: cross-track rms {g['rms']:.2f} m (max {g['max']:.2f}, mean {g['mean']:+.2f}); crosswind {g['w_cross']:+.2f} m/s; "
              f"the air velocity {g['crab_air']:+.2f} deg off the track (asin(-W_cross / V_a) {g['triangle']:+.2f}), the nose "
              f"{g['crab_nose']:+.2f}; ground speed {g['vg']:.2f} m/s")
    turn = a[(a[:, C["n"]] > ROUTE[1][0] - 80.0) & (a[:, C["e"]] < 80.0)]
    if len(turn):
        print(f"   the turn: past the corner {max(0.0, a[:, C['n']].max() - ROUTE[1][0]):.1f} m, off the route at most "
              f"{turn[:, C['cross_track']].max():.1f} m, bank up to {DEG * np.abs(turn[:, C['phi']]).max():.1f} deg, "
              f"sideslip up to {DEG * np.abs(turn[:, C['beta']]).max():.1f} deg")
    print(f"   height {a[:, C['h']].min():.2f}..{a[:, C['h']].max():.2f} m, airspeed {a[:, C['Va']].min():.2f}..{a[:, C['Va']].max():.2f} m/s, "
          f"throttle {a[:, C['throttle']].min():.2f}..{a[:, C['throttle']].max():.2f} (the thesis's model is at "
          f"{f.b.throttle_ref:.2f}); guards {f.b.guards}")


def telemetry(f):
    t0 = time.time()
    every = int(round(1.0 / f.b.dt))
    while not f.finished():
        f.step()
        if f.b.steps % every == 0:
            r = f.rows[-1]
            print(f"t {r[0]:5.1f}  n {r[1]:7.1f} e {r[2]:7.1f} h {r[3]:6.2f}  Va {r[4]:5.2f} Vg {r[5]:5.2f}  course {DEG * r[6]:+7.2f} "
                  f"heading {DEG * r[7]:+7.2f}  bank {DEG * r[8]:+6.2f}  leg {r[13]:.0f} e {r[14]:+7.2f}  thr {r[16]:.2f}")
    print(f"({time.time() - t0:.1f} s wall)")
    summary(f)
    if f.a.csv:
        np.savetxt(f.a.csv, np.array(f.rows), delimiter=",", header=",".join(COLS), comments="", fmt="%.6g")
        print(f"[bixler3] wrote {f.a.csv}")


# --------------------------------------------------------------------------- #
#  The checks
# --------------------------------------------------------------------------- #
def line(k, ok, text):
    print(f"[{k}] {'    ' if ok is None else 'PASS' if ok else 'FAIL'}  {text}")


def flown(spec, key, cache={}, **kw):
    """The route flown once per set of options (checks 7, 8 and 9 share the flights)."""
    if key not in cache:
        f = Flight(parse(sum(([f"--{k.replace('_', '-')}", str(v)] for k, v in kw.items()), [])), spec)
        while not f.finished():
            f.step()
        cache[key] = f
    return cache[key]


def step_metrics(t, y, y0, y1):
    """10 to 90 % rise time and overshoot (in the units of y)."""
    t, s = np.asarray(t), (np.asarray(y) - y0) / (y1 - y0)
    return t[np.argmax(s >= 0.9)] - t[np.argmax(s >= 0.1)], max(float(np.max(s)) - 1.0, 0.0) * abs(y1 - y0)


# 1 the thesis's model, untouched
def check_thesis_level(spec):
    b = Bixler3(spec)
    s = b.level_speed(apply=True, h=60.0)
    line(1, s["residual"] < 1e-9 and not s["guarded"],
         f"the thesis's model untouched (throttle held at throttle_ref = {b.throttle_ref}, our thrust difference zero) flies level at "
         f"{s['va']:.2f} m/s at 60 m (ISA, {b.out['rho']:.4f} kg/m^3): alpha {s['alpha'] * DEG:+.2f} deg, bank {s['phi'] * DEG:+.2f}, elevator "
         f"{s['de'] * DEG:+.2f}, aileron {s['da'] * DEG:+.3f}, rudder {s['dr'] * DEG:+.3f}; X = qbar S CX = {s['X']:+.3f} N against "
         f"m g sin(theta) = {b.m * b.g * math.sin(s['theta']):+.3f} N. Residual {s['residual']:.1e}.")
    line(1, None, f"that speed rests on CXu uh + CXo, the difference of two estimates each uncertain by as much as itself "
                  f"({spec['aero']['CXu']} +- {spec['aero']['uncertainty']['CXu']}, {spec['aero']['CXo']} +- {spec['aero']['uncertainty']['CXo']}); "
                  f"the bank is CYo's ({spec['aero']['CYo']} +- {spec['aero']['uncertainty']['CYo']}).")
    dT, va = 0.0, []
    for _ in range(int(30.0 / b.dt)):
        b.step()
        dT = max(dT, abs(b.out["dT"]))
        va.append(b.out["Va"])
    line(1, dT == 0.0 and not any(b.guards.values()) and abs(b.h - 60.0) < 0.01,
         f"flown hands-off from there for 30 s: airspeed {min(va):.3f}..{max(va):.3f} m/s, height {b.h:.3f} m, |T - T_ref| at most {dT}, guards {b.guards}.")
    b0 = Bixler3(spec)
    ref = b0.reference_throttle()
    v0 = b0.level_speed()["va"]
    line(1, abs(ref - b.throttle_ref) < 5e-4,
         f"throttle_ref recomputed: the ASSUMED propeller gives m g / (L/D = {spec['propulsion']['lift_to_drag']}) = "
         f"{b.m * b.g / spec['propulsion']['lift_to_drag']:.3f} N at the sea-level level speed {v0:.2f} m/s with the throttle at {ref:.4f} "
         f"(the spec carries {b.throttle_ref}), {ref * b.om_max / (2.0 * math.pi):.0f} rev/s, J {v0 / (ref * b.om_max / (2.0 * math.pi) * b.D):.2f}.")


# 2 the forces at throttle_ref are the thesis's coefficients
def thesis_forces(spec, rho, u, v, w, p, q, r, de, da, dr):
    """Eqs. (4.1)-(4.7) and (2.13) written out a second time, straight from the spec."""
    P, A = spec["physical"], spec["aero"]
    Vo, S, b, c = A["Vo"], P["S"], P["b"], P["cbar"]
    uh, vh, wh, ph, qh, rh = u / Vo, v / Vo, w / Vo, p * b / (2 * Vo), q * c / (2 * Vo), r * b / (2 * Vo)
    CX = A["CXu"] * uh + A["CXw"] * wh + A["CXw2"] * wh ** 2 + A["CXo"]
    CZ = A["CZw"] * wh + A["CZq"] * qh + A["CZde"] * de + A["CZw2"] * wh ** 2 + A["CZo"]
    Cm = A["Cmw"] * wh + A["Cmq"] * qh + A["Cmde"] * de + A["Cmo"]
    CY = A["CYv"] * vh + A["CYp"] * ph + A["CYr"] * rh + A["CYda"] * da + A["CYdr"] * dr + A["CYo"]
    Cl = A["Clv"] * vh + A["Clp"] * ph + A["Clr"] * rh + A["Clda"] * da + A["Cldr"] * dr + A["Clo"]
    Cn = A["Cnv"] * vh + A["Cnp"] * ph + A["Cnr"] * rh + A["Cnda"] * da + A["Cndr"] * dr + A["Cnv2"] * vh ** 2 + A["Cno"]
    qS = 0.5 * rho * (u * u + v * v + w * w) * S
    return np.array([qS * CX, qS * CY, qS * CZ, qS * b * Cl, qS * c * Cm, qS * b * Cn])


def check_forces(spec):
    b = Bixler3(spec)
    rng = np.random.default_rng(1)
    worst = worst_acc = 0.0
    only_x = True
    N = 500
    for _ in range(N):
        u, v, w = rng.uniform(9.0, 16.0), rng.uniform(-2.5, 2.5), rng.uniform(-1.5, 2.8)
        p, q, r = rng.uniform(-1.5, 1.5, 3)
        de, da, dr = rng.uniform(-0.3, 0.3, 3)
        phi, th = rng.uniform(-0.6, 0.6, 2)
        xr = [0.0, 0.0, -60.0, u, v, w, phi, th, 0.3, p, q, r, b.om_ref]
        o, o2 = {}, {}
        f = b.rigid_rates(xr, de, da, dr, b.throttle_ref, (0.0, 0.0, 0.0), o)
        ref = thesis_forces(spec, o["rho"], u, v, w, p, q, r, de, da, dr)
        got = np.array([o["Fx"] + b.m * b.g * math.sin(th), o["Y"], o["Z"], o["L"], o["M"], o["N"]])
        worst = max(worst, float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-9))))
        # (2.7): m (u' + q w - r v) = X - m g sin(theta), with nothing of ours in it
        worst_acc = max(worst_acc, abs(b.m * (f[3] + q * w - r * v) - (ref[0] - b.m * b.g * math.sin(th))))
        xr[12] = b.om_max
        b.rigid_rates(xr, de, da, dr, 1.0, (0.0, 0.0, 0.0), o2)
        d = np.array([o2[k] - o[k] for k in ("Fx", "Fy", "Fz", "L", "M", "N")])
        only_x = only_x and abs(d[0] - (o2["T"] - o2["T_ref"])) < 1e-12 and not np.any(d[1:])
    line(2, worst < 1e-12 and worst_acc < 1e-12 and only_x and not any(o[k] for k in ("g_wh", "g_vh", "g_j")),
         f"{N} random states inside the guards, propeller at its reference speed: the six forces and moments against eqs. (4.1)-(4.7), "
         f"(2.13) written out a second time agree to {worst:.1e} (relative), m (u' + q w - r v) against X - m g sin(theta) to "
         f"{worst_acc:.1e} N. At full throttle the same states differ in the x force alone, by T - T_ref.")


# 3 the rigid body
def check_rigid_body(spec):
    b = Bixler3(spec, rho=0.0)                               # no air: no aerodynamic force, and the propeller law gives none
    b.set_state(d=-500.0, u=12.0, v=1.0, w=-2.0, phi=0.1, theta=0.2, psi=0.5, p=0.3, q=0.2, r=3.0)
    I = np.diag([b.Ix, b.Iy, b.Iz])

    def H():
        x = b.x
        return rot_nb(x[6], x[7], x[8]) @ (I @ np.array(x[9:12]))
    E0, H0, th = b.energy(), H(), 0.0
    for _ in range(int(20.0 / b.dt)):
        b.step()
        th = max(th, abs(b.theta))
    dE, dH = abs(b.energy() - E0) / abs(E0), float(np.linalg.norm(H() - H0) / np.linalg.norm(H0))
    line(3, dE < 1e-7 and dH < 1e-7,
         f"no air (rho = 0: no aerodynamics, no thrust): a body thrown at 12 m/s turning ({b.x[9]:+.2f}, {b.x[10]:+.2f}, {b.x[11]:+.2f}) rad/s "
         f"at the end falls {500.0 - b.h:.0f} m in 20 s and keeps its energy to {dE:.1e} and its angular momentum (in NED) to {dH:.1e} "
         f"(|theta| stays under {th * DEG:.0f} deg; the Euler angles are singular at 90).")


# 4 the thesis's modes
TABLE_4_1 = dict(CXu=-0.190, CXw=0.345, CXw2=0.681, CXo=0.248, CZo=-0.114, CZw=-5.33, CZq=16.7, CZde=0.206, CZw2=4.83,
                 Cmw=-0.286, Cmq=-3.89, Cmde=-0.350, Cmo=0.0146)     # thesis Table 4.1: the flight-data-only longitudinal set


def modes_at_reference(spec, rho):
    """The modes of the implementation linearised numerically (central differences on
    rigid_rates) where the thesis states its own: u = Vo = 12 m/s, everything else zero,
    theta = 0, the propeller at its reference speed. Not an equilibrium, as the thesis's is not."""
    b = Bixler3(spec, rho=rho)
    A, _ = b.jacobian([0.0, 0.0, 0.0, b.Vo, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, b.om_ref], [0.0, 0.0, 0.0, b.throttle_ref, 0.0, 0.0, 0.0])
    el, ea = np.linalg.eigvals(A[np.ix_(LON, LON)]), np.linalg.eigvals(A[np.ix_(LAT, LAT)])
    cl = sorted([e for e in el if e.imag > 1e-9], key=lambda e: -abs(e))
    re = sorted([e.real for e in ea if abs(e.imag) < 1e-9])
    return {"short_period": cl[0], "phugoid": cl[1] if len(cl) > 1 else complex("nan"), "dutch_roll": [e for e in ea if e.imag > 1e-9][0],
            "roll": re[0], "spiral": re[-1]}


def vlm_spec(spec, rho):
    """The thesis's vortex-lattice derivatives (Table 3.7) in place of the identified ones, with
    CZo the lift that carries the weight at 12 m/s and CXo cancelling CXu there: the control."""
    s = copy.deepcopy(spec)
    a, V, P = s["aero"], spec["thesis_vlm"], spec["physical"]
    for k in list(a):
        if isinstance(a[k], float) and k != "Vo":
            a[k] = 0.0
    for k in ("CXu", "CXw", "CZw", "CZq", "Cmw", "Cmq", "CYv", "CYp", "CYr", "Clv", "Clp", "Clr", "Cnv", "Cnp", "Cnr"):
        a[k] = V[k]
    a["CXo"] = -V["CXu"]
    a["CZo"] = -P["mass"] * spec["environment"]["g"] / (0.5 * rho * a["Vo"] ** 2 * P["S"])
    return s


def check_modes(spec):
    cx = lambda z: f"{z.real:+.3f} +- {abs(z.imag):.3f}i"       # noqa: E731
    pct = lambda a, b: 100.0 * abs(a - b) / abs(b)             # noqa: E731
    T, V = spec["thesis_modes"], spec["thesis_vlm"]["modes"]
    line(4, None, "linearised numerically at the thesis's stated point (12 m/s along the nose, theta = 0, all else zero), the 4 x 4 "
                  "blocks (u, w, q, theta) and (v, p, r, phi). The thesis prints no air density, so first the control:")
    worst = {}
    for rho in (1.225, 1.18):
        m = modes_at_reference(vlm_spec(spec, rho), rho)
        worst[rho] = max(pct(m["short_period"], complex(*V["short_period"])), pct(m["dutch_roll"], complex(*V["dutch_roll"])),
                         pct(m["roll"], V["roll"]), pct(m["spiral"], V["spiral"]), pct(m["phugoid"].imag, V["phugoid"][1]))
        line(4, None, f"    its vortex-lattice derivatives (Table 3.7) at rho = {rho}: short period {cx(m['short_period'])}, phugoid "
                      f"{cx(m['phugoid'])}, dutch roll {cx(m['dutch_roll'])}, roll {m['roll']:+.2f}, spiral {m['spiral']:+.4f}: at most "
                      f"{worst[rho]:.1f} % from XFLR5's own modes of them (Table 3.8)")
    line(4, worst[1.18] < 2.0,
         f"Table 3.8 prints {V['short_period'][0]} +- {V['short_period'][1]}i, {V['phugoid'][0]} +- {V['phugoid'][1]}i, {V['dutch_roll'][0]} +- "
         f"{V['dutch_roll'][1]}i, {V['roll']}, {V['spiral']}: the linearisation and the non-dimensional states are the thesis's, and its "
         f"density was near 1.18 kg/m^3 ({worst[1.18]:.1f} % at the most; {worst[1.225]:.1f} % at 1.225).")
    m = modes_at_reference(spec, 1.18)
    m0 = modes_at_reference(spec, 1.225)
    d_dr, d_roll, d_sp = pct(m["dutch_roll"], complex(*T["dutch_roll"])), pct(m["roll"], T["roll"]), pct(m["spiral"], T["spiral"])
    line(4, max(d_dr, d_roll, d_sp) < 3.0,
         f"lateral-directional (Table 4.7's estimates, the model's) at 1.18 against Table 4.8: dutch roll {cx(m['dutch_roll'])} "
         f"({T['dutch_roll'][0]} +- {T['dutch_roll'][1]}i, {d_dr:.1f} % off), roll {m['roll']:+.3f} ({T['roll']}, {d_roll:.1f} %), spiral "
         f"{m['spiral']:+.4f} ({T['spiral']}, {d_sp:.1f} %). At 1.225: {cx(m0['dutch_roll'])}, {m0['roll']:+.3f}, {m0['spiral']:+.4f}.")
    s41 = copy.deepcopy(spec)
    s41["aero"].update(TABLE_4_1)
    m41 = modes_at_reference(s41, 1.18)
    tsp = complex(*T["short_period"])
    d_sp45, d_sp41 = pct(m["short_period"], tsp), pct(m41["short_period"], tsp)
    line(4, min(d_sp45, d_sp41) < 3.0,
         f"short period against Table 4.2 ({T['short_period'][0]} +- {T['short_period'][1]}i): the model's set (Table 4.5) gives "
         f"{cx(m['short_period'])} ({d_sp45:.0f} % off), and the flight-data-only set of Table 4.1, which is the one Table 4.2 belongs to "
         f"(the thesis prints no modes for Table 4.5), gives {cx(m41['short_period'])} ({d_sp41:.0f} % off). NOT REPRODUCED from either: "
         f"the printed real part is twice ours ({tsp.real / m41['short_period'].real:.2f} x Table 4.1's), the imaginary part is not. The "
         f"same linearisation reproduces Tables 3.8 and 4.8, so the difference is not in the method here; its cause is not known.")
    line(4, None, f"the phugoid, which the thesis leaves out: {cx(m['phugoid'])} at that point, barely damped (the model has almost no "
                  f"change of the x force with speed: d(u')/du = {-Autopilot(Bixler3(spec)).a_v[0]:+.3f} /s at the 12 m/s trim).")


# 5 the trim with our throttle
def check_trim(spec):
    b = Bixler3(spec)
    s = b.trim(12.0, h=60.0)
    o = b.out
    line(5, s["residual"] < 1e-9 and 0.0 < s["throttle"] < 1.0 and not s["guarded"],
         f"level at 12 m/s, 60 m up, still air, OUR throttle balancing it: alpha {s['alpha'] * DEG:+.2f} deg (wh {s['wh']:.3f}), bank "
         f"{s['phi'] * DEG:+.2f} with no sideslip (CYo), elevator {s['de'] * DEG:+.2f}, aileron {s['da'] * DEG:+.3f}, rudder {s['dr'] * DEG:+.3f}, "
         f"throttle {s['throttle']:.3f} ({s['omega'] / (2.0 * math.pi):.0f} rev/s, J {o['J']:.2f}), thrust {s['T']:.2f} N, {s['dT']:+.2f} N from "
         f"the reference; lift-to-drag by OUR split {b.m * b.g / s['T']:.1f}. Residual {s['residual']:.1e}.")
    line(5, None, "    V m/s  alpha   bank  elevator  throttle  thrust N  T - T_ref   L/D by our split")
    for V in (9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0):
        s = b.trim(V, h=60.0, apply=False)
        line(5, None, f"    {V:5.1f}  {s['alpha'] * DEG:+5.2f}  {s['phi'] * DEG:+5.2f}  {s['de'] * DEG:+7.2f}  {s['throttle']:8.3f}  {s['T']:8.2f}  "
                      f"{s['dT']:+8.2f}  {b.m * b.g / s['T']:8.1f}")


# 6 the autopilot's steps
def check_steps(spec):
    def fly(cmd, seconds, read):
        b = Bixler3(spec)
        b.trim(12.0, h=60.0)
        ap = Autopilot(b)
        y0 = read(b)
        ap.command(**cmd)
        t, y, rest = [], [], []
        for _ in range(int(seconds / b.dt)):
            ap.update()
            b.step()
            t.append(b.t)
            y.append(read(b))
            rest.append((b.h, b.out["Va"], b.phi * DEG, b.out["beta"] * DEG, b.out["throttle"], b.theta * DEG))
        return b, t, np.array(y), y0, np.array(rest)
    b, t, y, y0, r = fly(dict(altitude=70.0), 25.0, lambda b: b.h)
    tr, ov = step_metrics(t, y, y0, 70.0)
    line(6, tr < 8.0 and ov < 1.0 and abs(y[-1] - 70.0) < 0.1 and not any(b.guards.values()),
         f"+10 m of height from the 12 m/s trim: 10 to 90 % in {tr:.1f} s, {ov:.2f} m over, {y[-1]:.2f} m after 25 s; pitch up to {r[:, 5].max():.1f} deg, "
         f"airspeed {r[:, 1].min():.2f}..{r[:, 1].max():.2f} m/s, throttle up to {r[:, 4].max():.2f}.")
    b, t, y, y0, r = fly(dict(course=math.radians(90.0)), 25.0, lambda b: b.course() * DEG)
    tr, ov = step_metrics(t, y, y0, 90.0)
    line(6, tr < 6.0 and ov < 9.0 and abs(y[-1] - 90.0) < 0.5 and not any(b.guards.values()),
         f"a 90 deg course change: 10 to 90 % in {tr:.1f} s, {ov:.1f} deg over, {y[-1]:.2f} deg after 25 s; bank up to {np.abs(r[:, 2]).max():.1f} deg, "
         f"sideslip {r[:, 3].min():+.1f}..{r[:, 3].max():+.1f} deg (the roll rate yaws it: Cnp), height within {r[:, 0].min() - 60.0:+.2f} / "
         f"{r[:, 0].max() - 60.0:+.2f} m, airspeed {r[:, 1].min():.2f}..{r[:, 1].max():.2f} m/s.")
    b, t, y, y0, r = fly(dict(airspeed=14.0), 20.0, lambda b: b.out["Va"])
    tr, ov = step_metrics(t, y, y0, 14.0)
    line(6, tr < 5.0 and ov < 0.5 and abs(y[-1] - 14.0) < 0.05 and not any(b.guards.values()),
         f"+2 m/s of airspeed: 10 to 90 % in {tr:.1f} s, {ov:.2f} m/s over, {y[-1]:.2f} m/s after 20 s; throttle up to {r[:, 4].max():.2f}, height "
         f"within {r[:, 0].min() - 60.0:+.2f} / {r[:, 0].max() - 60.0:+.2f} m.")


# 7 the route in a crosswind
def check_crosswind(spec):
    f = flown(spec, "steady", wind=4.0)
    g1, g2 = leg_figures(f, 0), leg_figures(f, 1)
    a = np.array(f.rows)
    line(7, g1["rms"] < 1.0 and abs(g1["crab_air"] - g1["triangle"]) < 0.3,
         f"the route (300 m north, 250 m east, 60 m up, 12 m/s) in 4 m/s from the west, course hold. Leg 1, the wind across it: cross-track "
         f"rms {g1['rms']:.2f} m (max {g1['max']:.2f}); the air velocity {g1['crab_air']:+.2f} deg off the track against the wind triangle's "
         f"asin(4 / 12) = {g1['triangle']:+.2f}; the nose {g1['crab_nose']:+.2f}; ground speed {g1['vg']:.2f} m/s (sqrt(12^2 - 4^2) = {math.sqrt(128.0):.2f}).")
    line(7, g2["rms"] < 3.0 and f.los.done,
         f"leg 2, the wind behind it: cross-track rms {g2['rms']:.2f} m, ground speed {g2['vg']:.2f} m/s. The turn: {max(0.0, a[:, C['n']].max() - ROUTE[1][0]):.1f} m "
         f"past the corner, bank up to {DEG * np.abs(a[:, C['phi']]).max():.1f} deg, sideslip up to {DEG * np.abs(a[:, C['beta']]).max():.1f} deg. Height "
         f"{a[:, C['h']].min():.2f}..{a[:, C['h']].max():.2f} m, airspeed {a[:, C['Va']].min():.2f}..{a[:, C['Va']].max():.2f} m/s, {a[-1, C['t']]:.1f} s.")
    fh = flown(spec, "heading", wind=4.0, hold="heading", seconds=20.0)
    ah = np.array(fh.rows)
    line(7, None, f"heading hold in the same wind, for contrast: {abs(ah[-1, C['e_los']]):.1f} m off the first leg after 20 s and still drifting.")


# 8 the same route in gusts
def check_gusts(spec):
    f = flown(spec, "gusts", wind=4.0, turbulence=1.0)
    a = np.array(f.rows)
    g1, g2 = leg_figures(f, 0), leg_figures(f, 1)
    w = f.b.wind
    gust = a[:, [C["wind_n"], C["wind_e"], C["wind_d"]]]
    line(8, f.los.done and g1["rms"] < 5.0 and a[:, C["h"]].min() > 55.0 and np.abs(a[:, C["phi"]]).max() * DEG < 60.0,
         f"the same route in an m350_rig.Wind: 4 m/s from the west 10 m up ({math.hypot(*w.mean(0.0, 0.0, 60.0)[:2]):.1f} m/s at 60 m by its "
         f"profile) with Dryden gusts at the standard's intensities (seed 3; the wind it met: east {gust[:, 1].min():+.1f}..{gust[:, 1].max():+.1f}, "
         f"north {gust[:, 0].min():+.1f}..{gust[:, 0].max():+.1f}, down {gust[:, 2].min():+.1f}..{gust[:, 2].max():+.1f} m/s). Finished in {a[-1, C['t']]:.1f} s.")
    line(8, None, f"leg 1: cross-track rms {g1['rms']:.2f} m (max {g1['max']:.2f}), crab {g1['crab_air']:+.1f} deg (triangle for the mean wind there "
                  f"{g1['triangle']:+.1f}), bank {g1['bank'][0]:+.1f}..{g1['bank'][1]:+.1f} deg (sd {g1['bank_sd']:.1f}), pitch sd {g1['pitch_sd']:.1f} deg, height "
                  f"{g1['h'][0]:.2f}..{g1['h'][1]:.2f} m, airspeed {g1['Va'][0]:.2f}..{g1['Va'][1]:.2f} m/s. Leg 2: rms {g2['rms']:.2f} m (max {g2['max']:.2f}).")
    line(8, None, f"over the whole flight: bank up to {DEG * np.abs(a[:, C['phi']]).max():.1f} deg, sideslip {DEG * a[:, C['beta']].min():+.1f}..{DEG * a[:, C['beta']].max():+.1f} deg, "
                  f"angle of attack {DEG * a[:, C['alpha']].min():+.1f}..{DEG * a[:, C['alpha']].max():+.1f} deg, height {a[:, C['h']].min():.2f}..{a[:, C['h']].max():.2f} m, "
                  f"throttle {a[:, C['throttle']].min():.2f}..{a[:, C['throttle']].max():.2f}, surfaces up to {DEG * np.abs(a[:, C['d_e']]).max():.1f} / "
                  f"{DEG * np.abs(a[:, C['d_a']]).max():.1f} / {DEG * np.abs(a[:, C['d_r']]).max():.1f} deg (elevator / aileron / rudder).")
    f2 = Flight(parse(["--wind", "4", "--turbulence", "1"]), spec)
    for _ in range(600):
        f2.step()
    line(8, f2.rows[:600] == f.rows[:600], "the same seed gives the same flight: the first 600 steps of a second run are identical.")


# 9 the guards
def check_guards(spec):
    fs, fg = flown(spec, "steady", wind=4.0), flown(spec, "gusts", wind=4.0, turbulence=1.0)
    clean = not any(fs.b.guards.values()) and not any(fg.b.guards.values())
    line(9, clean, f"steps on a guard: the route in a steady 4 m/s {fs.b.guards} of {fs.b.steps}; the route in gusts {fg.b.guards} of {fg.b.steps}.")
    b = Bixler3(spec)
    b.trim(12.0, h=60.0)
    b.cmd = [-b.de_lim, 0.0, 0.0, 0.0]                       # the stick held back, the throttle shut
    lo, wh = 99.0, 0.0
    for _ in range(int(6.0 / b.dt)):
        b.step()
        lo, wh = min(lo, b.out["Va"]), max(wh, b.out["wh"])
    line(9, b.guards["wh"] > 0,
         f"and that they count: from the 12 m/s trim with the elevator held at its stop and the throttle shut for 6 s, {b.guards} of {b.steps} "
         f"steps (w / Vo reached {wh:.2f} against the guard's {b.wh_lim[1]}, the airspeed fell to {lo:.1f} m/s). On the wh guard the "
         f"lift holds its edge value: a plateau, not a stall.")


# 10 the slow end
def check_slow_end(spec):
    b = Bixler3(spec)

    def ok(V):
        s = b.trim(V, apply=False)
        return (s["residual"] < 1e-8 and 0.0 <= s["throttle"] <= 1.0 and abs(s["de"]) <= b.de_lim and not s["guarded"]), s

    def edge(good, bad):
        for _ in range(40):
            mid = 0.5 * (good + bad)
            good, bad = (mid, bad) if ok(mid)[0] else (good, mid)
        return good
    lo, hi = edge(12.0, 5.0), edge(12.0, 25.0)
    s, sh = ok(lo)[1], ok(hi)[1]
    vertex = -spec["aero"]["CZw"] / (2.0 * spec["aero"]["CZw2"])
    line(10, 5.0 < lo < 12.0 < hi,
         f"the slowest steady level flight the model has, at the sea: {lo:.2f} m/s, alpha {s['alpha'] * DEG:.1f} deg (w / Vo {s['wh']:.3f}; the guard "
         f"is at {b.wh_lim[1]}, the lift's vertex at {vertex:.3f}), elevator {s['de'] * DEG:+.1f} deg, throttle {s['throttle']:.2f}: it ends where the "
         f"ASSUMED propeller has no more to give, not at a stall. The model has none: the lift is the thesis's quadratic in w / 12 m/s "
         f"whatever the airspeed, so {s['alpha'] * DEG:.0f} deg of angle of attack at {lo:.0f} m/s is its arithmetic, not an aircraft's.")
    line(10, None, f"the fastest: {hi:.2f} m/s at full throttle, bank {sh['phi'] * DEG:+.1f} deg to fly straight (CYo's side force grows with "
                   f"qbar). Neither end is the thesis's: both rest on our thrust. Believe the model from about 10 to 15 m/s.")


CHECKS = {1: check_thesis_level, 2: check_forces, 3: check_rigid_body, 4: check_modes, 5: check_trim, 6: check_steps,
          7: check_crosswind, 8: check_gusts, 9: check_guards, 10: check_slow_end}


# --------------------------------------------------------------------------- #
#  The scene
# --------------------------------------------------------------------------- #
def material(tp, color, roughness=0.9, metalness=0.0):
    m = tp.MeshStandardMaterial()
    m.color, m.roughness, m.metalness = color, roughness, metalness
    return m


def polyline(tp, pts, color, capacity=None):
    pts = np.asarray(pts, np.float32)
    n = capacity or len(pts)
    buf = np.zeros((n, 3), np.float32)
    buf[:len(pts)] = pts
    g = tp.BufferGeometry()
    g.set_attribute("position", buf)
    g.set_draw_range(0, len(pts))
    m = tp.LineBasicMaterial()
    m.color = color
    ln = tp.Line(g, m)
    ln.frustum_culled = False
    return ln, g


POND = (250.0, 90.0, 110.0)                                  # north, east, radius


class Scene:
    def __init__(self, tp, f, w, h, headless):
        self.tp, self.f = tp, f
        a = f.a
        self.canvas = tp.Canvas("threepp - Bixler 3", width=w, height=h, antialiasing=4, headless=headless, vsync=not headless)
        r = self.r = tp.GLRenderer(self.canvas)
        r.tone_mapping = tp.ToneMapping.ACESFilmic
        r.tone_mapping_exposure = 1.0
        s = self.scene = tp.Scene()
        s.background = 0x9eb8d6
        s.set_fog(tp.Color(0x9eb8d6), 200.0, 1600.0)
        s.add(tp.HemisphereLight(0xdce6f2, 0x4a5a38, 1.1))
        sun = tp.DirectionalLight(0xfff4e0, 2.2)
        sun.position.set(-300.0, 500.0, 200.0)
        s.add(sun)
        ground = tp.Mesh(tp.PlaneGeometry(4000.0, 4000.0), material(tp, 0x5b7240, 0.95))
        ground.rotation.x = -math.pi / 2
        s.add(ground)
        pond = tp.Mesh(tp.CircleGeometry(POND[2], 96), material(tp, 0x3d5866, 0.25, 0.1))
        pond.rotation.x = -math.pi / 2
        pond.position.set(*ned_to_world(POND[0], POND[1], -0.05))
        s.add(pond)
        rng = np.random.default_rng(7)
        tree = material(tp, 0x2f4526, 0.9)
        cone = tp.ConeGeometry(2.6, 10.0, 8, 1)
        for _ in range(420):
            n, e = rng.uniform(-300.0, 700.0), rng.uniform(-400.0, 650.0)
            if math.hypot(n - POND[0], e - POND[1]) < POND[2] + 15.0:
                continue
            k = rng.uniform(0.7, 1.4)
            t = tp.Mesh(cone, tree)
            t.scale.set(k, k, k)
            t.position.set(*ned_to_world(n, e, -5.0 * k))
            s.add(t)
        # the route at the commanded height, poles at its waypoints
        hgt = a.height
        pts = [ned_to_world(START[0], START[1], -hgt)] + [ned_to_world(n, e, -hgt) for n, e in ROUTE]
        ln, _ = polyline(tp, pts, 0xffffff)
        s.add(ln)
        pole = material(tp, 0xe8e8e8, 0.6)
        knob = material(tp, 0xffffff, 0.4)
        for n, e in ROUTE:
            p = tp.Mesh(tp.CylinderGeometry(0.2, 0.2, hgt, 8, 1), pole)
            p.position.set(*ned_to_world(n, e, -0.5 * hgt))
            s.add(p)
            k = tp.Mesh(tp.SphereGeometry(0.7, 16, 10), knob)
            k.position.set(*ned_to_world(n, e, -hgt))
            s.add(k)
        # the aim point, the line of sight to it, the flown track
        aim_mat = material(tp, 0xff7a1a, 0.5)
        aim_mat.emissive = tp.Color(1.0, 0.45, 0.1)
        aim_mat.emissive_intensity = 0.6
        self.aim = tp.Mesh(tp.SphereGeometry(0.8, 16, 10), aim_mat)
        s.add(self.aim)
        self.sight, self.sight_g = polyline(tp, [(0, 0, 0), (0, 0, 0)], 0xff7a1a)
        s.add(self.sight)
        self.track_n = 0
        self.track, self.track_g = polyline(tp, [(0, 0, 0)], 0x2fd4ff, capacity=20000)
        self.track_buf = np.zeros((20000, 3), np.float32)
        s.add(self.track)
        # the wind
        if a.wind > 0.05:
            wv = np.asarray(wind_ned(a.wind, a.from_deg))
            d = np.array(ned_to_world(wv[0], wv[1], 0.0))
            d /= np.linalg.norm(d)
            arrow = tp.ArrowHelper(tp.Vector3(*map(float, d)), tp.Vector3(*ned_to_world(START[0] + 30.0, -18.0, -1.0)),
                                   a.wind * 1.0 + 2.0, tp.Color(0.95, 0.85, 0.2), 1.6, 1.2)
            s.add(arrow)
        # the aircraft
        mp = model_path()
        if mp is None:
            sys.exit("bixler3.glb is not built: blender --background --factory-startup --python build_bixler3_blender.py")
        self.model = tp.GLTFLoader().load(mp).scene
        s.add(self.model)
        self.vis = Visual(self.model)
        self.cam = tp.PerspectiveCamera(42.0, w / h, 0.2, 8000.0)
        self.mode = a.cam
        self.eye = None

    def sync(self):
        """The aircraft, the aim point, the line of sight and the track from the flight's state."""
        f, b = self.f, self.f.b
        self.vis.pose(b, h0=0.0)
        p = ned_to_world(b.n, b.e, b.d)
        an, ae = f.los.aim
        q = ned_to_world(an, ae, -f.a.height)
        self.aim.position.set(*q)
        self.sight_g.update_attribute("position", np.array([p, q], np.float32))
        if self.track_n < len(self.track_buf):
            if self.track_n == 0 or np.linalg.norm(self.track_buf[self.track_n - 1] - p) > 0.5:
                self.track_buf[self.track_n] = p
                self.track_n += 1
                self.track_g.update_attribute("position", self.track_buf)
                self.track_g.set_draw_range(0, self.track_n)

    def camera(self, smooth=None):
        tp, b = self.tp, self.f.b
        p = np.array(ned_to_world(b.n, b.e, b.d))
        vg = b.ground_velocity()
        fwd = np.array(ned_to_world(vg[0], vg[1], 0.0))
        fwd /= max(np.linalg.norm(fwd), 1e-6)
        if self.mode == "chase":
            eye, at = p - 9.0 * fwd + np.array([0.0, 2.3, 0.0]), p + 4.0 * fwd
        elif self.mode == "high":
            eye, at = p - 50.0 * fwd + np.array([0.0, 32.0, 0.0]) + 18.0 * np.array([-fwd[2], 0.0, fwd[0]]), p + 28.0 * fwd
        else:
            eye, at = np.array(ned_to_world(270.0, -50.0, -2.0)), p
        # the lens: 42 deg, or narrower so that the aircraft spans about a sixth of the frame
        dist = float(np.linalg.norm(at - eye)) if self.mode == "ground" else 0.0
        fov = min(42.0, math.degrees(2.0 * math.atan(4.4 / max(dist, 1.0)))) if self.mode == "ground" else 42.0
        if abs(self.cam.fov - fov) > 1e-6:
            self.cam.fov = fov
            self.cam.update_projection_matrix()
        if smooth is not None and self.eye is not None:
            k = 1.0 - math.exp(-smooth)
            eye = self.eye + (eye - self.eye) * k
        self.eye = eye
        self.cam.position.set(*map(float, eye))
        self.cam.look_at(tp.Vector3(*map(float, at)))

    def render(self):
        self.r.render(self.scene, self.cam)


def stills(tp, f, times, paths, w, h):
    sc = Scene(tp, f, w, h, headless=True)
    frame = 1.0 / 60.0
    for t, path in sorted(zip(times, paths)):
        while f.b.t < t - 1e-9 and not f.finished():
            f.advance(min(frame, t - f.b.t))
            sc.sync()
        sc.camera()
        sc.render()
        sc.r.save_frame(path)
        print(f"[bixler3] t {f.b.t:.2f} s -> {path}")
    summary(f)


def window(tp, f, w, h):
    sc = Scene(tp, f, w, h, headless=False)
    clock = tp.Clock()
    state = {"debt": 0.0, "c": False}
    modes = ("chase", "high", "ground")

    def animate():
        dt = min(clock.get_delta(), 0.1)
        state["debt"] += dt
        n = int(state["debt"] / f.b.dt)
        state["debt"] -= n * f.b.dt
        for _ in range(n):
            if f.finished():
                f.reset()
                sc.track_n = 0
            f.step()
        c = sc.canvas.is_key_down("C")
        if c and not state["c"]:
            sc.mode = modes[(modes.index(sc.mode) + 1) % len(modes)]
            sc.eye = None
        state["c"] = c
        sc.sync()
        sc.camera(smooth=dt / 0.25)
        sc.render()
    sc.canvas.animate(animate)


def main(argv=None):
    a = parse(argv)
    if a.checks:
        spec = load_spec()
        for k in (sorted(CHECKS) if a.checks == "all" else [int(k) for k in a.checks.split(",")]):
            CHECKS[k](spec)
        return
    f = Flight(a)
    if a.telemetry:
        telemetry(f)
        return
    import threepp as tp
    w, h = (int(v) for v in a.size.lower().split("x"))
    if a.shot is not None or a.stills:
        times = [a.shot] if a.shot is not None else [float(v) for v in a.stills.split(",")]
        if a.shot is not None:
            paths = [a.out]
        else:
            os.makedirs(a.out_dir, exist_ok=True)
            paths = [os.path.join(a.out_dir, f"bixler3_{a.cam}_{t:05.1f}s.png") for t in times]
        stills(tp, f, times, paths, w, h)
        return
    window(tp, f, w, h)


if __name__ == "__main__":
    main()
