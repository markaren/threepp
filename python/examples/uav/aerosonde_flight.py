"""The Aerosonde flying a route on the textbook's flight model: a kilometre north, a right turn,
a kilometre east, over a lake, with line-of-sight guidance and a wind you set; and the checks
of that model.

The aircraft is ../rigs/aerosonde_rig.py, which any scene can import: Aerosonde (the 6-DOF model
of Beard and McLain, Small Unmanned Aircraft: Theory and Practice, 2012, with the numbers its
authors give for the Aerosonde, in aerosonde_spec.json), Autopilot and LOS (ours: successive loop
closure on the three surfaces and the throttle, and line-of-sight guidance), and Visual
(aerosonde.glb posed from the model). This file is the scene, the cameras, the telemetry and
the checks.

    python aerosonde_flight.py                              # a window; C cycles the camera
    python aerosonde_flight.py --wind 0                     # calm
    python aerosonde_flight.py --wind 8 --from 315          # 8 m/s from the north-west
    python aerosonde_flight.py --turbulence 1               # gusts on the wind (m350_rig.Wind)
    python aerosonde_flight.py --hold heading               # hold the heading, not the track: it drifts
    python aerosonde_flight.py --shot 30 --out aerosonde.png       # headless still at t = 30 s
    python aerosonde_flight.py --stills 10,45,60 --out-dir out
    python aerosonde_flight.py --telemetry                  # no renderer: a line a second and a summary
    python aerosonde_flight.py --telemetry --csv aerosonde.csv     # and every step to a file
    python aerosonde_flight.py --checks                     # the ten checks, each line numbered
    python aerosonde_flight.py --checks 1,3                 # some of them

Options: --wind m/s (default 8), --from deg (where it blows from: 270, the default, is west, a
crosswind on the first leg and a tailwind on the second), --turbulence (0, the default, is a
uniform steady wind; above 0 the wind is m350_rig.Wind: --wind is then the mean 10 m above the
ground on a logarithmic profile, half as much again at the route's height, with Dryden gusts
scaled by this number), --seed (the gusts'), --airspeed 25, --height 100, --lookahead (m, the
guidance's look-ahead), --hold course|heading, --seconds (default: to the end of the route),
--size 1280x720, --cam chase|high|ground (the stills' camera).

What is drawn: the route at the commanded height (white), the poles at its waypoints, the
aim point the guidance steers at (orange) and the line of sight to it, the flown track (cyan),
and the wind as an arrow on the ground at the start, as long as the wind is strong (1 m per m/s).
Needs aerosonde.glb (build it once: see build_aerosonde_blender.py). The telemetry and the checks
need only numpy.

Read the numbers for what they are: the textbook's model with the parameters its authors give,
servos and a throttle lag that are ASSUMED, an autopilot that is OURS with exact state feedback.
Nothing is validated against a real Aerosonde.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rigs"))   # examples/rigs/ (aerosonde_rig)

from aerosonde_rig import (LOS, Aerosonde, Autopilot, Visual, Wind, cross_track, load_spec, model_path,  # noqa: E402
                           modes, ned_to_world)

ROUTE = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0)]       # NED (north, east), m
START = (-200.0, 0.0)                                        # trimmed here, on the first leg's course
DEG = 180.0 / math.pi


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", nargs="?", const="all", default=None, help="all, or a comma-separated list of 1..10")
    ap.add_argument("--wind", type=float, default=8.0, help="wind speed, m/s")
    ap.add_argument("--from", dest="from_deg", type=float, default=270.0, help="the direction the wind blows from, deg (0 north, 90 east)")
    ap.add_argument("--turbulence", type=float, default=0.0, help="above 0: m350_rig.Wind, its Dryden sigmas scaled by this")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--airspeed", type=float, default=25.0)
    ap.add_argument("--height", type=float, default=100.0)
    ap.add_argument("--lookahead", type=float, default=None)
    ap.add_argument("--hold", choices=("course", "heading"), default="course")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--shot", type=float, default=None)
    ap.add_argument("--stills", default=None, help="comma-separated times, s")
    ap.add_argument("--out", default="aerosonde_flight.png")
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
    """The Aerosonde, its autopilot and its guidance on ROUTE."""

    def __init__(self, a, spec=None):
        self.a = a
        self.spec = spec or load_spec()
        self.reset()

    def reset(self):
        a = self.a
        if a.turbulence > 0.0 and a.wind > 0.0:
            wind = Wind(a.wind, a.from_deg, turbulence=a.turbulence, seed=a.seed)     # the same gusts every time round
            self.wind = wind.mean(0.0, 0.0, a.height)                                 # the mean at the route's height
        else:
            wind = self.wind = wind_ned(a.wind, a.from_deg)
        self.ac = Aerosonde(self.spec, wind=wind, h0=0.0)
        # trimmed on the first leg's course, crabbed into the wind
        self.ac.trim(a.airspeed, n=START[0], e=START[1], h=a.height, course=math.atan2(ROUTE[1][1] - ROUTE[0][1], ROUTE[1][0] - ROUTE[0][0]))
        self.ap = Autopilot(self.ac, airspeed=a.airspeed, hold=a.hold)
        self.los = LOS(ROUTE, lookahead=a.lookahead, spec=self.spec)
        self.rows = []

    def step(self):
        ac, a = self.ac, self.a
        vg = ac.ground_velocity()
        chi = self.los.update(ac.n, ac.e, math.hypot(vg[0], vg[1]))
        if a.hold == "course":
            self.ap.command(course=chi, altitude=a.height, airspeed=a.airspeed)
        else:
            self.ap.command(heading=chi, altitude=a.height, airspeed=a.airspeed)
        self.ap.update()
        o = ac.out
        va = ac.air_velocity_ned()
        self.rows.append((ac.t, ac.n, ac.e, ac.h, o["Va"], math.hypot(vg[0], vg[1]), math.atan2(vg[1], vg[0]),
                          ac.psi, ac.phi, ac.theta, o["alpha"], o["beta"], math.atan2(va[1], va[0]),
                          self.los.k, self.los.e, cross_track(ROUTE, ac.n, ac.e), o["throttle"], o["omega"],
                          o["T"], o["Im"], o["throttle"] * o["Im"], o["de"], o["da"], o["dr"]))
        ac.step()

    def advance(self, seconds):
        for _ in range(int(round(seconds / self.ac.dt))):
            if self.finished():
                return
            self.step()

    def finished(self):
        return self.los.done or (self.a.seconds is not None and self.ac.t >= self.a.seconds - 1e-9)


COLS = ("t", "n", "e", "h", "Va", "Vg", "chi", "psi", "phi", "theta", "alpha", "beta", "chi_air", "leg", "e_los",
        "cross_track", "throttle", "omega", "T", "I_motor", "I_battery", "d_e", "d_a", "d_r")
C = {k: i for i, k in enumerate(COLS)}
LEGS = ((0, "leg 1 (north)"), (1, "leg 2 (east)"))


def leg_stats(f):
    """Per leg, over its settled part (from 500 m after its start to 200 m before a corner, or
    to its end): the cross-track error, the crosswind, the air velocity's and the nose's angle
    off the track against the wind triangle, the ground speed."""
    a = np.array(f.rows)
    W = np.asarray(f.wind)
    out = []
    for k, name in LEGS:
        leg = a[a[:, C["leg"]] == k]
        if len(leg) == 0:
            continue
        (an, ae), (bn, be) = ROUTE[k], ROUTE[k + 1]
        L = math.hypot(bn - an, be - ae)
        s = ((leg[:, C["n"]] - an) * (bn - an) + (leg[:, C["e"]] - ae) * (be - ae)) / L
        mid = leg[(s > 500.0) & (s < (L - 200.0 if k + 2 < len(ROUTE) else L))]
        if len(mid) == 0:
            continue
        chi = math.atan2(be - ae, bn - an)
        t_hat = np.array([math.cos(chi), math.sin(chi), 0.0])
        w_cross = float(np.cross(t_hat, W)[2])                      # + from the left of the track
        out.append({"name": name, "rms": float(np.sqrt(np.mean(mid[:, C["e_los"]] ** 2))),
                    "max": float(np.abs(mid[:, C["e_los"]]).max()), "mean": float(mid[:, C["e_los"]].mean()),
                    "w_cross": w_cross, "triangle": DEG * math.asin(max(-1.0, min(1.0, -w_cross / f.a.airspeed))),
                    "crab_air": DEG * float(np.mean([math.remainder(x, 2 * math.pi) for x in mid[:, C["chi_air"]] - mid[:, C["chi"]]])),
                    "crab_nose": DEG * float(np.mean([math.remainder(x, 2 * math.pi) for x in mid[:, C["psi"]] - mid[:, C["chi"]]])),
                    "vg": float(mid[:, C["Vg"]].mean())})
    return out


def turn_stats(f):
    """The corner: how far past it, how far off the route, the steepest bank."""
    a = np.array(f.rows)
    turn = a[(a[:, C["n"]] > ROUTE[1][0] - 300.0) & (a[:, C["e"]] < 350.0)]
    return (max(0.0, float(a[:, C["n"]].max()) - ROUTE[1][0]), float(turn[:, C["cross_track"]].max()) if len(turn) else 0.0,
            DEG * float(np.abs(a[:, C["phi"]]).max()))


def summary(f):
    """The figures the wind is about: per leg the crab angle against the wind triangle, the
    track's error, and what the aircraft did."""
    a = np.array(f.rows)
    gusts = f" at 10 m ({math.hypot(f.wind[0], f.wind[1]):.1f} at the route's height), turbulence x{f.a.turbulence}" if f.a.turbulence > 0.0 else ""
    print(f"[aerosonde] {a[-1, C['t']]:.1f} s, wind {f.a.wind:.1f} m/s from {f.a.from_deg:.0f} deg{gusts}, {f.a.hold} hold, "
          f"look-ahead {f.los.delta:.0f} m")
    for s in leg_stats(f):
        print(f"   {s['name']}: cross-track rms {s['rms']:.2f} m (max {s['max']:.2f}, mean {s['mean']:+.2f}); crosswind {s['w_cross']:+.2f} m/s; "
              f"the air velocity {s['crab_air']:+.2f} deg off the track (asin(-W_cross / V_a) {s['triangle']:+.2f}), the nose "
              f"{s['crab_nose']:+.2f}; ground speed {s['vg']:.2f} m/s")
    past, off, bank = turn_stats(f)
    print(f"   the turn: past the corner {past:.1f} m, off the route at most {off:.1f} m, bank up to {bank:.1f} deg")
    print(f"   height {a[:, C['h']].min():.2f}..{a[:, C['h']].max():.2f} m, airspeed {a[:, C['Va']].min():.2f}..{a[:, C['Va']].max():.2f} m/s, "
          f"throttle {a[:, C['throttle']].min():.2f}..{a[:, C['throttle']].max():.2f}, motor {a[:, C['I_motor']].mean():.1f} A "
          f"(battery {a[:, C['I_battery']].mean():.1f} A) on average; guards {f.ac.guards}")


def telemetry(f):
    t0 = time.time()
    every = int(round(1.0 / f.ac.dt))
    while not f.finished():
        f.step()
        if f.ac.steps % every == 0:
            r = f.rows[-1]
            print(f"t {r[0]:5.1f}  n {r[1]:7.1f} e {r[2]:7.1f} h {r[3]:6.2f}  Va {r[4]:5.2f} Vg {r[5]:5.2f}  course {DEG * r[6]:+7.2f} "
                  f"heading {DEG * r[7]:+7.2f}  bank {DEG * r[8]:+6.2f}  leg {r[13]:.0f} e {r[14]:+7.2f}  thr {r[16]:.2f}")
    print(f"({time.time() - t0:.1f} s wall)")
    summary(f)
    if f.a.csv:
        np.savetxt(f.a.csv, np.array(f.rows), delimiter=",", header=",".join(COLS), comments="", fmt="%.6g")
        print(f"[aerosonde] wrote {f.a.csv}")


# --------------------------------------------------------------------------- #
#  The checks
# --------------------------------------------------------------------------- #
def line(k, ok, text):
    print(f"[{k}] {'PASS' if ok else 'FAIL'}  {text}")


def note(k, text):
    print(f"[{k}]       {text}")


def fly(a, ap, seconds, record=None):
    for _ in range(int(round(seconds / a.dt))):
        ap.update()
        a.step()
        if record is not None:
            record(a)


def step_metrics(t, y, y0, y1):
    """10-90 % rise time and overshoot (% of the step)."""
    t, y = np.asarray(t), (np.asarray(y) - y0) / (y1 - y0)
    t10 = t[np.argmax(y >= 0.1)]
    t90 = t[np.argmax(y >= 0.9)] if np.any(y >= 0.9) else float("nan")
    return t90 - t10, 100.0 * max(float(np.max(y)) - 1.0, 0.0)


def check_trim(spec):
    """1: straight and level at the design airspeed."""
    a = Aerosonde(spec)
    s = a.trim(25.0, h=100.0)
    f = a.rates(a.x, a.cmd)
    o = a.out
    acc = max(abs(v) for v in f[3:6] + f[9:12])
    line(1, s["residual"] < 1e-9 and acc < 1e-9 and not s["limited"],
         f"level at 25 m/s in still air, rho {o['rho']} kg/m^3 (Aerosonde.trim): residual {s['residual']:.1e}, accelerations of the placed "
         f"aircraft {acc:.1e}; alpha {s['alpha'] * DEG:.2f} deg, elevator {s['de'] * DEG:.2f} deg, throttle {s['throttle']:.3f}, propeller "
         f"{s['omega']:.1f} rad/s ({s['omega'] * 30.0 / math.pi:.0f} rev/min, J {s['J']:.3f}).")
    note(1, f"thrust {s['T']:.2f} N against {s['drag']:.2f} N of drag, lift {s['lift']:.1f} N against {a.m * a.g:.1f} N of weight (L/D {s['lift'] / s['drag']:.1f}); "
            f"the propeller's torque {s['Q']:.3f} N m is held by {s['da'] * DEG:+.2f} deg of aileron, {s['dr'] * DEG:+.2f} of rudder and "
            f"{s['phi'] * DEG:+.3f} deg of bank; motor {o['V_in']:.1f} V, {s['Im']:.1f} A, {s['P_el']:.0f} W.")
    note(1, "    V m/s  alpha  elevator  throttle  propeller rad/s    J    thrust N   L/D   motor W")
    for V in (18.0, 20.0, 25.0, 30.0):
        s = a.trim(V, h=100.0, apply=False)
        note(1, f"    {V:5.1f}  {s['alpha'] * DEG:5.2f}  {s['de'] * DEG:7.2f}  {s['throttle']:8.3f}  {s['omega']:13.1f}  {s['J']:5.3f}  {s['T']:7.2f}  "
                f"{s['lift'] / s['drag']:5.1f}  {s['P_el']:7.0f}")


def check_rigid_body(spec):
    """2: no air, so no aerodynamics and no thrust: a thrown, tumbling body."""
    a = Aerosonde(spec, rho=0.0)
    a.set_state(d=-100.0, u=15.0, v=2.0, w=-5.0, theta=0.1, p=1.5, q=0.2, r=0.15)
    e0, h0, vd0 = a.energy(), a.angular_momentum_ned(), float(a.ground_velocity()[2])
    de = dh = th = 0.0
    for _ in range(int(round(10.0 / a.dt))):
        a.step()
        de = max(de, abs(a.energy() - e0) / e0)
        dh = max(dh, float(np.linalg.norm(a.angular_momentum_ned() - h0)) / float(np.linalg.norm(h0)))
        th = max(th, abs(a.x[7]))
    drop, want = a.x[2] + 100.0, vd0 * a.t + 0.5 * a.g * a.t * a.t
    line(2, de < 1e-8 and dh < 1e-8 and abs(drop - want) < 1e-6,
         f"thrown at 16 m/s with 1.5 rad/s of tumble where there is no air (rho = 0: no aerodynamic force, no thrust), 10 s: the energy "
         f"(kinetic, rotational and potential) is kept to {de:.0e} and the angular momentum about the centre of mass to {dh:.0e}; it is "
         f"{drop:.4f} m lower, and v t + g t^2 / 2 is {want:.4f}; |theta| stayed under {th * DEG:.0f} deg (the Euler angles are singular at 90).")


def check_modes(spec):
    """3: the linearised model at the trim."""
    a = Aerosonde(spec)
    a.trim(25.0, h=100.0)
    A, _ = a.jacobian()
    m = modes(A)
    sp, ph, dr = m["short_period"], m["phugoid"], m["dutch_roll"]

    def wz(e):
        return f"{abs(e):.2f} rad/s, damping {-e.real / abs(e):.2f} ({e.real:+.3f} +- {abs(e.imag):.3f}i)"
    ok = (dr is not None and all(abs(e.imag) > 1e-9 and e.real < 0.0 for e in (sp, ph, dr)) and m["roll"] < 0.0)
    line(3, ok, f"modes at the 25 m/s trim (Aerosonde.jacobian, central differences; the 4 x 4 blocks u, w, q, theta and v, p, r, phi): "
                f"short period {wz(sp)}; phugoid {wz(ph)}; roll {m['roll']:+.2f} /s ({-1.0 / m['roll']:.3f} s); dutch roll {wz(dr)}; "
                f"spiral {m['spiral']:+.4f} /s.")
    if m["spiral"] > 0.0:
        note(3, f"the spiral mode is UNSTABLE: it doubles in {math.log(2.0) / m['spiral']:.1f} s. That is the parameter set's "
                f"(Clbeta Cnr - Cnbeta Clr = {a.a['Clbeta'] * a.a['Cnr'] - a.a['Cnbeta'] * a.a['Clr']:+.4f} < 0), not the implementation's; "
                f"the autopilot's roll loop holds it.")
    full = np.linalg.eigvals(A[np.ix_((3, 4, 5, 6, 7, 9, 10, 11), (3, 4, 5, 6, 7, 9, 10, 11))])
    note(3, "the coupled 8-state model has the same roots to " +
            f"{max(min(abs(e - g) for g in full) for e in (sp, ph, dr, m['roll'], m['spiral'])):.0e} (the propeller's torque and the "
            f"trim's 0.03 deg of bank are all that couples the two blocks).")


def check_open_loop(spec):
    """4: the commands held at their trim, nothing closed."""
    a = Aerosonde(spec)
    s = a.trim(25.0, h=100.0)
    chi0 = a.course()
    a.run(60.0)
    o = a.evaluate()
    dv, dh, dpsi, dchi, dphi = (abs(o["Va"] - 25.0), abs(a.h - 100.0), abs(a.psi) * DEG, abs(a.course() - chi0) * DEG,
                                abs(a.x[6] - s["phi"]) * DEG)
    line(4, dv < 1e-6 and dh < 1e-4 and dpsi < 1e-4 and dchi < 1e-4 and dphi < 1e-4,
         f"open loop from the 25 m/s trim, 60 s and {a.n:.0f} m with the commands held: airspeed off by {dv:.1e} m/s, height by {dh:.1e} m, "
         f"heading by {dpsi:.1e} deg, course by {dchi:.1e} deg, bank by {dphi:.1e} deg.")
    b = Aerosonde(spec)
    s = b.trim(25.0, h=100.0)
    x = list(b.x)
    b.set_state(*x[:3], x[3], x[4], x[5], x[6] + math.radians(0.5), x[7], x[8], de=s["de"], da=s["da"], dr=s["dr"], throttle=s["throttle"])
    b.run(30.0)
    note(4, f"it is not a stable equilibrium: started 0.5 deg of bank off the trim, it is banked {(b.x[6] - s['phi']) * DEG:+.1f} deg and has "
            f"turned {b.x[8] * DEG:+.1f} deg after 30 s, {100.0 - b.h:.1f} m lower (the spiral of check 3).")


def check_steps(spec):
    """5: the autopilot's loops, one step each."""
    a = Aerosonde(spec)
    a.trim(25.0, h=100.0)
    ap = Autopilot(a, airspeed=25.0)
    ap.command(course=0.0, altitude=100.0)
    fly(a, ap, 5.0)
    rec = {"t": [], "h": [], "chi": [], "V": [], "phi": [], "beta": [], "thr": []}

    def record(a):
        rec["t"].append(a.t)
        rec["h"].append(a.h)
        rec["chi"].append(a.course())
        rec["V"].append(a.out["Va"])
        rec["phi"].append(abs(a.x[6]))
        rec["beta"].append(abs(a.out["beta"]))
        rec["thr"].append(a.out["throttle"])
    ap.command(altitude=110.0)
    fly(a, ap, 30.0, record)
    tr, ov = step_metrics(rec["t"], rec["h"], 100.0, 110.0)
    line(5, ov < 15.0 and abs(rec["h"][-1] - 110.0) < 0.1,
         f"from the 25 m/s trim in still air: +10 m of height in {tr:.1f} s (10 to 90 %), {ov * 0.1:.2f} m over, the airspeed within "
         f"{min(rec['V']):.2f}..{max(rec['V']):.2f} m/s, throttle up to {max(rec['thr']):.2f}.")
    for k in rec:
        rec[k].clear()
    ap.command(course=math.radians(90.0))
    fly(a, ap, 30.0, record)
    tr, ov = step_metrics(rec["t"], np.unwrap(rec["chi"]), 0.0, math.radians(90.0))
    line(5, ov < 10.0 and abs(rec["chi"][-1] - math.radians(90.0)) * DEG < 0.2,
         f"a 90 deg course change in {tr:.1f} s, {ov * 0.9:.1f} deg over, the height within {max(abs(h - 110.0) for h in rec['h']):.2f} m, bank up to "
         f"{max(rec['phi']) * DEG:.0f} deg, sideslip up to {max(rec['beta']) * DEG:.1f} deg (the rudder is a yaw damper, not a sideslip hold).")
    for k in rec:
        rec[k].clear()
    ap.command(airspeed=28.0)
    fly(a, ap, 30.0, record)
    tr, ov = step_metrics(rec["t"], rec["V"], 25.0, 28.0)
    line(5, ov < 20.0 and abs(rec["V"][-1] - 28.0) < 0.05,
         f"+3 m/s of airspeed in {tr:.1f} s, {ov * 0.03:.2f} m/s over, the height within {max(abs(h - 110.0) for h in rec['h']):.2f} m, throttle up to "
         f"{max(rec['thr']):.2f}. Guards: {a.guards}.")


def route_flight(spec, wind=8.0, from_deg=270.0, turbulence=0.0, seed=3):
    f = Flight(parse(["--wind", str(wind), "--from", str(from_deg), "--turbulence", str(turbulence), "--seed", str(seed)]), spec)
    t0 = time.perf_counter()
    while not f.finished() and f.ac.t < 400.0:
        f.step()
    return f, time.perf_counter() - t0


def check_route(spec):
    """6 and 7: the route in a steady crosswind; the guards on that flight."""
    f, _ = route_flight(spec)
    st = leg_stats(f)
    a = np.array(f.rows)
    ok = f.los.done and len(st) == 2 and all(s["rms"] < 1.0 and abs(s["crab_air"] - s["triangle"]) < 0.2 for s in st)
    line(6, ok, f"the route ({ROUTE[1][0]:.0f} m north, a right turn, {ROUTE[2][1]:.0f} m east, {f.a.height:.0f} m up at 25 m/s) in a steady 8 m/s from "
                f"the west, course hold: done in {f.ac.t:.1f} s.")
    for s in st:
        note(6, f"{s['name']}, settled part: cross-track rms {s['rms']:.2f} m (max {s['max']:.2f}); crosswind {s['w_cross']:+.2f} m/s; the air "
                f"velocity {s['crab_air']:+.2f} deg off the track against the wind triangle's asin(-W_cross / V_a) = {s['triangle']:+.2f}; the "
                f"nose {s['crab_nose']:+.2f}; ground speed {s['vg']:.2f} m/s.")
    past, off, bank = turn_stats(f)
    note(6, f"the turn, handed over a turn radius before the corner: {past:.1f} m past the corner, at most {off:.1f} m off the route (the tailwind "
            f"takes the ground speed from {st[0]['vg']:.0f} to {st[1]['vg']:.0f} m/s through it), bank up to {bank:.1f} deg.")
    g = f.ac.guards
    line(7, not any(g.values()),
         f"guards on that flight: {g} steps (airspeed under the floor, the propeller stopped). Height {a[:, C['h']].min():.2f}..{a[:, C['h']].max():.2f} m, "
         f"airspeed {a[:, C['Va']].min():.2f}..{a[:, C['Va']].max():.2f} m/s, alpha {a[:, C['alpha']].min() * DEG:.1f}..{a[:, C['alpha']].max() * DEG:.1f} deg, "
         f"sideslip within {np.abs(a[:, C['beta']]).max() * DEG:.1f} deg, throttle {a[:, C['throttle']].min():.2f}..{a[:, C['throttle']].max():.2f}, "
         f"elevator {a[:, C['d_e']].min() * DEG:.1f}..{a[:, C['d_e']].max() * DEG:.1f}, aileron within {np.abs(a[:, C['d_a']]).max() * DEG:.1f}, "
         f"rudder within {np.abs(a[:, C['d_r']]).max() * DEG:.1f} deg.")


def check_glide(spec):
    """8: the throttle closed."""
    a = Aerosonde(spec)
    rows, ok = [], True
    for V in (18.0, 20.0, 25.0, 30.0):
        s = a.trim(V, h=100.0, apply=False, throttle=0.0)
        ok = ok and s["residual"] < 1e-9
        rows.append((V, s))
    line(8, ok, "the glide, throttle closed (Aerosonde.trim(V, throttle=0.0) solves for the path angle):")
    note(8, "    V m/s  path deg  sink m/s  glide ratio  alpha  propeller rad/s  its force N  airframe drag N  airframe L/D")
    for V, s in rows:
        note(8, f"    {V:5.1f}  {s['gamma'] * DEG:8.2f}  {-V * math.sin(s['gamma']):8.2f}  {1.0 / math.tan(-s['gamma']):11.2f}  {s['alpha'] * DEG:5.2f}  "
                f"{s['omega']:15.1f}  {s['T']:11.2f}  {s['drag']:15.2f}  {s['lift'] / s['drag']:12.1f}")
    V, s = rows[2]
    note(8, f"this is the propeller model, not a measured glide: with no voltage the motor is a short across its winding and holds the "
            f"propeller at {s['omega']:.0f} rad/s (J = {s['J']:.0f}), and the thrust polynomial, fitted where a propeller drives, gives it "
            f"CT2 rho D^2 Va^2 = {a.ct[2] * a.rho_at(0.0) * a.D ** 2 * V * V:.1f} N at 25 m/s, twice the airframe's drag.")


def check_slowest(spec):
    """9: the ends of steady level flight, and the stall blend."""
    a = Aerosonde(spec)

    def good(V):
        s = a.trim(V, h=100.0, apply=False)
        return s["residual"] < 1e-9 and not s["limited"]
    lo, hi = 10.0, 25.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if good(mid) else (mid, hi)
    vmin = hi
    lo, hi = 25.0, 45.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if good(mid) else (lo, mid)
    vmax = lo
    s, sx = a.trim(vmin, h=100.0, apply=False), a.trim(vmax, h=100.0, apply=False)
    line(9, good(vmin) and good(vmax),
         f"slowest steady level flight {vmin:.2f} m/s: alpha {s['alpha'] * DEG:.2f} deg, C_L {s['CL']:.3f}, elevator {s['de'] * DEG:.1f} deg, the end "
         f"of its ASSUMED {a.limits[0] * DEG:.0f} deg throw, throttle {s['throttle']:.2f}. Fastest {vmax:.2f} m/s, throttle {sx['throttle']:.2f} "
         f"(propeller {sx['omega']:.0f} rad/s of the {a.V_max / a.KV:.0f} the motor turns unloaded at {a.V_max} V).")
    # the lift curve itself: C_L(alpha) with no pitch rate and no elevator, the blend sigma
    best = (0.0, 0.0, 0.0)
    for k in range(0, 901):
        al = math.radians(0.1 * k)
        o = {}
        a.rigid_rates([0.0, 0.0, -100.0, 25.0 * math.cos(al), 0.0, 25.0 * math.sin(al), 0.0, al, 0.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), o)
        if o["CL"] > best[1]:
            best = (al, o["CL"], o["sigma"])
    al, cl, sg = best
    A = a.a
    de_need = (A["Cm0"] + A["Cmalpha"] * al) / -A["Cmde"]
    note(9, f"the stall blend does nothing there: sigma(alpha) is {s['sigma']:.0e} at {s['alpha'] * DEG:.1f} deg, the lift is the linear wing's. "
            f"The book's lift curve peaks at C_L {cl:.2f}, alpha {al * DEG:.1f} deg (sigma {sg:.2f}; alpha0 = {A['alpha0'] * DEG:.1f} deg, M = {A['M']:.0f}), "
            f"which would carry the weight at {math.sqrt(2.0 * a.m * a.g / (a.rho_at(0.0) * a.S * cl)):.1f} m/s; trimming there takes "
            f"{de_need * DEG:.0f} deg of elevator (Cmalpha / Cmde = {A['Cmalpha'] / A['Cmde']:.2f}). So no steady flight of this model reaches its "
            f"stall, and a C_L of {cl:.1f} is not a real wing's: the slow end is the elevator's throw, an ASSUMED number.")


def check_turbulence(spec):
    """10: the same route in a wind with gusts, twice."""
    runs = [route_flight(spec, turbulence=1.0) for _ in range(2)]
    f, wall = runs[1]
    st = leg_stats(f)
    a = np.array(f.rows)
    same = runs[0][0].ac.x == f.ac.x and runs[0][0].ac.t == f.ac.t
    g = f.ac.guards
    line(10, f.los.done and same and not any(g.values()),
         f"the route in m350_rig.Wind: 8 m/s from the west 10 m up, {math.hypot(f.wind[0], f.wind[1]):.1f} m/s at the route's height (the "
         f"logarithmic profile), Dryden gusts at the standard's intensities (seed 3): done in {f.ac.t:.1f} s; two runs are "
         f"{'bit-identical' if same else 'NOT identical'}; {f.ac.t / wall:.0f} times real time, autopilot and wind included.")
    for s in st:
        note(10, f"{s['name']}: cross-track rms {s['rms']:.2f} m (max {s['max']:.2f}); the air velocity {s['crab_air']:+.2f} deg off the track "
                 f"(the mean wind's triangle {s['triangle']:+.2f}; the gusts along the wind are some 260 m long up there and do not "
                 f"average out over a leg's settled part).")
    note(10, f"height {a[:, C['h']].min():.2f}..{a[:, C['h']].max():.2f} m, airspeed {a[:, C['Va']].min():.2f}..{a[:, C['Va']].max():.2f} m/s, bank up to "
             f"{np.abs(a[:, C['phi']]).max() * DEG:.1f} deg, alpha {a[:, C['alpha']].min() * DEG:.1f}..{a[:, C['alpha']].max() * DEG:.1f} deg, sideslip within "
             f"{np.abs(a[:, C['beta']]).max() * DEG:.1f} deg, throttle {a[:, C['throttle']].min():.2f}..{a[:, C['throttle']].max():.2f}; guards {g}.")


CHECKS = {1: check_trim, 2: check_rigid_body, 3: check_modes, 4: check_open_loop, 5: check_steps, 6: check_route, 7: check_route,
          8: check_glide, 9: check_slowest, 10: check_turbulence}


# --------------------------------------------------------------------------- #
#  The scene
# --------------------------------------------------------------------------- #
LAKE = (820.0, 330.0, 360.0)                                # north, east, radius


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


class Scene:
    def __init__(self, tp, f, w, h, headless):
        self.tp, self.f = tp, f
        a = f.a
        self.canvas = tp.Canvas("threepp - Aerosonde", width=w, height=h, antialiasing=4, headless=headless, vsync=not headless)
        r = self.r = tp.GLRenderer(self.canvas)
        r.tone_mapping = tp.ToneMapping.ACESFilmic
        r.tone_mapping_exposure = 1.0
        s = self.scene = tp.Scene()
        s.background = 0x9eb8d6
        s.set_fog(tp.Color(0x9eb8d6), 500.0, 4500.0)
        s.add(tp.HemisphereLight(0xdce6f2, 0x4a5a38, 1.1))
        sun = tp.DirectionalLight(0xfff4e0, 2.2)
        sun.position.set(-300.0, 500.0, 200.0)
        s.add(sun)
        ground = tp.Mesh(tp.PlaneGeometry(10000.0, 10000.0), material(tp, 0x5b7240, 0.95))
        ground.rotation.x = -math.pi / 2
        s.add(ground)
        lake = tp.Mesh(tp.CircleGeometry(LAKE[2], 96), material(tp, 0x3d5866, 0.25, 0.1))
        lake.rotation.x = -math.pi / 2
        lake.position.set(*ned_to_world(LAKE[0], LAKE[1], -0.05))
        s.add(lake)
        rng = np.random.default_rng(7)
        tree = material(tp, 0x2f4526, 0.9)
        cone = tp.ConeGeometry(2.6, 10.0, 8, 1)
        for _ in range(900):
            n, e = rng.uniform(-600.0, 1700.0), rng.uniform(-700.0, 1700.0)
            if math.hypot(n - LAKE[0], e - LAKE[1]) < LAKE[2] + 20.0:
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
            p = tp.Mesh(tp.CylinderGeometry(0.3, 0.3, hgt, 8, 1), pole)
            p.position.set(*ned_to_world(n, e, -0.5 * hgt))
            s.add(p)
            b = tp.Mesh(tp.SphereGeometry(1.2, 16, 10), knob)
            b.position.set(*ned_to_world(n, e, -hgt))
            s.add(b)
        # the aim point, the line of sight to it, the flown track
        aim_mat = material(tp, 0xff7a1a, 0.5)
        aim_mat.emissive = tp.Color(1.0, 0.45, 0.1)
        aim_mat.emissive_intensity = 0.6
        self.aim = tp.Mesh(tp.SphereGeometry(1.4, 16, 10), aim_mat)
        s.add(self.aim)
        self.sight, self.sight_g = polyline(tp, [(0, 0, 0), (0, 0, 0)], 0xff7a1a)
        s.add(self.sight)
        self.track_n = 0
        self.track, self.track_g = polyline(tp, [(0, 0, 0)], 0x2fd4ff, capacity=20000)
        self.track_buf = np.zeros((20000, 3), np.float32)
        s.add(self.track)
        # the wind
        if a.wind > 0.05:
            wv = np.asarray(f.wind)
            d = np.array(ned_to_world(wv[0], wv[1], 0.0))
            d /= np.linalg.norm(d)
            arrow = tp.ArrowHelper(tp.Vector3(*map(float, d)), tp.Vector3(*ned_to_world(START[0] + 40.0, -25.0, -1.0)),
                                   a.wind * 1.0 + 2.0, tp.Color(0.95, 0.85, 0.2), 1.6, 1.2)
            s.add(arrow)
        # the aircraft
        mp = model_path()
        if mp is None:
            sys.exit("aerosonde.glb is not built: blender --background --factory-startup --python build_aerosonde_blender.py")
        self.model = tp.GLTFLoader().load(mp).scene
        s.add(self.model)
        self.vis = Visual(self.model)
        self.cam = tp.PerspectiveCamera(42.0, w / h, 0.2, 12000.0)
        self.mode = a.cam
        self.eye = None

    def sync(self):
        """The aircraft, the aim point, the line of sight and the track from the flight's state."""
        f, ac = self.f, self.f.ac
        self.vis.pose(ac, h0=0.0)
        p = ned_to_world(ac.n, ac.e, ac.d)
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
        tp, ac = self.tp, self.f.ac
        p = np.array(ned_to_world(ac.n, ac.e, ac.d))
        vg = ac.ground_velocity()
        fwd = np.array(ned_to_world(vg[0], vg[1], 0.0))
        fwd /= max(np.linalg.norm(fwd), 1e-6)
        if self.mode == "chase":
            eye, at = p - 15.0 * fwd + np.array([0.0, 3.6, 0.0]), p + 7.0 * fwd
        elif self.mode == "high":
            eye, at = p - 90.0 * fwd + np.array([0.0, 55.0, 0.0]) + 30.0 * np.array([-fwd[2], 0.0, fwd[0]]), p + 50.0 * fwd
        else:
            eye, at = np.array(ned_to_world(ROUTE[1][0] - 60.0, -90.0, -2.0)), p
        # the lens: 42 deg, or narrower so that the aircraft spans about a sixth of the frame
        dist = float(np.linalg.norm(at - eye)) if self.mode == "ground" else 0.0
        fov = min(42.0, math.degrees(2.0 * math.atan(8.0 / max(dist, 1.0)))) if self.mode == "ground" else 42.0
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
        while f.ac.t < t - 1e-9 and not f.finished():
            f.advance(min(frame, t - f.ac.t))
            sc.sync()
        sc.camera()
        sc.render()
        sc.r.save_frame(path)
        print(f"[aerosonde] t {f.ac.t:.2f} s -> {path}")
    summary(f)


def window(tp, f, w, h):
    sc = Scene(tp, f, w, h, headless=False)
    clock = tp.Clock()
    state = {"debt": 0.0, "c": False}
    cams = ("chase", "high", "ground")

    def animate():
        dt = min(clock.get_delta(), 0.1)
        state["debt"] += dt
        n = int(state["debt"] / f.ac.dt)
        state["debt"] -= n * f.ac.dt
        for _ in range(n):
            if f.finished():
                f.reset()
                sc.track_n = 0
            f.step()
        c = sc.canvas.is_key_down("C")
        if c and not state["c"]:
            sc.mode = cams[(cams.index(sc.mode) + 1) % len(cams)]
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
        which = sorted(CHECKS) if a.checks == "all" else [int(k) for k in a.checks.split(",")]
        done = []
        for k in which:
            if CHECKS[k] not in done:                        # 6 and 7 are one flight
                CHECKS[k](spec)
                done.append(CHECKS[k])
        return
    f = Flight(a)
    if a.telemetry:
        telemetry(f)
        return
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # python/ (threepp)
    import threepp as tp
    w, h = (int(v) for v in a.size.lower().split("x"))
    if a.shot is not None or a.stills:
        times = [a.shot] if a.shot is not None else [float(v) for v in a.stills.split(",")]
        if a.shot is not None:
            paths = [a.out]
        else:
            os.makedirs(a.out_dir, exist_ok=True)
            paths = [os.path.join(a.out_dir, f"aerosonde_{a.cam}_{t:05.1f}s.png") for t in times]
        stills(tp, f, times, paths, w, h)
        return
    window(tp, f, w, h)


if __name__ == "__main__":
    main()
