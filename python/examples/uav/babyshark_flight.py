"""
The Babyshark 260 VTOL's flight model on the command line: its checks and a flight.

The aircraft, the autopilot and the mission are rigs/babyshark_rig.py (the wind is m350_rig's,
the route guidance x8_rig's); this file runs them without a renderer (numpy only), or draws a
still of the aircraft in flight.

    python babyshark_flight.py --checks                 # the ten checks, each line numbered
    python babyshark_flight.py --checks 2,3             # some of them
    python babyshark_flight.py --telemetry              # take off, transition, a circuit, back, land: a line a second
    python babyshark_flight.py --telemetry --wind 8 --from 250 --turbulence 1 --csv babyshark.csv
    python babyshark_flight.py --shot 45 --out babyshark.png      # a headless still at t = 45 s (needs babyshark.glb)
    python babyshark_flight.py --stills 12,26,45,150 --out-dir out/babyshark --cam side
    python babyshark_flight.py --window                 # the same flight in a window; C cycles the camera
    python babyshark_fly.py                             # fly it yourself: the keyboard and the mouse

Read the numbers for what they are. The wing's model is B. P. Graesdal's (Master's thesis, NTNU,
2021), identified in flight around 21 m/s; check 2 compares this implementation with the
thesis's own linearisation. The hover, the stall, the transitions, the pusher's loss of thrust
with airspeed, the motors and the batteries are OURS or ASSUMED (babyshark_spec.json says which
and why) and nothing of them is fitted or validated: where the product page or the thesis gives
a number to hold them against, check 3 prints both. The autopilot is OURS, with exact state
feedback (no estimator, no sensor noise).
"""
import argparse
import copy
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rigs"))
from babyshark_rig import (LAT, LON, Autopilot, Babyshark, Mission, Pilot, Visual, Wind, cross_track,  # noqa: E402
                           load_spec, model_path, ned_to_world, wrap)

DEG = 180.0 / math.pi


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", nargs="?", const="all", default=None, help="all, or a comma-separated list of 1..10")
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--wind", type=float, default=6.0, help="wind speed 10 m up, m/s")
    ap.add_argument("--from", dest="from_deg", type=float, default=250.0, help="the direction the wind blows from, deg (0 north, 90 east)")
    ap.add_argument("--turbulence", type=float, default=1.0, help="scales the Dryden sigmas")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--window", action="store_true", help="fly it in a window; C cycles the camera")
    ap.add_argument("--shot", type=float, default=None, help="a still at this time of the flight, s")
    ap.add_argument("--stills", default=None, help="comma-separated times, s")
    ap.add_argument("--out", default="babyshark_flight.png")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--cam", choices=("chase", "side", "high"), default="chase")
    ap.add_argument("--spec", default=None, help="another spec file in place of babyshark_spec.json")
    return ap.parse_args(argv)


def line(k, text):
    print(f"[{k}] {text}")


def fly(a, ap, seconds, record=None, mission=None):
    for _ in range(int(round(seconds / a.dt))):
        if mission is not None:
            mission.update(a, ap)
        ap.update()
        a.step()
        if record is not None:
            record(a)


def on_wing(spec, va=21.0, h=100.0, wind=None, course=0.0):
    a = Babyshark(spec, wind=wind)
    a.trim(va, h=h, course=course)
    ap = Autopilot(a, airspeed=va)
    ap.reset()
    ap.mode = "fw"
    ap.command(course=course, altitude=h, airspeed=va)
    return a, ap


def hovering(spec, h=30.0, wind=None, yaw=0.0):
    a = Babyshark(spec, wind=wind)
    a.set_hover(0.0, 0.0, h, yaw)
    ap = Autopilot(a)
    ap.reset()
    return a, ap


def step_metrics(t, y, y0, y1, band=0.02):
    """10-90 % rise time, overshoot (% of the step) and the time after which it stays inside
    +-band of the step."""
    t, y = np.asarray(t), (np.asarray(y) - y0) / (y1 - y0)
    t10 = t[np.argmax(y >= 0.1)]
    t90 = t[np.argmax(y >= 0.9)] if np.any(y >= 0.9) else float("nan")
    out = np.nonzero(np.abs(y - 1.0) > band)[0]
    return t90 - t10, 100.0 * max(float(np.max(y)) - 1.0, 0.0), (t[out[-1]] if len(out) else t[0]) - t[0]


# --------------------------------------------------------------------------- #
#  1 the airframe and the hover
# --------------------------------------------------------------------------- #
def check_airframe(spec):
    a = Babyshark(spec)
    printed = (0.1419, 0.8795, 1.3851, 0.1045, 0.9003, 0.1197, -0.1872, 0.5990)
    worst = max(abs(g - p) for g, p in zip(a.gamma, printed))
    line(1, f"{a.m:.2f} kg, b {a.b} m, S {a.S} m^2 (wing and tail projected), cbar {a.cbar} m. The eight Gamma constants from "
            f"Table 6.2's four inertias against the numbers the model repository's code carries: off by {worst:.1e} at the most.")
    n = a.hover_speeds()
    kT = a._kTm
    a.set_hover(0.0, 0.0, 30.0)
    out = a.evaluate()
    f = a.rates(a.x, a._clipped())
    line(1, f"hover: front rotors {n[0]:.1f} rev/s ({kT * n[0] ** 2:.1f} N each), rear {n[1]:.1f} rev/s ({kT * n[1] ** 2:.1f} N): the centre "
            f"of gravity is 0.353 m behind the front axes and 0.447 m ahead of the rear. The thesis's thrust-stand run ends at 99 rev/s; "
            f"the rotors' limit here is {a.eta_mr_max:.1f} (ASSUMED). Residual accelerations {max(abs(v) for v in f[3:9]):.1e}.")
    line(1, f"hover power: {out['shaft_mr_w']:.0f} W at the shafts (the thesis's torque coefficient), {out['lift_w']:.0f} W from the lift "
            f"battery at an ASSUMED drive efficiency of {a.eta_drive}: {a.lift_wh / out['lift_w'] * 60.0:.1f} min of hover on its "
            f"{a.lift_wh:.0f} Wh (ASSUMED pack). Figure of merit {(a.m * a.g) ** 1.5 / math.sqrt(2.0 * a.rho * 4.0 * math.pi * (0.5 * a.D_mr) ** 2) / out['shaft_mr_w']:.2f}.")


# --------------------------------------------------------------------------- #
#  2 the thesis's linearisation
# --------------------------------------------------------------------------- #
def linear(spec):
    """The longitudinal and lateral blocks of this implementation, linearised numerically at
    the thesis's trim (Table 6.5), with the thesis's static propeller law."""
    a = Babyshark(spec, rho=spec["actuators"]["rho"], thesis_propeller=True)
    T = spec["trim"]
    al = math.radians(T["alpha_deg"])
    a.set_state(h=100.0, u=T["V"] * math.cos(al), w=T["V"] * math.sin(al), theta=al, eta_fw=T["eta_fw"])
    A, B = a.jacobian()
    return (A[np.ix_(LON, LON)], B[np.ix_(LON, (1, 4))], A[np.ix_(LAT, LAT)], B[np.ix_(LAT, (0, 2))])


def compare(mats, spec):
    """Entries outside the band: half a unit of the thesis's last printed digit and a thousandth
    of the entry (what rounding the printed inputs can do)."""
    TL = spec["thesis_linearisation"]
    names = {"A_lon": ("u'", "w'", "q'", "theta'", "u", "w", "q", "theta"), "B_lon": ("u'", "w'", "q'", "theta'", "delta_e", "delta_t"),
             "A_lat": ("v'", "p'", "r'", "phi'", "v", "p", "r", "phi"), "B_lat": ("v'", "p'", "r'", "phi'", "delta_a", "delta_r")}
    out, count = [], 0
    for key, M in zip(("A_lon", "B_lon", "A_lat", "B_lat"), mats):
        Tm = np.array(TL[key])
        for i in range(Tm.shape[0]):
            for j in range(Tm.shape[1]):
                count += 1
                if abs(M[i, j] - Tm[i, j]) > 5.1e-5 + 1e-3 * abs(Tm[i, j]):
                    out.append(f"{key}[{names[key][i]}, {names[key][4 + j]}] thesis {Tm[i, j]:+.4f}, here {M[i, j]:+.4f}")
    return count, out


def modes(Alon, Alat):
    el, ea = np.linalg.eigvals(Alon), np.linalg.eigvals(Alat)
    cl = sorted([e for e in el if e.imag > 1e-9], key=lambda e: -abs(e))
    re = sorted([e.real for e in ea if abs(e.imag) < 1e-9])
    dr = [e for e in ea if e.imag > 1e-9][0]
    return {"short_period": cl[0], "phugoid": cl[1], "roll": re[0], "dutch_roll": dr, "spiral": re[-1]}


def check_linearisation(spec):
    mats = linear(spec)
    count, out = compare(mats, spec)
    line(2, f"against the thesis's printed matrices (6.17), (6.18), linearised at its trim with its static propeller law: "
            f"{count - len(out)} of {count} entries inside the band (half a unit of the last printed digit and 0.1 % of the entry). Outside:")
    for o in out:
        line(2, "    " + o)
    s2 = copy.deepcopy(spec)
    s2["aero"]["cDa"] = s2["aero"]["cDa2"] = 0.0
    count2, out2 = compare(linear(s2), spec)
    line(2, f"those are the three derivatives the drag's change with alpha enters. With cD = cD0 in the linearisation (the alpha terms "
            f"of (6.8d) left out) {count2 - len(out2)} of {count2} are inside" + (": " + "; ".join(out2) if out2 else "") +
            ". The thesis's matrices are its model's with the drag held at cD0; the model here keeps (6.8d) whole.")
    m, T = modes(mats[0], mats[2]), spec["thesis_linearisation"]["modes"]
    line(2, f"modes, here against Table 6.7: short period {m['short_period'].real:+.3f} +- {m['short_period'].imag:.3f}i "
            f"({T['short_period'][0]:+.3f} +- {T['short_period'][1]:.3f}i), phugoid {m['phugoid'].real:+.4f} +- {m['phugoid'].imag:.3f}i "
            f"({T['phugoid'][0]:+.4f} +- {T['phugoid'][1]:.3f}i), roll {m['roll']:+.2f} ({T['roll']:+.2f}), dutch roll "
            f"{m['dutch_roll'].real:+.3f} +- {m['dutch_roll'].imag:.3f}i ({T['dutch_roll'][0]:+.3f} +- {T['dutch_roll'][1]:.3f}i), "
            f"spiral {m['spiral']:+.3f} ({T['spiral']:+.3f}). The thesis trusts neither its phugoid nor its spiral.")
    a = Babyshark(spec, rho=spec["actuators"]["rho"], thesis_propeller=True)
    Tr = spec["trim"]
    al = math.radians(Tr["alpha_deg"])
    a.set_state(h=100.0, u=Tr["V"] * math.cos(al), w=Tr["V"] * math.sin(al), theta=al, eta_fw=Tr["eta_fw"])
    f = a.rates(a.x, a._clipped())
    line(2, f"the thesis's trim point is not an equilibrium of its model: u' {f[3]:+.2f}, w' {f[5]:+.2f} m/s^2, q' {f[7] * DEG:+.1f} deg/s^2 "
            f"there (it was read off manual flights; lift {a.evaluate()['cL'] * 0.5 * a.rho * Tr['V'] ** 2 * a.S:.0f} N against {a.m * a.g:.0f} N of weight).")


# --------------------------------------------------------------------------- #
#  3 the wing's trim, its slowest and fastest flight, the stall
# --------------------------------------------------------------------------- #
def check_trim(spec):
    a = Babyshark(spec)
    line(3, "straight and level in still air at the sea (Babyshark.trim), the lift rotors stopped:")
    line(3, "    V m/s  alpha  elevator  pusher rev/s  thrust N   J     cL    L/D   cruise battery W")
    for V in (16.0, 17.0, 19.0, 21.0, 22.0):
        s = a.trim(V, apply=False)
        line(3, f"    {V:5.1f}  {s['alpha'] * DEG:5.2f}  {s['de'] * DEG:7.2f}  {s['eta']:10.1f}  {s['T']:9.1f}  {s['J']:5.2f}  {s['cL']:5.2f}  "
                f"{s['cL'] / s['cD']:5.1f}  {s['cruise_w']:8.0f}")

    def edge(lo, hi, ok):
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if ok(a.trim(mid, apply=False)):
                hi = mid
            else:
                lo = mid
        return hi
    vmin = edge(13.0, 18.0, lambda s: not s["limited"] and s["residual"] < 1e-6)
    lo, hi = 20.0, 27.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        s = a.trim(mid, apply=False)
        if s["limited"] or s["residual"] > 1e-6:
            hi = mid
        else:
            lo = mid
    vmax = lo
    smin = a.trim(vmin, apply=False)
    line(3, f"slowest steady flight {vmin:.1f} m/s (alpha {smin['alpha'] * DEG:.1f} deg, the elevator at its {a.d_max[1] * DEG:.0f} deg throw; "
            f"Foxtech gives a stall speed of 15 to 16 m/s). Fastest {vmax:.1f} m/s = {vmax * 3.6:.0f} km/h (the pusher at its "
            f"{a.eta_fw_max:.0f} rev/s; Foxtech: about 100 km/h). Neither is fitted.")
    s = a.trim(21.0, apply=False)
    b = Babyshark(spec, thesis_propeller=True)
    sb = b.trim(21.0, apply=False)
    line(3, f"at the thesis's 21 m/s the model's level flight needs {s['T']:.1f} N. With the pusher's thrust falling with the advance ratio "
            f"(OURS, J0 = {a.J0}) that is {s['eta']:.0f} rev/s (the thesis's Table 6.5 prints 125); with the thesis's static law it is "
            f"{sb['eta']:.0f} rev/s, delta_t = {sb['delta_t']:.0f} (the recorded inputs its repository ships sit near 9750).")
    # the lift curve past the identified range: the wing alone at 16 m/s, the surfaces at their trim
    a.set_state(h=0.0)
    best = (0.0, 0.0)
    for k in range(0, 451):
        al = math.radians(0.1 * k)
        V = 16.0
        x = list(a.x)
        x[3], x[5] = V * math.cos(al), V * math.sin(al)
        out = {}
        a.rates(x, x[12:], out)
        ax, _, az = out["accel_body"]
        cl = a.m * (ax * math.sin(al) - az * math.cos(al)) / (0.5 * a.rho * V * V * a.S)
        if cl > best[1]:
            best = (al, cl)
    line(3, f"past the thesis's data (alpha {a.alpha_lim[0] * DEG:.0f}..{a.alpha_lim[1] * DEG:.0f} deg) the lift goes over to a bluff body's "
            f"(OURS, ASSUMED): it peaks at cL {best[1]:.2f}, alpha {best[0] * DEG:.1f} deg, surfaces at their trim. The thesis's own "
            f"quadratic would rise to alpha 38 deg.")


# --------------------------------------------------------------------------- #
#  4 the wing's loops
# --------------------------------------------------------------------------- #
def check_wing_steps(spec):
    a, ap = on_wing(spec)
    fly(a, ap, 5.0)
    rec = {"t": [], "h": [], "chi": [], "V": [], "phi": [], "beta": []}

    def record(a):
        rec["t"].append(a.t)
        rec["h"].append(a.h)
        rec["chi"].append(a.course())
        rec["V"].append(a.out["Va"])
        rec["phi"].append(abs(a.x[9]))
        rec["beta"].append(abs(a.out["beta"]))
    ap.command(altitude=110.0)
    fly(a, ap, 40.0, record)
    tr, ov, _ = step_metrics(rec["t"], rec["h"], 100.0, 110.0)
    vmin = min(rec["V"])
    line(4, f"from the 21 m/s trim in still air: +10 m of height in {tr:.1f} s (10 to 90 %), {ov * 0.1:.2f} m over, the airspeed not under "
            f"{vmin:.1f} m/s (the climb the pusher can hold at 21 m/s is {ap.climb_max * 21.0:.2f} m/s, and the pitch command stops at "
            f"{spec['autopilot']['altitude']['climb_fraction']:.0%} of it).")
    for k in rec:
        rec[k].clear()
    ap.command(course=math.radians(90.0))
    fly(a, ap, 25.0, record)
    tr, ov, _ = step_metrics(rec["t"], np.unwrap(rec["chi"]), 0.0, math.radians(90.0))
    line(4, f"a 90 deg course change in {tr:.1f} s, {ov * 0.9:.1f} deg over, the height within {max(abs(h - 110.0) for h in rec['h']):.1f} m, "
            f"bank up to {max(rec['phi']) * DEG:.0f} deg, sideslip up to {max(rec['beta']) * DEG:.1f} deg.")
    for k in rec:
        rec[k].clear()
    ap.command(airspeed=19.0)
    fly(a, ap, 25.0, record)
    tr, ov, _ = step_metrics(rec["t"], rec["V"], 21.0, 19.0)
    line(4, f"-2 m/s of airspeed in {tr:.1f} s, {ov * 0.02:.2f} m/s over, the height within {max(abs(h - 110.0) for h in rec['h']):.1f} m. "
            f"Guards: {a.guards} (steps outside the identified alpha and beta).")


# --------------------------------------------------------------------------- #
#  5 the hover's loops, 6 the hover in wind
# --------------------------------------------------------------------------- #
def check_hover_steps(spec):
    a, ap = hovering(spec)
    ap.command(yaw=0.0)
    fly(a, ap, 2.0)
    rec = {"t": [], "e": [], "h": [], "psi": [], "tilt": []}

    def record(a):
        rec["t"].append(a.t)
        rec["e"].append(a.x[1])
        rec["h"].append(a.h)
        rec["psi"].append(a.x[11])
        rec["tilt"].append(a.tilt())
    ap.command(position=(0.0, 5.0, 30.0))
    fly(a, ap, 15.0, record)
    tr, ov, ts = step_metrics(rec["t"], rec["e"], 0.0, 5.0)
    tilt = max(rec["tilt"])
    for k in rec:
        rec[k].clear()
    ap.command(position=(0.0, 5.0, 32.0))
    fly(a, ap, 10.0, record)
    tr2, ov2, _ = step_metrics(rec["t"], rec["h"], 30.0, 32.0)
    for k in rec:
        rec[k].clear()
    ap.command(yaw=math.radians(90.0))
    fly(a, ap, 12.0, record)
    tr3, ov3, _ = step_metrics(rec["t"], rec["psi"], 0.0, math.radians(90.0))
    line(5, f"steps from a hover in still air: 5 m sideways in {tr:.1f} s (10 to 90 %) with {ov:.1f} % overshoot, inside 2 % after {ts:.1f} s, "
            f"tilt up to {tilt * DEG:.0f} deg; 2 m up in {tr2:.1f} s with {ov2:.1f} %; 90 deg of yaw in {tr3:.1f} s with {ov3:.1f} % "
            f"(the rotors' drag torque is all the yaw there is).")


def check_hover_wind(spec):
    for speed, turb in ((8.0, 0.0), (8.0, 1.0)):
        wind = Wind(speed, from_deg=90.0, turbulence=turb, seed=2, spec=spec)
        a, ap = hovering(spec, wind=wind)                    # nose north, the wind on the beam
        t_turn, err, far, yaw = 0.0, [], 0.0, []

        def record(a):
            nonlocal t_turn, far
            if a.t <= 40.0:
                if abs(math.degrees(a.x[11]) - 90.0) > 10.0:
                    t_turn = a.t
                far = max(far, math.hypot(a.x[0], a.x[1]))
            else:
                err.append(math.hypot(a.x[0], a.x[1]))
                yaw.append(math.degrees(a.x[11]) - 90.0)
        fly(a, ap, 100.0, record)
        o = a.out
        line(6, f"hovering 30 m up in {speed:.0f} m/s on the beam ({o['Va']:.1f} m/s at that height)"
                + (", turbulence at the standard's intensities" if turb else ", steady")
                + f": the weathervane has the nose within 10 deg of the wind after {t_turn:.0f} s, the aircraft pushed {far:.2f} m off the "
                  f"point at the most while it turns; from 40 s on the nose is {np.sqrt(np.mean(np.square(yaw))):.1f} deg rms off the wind and the "
                  f"aircraft {np.sqrt(np.mean(np.square(err))):.2f} m rms from the point ({max(err):.2f} m at the most), pitch {a.x[10] * DEG:+.1f} deg, the rotors at "
                  f"{a.x[16]:.0f} / {a.x[17]:.0f} rev/s (the wing carries part of the weight), {o['lift_w']:.0f} W.")


# --------------------------------------------------------------------------- #
#  7 the transitions
# --------------------------------------------------------------------------- #
def check_transitions(spec):
    for speed in (0.0, 8.0):
        wind = Wind(speed, from_deg=0.0, turbulence=0.0, spec=spec) if speed else None
        a, ap = hovering(spec, h=60.0, wind=wind)
        ap.command(yaw=0.0)
        fly(a, ap, 3.0)
        h = [a.h, a.h]

        def record(a):
            h[0], h[1] = min(h[0], a.h), max(h[1], a.h)
        t0, n0 = a.t, a.x[0]
        ap.transition("fw", course=0.0)
        while ap.mode != "fw" and a.t - t0 < 60.0:
            fly(a, ap, a.dt, record)
        tf, nf = a.t - t0, a.x[0] - n0
        fly(a, ap, 15.0, record)
        hf = (h[0] - 60.0, h[1] - 60.0)
        h = [a.h, a.h]
        href = ap.h_c
        t0, n0 = a.t, a.x[0]
        ap.transition("mc")
        while (ap.mode != "mc" or float(np.linalg.norm(a.ground_velocity())) > 0.3) and a.t - t0 < 60.0:
            fly(a, ap, a.dt, record)
        line(7, ("in still air" if not speed else f"into {speed:.0f} m/s ({a.out['Va']:.1f} m/s at 60 m)")
                + f": the front transition takes {tf:.1f} s and {nf:.0f} m over the ground to {ap.tr_speed:.0f} m/s of airspeed, the height "
                  f"within {hf[0]:+.1f} / {hf[1]:+.1f} m through it and the 15 s after; the back transition from "
                  f"{ap._v0:.0f} m/s over the ground to a standstill takes {a.t - t0:.1f} s and {a.x[0] - n0:.0f} m, the height within "
                  f"{h[0] - href:+.1f} / {h[1] - href:+.1f} m." + (" ABORTED." if ap.aborted else ""))


# --------------------------------------------------------------------------- #
#  8 a mission
# --------------------------------------------------------------------------- #
ALTITUDE = 60.0


def circuit(from_deg):
    """Take off to the circuit's height, a circuit of 800 x 250 m at 20 m/s laid out so the last
    leg and the landing are into the wind, land where it took off."""
    c, s = math.cos(math.radians(from_deg)), math.sin(math.radians(from_deg))

    def P(f, r):
        return (f * c - r * s, f * s + r * c)
    route = [P(350.0, 0.0), P(350.0, 250.0), P(-450.0, 250.0), P(-450.0, 0.0)]
    return [dict(kind="takeoff", height=ALTITUDE, hold=2.0), dict(kind="cruise", route=route, altitude=ALTITUDE, airspeed=20.0),
            dict(kind="land", pos=(0.0, 0.0))], route


def check_mission(spec):
    for speed, turb in ((0.0, 0.0), (8.0, 1.0)):
        wind = Wind(speed, from_deg=250.0, turbulence=turb, seed=3, spec=spec) if speed else None
        a = Babyshark(spec, wind=wind)
        a.place_on_ground(0.0, 0.0, yaw=math.radians(250.0))
        ap = Autopilot(a)
        legs, route = circuit(250.0)
        mission = Mission(legs, spec)
        sink, xt, hband, poly = None, 0.0, [0.0, 0.0], None
        while not mission.done and a.t < 600.0:
            mission.update(a, ap)
            ap.update()
            a.step()
            if mission.phase == "descend" and a.on_ground and sink is None:
                sink = float(a.ground_velocity()[2])
            if mission.phase == "cruise" and mission.los is not None:
                poly = poly or mission.los.wp
                xt = max(xt, cross_track(poly, a.x[0], a.x[1]))
                if mission.los.k >= 1:
                    hband = [min(hband[0], a.h - ALTITUDE), max(hband[1], a.h - ALTITUDE)]
        line(8, ("still air" if not speed else f"{speed:.0f} m/s from 250 deg with turbulence")
                + f": take-off to {ALTITUDE:.0f} m, the circuit there at 20 m/s, a landing into the wind: "
                + (f"down after {a.t:.0f} s, sinking {sink:.2f} m/s at the touch, {math.hypot(a.x[0], a.x[1]):.2f} m from where it took off"
                   if mission.done and sink is not None else "NOT landed")
                + f"; on the circuit at most {xt:.0f} m from the route (it cuts each corner by its turn radius) and {hband[0]:+.1f} / {hband[1]:+.1f} m "
                  f"off the height after the first leg; "
                  f"the lift battery gave {100 * (1 - a.lift_fraction):.0f} %, the cruise battery {100 * (1 - a.cruise_fraction):.1f} %."
                + (" A transition was ABORTED." if ap.aborted else ""))


# --------------------------------------------------------------------------- #
#  9 the integration
# --------------------------------------------------------------------------- #
def check_integrity(spec):
    a = Babyshark(spec, rho=0.0)
    a.g = 0.0
    a.set_state(h=100.0, u=3.0, v=-1.0, w=0.5, p=1.2, q=-0.7, r=0.9)
    e0, h0 = a.energy(), a.angular_momentum_ned()
    de = dh = 0.0
    for _ in range(int(60.0 / a.dt)):
        a.step()
        if abs(a.x[10]) < math.radians(60.0):
            de = max(de, abs(a.energy() - e0) / e0)
            dh = max(dh, float(np.linalg.norm(a.angular_momentum_ned() - h0)) / float(np.linalg.norm(h0)))
    line(9, f"a torque-free tumble (no air, no gravity) keeps its energy to {de:.0e} and its angular momentum to {dh:.0e} over 60 s "
            f"while |theta| is under 60 deg (the Euler angles of (6.6g-i) are singular at 90).")
    a, ap = on_wing(spec, wind=None)
    ap.command(course=math.radians(120.0), altitude=115.0)
    e0, work = a.energy(), 0.0
    for _ in range(int(60.0 / a.dt)):
        ap.update()
        a.step()
        work += a.out["power_nc"] * a.dt
    line(9, f"a minute on the wing in still air (a turn and a climb): the energy changes by {a.energy() - e0:+.1f} J, the work of the "
            f"aerodynamic and propeller forces and moments is {work:+.1f} J (a first-order sum of the power at each step's start).")
    runs = []
    for _ in range(2):
        wind = Wind(7.0, from_deg=200.0, turbulence=1.0, seed=5, spec=spec)
        a, ap = hovering(spec, wind=wind)
        ap.transition("fw", course=math.radians(200.0))
        t0 = time.perf_counter()
        fly(a, ap, 30.0)
        runs.append((list(a.x), time.perf_counter() - t0))
    line(9, f"two runs of a front transition in turbulence are {'bit-identical' if runs[0][0] == runs[1][0] else 'NOT identical'}; "
            f"{30.0 / runs[1][1]:.0f} times real time, autopilot and wind included.")


# --------------------------------------------------------------------------- #
#  10 the pilot's sticks
# --------------------------------------------------------------------------- #
def check_pilot(spec):
    a = Babyshark(spec)
    a.place_on_ground(0.0, 0.0)
    ap = Autopilot(a)
    ap.armed = False
    pilot = Pilot(a, ap, spec)

    def hold(seconds, record=None, until=None, **sticks):
        pilot.sticks(**sticks)
        for _ in range(int(round(seconds / a.dt))):
            pilot.update()
            ap.update()
            a.step()
            if record is not None:
                record(a)
            if until is not None and until():
                break

    def speed():
        g = a.ground_velocity()
        return math.hypot(g[0], g[1])

    # the hover
    t0 = a.t
    hold(6.0, until=lambda: not a.on_ground, up=1.0)
    t_off = a.t - t0
    hold(10.0, up=1.0)
    climb, h1 = -float(a.ground_velocity()[2]), a.h
    hold(5.0)
    rise = a.h - h1
    hold(8.0, forward=1.0)
    v_fwd, p1, t1 = speed(), (a.x[0], a.x[1]), a.t
    hold(15.0, until=lambda: speed() < 0.3)
    stop, t_stop, p2 = math.hypot(a.x[0] - p1[0], a.x[1] - p1[1]), a.t - t1, (a.x[0], a.x[1])
    hold(8.0)
    drift = math.hypot(a.x[0] - p2[0], a.x[1] - p2[1])
    hold(4.0, turn=1.0)
    rate, psi1 = a.x[8] * DEG, a.x[11]
    swing = [0.0]

    def turned(a):
        swing[0] = max(swing[0], wrap(a.x[11] - psi1))
    hold(8.0, turned)
    more = wrap(a.x[11] - psi1)
    line(10, f"the pilot's sticks, hovering in still air: up from the ground, the feet leave it after {t_off:.1f} s and it climbs at "
             f"{climb:.1f} m/s; let go, it rises {rise:.1f} m more. Forward: {v_fwd:.1f} m/s after 8 s; let go, it stops in {stop:.0f} m and "
             f"{t_stop:.1f} s and is {drift:.2f} m from there 8 s on. Turn: {rate:.0f} deg/s after 4 s; let go, the nose goes "
             f"{more * DEG:.0f} deg further and stops ({(swing[0] - more) * DEG:.1f} deg of swing back).")
    # the wing
    t0, hb = a.t, [a.h, a.h, a.h]

    def band(a):
        hb[0], hb[1] = min(hb[0], a.h), max(hb[1], a.h)
    pilot.toggle()
    hold(60.0, band, until=lambda: ap.mode == "fw")
    t_front = a.t - t0
    hold(10.0, band)
    front = (hb[0] - hb[2], hb[1] - hb[2])
    h2 = a.h
    hold(4.0, turn=-1.0)
    psi2 = a.x[11]
    hold(4.0, turn=-1.0)
    bank, turn_rate, sag = a.x[9] * DEG, wrap(a.x[11] - psi2) * DEG / 4.0, a.h - h2
    hold(4.0)
    chi = a.course()
    off = [0.0]

    def course(a):
        off[0] = max(off[0], abs(wrap(a.course() - chi)))
    hold(10.0, course)
    can = ap.fw_cfg["altitude"]["climb_fraction"] * ap.climb_max * a.out["Va"]
    hold(6.0, up=1.0)
    h3 = a.h
    hold(12.0, up=1.0)
    up_rate = (a.h - h3) / 12.0
    hold(4.0)
    low = [a.out["Va"]]

    def slowest(a):
        low[0] = min(low[0], a.out["Va"])
    hold(6.0, slowest, forward=-1.0)
    hold(4.0, slowest)
    h4 = a.h
    hold(8.0, slowest, up=-1.0)
    down_rate = (a.h - h4) / 8.0
    hold(12.0, slowest)
    hold(6.0, slowest, turn=1.0)
    bank_slow = a.x[9] * DEG
    hold(6.0, slowest)
    line(10, f"on the wing: T, and {t_front:.1f} s later the wing has it, the height within {front[0]:+.1f} / {front[1]:+.1f} m. Turn: "
             f"banked {bank:.0f} deg it turns {turn_rate:.1f} deg/s and is {sag:+.1f} m off its height after 8 s; let go, it levels and "
             f"holds the course it then has within {off[0] * DEG:.1f} deg. Up: {up_rate:.2f} m/s (the pusher has {can:.2f} to give at "
             f"21 m/s). Slower: the setpoint stops at {ap.va_c:.0f} m/s; down ({down_rate:.1f} m/s), level off and a turn at that speed "
             f"(banked {bank_slow:.0f} deg) never see less than {low[0]:.1f} m/s of airspeed.")
    # back, down, and the autopilot's turn
    pilot.toggle()
    hold(60.0, until=lambda: ap.mode == "mc" and speed() < 0.3)
    sink, t0 = [None], a.t

    def touch(a):
        if sink[0] is None and a.on_ground:
            sink[0] = float(a.ground_velocity()[2])
    hold(120.0, touch, until=lambda: pilot.state == "ground", up=-1.0)
    t_down, away = a.t - t0, math.hypot(a.x[0], a.x[1])
    legs, _ = circuit(0.0)
    pilot.fly(legs)
    hold(200.0, until=lambda: pilot.phase == "auto: cruise")
    hold(10.0)
    hold(0.5, turn=1.0)
    took = pilot.mission is None and pilot.state == "fly" and ap.mode == "fw"
    hold(6.0)
    pilot.fly([dict(kind="land", pos=(0.0, 0.0))])
    t0 = a.t
    hold(600.0, until=lambda: pilot.state == "ground" and pilot.mission is None)
    line(10, f"T again and down: hovering, then on the ground {t_down:.0f} s after the stick went down ({away:.0f} m from the pad), "
             f"sinking {sink[0]:.2f} m/s at the touch, the rotors stopped. Handed to the autopilot there (the circuit, from the ground) "
             f"and taken back on the first leg with a touch of the turn stick: " + ("the pilot has it, on the wing" if took else "NOT taken")
             + f"; handed back to land, it is on the pad {a.t - t0:.0f} s later, {math.hypot(a.x[0], a.x[1]):.2f} m from its middle, "
             + ("the motors stopped." if not ap.armed else "the motors NOT stopped."))


CHECKS = {1: check_airframe, 2: check_linearisation, 3: check_trim, 4: check_wing_steps, 5: check_hover_steps,
          6: check_hover_wind, 7: check_transitions, 8: check_mission, 9: check_integrity, 10: check_pilot}


# --------------------------------------------------------------------------- #
#  The telemetry flight
# --------------------------------------------------------------------------- #
class Flight:
    """The example's flight: on the ground nose into the wind, the circuit of check 8."""

    def __init__(self, a, spec):
        wind = Wind(a.wind, a.from_deg, turbulence=a.turbulence, seed=a.seed, spec=spec) if a.wind > 0.0 else None
        self.aircraft = Babyshark(spec, wind=wind)
        self.aircraft.place_on_ground(0.0, 0.0, yaw=math.radians(a.from_deg))
        self.ap = Autopilot(self.aircraft)
        legs, self.route = circuit(a.from_deg)
        self.mission = Mission(legs, spec)
        self.track = []

    def step(self):
        ac = self.aircraft
        self.mission.update(ac, self.ap)
        self.ap.update()
        ac.step()
        if ac.steps % 30 == 0:
            self.track.append((ac.x[0], ac.x[1], ac.x[2]))

    def advance(self, seconds):
        for _ in range(int(round(seconds / self.aircraft.dt))):
            if self.mission.done:
                break
            self.step()


def telemetry(a, spec):
    f = Flight(a, spec)
    ac, ap, mission = f.aircraft, f.ap, f.mission
    print(f"Babyshark 260 VTOL, {ac.m:.2f} kg, wind {a.wind:.1f} m/s from {a.from_deg:.0f} deg (10 m up), turbulence x{a.turbulence}, "
          f"seed {a.seed}. Autopilot OURS, exact state feedback.")
    print("    t  phase              mode       N       E      H     Va     Vg   roll  pitch    yaw  alpha   wind(N,E,D)       "
          "pusher  lift rotors rev/s    lift W cruise W  lift % cruise %")
    csv = open(a.csv, "w", encoding="utf-8") if a.csv else None
    if csv:
        csv.write("t,phase,mode,n,e,h,u,v,w,p,q,r,phi,theta,psi,delta_a,delta_e,delta_r,eta_fw,eta_1,eta_2,eta_3,eta_4,"
                  "va,alpha,beta,sigma,wind_n,wind_e,wind_d,lift_w,cruise_w,lift_fraction,cruise_fraction\n")
    per = int(round(1.0 / ac.dt))
    sink = None
    while not mission.done and ac.t < 900.0:
        f.step()
        x, o, w = ac.x, ac.out, ac.wind_ned()
        if mission.phase == "descend" and ac.on_ground and sink is None:
            sink = float(ac.ground_velocity()[2])
        if csv:
            csv.write(f"{ac.t:.6g},{mission.phase},{ap.mode}," + ",".join(f"{v:.6g}" for v in (
                x[0], x[1], -x[2], *x[3:20], o["Va"], o["alpha"], o["beta"], o["sigma"], w[0], w[1], w[2], o["lift_w"], o["cruise_w"],
                ac.lift_fraction, ac.cruise_fraction)) + "\n")
        if ac.steps % per == 0:
            vg = ac.ground_velocity()
            print(f"{ac.t:5.0f}  {mission.phase:18s} {ap.mode:5s} {x[0]:7.1f} {x[1]:7.1f} {-x[2]:6.1f} {o['Va']:6.1f} {math.hypot(vg[0], vg[1]):6.1f} "
                  f"{x[9] * DEG:+6.1f} {x[10] * DEG:+6.1f} {x[11] % (2 * math.pi) * DEG:6.1f} {o['alpha'] * DEG:+6.1f}  "
                  f"({w[0]:+5.1f} {w[1]:+5.1f} {w[2]:+5.1f})  {x[15]:6.1f}  {x[16]:4.0f} {x[17]:4.0f} {x[18]:4.0f} {x[19]:4.0f}  "
                  f"{o['lift_w']:7.0f} {o['cruise_w']:7.0f}  {100 * ac.lift_fraction:6.1f} {100 * ac.cruise_fraction:7.2f}")
    if csv:
        csv.close()
    print(f"done at {ac.t:.1f} s: " + (f"landed, sinking {sink:.2f} m/s at the touch, {math.hypot(ac.x[0], ac.x[1]):.2f} m from the take-off point"
                                        if mission.done and sink is not None else "NOT landed")
          + f"; lift battery used {100 * (1 - ac.lift_fraction):.0f} %, cruise battery {100 * (1 - ac.cruise_fraction):.1f} %; "
            f"{ac.guards['alpha']} steps outside the identified alpha (the hover and the transitions' ends), {ac.guards['beta']} outside the beta.")


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


class Scene:
    """A field, the pad it takes off from, the route at its height, the flown track, the wind
    as an arrow on the ground (1 m per m/s), and the aircraft."""

    CAMS = ("chase", "side", "high")

    def __init__(self, tp, f, a, w, h, headless):
        self.tp, self.f = tp, f
        self.canvas = tp.Canvas("threepp - Babyshark 260 VTOL", width=w, height=h, antialiasing=4, headless=headless,
                                vsync=not headless)
        r = self.r = tp.GLRenderer(self.canvas)
        r.tone_mapping = tp.ToneMapping.ACESFilmic
        r.tone_mapping_exposure = 1.0
        s = self.scene = tp.Scene()
        s.background = 0x9eb8d6
        s.set_fog(tp.Color(0x9eb8d6), 300.0, 2600.0)
        s.add(tp.HemisphereLight(0xdce6f2, 0x4a5a38, 1.1))
        sun = tp.DirectionalLight(0xfff4e0, 2.2)
        sun.position.set(-300.0, 500.0, 200.0)
        s.add(sun)
        ground = tp.Mesh(tp.PlaneGeometry(6000.0, 6000.0), material(tp, 0x5b7240, 0.95))
        ground.rotation.x = -math.pi / 2
        s.add(ground)
        pad = tp.Mesh(tp.CylinderGeometry(3.0, 3.0, 0.04, 48, 1), material(tp, 0x8a8d90, 0.8))
        pad.position.set(0.0, 0.02, 0.0)
        s.add(pad)
        rng = np.random.default_rng(7)
        tree = material(tp, 0x2f4526, 0.9)
        cone = tp.ConeGeometry(2.6, 10.0, 8, 1)
        for _ in range(520):
            n, e = rng.uniform(-900.0, 900.0), rng.uniform(-900.0, 900.0)
            if math.hypot(n, e) < 60.0:
                continue
            k = rng.uniform(0.7, 1.4)
            t = tp.Mesh(cone, tree)
            t.scale.set(k, k, k)
            t.position.set(*ned_to_world(n, e, -5.0 * k))
            s.add(t)
        hgt = ALTITUDE
        ln, _ = polyline(tp, [ned_to_world(n, e, -hgt) for n, e in f.route], 0xffffff)
        s.add(ln)
        pole, knob = material(tp, 0xe8e8e8, 0.6), material(tp, 0xffffff, 0.4)
        for n, e in f.route:
            p = tp.Mesh(tp.CylinderGeometry(0.25, 0.25, hgt, 8, 1), pole)
            p.position.set(*ned_to_world(n, e, -0.5 * hgt))
            s.add(p)
            b = tp.Mesh(tp.SphereGeometry(0.9, 16, 10), knob)
            b.position.set(*ned_to_world(n, e, -hgt))
            s.add(b)
        self.track_n = 0
        self.track, self.track_g = polyline(tp, [(0, 0, 0)], 0x2fd4ff, capacity=20000)
        self.track_buf = np.zeros((20000, 3), np.float32)
        s.add(self.track)
        if a.wind > 0.05:
            ang = math.radians(a.from_deg)
            d = np.array(ned_to_world(-math.cos(ang), -math.sin(ang), 0.0))
            s.add(tp.ArrowHelper(tp.Vector3(*map(float, d)), tp.Vector3(*ned_to_world(8.0, 8.0, -0.3)),
                                 a.wind * 1.0 + 2.0, tp.Color(0.95, 0.85, 0.2), 1.6, 1.2))
        mp = model_path()
        if mp is None:
            sys.exit("babyshark.glb is not built: blender --background --factory-startup --python build_babyshark_blender.py "
                     "-- --spec babyshark_spec.json --out babyshark.glb")
        self.model = tp.GLTFLoader().load(mp).scene
        s.add(self.model)
        self.vis = Visual(self.model)
        self.cam = tp.PerspectiveCamera(42.0, w / h, 0.1, 9000.0)
        self.mode = a.cam
        self.eye = None

    def sync(self):
        """The aircraft and the track from the flight's state."""
        ac = self.f.aircraft
        self.vis.pose(ac, h0=0.0)
        p = ac.world_position()
        if self.track_n < len(self.track_buf):
            if self.track_n == 0 or np.linalg.norm(self.track_buf[self.track_n - 1] - p) > 0.5:
                self.track_buf[self.track_n] = p
                self.track_n += 1
                self.track_g.update_attribute("position", self.track_buf)
                self.track_g.set_draw_range(0, self.track_n)

    def camera(self, smooth=None):
        tp, ac = self.tp, self.f.aircraft
        p = np.array(ac.world_position())
        psi = ac.x[11]
        fwd = np.array(ned_to_world(math.cos(psi), math.sin(psi), 0.0))        # along the nose, level
        right = np.array([-fwd[2], 0.0, fwd[0]])
        up = np.array([0.0, 1.0, 0.0])
        if self.mode == "chase":
            eye, at = p - 4.6 * fwd + 0.9 * right + 1.0 * up, p + 0.2 * fwd
        elif self.mode == "side":
            eye, at = p + 4.0 * right + 1.6 * fwd + 0.7 * up, p
        else:
            eye, at = p - 45.0 * fwd + 30.0 * up + 16.0 * right, p + 20.0 * fwd
        if smooth is not None and self.eye is not None:
            eye = self.eye + (eye - self.eye) * (1.0 - math.exp(-smooth))
        self.eye = eye
        self.cam.position.set(*map(float, eye))
        self.cam.look_at(tp.Vector3(*map(float, at)))

    def render(self):
        self.r.render(self.scene, self.cam)


def stills(tp, f, a, times, paths, w, h):
    sc = Scene(tp, f, a, w, h, headless=True)
    ac = f.aircraft
    for t, path in sorted(zip(times, paths)):
        while ac.t < t - 1e-9 and not f.mission.done:
            f.advance(min(1.0 / 60.0, t - ac.t))
            sc.sync()
        sc.sync()
        sc.camera()
        sc.render()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sc.r.save_frame(path)
        print(f"[babyshark] t {ac.t:.2f} s, {f.mission.phase}, {f.ap.mode}, h {ac.h:.1f} m, Va {ac.out.get('Va', 0.0):.1f} m/s -> {path}")


def window(tp, f, a, w, h):
    sc = Scene(tp, f, a, w, h, headless=False)
    clock = tp.Clock()
    state = {"debt": 0.0, "c": False}

    def animate():
        dt = min(clock.get_delta(), 0.1)
        state["debt"] += dt
        n = int(state["debt"] / f.aircraft.dt)
        state["debt"] -= n * f.aircraft.dt
        for _ in range(n):
            if not f.mission.done:
                f.step()
        c = sc.canvas.is_key_down("C")
        if c and not state["c"]:
            sc.mode = Scene.CAMS[(Scene.CAMS.index(sc.mode) + 1) % len(Scene.CAMS)]
            sc.eye = None
        state["c"] = c
        sc.sync()
        sc.camera(smooth=dt / 0.25)
        sc.render()
    sc.canvas.animate(animate)


def main(argv=None):
    a = parse(argv)
    spec = load_spec(a.spec)
    if a.checks:
        which = sorted(CHECKS) if a.checks == "all" else [int(k) for k in a.checks.split(",")]
        for k in which:
            CHECKS[k](spec)
    if a.shot is not None or a.stills or a.window:
        sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))       # python/ (threepp)
        import threepp as tp
        w, h = (int(v) for v in a.size.lower().split("x"))
        if a.window:
            window(tp, Flight(a, spec), a, w, h)
        else:
            times = [a.shot] if a.shot is not None else [float(t) for t in a.stills.split(",")]
            paths = [a.out] if a.shot is not None else [os.path.join(a.out_dir, f"babyshark_{t:05.1f}s.png") for t in times]
            stills(tp, Flight(a, spec), a, times, paths, w, h)
    elif a.telemetry or not a.checks:
        telemetry(a, spec)


if __name__ == "__main__":
    main()
