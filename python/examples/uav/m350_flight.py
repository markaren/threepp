"""
The Matrice 350 RTK's flight model on the command line: its checks and a short flight.

The aircraft, the wind, the autopilot and the mission are rigs/m350_rig.py; this file runs them
without a window or a renderer (numpy only). The film scene that flies the model is elsewhere.

    python m350_flight.py --checks                    # the eight checks, each line numbered
    python m350_flight.py --checks 3,5                # some of them
    python m350_flight.py --telemetry                 # take off, 60 m out, a slow pass, back, land: a line a second
    python m350_flight.py --telemetry --wind 8 --from 250 --turbulence 1.5 --csv m350.csv

Read the numbers for what they are. The product page gives masses, limits and a flight time;
the thrust, torque, drag and motor figures are ASSUMED (m350_spec.json), two of them CALIBRATED
so the hover power and the tilt at top speed match the product page, which is therefore no
validation. The autopilot is OURS, with exact state feedback (no estimator, no sensor noise).
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rigs"))
from m350_rig import M350, Autopilot, Gimbal, Mission, Wind, load_spec    # noqa: E402

DEG = 180.0 / math.pi


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", nargs="?", const="all", default=None, help="all, or a comma-separated list of 1..8")
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--wind", type=float, default=6.0, help="wind speed 10 m up, m/s")
    ap.add_argument("--from", dest="from_deg", type=float, default=270.0, help="the direction the wind blows from, deg (0 north, 90 east)")
    ap.add_argument("--turbulence", type=float, default=1.0, help="scales the Dryden sigmas")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-payload", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--spec", default=None, help="another spec file in place of m350_spec.json")
    return ap.parse_args(argv)


def line(k, text):
    print(f"[{k}] {text}")


def fly(m, ap, seconds, record=None, mission=None):
    for _ in range(int(round(seconds / m.dt))):
        if mission is not None:
            mission.update(m, ap)
        ap.update()
        m.step()
        if record is not None:
            record(m)


def hover(spec, payload=True, wind=None, h=30.0, ground=None):
    m = M350(spec, payload=payload, wind=wind, ground=ground)
    m.set_state(h=h)
    return m, Autopilot(m)


# --------------------------------------------------------------------------- #
#  1 mass and hover, 2 calibration
# --------------------------------------------------------------------------- #
def check_mass(spec):
    for payload in (False, True):
        m = M350(spec, payload=payload)
        I, c = m.I, m.cg_model
        line(1, f"{'with' if payload else 'without'} the payload: mass {m.m:.3f} kg, centre of gravity in the .glb's frame "
                f"({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}) m; inertia about it, body axes: Ixx {I[0, 0]:.4f} Iyy {I[1, 1]:.4f} "
                f"Izz {I[2, 2]:.4f} Ixz {I[0, 2]:+.4f} kg m^2")
    m = M350(spec, payload=False)
    line(1, f"rotor: kT {m.kT_rho * m.rho0:.4e} N s^2, kz {m.kz_rho * m.rho0:.4e} N s^2/m (solidity {m.solidity:.4f}, "
            f"lambda_h {m.lambda_h:.4f}), kQ {m.kQ_rho * m.rho0:.4e} N m s^2; idle {m.w_idle:.1f} rad/s, "
            f"maximum {m.w_max:.1f} rad/s ({m.spec['propulsion']['max_thrust_n']:.0f} N)")
    target = spec["aircraft"]["limits"]["max_flight_time_min"]
    for mass, what in ((None, "6.47 kg, the product page's point"), (M350(spec, payload=True, calibrate=False).m, "with the payload")):
        hp = m.hover_power(mass)
        line(1, f"hover at {hp['mass']:.3f} kg ({what}), sea level, still air: {hp['omega']:.1f} rad/s = {hp['rpm']:.0f} rpm, "
                f"tip {hp['tip_speed']:.1f} m/s, {hp['thrust_per_rotor']:.2f} N a rotor, induced {hp['induced_w']:.0f} W, "
                f"shaft {hp['shaft_w']:.0f} W, battery {hp['battery_w']:.0f} W, {hp['endurance_min']:.1f} min from "
                f"{m.battery_wh:.1f} Wh")
    hp = m.hover_power()
    line(1, f"against the product page's {target:.0f} min: {hp['endurance_min']:.1f} min. CALIBRATED, not validated: the figure "
            f"of merit ({m.fm}) and the drive efficiency ({m.eta}) are ASSUMED and were chosen together so it comes out "
            f"({m.battery_wh:.1f} Wh / {target:.0f} min = {m.battery_wh / target * 60.0:.0f} W)")


def check_calibration(spec):
    m = M350(spec, payload=False)
    c = m.calibration
    line(2, f"calibrate: kh = {c['kh']:.4e} N s^2/m at sea level, so {c['speed']:.0f} m/s needs {c['tilt'] * DEG:.2f} deg of tilt at "
            f"{c['mass']:.2f} kg (the product page's {c['tilt_target'] * DEG:.0f} deg; residual {c['residual']:.1e}). Drag along the "
            f"flight there: airframe {c['body_drag']:.1f} N, rotors {c['rotor_drag']:.1f} N = {100 * c['rotor_share']:.0f} % "
            f"the rotors'. CALIBRATED, not validated" + (f". NOTE: {c['note']}" if "note" in c else ""))
    if not c["saturated"]:
        kT0 = m.kT_rho * m.rho0
        line(2, f"that point asks {c['omega']:.0f} rad/s of the rear rotors ({kT0 * c['omega'] ** 2:.1f} N static), inside the "
                f"{c['omega_max']:.0f} rad/s of the spec's ASSUMED maximum thrust ({m.spec['propulsion']['max_thrust_n']:.0f} N): "
                f"{100 * (c['omega_max'] ** 2 / c['omega'] ** 2 - 1):.0f} % of thrust left for control there")
    if c["saturated"]:
        lo, hi = 0.0, c["speed"]
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if max(m.trim((mid, 0.0, 0.0))["omega"]) > m.w_max:
                hi = mid
            else:
                lo = mid
        line(2, f"FINDING: that point asks {c['omega']:.0f} rad/s of the rear rotors, past the {c['omega_max']:.0f} rad/s of the "
                f"spec's ASSUMED maximum thrust: the thrust lost to the air through the tilted discs (kz) is what costs it. "
                f"Inside its rotor speeds the model's steady level flight ends at {lo:.1f} m/s; the trim rows below ignore "
                f"the limit and flag it")
    for v in (5.0, 10.0, 15.0, 23.0):
        s = m.trim((v, 0.0, 0.0))
        line(2, f"steady level flight at {v:4.1f} m/s, still air, {m.m:.2f} kg: tilt {s['tilt'] * DEG:5.2f} deg, rotors "
                f"{min(s['omega']):.0f}..{max(s['omega']):.0f} rad/s, battery {s['battery_w']:.0f} W"
                + ("  (past the rotors' maximum)" if s["saturated"] else ""))
    mp = M350(spec, payload=True)
    for w in (5.0, 8.0, 12.0):
        a, b = mp.trim((w, 0.0, 0.0)), mp.trim((0.0, w, 0.0))
        line(2, f"holding position in a steady {w:4.1f} m/s, {mp.m:.3f} kg: nose into it tilt {a['tilt'] * DEG:5.2f} deg, battery "
                f"{a['battery_w']:.0f} W; beam on tilt {b['tilt'] * DEG:5.2f} deg, {b['battery_w']:.0f} W "
                f"(hover in still air {mp.hover_power()['battery_w']:.0f} W)"
                + ("  (past the rotors' maximum)" if a["saturated"] or b["saturated"] else ""))


# --------------------------------------------------------------------------- #
#  3 steps
# --------------------------------------------------------------------------- #
def step_metrics(t, y, y0, y1, band=0.02):
    """Rise time (10 to 90 %), overshoot (% of the step), settling time (inside +-band of the step for good)."""
    s = (np.asarray(y) - y0) / (y1 - y0)
    t = np.asarray(t)
    t10 = t[np.argmax(s >= 0.1)]
    t90 = t[np.argmax(s >= 0.9)] if np.any(s >= 0.9) else float("nan")
    out = np.nonzero(np.abs(s - 1.0) > band)[0]
    settle = t[min(out[-1] + 1, len(t) - 1)] if len(out) else 0.0
    return t90 - t10, 100.0 * max(float(np.max(s)) - 1.0, 0.0), settle


def check_steps(spec):
    cases = (("5 m sideways", "e"), ("2 m up", "h"), ("90 deg of yaw", "yaw"), ("5 m/s of velocity north", "v"))
    lim = spec["aircraft"]["limits"]
    for name, kind in cases:
        m, ap = hover(spec)
        fly(m, ap, 2.0)
        if kind == "e":
            ap.command(position=(0.0, 5.0, 30.0))
            y0, y1 = 0.0, 5.0
            read = lambda m: m.x[1]
        elif kind == "h":
            ap.command(position=(0.0, 0.0, 32.0))
            y0, y1 = 30.0, 32.0
            read = lambda m: -m.x[2]
        elif kind == "yaw":
            ap.command(yaw=math.pi / 2.0)
            y0, y1 = 0.0, 90.0
            read = lambda m: m.euler()[2] * DEG
        else:
            ap.command(position=None, velocity=(5.0, 0.0, 0.0))
            y0, y1 = 0.0, 5.0
            read = lambda m: m.x[3]
        ts, ys, peak = [], [], [0.0, 0.0, 0.0, 0.0, 0.0]
        t0 = m.t

        def rec(m):
            ts.append(m.t - t0)
            ys.append(read(m))
            x = m.x
            peak[0] = max(peak[0], m.tilt())
            peak[1] = max(peak[1], abs(x[10]), abs(x[11]))
            peak[2] = max(peak[2], abs(x[12]))
            peak[3] = max(peak[3], abs(-x[2] - (32.0 if kind == "h" else 30.0)) if kind != "h" else 0.0)
            peak[4] = max(peak[4], max(x[13:17]))
        fly(m, ap, 14.0, rec)
        rise, over, settle = step_metrics(ts, ys, y0, y1)
        line(3, f"{name}: rise {rise:.2f} s (10 to 90 %), overshoot {over:.1f} %, inside 2 % after {settle:.2f} s; peak tilt "
                f"{peak[0] * DEG:.1f} deg (limit {lim['max_tilt_forward_vision_deg']:.0f}), roll/pitch rate {peak[1] * DEG:.0f} deg/s "
                f"(product {lim['max_pitch_rate_dps']:.0f}), yaw rate {peak[2] * DEG:.0f} deg/s (limit {lim['max_yaw_rate_dps']:.0f}), "
                f"height within {peak[3]:.3f} m, rotors to {peak[4]:.0f} rad/s")


# --------------------------------------------------------------------------- #
#  4 hover in turbulence, 5 a gust
# --------------------------------------------------------------------------- #
def check_wind_filter(spec):
    """The Dryden filters alone: 600 s at a fixed point 10 m up in a steady 8 m/s."""
    dt, T, h, U, more = 1.0 / 300.0, 600.0, 10.0, 8.0, 12
    n = int(round(T / dt))

    def record(seed):
        w = Wind(U, 270.0, seed=seed, spec=spec)
        g = np.empty((n, 3))
        for i in range(n):
            w.advance(dt, 0.0, 0.0, h)
            g[i] = w.uvw
        return w, g

    def autocorrelation(y):
        f = np.fft.rfft(y, 2 * n)
        return np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)

    def e_time(ac):
        return float(np.argmax(ac < ac[0] / math.e)) * dt
    # the second-order filter on its own: its unit-variance output over many correlation times
    rng, th, nn, z1, z2, acc = np.random.default_rng(7), 0.05, 1000000, 0.0, 0.0, 0.0
    for x1, x2 in rng.standard_normal((nn, 2)).tolist():
        z1, z2 = Wind._second_order(z1, z2, th, x1, x2)
        acc += (0.5 * z1 + 0.8660254037844386 * z2) ** 2
    line(4, f"the lateral / vertical shaping filter alone, {nn} steps at V dt / L = {th} ({nn * th:.0f} L/V): variance of its "
            f"unit output {acc / nn:.4f} (1; scatter about {math.sqrt(1.25 / (nn * th)):.4f})")
    w, g = record(1)
    records = [g] + [record(s)[1] for s in range(2, 2 + more)]
    # the 1/e time of each autocorrelation: L / V for the first-order filter; the second-order
    # one's is exp(-x) (1 - x / 2), x = V t / L, which falls to 1/e at x = 0.625
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if math.exp(-mid) * (1.0 - 0.5 * mid) > math.exp(-1.0) else (lo, mid)
    theory = [w.length[0] / w.V, lo * w.length[1] / w.V, lo * w.length[2] / w.V]
    for k, name in enumerate(("longitudinal", "lateral", "vertical")):
        sd = math.sqrt(float((g[:, k] ** 2).mean()))
        sd_all = math.sqrt(float(np.mean([(r[:, k] ** 2).mean() for r in records])))
        tau = e_time(autocorrelation(g[:, k]))
        tau_all = e_time(np.mean([autocorrelation(r[:, k]) for r in records], axis=0))
        samp = 100.0 * math.sqrt(theory[k] / (2.0 * T))
        line(4, f"Dryden {name}, {T:.0f} s at a point {h:.0f} m up in {U:.0f} m/s (W20 {w.w20(0, 0):.2f} m/s, L {w.length[k]:.1f} m, "
                f"L/V {w.length[k] / w.V:.2f} s): rms {sd:.3f} m/s against sigma {w.sigma[k]:.3f} ({100 * (sd / w.sigma[k] - 1):+.1f} %; "
                f"asked: 10 %; one {T:.0f} s record's own scatter is about +-{samp:.0f} %), over {more + 1} seeds {sd_all:.3f} "
                f"({100 * (sd_all / w.sigma[k] - 1):+.1f} %); autocorrelation down to 1/e in {tau:.2f} s against {theory[k]:.2f} s "
                f"({100 * (tau / theory[k] - 1):+.0f} %), the {more + 1} seeds' mean in {tau_all:.2f} s ({100 * (tau_all / theory[k] - 1):+.0f} %)")


def check_hover(spec):
    check_wind_filter(spec)
    acc = spec["aircraft"]["limits"]["hover_accuracy_rtk_m"]
    for U in (4.0, 8.0, 12.0):
        wind = Wind(U, 270.0, turbulence=1.0, seed=1, spec=spec)
        m, ap = hover(spec, wind=wind)
        fly(m, ap, 20.0)                                   # the velocity integral finds the lean
        rows = []
        fly(m, ap, 120.0, lambda m: rows.append((m.x[0], m.x[1], -m.x[2] - 30.0, m.tilt(), min(m.x[13:17]), max(m.x[13:17]),
                                                 m.out["battery_w"])))
        a = np.array(rows)
        rms, mx = np.sqrt((a[:, :3] ** 2).mean(axis=0)), np.abs(a[:, :3]).max(axis=0)
        line(4, f"hover 30 m up, 120 s in {U:.0f} m/s from the west (W at 30 m {wind.profile(30.0):.1f} m/s, sigma "
                f"{wind.sigma[0]:.2f}/{wind.sigma[1]:.2f}/{wind.sigma[2]:.2f} m/s): position error rms N {rms[0]:.3f} E {rms[1]:.3f} "
                f"H {rms[2]:.3f} m, max N {mx[0]:.3f} E {mx[1]:.3f} H {mx[2]:.3f} m; tilt {a[:, 3].mean() * DEG:.1f} +- "
                f"{a[:, 3].std() * DEG:.1f} deg (max {a[:, 3].max() * DEG:.1f}); rotors {a[:, 4].min():.0f}..{a[:, 5].max():.0f} rad/s; "
                f"battery {a[:, 6].mean():.0f} W")
    line(4, f"the product page's hover accuracy is +-{acc['horizontal']} m horizontal, +-{acc['vertical']} m vertical with RTK, and it "
            f"states no wind for that figure. The numbers above are with EXACT state feedback (no estimator, no sensor noise): "
            f"what the airframe and this controller do to the gusts, a floor under what a real aircraft holds")


def check_gust(spec):
    A, dur = 5.0, 2.0
    for payload in (True,):
        m = M350(spec, payload=payload)
        t0 = 2.0
        m.wind = lambda t, n, e, d: (0.0, 0.5 * A * (1.0 - math.cos(2.0 * math.pi * (t - t0) / dur)) if t0 < t < t0 + dur else 0.0, 0.0)
        m.set_state(h=30.0)
        ap = Autopilot(m)
        rows = []
        fly(m, ap, 20.0, lambda m: rows.append((m.t, m.x[0], m.x[1], -m.x[2] - 30.0, m.tilt())))
        a = np.array(rows)
        dist = np.hypot(a[:, 1], a[:, 2])
        k = int(np.argmax(dist))

        def back(r):
            out = np.nonzero(dist > r)[0]
            if not len(out):
                return f"never more than {r} m off"
            return f"back inside {r} m for good {a[min(out[-1] + 1, len(a) - 1), 0] - t0:.2f} s after the gust began"
        line(5, f"a one-minus-cosine gust of {A:.0f} m/s over {dur:.0f} s from the west on a hover (heading north, the gust on the "
                f"beam): pushed {dist[k]:.3f} m ({a[k, 2]:+.3f} m east) at {a[k, 0] - t0:.2f} s, height within "
                f"{np.abs(a[:, 3]).max():.3f} m, peak tilt {a[:, 4].max() * DEG:.1f} deg; {back(0.1)}; {back(0.02)}")


# --------------------------------------------------------------------------- #
#  6 take-off and landing
# --------------------------------------------------------------------------- #
def check_takeoff(spec):
    m = M350(spec)
    m.place_on_ground(0.0, 0.0)
    ap = Autopilot(m)
    mission = Mission([dict(kind="takeoff", height=10.0, speed=2.0, hold=5.0), dict(kind="land")], spec)
    rest0 = m.height_above_ground()
    rows = []
    while not mission.done and m.t < 120.0:
        mission.update(m, ap)
        ap.update()
        m.step()
        rows.append((m.t, m.height_above_ground(), m.x[5], math.hypot(m.x[0], m.x[1]), m.tilt(), 1.0 if mission.phase == "landed" else 0.0))
    a = np.array(rows)
    td = mission.touchdown
    after = a[a[:, 0] > td["t"]]
    line(6, f"take-off from rest on the ground (skids {rest0 * 1000:+.1f} mm: the springs' sag) to 10 m, 5 s hold, land: top at "
            f"{a[:, 1].max():.2f} m above the ground, horizontal wander {a[:, 3].max():.3f} m; touchdown at {td['t']:.1f} s sinking "
            f"{td['sink']:.2f} m/s; bounce {max(after[:, 1].max(), 0.0) * 1000:.1f} mm (after it the lowest skid point is never above "
            f"{after[:, 1].max() * 1000:+.1f} mm: it stays in the springs); final rest: skids {a[-1, 1] * 1000:+.1f} mm, speed {abs(a[-1, 2]):.4f} m/s, tilt {a[-1, 4] * DEG:.2f} deg, rotors at "
            f"idle {m.x[13]:.1f} rad/s")
    slope = math.tan(math.radians(5.0))
    m = M350(spec, ground=lambda n, e: slope * e)
    m.place_on_ground(0.0, 0.0)
    ap = Autopilot(m)
    ap.idle = True
    p0 = (m.x[0], m.x[1], m.x[2])
    fly(m, ap, 30.0)
    slip = math.sqrt(sum((a - b) ** 2 for a, b in zip(p0, m.x[:3])))
    line(6, f"on a 5 deg slope, rotors at idle ({m.x[13]:.1f} rad/s, {4 * m.out['thrust'][0]:.1f} N of thrust), 30 s: moved "
            f"{slip * 1000:.2f} mm, speed {math.sqrt(m.x[3] ** 2 + m.x[4] ** 2 + m.x[5] ** 2) * 1000:.4f} mm/s, tilt {m.tilt() * DEG:.2f} deg "
            f"(friction {m.spec['ground']['friction']}, ASSUMED)")


# --------------------------------------------------------------------------- #
#  7 integrity, 8 speed
# --------------------------------------------------------------------------- #
def flight(spec, seconds, seed=1, wind_speed=8.0):
    """A flight from a hover 30 m up: a climbing leg along the track, then a leg back turning to
    a heading. Returns the aircraft, every step's state, and the energy and the power of the
    non-conservative forces and moments at every step's start."""
    wind = Wind(wind_speed, 250.0, turbulence=1.0, seed=seed, spec=spec) if wind_speed else None
    m, ap = hover(spec, wind=wind)
    mission = Mission([dict(pos=(20.0, 8.0, 38.0), speed=5.0, yaw="track", hold=1.0),
                       dict(pos=(0.0, 0.0, 30.0), speed=5.0, yaw=90.0, hold=1.0)], spec)
    trace, energy, power = [], [], []
    for _ in range(int(round(seconds / m.dt))):
        mission.update(m, ap)
        ap.update()
        energy.append(m.energy())
        m.step()
        power.append(m.out["power_nc"])                    # evaluated at the step's start
        trace.append(tuple(m.x))
    return m, trace, np.array(energy), np.array(power)


def check_integrity(spec):
    _, a, _, _ = flight(spec, 20.0)
    _, b, _, _ = flight(spec, 20.0)
    same = all(x == y for x, y in zip(a, b))
    line(7, f"two runs of a 20 s flight in 8 m/s of turbulent wind ({len(a)} steps, 17 states): "
            f"{'bit-identical at every step' if same else 'DIFFERENT'}")
    m = M350(spec, rho=0.0)
    w = m.hover_speed(rho=m.rho0)
    m.g = 0.0
    m.set_state(h=30.0, rates=(1.1, -0.7, 0.4), rotor_speed=w)
    e0, h0 = m.energy(), m.angular_momentum_ned()
    de = dh = 0.0
    for i in range(int(round(60.0 / m.dt))):
        m.step()
        if i % 300 == 299:
            de = max(de, abs(m.energy() / e0 - 1.0))
            dh = max(dh, float(np.linalg.norm(m.angular_momentum_ned() - h0) / np.linalg.norm(h0)))
    line(7, f"a torque-free tumble, no air, no gravity, rotors held at {w:.0f} rad/s (their momentum {m.Jr * w * 4:.2f} N m s "
            f"cancels in pairs; one pair alone: see below), 60 s from (63, -40, 23) deg/s: energy within {de:.1e}, angular "
            f"momentum (airframe and rotors) within {dh:.1e} (asked: 1e-8)")
    m.set_state(h=30.0, rates=(1.1, -0.7, 0.4), rotor_speed=w)
    m.x[14] = m.x[16] = 0.0                                # the two clockwise rotors stopped: a net rotor momentum
    m.cmd = list(m.x[13:17])
    e0, h0 = m.energy(), m.angular_momentum_ned()
    de = dh = 0.0
    for i in range(int(round(60.0 / m.dt))):
        m.step()
        if i % 300 == 299:
            de = max(de, abs(m.energy() / e0 - 1.0))
            dh = max(dh, float(np.linalg.norm(m.angular_momentum_ned() - h0) / np.linalg.norm(h0)))
    line(7, f"the same with only the two counter-clockwise rotors turning (a net rotor momentum of {2 * m.Jr * w:.2f} N m s against "
            f"the airframe's {float(np.linalg.norm(m.I @ np.array([1.1, -0.7, 0.4]))):.2f}: the gyroscopic term at work): energy "
            f"within {de:.1e}, angular momentum within {dh:.1e}")
    m, _, energy, power = flight(spec, 30.0, wind_speed=0.0)
    # trapezoidal rule over the steps' starts: the work up to step k against the energy's change there
    work = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * m.dt)])
    de = energy - energy[0]
    k = int(np.argmax(np.abs(de)))
    worst = float(np.max(np.abs(work - de)))
    line(7, f"a 30 s flight in still air (a climbing leg, a leg back with a yaw): the energy "
            f"(kinetic plus potential) is at most {de[k]:+.2f} J from its start, at {k * m.dt:.1f} s, where the work of the rotors', "
            f"the drag's and the reaction torques' forces and moments is {work[k]:+.2f} J ({100 * abs(work[k] - de[k]) / abs(de[k]):.4f} % "
            f"apart; asked: 0.1 %); the two are never more than {worst:.3f} J apart over the flight")


def check_speed(spec):
    wind = Wind(8.0, 270.0, turbulence=1.0, seed=1, spec=spec)
    m, ap = hover(spec, wind=wind)
    fly(m, ap, 1.0)
    t0 = time.perf_counter()
    fly(m, ap, 30.0)
    el = time.perf_counter() - t0
    line(8, f"a hover in turbulence, autopilot and wind included: 30 s simulated in {el:.2f} s of wall time = {30.0 / el:.1f} x "
            f"real time ({el / 9000 * 1e6:.0f} us a step; asked: more than 10 x)")


CHECKS = {1: check_mass, 2: check_calibration, 3: check_steps, 4: check_hover, 5: check_gust, 6: check_takeoff,
          7: check_integrity, 8: check_speed}


# --------------------------------------------------------------------------- #
#  The telemetry flight
# --------------------------------------------------------------------------- #
def telemetry(a, spec):
    wind = Wind(a.wind, a.from_deg, turbulence=a.turbulence, seed=a.seed, spec=spec) if a.wind > 0.0 else None
    m = M350(spec, payload=not a.no_payload, wind=wind)
    m.place_on_ground(0.0, 0.0)
    ap, gimbal = Autopilot(m), Gimbal(spec)
    poi = (68.0, 6.0, 12.0)
    names = ("take-off to 15 m", "60 m north at 6 m/s", "slow pass, 12 m east at 1.2 m/s, nose on a point 8 m off",
             "back at 6 m/s", "land")
    mission = Mission([dict(kind="takeoff", height=15.0, speed=2.5, hold=2.0),
                       dict(pos=(60.0, 0.0, 15.0), speed=6.0, yaw="track", hold=1.0),
                       dict(pos=(60.0, 12.0, 15.0), speed=1.2, yaw="poi", poi=poi, hold=5.0),
                       dict(pos=(0.0, 0.0, 15.0), speed=6.0, yaw="track", hold=2.0),
                       dict(kind="land")], spec)
    legs = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in names]      # per leg: tilt, tilt rate, yaw rate, setpoint error, tilt command rate
    tilt_c = None
    print(f"M350, {m.m:.3f} kg, wind {a.wind:.1f} m/s from {a.from_deg:.0f} deg (10 m up), turbulence x{a.turbulence}, seed {a.seed}. "
          f"Autopilot OURS, exact state feedback.")
    print("    t  leg phase        N       E      H     Vg   roll  pitch    yaw  tilt   wind(N,E,D)        rotors rad/s        "
          "batt W   batt %  gimbal pan/tilt")
    csv = open(a.csv, "w", encoding="utf-8") if a.csv else None
    if csv:
        csv.write("t,leg,n,e,h,vn,ve,vd,roll,pitch,yaw,p,q,r,w_fr,w_fl,w_rl,w_rr,wind_n,wind_e,wind_d,battery_w,battery_fraction,"
                  "sp_n,sp_e,sp_h,gimbal_pan,gimbal_roll,gimbal_tilt\n")
    per = int(round(1.0 / m.dt))
    worst = 0.0
    while not mission.done and m.t < 300.0:
        mission.update(m, ap)
        ap.update()
        m.step()
        gimbal.update(m, mission.look)
        x = m.x
        ro, pi, ya = m.euler()
        sp = mission.setpoint or (x[0], x[1], -x[2])
        if mission.phase in ("fly", "settle", "hold"):
            err = math.sqrt((x[0] - sp[0]) ** 2 + (x[1] - sp[1]) ** 2 + (-x[2] - sp[2]) ** 2)
            worst = max(worst, err)
            g = legs[min(mission.leg, len(legs) - 1)]
            g[0], g[1], g[2], g[3] = max(g[0], m.tilt()), max(g[1], math.hypot(x[10], x[11])), max(g[2], abs(x[12])), max(g[3], err)
            if tilt_c is not None:
                g[4] = max(g[4], abs(ap.tilt_c - tilt_c) / m.dt)
            tilt_c = ap.tilt_c
        else:
            tilt_c = None
        w = m.wind_ned()
        if csv:
            csv.write(",".join(f"{v:.6g}" for v in (m.t, mission.leg, x[0], x[1], -x[2], x[3], x[4], x[5], ro, pi, ya, x[10], x[11], x[12],
                                                    x[13], x[14], x[15], x[16], w[0], w[1], w[2], m.out["battery_w"], m.battery_fraction,
                                                    sp[0], sp[1], sp[2], gimbal.pan, gimbal.roll, gimbal.tilt)) + "\n")
        if m.steps % per == 0:
            print(f"{m.t:5.0f}  {mission.leg:3d} {mission.phase:8s} {x[0]:7.2f} {x[1]:7.2f} {-x[2]:6.2f} {math.hypot(x[3], x[4]):6.2f} "
                  f"{ro * DEG:+6.1f} {pi * DEG:+6.1f} {ya * DEG:+6.1f} {m.tilt() * DEG:5.1f}  ({w[0]:+5.1f} {w[1]:+5.1f} {w[2]:+5.1f})  "
                  f"{x[13]:4.0f} {x[14]:4.0f} {x[15]:4.0f} {x[16]:4.0f}  {m.out['battery_w']:6.0f}  {100 * m.battery_fraction:6.2f}  "
                  f"{gimbal.pan * DEG:+6.1f} {gimbal.tilt * DEG:+6.1f}")
    if csv:
        csv.close()
    for name, g in zip(names, legs):
        if g[0] > 0.0:
            print(f"  {name}: worst tilt {g[0] * DEG:.1f} deg, tilt rate {g[1] * DEG:.1f} deg/s, tilt command rate {g[4] * DEG:.1f} deg/s, "
                  f"yaw rate {g[2] * DEG:.1f} deg/s, {g[3]:.2f} m from the setpoint")
    td = mission.touchdown
    print(f"done at {m.t:.1f} s: {'landed' if td else 'NOT landed'}"
          + (f", touchdown sinking {td['sink']:.2f} m/s, {math.hypot(m.x[0], m.x[1]):.2f} m from the take-off point" if td else "")
          + f"; farthest from the setpoint in flight {worst:.2f} m; battery used {100 * (1 - m.battery_fraction):.2f} % "
            f"({(1 - m.battery_fraction) * m.battery_wh:.1f} Wh)")


def main(argv=None):
    a = parse(argv)
    spec = load_spec(a.spec)
    if a.checks:
        which = sorted(CHECKS) if a.checks == "all" else [int(k) for k in a.checks.split(",")]
        for k in which:
            CHECKS[k](spec)
    if a.telemetry or not a.checks:
        telemetry(a, spec)


if __name__ == "__main__":
    main()
