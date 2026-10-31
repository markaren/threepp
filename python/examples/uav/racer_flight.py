"""
The racing quad's flight on the command line, in stills and in a window: its checks and a flown lap.

The aircraft and its flight controller are rigs/racer_rig.py, the wind m350_rig's. --checks and
--telemetry run them without a renderer (numpy only).

    python racer_flight.py --checks                   # the nine checks, each line numbered
    python racer_flight.py --checks 2,4               # some of them
    python racer_flight.py --telemetry                # off the pad, three laps on AUTO: a line a second
    python racer_flight.py --telemetry --wind 8 --from 250 --turbulence 1.5 --csv racer.csv
    python racer_flight.py --stills 0.5,6,14 --out-dir out/racer [--cam chase|side|high]
    python racer_flight.py --window                   # the same flight in a window; C cycles the camera

Read the numbers for what they are: there is no product behind this aircraft, the spec's figures
are the class's, tuned for a simulator, and nothing is checked against a flight log or by a pilot.
The flight controller is OURS, with exact state feedback (no estimator, no sensor noise).
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rigs"))
from racer_rig import (FlightController, Line, Racer, Sticks, Visual, Wind, load_spec, model_path,  # noqa: E402
                       ned_to_world)

DEG = 180.0 / math.pi
COURSE = ((0.0, 0.0, 4.0), (90.0, 10.0, 6.0), (150.0, 60.0, 4.0), (130.0, 150.0, 12.0), (40.0, 175.0, 5.0), (-50.0, 100.0, 8.0))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", nargs="?", const="all", default=None, help="all, or a comma-separated list of 1..9")
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--wind", type=float, default=4.0, help="wind speed 10 m up, m/s")
    ap.add_argument("--from", dest="from_deg", type=float, default=250.0, help="the direction the wind blows from, deg (0 north, 90 east)")
    ap.add_argument("--turbulence", type=float, default=1.0, help="scales the Dryden sigmas")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--spec", default=None, help="another spec file in place of racer_spec.json")
    ap.add_argument("--window", action="store_true", help="fly it in a window; C cycles the camera")
    ap.add_argument("--stills", default=None, help="comma-separated times, s")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--cam", choices=("chase", "side", "high"), default="chase")
    return ap.parse_args(argv)


def line(k, text):
    print(f"[{k}] {text}")


def fly(r, fc, seconds, sticks=None, record=None):
    for _ in range(int(round(seconds / r.dt))):
        if sticks is not None:
            fc.sticks = sticks
        fc.update()
        r.step()
        if record is not None:
            record(r)


def high(spec, assist, h=900.0, ground=None, **state):
    """In the air, far from anything, on an assist."""
    r = Racer(spec, ground=ground)
    r.set_state(h=h, **state)
    return r, FlightController(r, assist=assist)


def hover_stick(r, fc):
    return r.hover_share() ** (1.0 / fc.th_power)


# --------------------------------------------------------------------------- #
#  1 the airframe, 2 the rate loop
# --------------------------------------------------------------------------- #
def check_airframe(spec):
    r, fc = high(spec, "acro")
    w = math.sqrt(r.hover_share()) * r.w_max
    line(1, f"{r.m:.2f} kg, {2.0 * math.sqrt(2.0) * r.arm * 1000:.0f} mm between diagonal motor axes; {4.0 * r.full:.1f} N of static thrust, "
            f"{r.twr:.0f} to 1; hover at {100 * r.hover_share():.1f} % of it, stick {hover_stick(r, fc):.3f}, the rotors at "
            f"{w * 60.0 / (2.0 * math.pi):.0f} rpm (full {r.w_max * 60.0 / (2.0 * math.pi):.0f})")


def step_response(spec, rate, throttle, sticks, target):
    """(seconds to 90 % of the rate, overshoot) of a full stick from a hover at this throttle."""
    r, fc = high(spec, "acro")
    fly(r, fc, 0.3, Sticks(throttle=throttle))
    t90, peak = None, 0.0
    fc.sticks = Sticks(throttle=throttle, **sticks)
    for k in range(int(0.6 / r.dt)):
        fc.update()
        r.step()
        v = abs(r.x[rate])
        peak = max(peak, v)
        if t90 is None and v > 0.9 * target:
            t90 = (k + 1) * r.dt
    return t90 if t90 is not None else float("nan"), peak / target - 1.0


def check_rates(spec):
    r, fc = high(spec, "acro")
    hover = hover_stick(r, fc)
    t0, over = step_response(spec, 10, hover, {"roll": 1.0}, fc.r_max)
    line(2, f"a full roll stick at the hover: {fc.r_max * DEG:.0f} deg/s, 90 % in {t0 * 1000:.0f} ms, {over * 100:.0f} % overshoot")
    t1, _ = step_response(spec, 10, 1.0, {"roll": 1.0}, fc.r_max)
    t2, _ = step_response(spec, 10, 0.0, {"roll": 1.0}, fc.r_max)
    line(2, f"at full throttle {t1 * 1000:.0f} ms (the rotors have no headroom); at zero throttle {t2 * 1000:.0f} ms (airmode adds thrust to find it)")
    t3, _ = step_response(spec, 12, hover, {"yaw": 1.0}, fc.r_yaw)
    line(2, f"a full yaw stick: {fc.r_yaw * DEG:.0f} deg/s, 90 % in {t3 * 1000:.0f} ms: the weak axis")
    r, fc = high(spec, "acro")
    flips, last = 0, 1.0

    def count(r):
        nonlocal flips, last
        up = 1.0 - 2.0 * (r.x[7] * r.x[7] + r.x[8] * r.x[8])
        flips += last >= 0.0 > up
        last = up
    fly(r, fc, 1.0, Sticks(throttle=hover, roll=1.0), count)
    line(2, f"a roll held for a second goes inverted {flips} times")


# --------------------------------------------------------------------------- #
#  3 the assists, 4 speed, 5 the pack
# --------------------------------------------------------------------------- #
def check_assists(spec):
    r, fc = high(spec, "arcade", h=3.0)
    fly(r, fc, 6.0, Sticks())
    line(3, f"ARCADE, sticks at rest for 6 s: {r.height_above_ground():.2f} m up (it holds 3.00), {r.speed():.3f} m/s")
    fly(r, fc, 10.0, Sticks(throttle=1.0))
    line(3, f"ARCADE, full throttle for 10 s: {r.speed():.1f} m/s (asked {fc.AR['v_max_ms']:.0f}), {r.height_above_ground():.2f} m up, tilt {r.tilt() * DEG:.0f} deg")
    fly(r, fc, 2.5, Sticks(brake=1.0))
    line(3, f"ARCADE, the brake for 2.5 s: {r.speed():.1f} m/s")
    r, fc = high(spec, "arcade", h=3.0)
    fly(r, fc, 2.0, Sticks(throttle=0.6))
    y0 = r.euler()[2]
    fly(r, fc, 0.8, Sticks(throttle=0.6, yaw=1.0))
    line(3, f"ARCADE, yaw right for 0.8 s at speed: the heading goes {(r.euler()[2] - y0) * DEG:+.0f} deg")
    r, fc = high(spec, "acro", h=30.0)
    fly(r, fc, 2.0, Sticks(throttle=hover_stick(r, fc)))
    line(3, f"ACRO, the hover stick for 2 s: {r.height_above_ground():.1f} m up, from 30.0")
    r, fc = high(spec, "angle", h=30.0)
    fly(r, fc, 2.5, Sticks(throttle=hover_stick(r, fc) * 1.25, pitch=0.5))
    line(3, f"ANGLE, half stick forward for 2.5 s: tilt {r.tilt() * DEG:.0f} deg (asked {0.5 * fc.angle_max * DEG:.0f}), {r.x[0]:.1f} m north")


def check_speed(spec):
    r, fc = high(spec, "angle")
    fc.angle_max = math.radians(76.0)
    t100 = None

    def record(r):
        nonlocal t100
        if t100 is None and math.hypot(r.x[3], r.x[4]) > 100.0 / 3.6:
            t100 = r.t
    fly(r, fc, 7.0, Sticks(throttle=1.0, pitch=1.0), record)
    v = math.hypot(r.x[3], r.x[4])
    line(4, f"flat out at a held 76 deg lean: 0 to 100 km/h in {t100:.2f} s, {v * 3.6:.0f} km/h after 7 s, climbing {-r.x[5]:+.1f} m/s, the props "
            f"at {r.thrust / (4.0 * r.full) * 100:.0f} % of their static thrust")
    r, fc = high(spec, "angle", velocity=(15.0, 0.0, 0.0))
    fly(r, fc, 3.0, Sticks(throttle=hover_stick(r, fc)))
    line(4, f"level at the hover's thrust, 15 m/s: {math.hypot(r.x[3], r.x[4]):.1f} m/s after 3 s, the rotors' drag")
    r, fc = high(spec, "acro")
    fly(r, fc, 1.4, Sticks(throttle=0.05))
    wash, rate = 0.0, 0.0

    def washed(r):
        nonlocal wash, rate
        wash, rate = max(wash, r.wash), max(rate, abs(r.x[10]), abs(r.x[11]))
    fly(r, fc, 0.8, Sticks(throttle=0.42), washed)
    line(4, f"down at low throttle, then a catch at 0.42: into its own wash {wash:.2f} of the way, shaken to {rate * DEG:.0f} deg/s")


def check_pack(spec):
    r, fc = high(spec, "acro", h=100000.0)
    fly(r, fc, 90.0, Sticks(throttle=0.6))
    left = r.charge
    t = 90.0
    while r.charge > 0.0 and t < 900.0:
        fly(r, fc, 1.0, Sticks(throttle=0.6))
        t += 1.0
    line(5, f"0.6 stick from a full pack: {100 * left:.0f} % left after 90 s, empty after {t / 60.0:.1f} min, {100 * r.sag:.0f} % of the thrust gone then")

    def climb(seconds):
        r, fc = high(spec, "acro", h=100000.0)
        fly(r, fc, seconds, Sticks(throttle=0.6))
        r.set_state(h=100000.0)
        fc.reset()
        fly(r, fc, 0.25, Sticks(throttle=1.0))
        return -r.x[5]
    line(5, f"a quarter second flat out from a hover: {climb(0.0):.1f} m/s up on a fresh pack, {climb(120.0):.1f} m/s after two minutes")


# --------------------------------------------------------------------------- #
#  6 what it hits, 7 the line, 8 the wind
# --------------------------------------------------------------------------- #
def check_hits(spec):
    r = Racer(spec)
    r.set_state(h=1.0, velocity=(4.0, 0.0, 6.0))
    fc = FlightController(r, assist="acro")
    fly(r, fc, 0.3, Sticks(throttle=hover_stick(r, fc)))
    line(6, f"into the ground at 6 m/s: a {r.take_hit():.1f} m/s hit, crashed {r.crashed}")
    r = Racer(spec)
    r.set_state(h=1.0, velocity=(0.0, 0.0, 14.0))
    fc = FlightController(r, assist="acro")
    fly(r, fc, 0.3, Sticks(throttle=hover_stick(r, fc)))
    line(6, f"at 14 m/s: a {r.take_hit():.1f} m/s hit, crashed {r.crashed}, the motors at {r.rev_s():.0f} rev/s")
    r = Racer(spec)
    r.set_state(h=r.radius - 0.001, roll=math.pi / 2.0)
    fc = FlightController(r, assist="acro")
    fly(r, fc, 0.1)
    line(6, f"on its side on the ground: crashed {r.crashed}")
    r = Racer(spec)
    r.place_on_ground(10.0, 5.0, yaw=1.0)
    fc = FlightController(r, assist="arcade")
    fly(r, fc, 1.0)
    h0 = r.height_above_ground()
    r.release()
    fly(r, fc, 2.0)
    line(6, f"parked on the pad for a second: {h0:.2f} m up, then released on ARCADE: {r.height_above_ground():.2f} m up after 2 s")


def lap_flight(spec, wind=None, laps=3, record=None):
    """AUTO round the course from the air at its start. Returns the aircraft, each lap's time and
    how far from the line it got."""
    L = Line(COURSE, spec=spec)
    r = Racer(spec, wind=wind)
    r.set_state(L.n[0], L.e[0], L.h[0], yaw=math.atan2(L.e[1] - L.e[0], L.n[1] - L.n[0]))
    fc = FlightController(r, assist="auto", line=L)
    times, t0, last, off = [], 0.0, 0, 0.0
    while len(times) < laps and r.t < 300.0:
        fc.update()
        r.step()
        i = fc.i
        off = max(off, math.sqrt((L.n[i] - r.x[0]) ** 2 + (L.e[i] - r.x[1]) ** 2 + (L.h[i] + r.x[2]) ** 2))
        if i < last - L.size // 2:
            times.append(r.t - t0)
            t0 = r.t
        last = i
        if record is not None:
            record(r, fc)
    return r, L, times, off


def check_line(spec):
    r, L, times, off = lap_flight(spec)
    line(7, f"{L.length:.0f} m, {min(L.h):.0f} to {max(L.h):.0f} m up, its speed profile {min(L.v):.1f} to {max(L.v):.1f} m/s: {L.lap_time():.2f} s at the "
            f"profile's own speed")
    line(7, f"AUTO, still air: laps {', '.join(f'{t:.2f}' for t in times)} s, at most {off:.2f} m off the line, the pack at {100 * r.charge:.0f} %")
    r, _, times, off = lap_flight(spec, Wind(8.0, 250.0, turbulence=1.0, seed=1))
    line(7, f"AUTO, 8 m/s from 250 deg with turbulence: laps {', '.join(f'{t:.2f}' for t in times)} s, at most {off:.2f} m off the line, crashed {r.crashed}")


def check_wind(spec):
    for U in (4.0, 8.0, 12.0):
        wind = Wind(U, 270.0, turbulence=1.0, seed=1)
        r = Racer(spec, wind=wind)
        r.set_state(h=10.0)
        fc = FlightController(r, assist="arcade")
        fly(r, fc, 10.0, Sticks())
        rows = []
        fly(r, fc, 60.0, Sticks(), lambda r: rows.append((r.x[3], r.x[4], r.height_above_ground(), r.tilt())))
        a = np.array(rows)
        line(8, f"ARCADE, sticks at rest 10 m up in {U:.0f} m/s from the west: it has no position loop, so it holds a ground speed, "
                f"{a[:, 1].mean():.2f} m/s east, {math.sqrt((a[:, 2] - 10.0).var() + (a[:, 2].mean() - 10.0) ** 2):.2f} m rms off its height, "
                f"tilt {a[:, 3].mean() * DEG:.1f} +- {a[:, 3].std() * DEG:.1f} deg")


# --------------------------------------------------------------------------- #
#  9 integrity, speed
# --------------------------------------------------------------------------- #
def check_integrity(spec):
    def run():
        out = []
        lap_flight(spec, Wind(8.0, 250.0, turbulence=1.0, seed=1), laps=1, record=lambda r, fc: out.append(tuple(r.x)))
        return out
    a, b = run(), run()
    line(9, f"two runs of a lap in turbulent wind ({len(a)} steps, 17 states): {'bit-identical at every step' if a == b else 'DIFFERENT'}")
    r, fc = high(spec, "acro")
    fly(r, fc, 1.0, Sticks(throttle=hover_stick(r, fc)))
    t0 = time.perf_counter()
    fly(r, fc, 60.0, Sticks(throttle=hover_stick(r, fc), roll=0.1))
    el = time.perf_counter() - t0
    line(9, f"ACRO: 60 s simulated in {el:.2f} s of wall time = {60.0 / el:.0f} x real time ({el / 14400 * 1e6:.0f} us a step)")
    t0 = time.perf_counter()
    r, _, times, _ = lap_flight(spec, Wind(8.0, 250.0, turbulence=1.0, seed=1), laps=1)
    el = time.perf_counter() - t0
    line(9, f"AUTO in turbulence: {r.t:.1f} s simulated in {el:.2f} s = {r.t / el:.0f} x real time")


CHECKS = {1: check_airframe, 2: check_rates, 3: check_assists, 4: check_speed, 5: check_pack, 6: check_hits, 7: check_line,
          8: check_wind, 9: check_integrity}


# --------------------------------------------------------------------------- #
#  The flight
# --------------------------------------------------------------------------- #
class Flight:
    """Parked on the pad under the line's start, released after a second, AUTO round the course."""

    def __init__(self, a, spec):
        wind = Wind(a.wind, a.from_deg, turbulence=a.turbulence, seed=a.seed) if a.wind > 0.0 else None
        self.line = L = Line(COURSE, spec=spec)
        self.r = r = Racer(spec, wind=wind, seed=a.seed)
        r.place_on_ground(L.n[0], L.e[0], yaw=math.atan2(L.e[1] - L.e[0], L.n[1] - L.n[0]))
        self.fc = FlightController(r, assist="auto", line=L)

    def step(self):
        if self.r.parked and self.r.t >= 1.0:
            self.r.release()
        self.fc.update()
        self.r.step()


def telemetry(a, spec):
    f = Flight(a, spec)
    r, fc = f.r, f.fc
    print(f"racer, {r.m:.2f} kg, wind {a.wind:.1f} m/s from {a.from_deg:.0f} deg (10 m up), turbulence x{a.turbulence}, seed {a.seed}. "
          f"Flight controller OURS, exact state feedback.")
    print("    t  lap       N       E      H     Vg   roll  pitch    yaw  tilt  thrust N  rev/s  wash  pack %")
    csv = open(a.csv, "w", encoding="utf-8") if a.csv else None
    if csv:
        csv.write("t,lap,n,e,h,vn,ve,vd,roll,pitch,yaw,p,q,r,w_fr,w_fl,w_rl,w_rr,thrust,wash,charge,wind_n,wind_e,wind_d\n")
    lap, last, per = 0, 0, int(round(1.0 / r.dt))
    while lap < a.laps and r.t < 300.0 and not r.crashed:
        f.step()
        x = r.x
        ro, pi, ya = r.euler()
        if fc.i < last - f.line.size // 2:
            lap += 1
        last = max(fc.i, 0)
        if csv:
            w = r.wind_ned()
            csv.write(",".join(f"{v:.6g}" for v in (r.t, lap, x[0], x[1], -x[2], x[3], x[4], x[5], ro, pi, ya, x[10], x[11], x[12],
                                                    x[13], x[14], x[15], x[16], r.thrust, r.wash, r.charge, w[0], w[1], w[2])) + "\n")
        if r.steps % per == 0:
            print(f"{r.t:5.0f}  {lap:3d} {x[0]:7.2f} {x[1]:7.2f} {-x[2]:6.2f} {math.hypot(x[3], x[4]):6.2f} {ro * DEG:+6.1f} {pi * DEG:+6.1f} "
                  f"{ya * DEG:+6.1f} {r.tilt() * DEG:5.1f}  {r.thrust:8.1f}  {r.rev_s():5.0f}  {r.wash:4.2f}  {100 * r.charge:6.2f}")
    if csv:
        csv.close()
    print(f"done at {r.t:.1f} s: {lap} laps, crashed {r.crashed}, the pack at {100 * r.charge:.1f} %")


# --------------------------------------------------------------------------- #
#  The scene
# --------------------------------------------------------------------------- #
def material(tp, color, roughness=0.9, metalness=0.0):
    m = tp.MeshStandardMaterial()
    m.color, m.roughness, m.metalness = color, roughness, metalness
    return m


def polyline(tp, pts, color):
    g = tp.BufferGeometry()
    g.set_attribute("position", np.asarray(pts, np.float32))
    m = tp.LineBasicMaterial()
    m.color = color
    ln = tp.Line(g, m)
    ln.frustum_culled = False
    return ln


class Scene:
    """A field with the line drawn over it, a pole at each of the course's points, and the aircraft."""

    CAMS = ("chase", "side", "high")

    def __init__(self, tp, f, a, w, h, headless):
        self.tp, self.f = tp, f
        self.canvas = tp.Canvas("threepp - racing quad", width=w, height=h, antialiasing=4, headless=headless, vsync=not headless)
        r = self.r = tp.GLRenderer(self.canvas)
        r.tone_mapping = tp.ToneMapping.ACESFilmic
        r.tone_mapping_exposure = 1.0
        s = self.scene = tp.Scene()
        s.background = 0x9eb8d6
        s.set_fog(tp.Color(0x9eb8d6), 200.0, 1500.0)
        s.add(tp.HemisphereLight(0xdce6f2, 0x4a5a38, 1.1))
        sun = tp.DirectionalLight(0xfff4e0, 2.2)
        sun.position.set(-300.0, 500.0, 200.0)
        s.add(sun)
        ground = tp.Mesh(tp.PlaneGeometry(4000.0, 4000.0), material(tp, 0x5b7240, 0.95))
        ground.rotation.x = -math.pi / 2
        s.add(ground)
        L = f.line
        s.add(polyline(tp, [ned_to_world(n, e, -h_) for n, e, h_ in zip(L.n + L.n[:1], L.e + L.e[:1], L.h + L.h[:1])], 0xffffff))
        pole = material(tp, 0xe0562a, 0.6)
        for n, e, h_ in COURSE:
            p = tp.Mesh(tp.CylinderGeometry(0.08, 0.08, h_, 8, 1), pole)
            p.position.set(*ned_to_world(n + 3.0, e + 3.0, -0.5 * h_))         # beside the line, not on it
            s.add(p)
        rng = np.random.default_rng(7)
        tree = material(tp, 0x2f4526, 0.9)
        cone = tp.ConeGeometry(2.6, 10.0, 8, 1)
        for _ in range(400):
            n, e = rng.uniform(-300.0, 400.0), rng.uniform(-200.0, 450.0)
            if min(math.hypot(a_ - n, b_ - e) for a_, b_ in zip(L.n[::4], L.e[::4])) < 12.0:
                continue
            k = rng.uniform(0.7, 1.4)
            t = tp.Mesh(cone, tree)
            t.scale.set(k, k, k)
            t.position.set(*ned_to_world(n, e, -5.0 * k))
            s.add(t)
        mp = model_path()
        if mp is None:
            sys.exit("racer.glb is not built: blender --background --factory-startup --python build_racer_blender.py "
                     "-- --spec racer_spec.json --out racer.glb")
        self.model = tp.GLTFLoader().load(mp).scene
        s.add(self.model)
        self.vis = Visual(self.model, tp=tp)
        self.cam = tp.PerspectiveCamera(50.0, w / h, 0.02, 3000.0)
        self.mode = a.cam
        self.eye = None

    def camera(self, smooth=None):
        tp, ac = self.tp, self.f.r
        p = np.array(ac.world_position())
        psi = self.f.fc.psi                                                    # the way it is going: at this lean the nose says little
        fwd = np.array(ned_to_world(math.cos(psi), math.sin(psi), 0.0))
        right = np.array([-fwd[2], 0.0, fwd[0]])
        up = np.array([0.0, 1.0, 0.0])
        if self.mode == "chase":
            eye, at = p - 0.85 * fwd + 0.22 * up, p + 0.25 * fwd
        elif self.mode == "side":
            eye, at = p + 0.55 * right + 0.18 * fwd + 0.14 * up, p
        else:
            eye, at = p - 5.0 * fwd + 2.4 * up + 1.8 * right, p + 2.0 * fwd
        if smooth is not None and self.eye is not None:
            eye = self.eye + (eye - self.eye) * (1.0 - math.exp(-smooth))
        self.eye = eye
        self.cam.position.set(*map(float, eye))
        self.cam.look_at(tp.Vector3(*map(float, at)))

    def render(self):
        self.vis.pose(self.f.r)
        self.r.render(self.scene, self.cam)


def stills(tp, f, a, times, w, h):
    sc = Scene(tp, f, a, w, h, headless=True)
    ac = f.r
    for t in sorted(times):
        while ac.t < t - 1e-9:
            f.step()
        sc.vis.pose(ac)
        sc.camera()
        sc.render()
        path = os.path.join(a.out_dir, f"racer_{t:05.1f}s_{sc.mode}.png")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sc.r.save_frame(path)
        print(f"[racer] t {ac.t:.2f} s, {ac.height_above_ground():.1f} m up, {ac.speed():.1f} m/s, tilt {ac.tilt() * DEG:.0f} deg -> {path}")


def window(tp, f, a, w, h):
    sc = Scene(tp, f, a, w, h, headless=False)
    clock = tp.Clock()
    state = {"debt": 0.0, "c": False}

    def animate():
        dt = min(clock.get_delta(), 0.1)
        state["debt"] += dt
        n = int(state["debt"] / f.r.dt)
        state["debt"] -= n * f.r.dt
        for _ in range(n):
            f.step()
        c = sc.canvas.is_key_down("C")
        if c and not state["c"]:
            sc.mode = Scene.CAMS[(Scene.CAMS.index(sc.mode) + 1) % len(Scene.CAMS)]
            sc.eye = None
        state["c"] = c
        sc.camera(smooth=dt / 0.12)
        sc.render()
    sc.canvas.animate(animate)


def main(argv=None):
    a = parse(argv)
    spec = load_spec(a.spec)
    if a.checks:
        which = sorted(CHECKS) if a.checks == "all" else [int(k) for k in a.checks.split(",")]
        for k in which:
            CHECKS[k](spec)
    if a.stills or a.window:
        sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))       # python/ (threepp)
        import threepp as tp
        w, h = (int(v) for v in a.size.lower().split("x"))
        if a.window:
            window(tp, Flight(a, spec), a, w, h)
        else:
            stills(tp, Flight(a, spec), a, [float(t) for t in a.stills.split(",")], w, h)
    elif a.telemetry or not a.checks:
        telemetry(a, spec)


if __name__ == "__main__":
    main()
