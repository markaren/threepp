"""The Skywalker X8 flying a route on its published flight model: a straight leg north, a right
turn, a leg east, over a lake, with line-of-sight guidance and a wind you set.

The aircraft is ../x8_rig.py, which any scene can import: X8 (the 6-DOF model of Løw-Hansen,
Hann, Gryte, Johansen and Deiler, CEAS Aeronautical Journal 16, 501-523 (2025), with its
numbers from x8_spec.json), Autopilot and LOS (ours: successive loop closure on the elevons
and the throttle, and line-of-sight guidance), and Visual (x8.glb posed from the model). This
file is the scene, the cameras and the telemetry.

    python x8_flight.py                              # a window; C cycles the camera
    python x8_flight.py --wind 0                     # calm
    python x8_flight.py --wind 8 --from 315          # 8 m/s from the north-west
    python x8_flight.py --hold heading               # hold the heading, not the track: it drifts
    python x8_flight.py --shot 20 --out x8.png       # headless still at t = 20 s
    python x8_flight.py --stills 8,28,38 --out-dir out
    python x8_flight.py --telemetry                  # no renderer: a line a second and a summary
    python x8_flight.py --telemetry --csv x8.csv     # and every step to a file

Options: --wind m/s (default 6), --from deg (where it blows from: 270, the default, is west, a
crosswind on the first leg and a tailwind on the second), --airspeed 18, --height 60,
--lookahead 60 (m, the guidance's look-ahead), --hold course|heading, --seconds (default: to the
end of the route), --size 1280x720, --cam chase|high|ground (the stills' camera).

What is drawn: the route at the commanded height (white), the poles at its waypoints, the
aim point the guidance steers at (orange) and the line of sight to it, the flown track (cyan),
and the wind as an arrow on the ground at the start, as long as the wind is strong (1 m per m/s).
Needs x8.glb (build it once: see build_x8_blender.py). The telemetry mode needs only numpy.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # python/ (threepp)
sys.path.insert(0, os.path.dirname(HERE))                    # examples/ (x8_rig)

from x8_rig import LOS, X8, Autopilot, Visual, cross_track, model_path, ned_to_world  # noqa: E402

ROUTE = [(0.0, 0.0), (600.0, 0.0), (600.0, 450.0)]          # NED (north, east), m
START = (-120.0, 0.0)                                        # trimmed here, on the first leg's course


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wind", type=float, default=6.0, help="wind speed, m/s")
    ap.add_argument("--from", dest="from_deg", type=float, default=270.0, help="the direction the wind blows from, deg (0 north, 90 east)")
    ap.add_argument("--airspeed", type=float, default=18.0)
    ap.add_argument("--height", type=float, default=60.0)
    ap.add_argument("--lookahead", type=float, default=None)
    ap.add_argument("--hold", choices=("course", "heading"), default="course")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--shot", type=float, default=None)
    ap.add_argument("--stills", default=None, help="comma-separated times, s")
    ap.add_argument("--out", default="x8_flight.png")
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
    """The X8, its autopilot and its guidance on ROUTE."""

    def __init__(self, a):
        self.a = a
        self.wind = wind_ned(a.wind, a.from_deg)
        self.reset()

    def reset(self):
        a = self.a
        self.x8 = X8(wind=self.wind, h0=0.0)
        # trimmed on the first leg's course, crabbed into the wind
        self.x8.trim(a.airspeed, n=START[0], e=START[1], h=a.height, course=math.atan2(ROUTE[1][1] - ROUTE[0][1], ROUTE[1][0] - ROUTE[0][0]))
        self.ap = Autopilot(self.x8, airspeed=a.airspeed, hold=a.hold)
        self.los = LOS(ROUTE, lookahead=a.lookahead)
        self.rows = []

    def step(self):
        x8, a = self.x8, self.a
        vg = x8.ground_velocity()
        chi = self.los.update(x8.n, x8.e, math.hypot(vg[0], vg[1]))
        if a.hold == "course":
            self.ap.command(course=chi, altitude=a.height, airspeed=a.airspeed)
        else:
            self.ap.command(heading=chi, altitude=a.height, airspeed=a.airspeed)
        self.ap.update()
        o = x8.out
        va = x8.air_velocity_ned()
        self.rows.append((x8.t, x8.n, x8.e, x8.h, o["Va"], math.hypot(vg[0], vg[1]), math.atan2(vg[1], vg[0]),
                          x8.psi, x8.phi, x8.theta, o["alpha"], o["beta"], math.atan2(va[1], va[0]),
                          self.los.k, self.los.e, cross_track(ROUTE, x8.n, x8.e), o["throttle"], x8.omega,
                          o["T"], o["Im"], o["U"] * o["Im"] / x8.Ub, o["de"], o["da"]))
        x8.step()

    def advance(self, seconds):
        for _ in range(int(round(seconds / self.x8.dt))):
            if self.finished():
                return
            self.step()

    def finished(self):
        return self.los.done or (self.a.seconds is not None and self.x8.t >= self.a.seconds - 1e-9)


COLS = ("t", "n", "e", "h", "Va", "Vg", "chi", "psi", "phi", "theta", "alpha", "beta", "chi_air", "leg", "e_los",
        "cross_track", "throttle", "omega", "T", "I_motor", "I_battery", "d_e", "d_a")


def summary(f):
    """The figures the wind is about: per leg the crab angle against the wind triangle, the
    track's error, and what the aircraft did."""
    a = np.array(f.rows)
    c = {k: i for i, k in enumerate(COLS)}
    D = math.degrees
    W = np.asarray(f.wind)
    print(f"[x8] {a[-1, c['t']]:.1f} s, wind {f.a.wind:.1f} m/s from {f.a.from_deg:.0f} deg, {f.a.hold} hold, "
          f"look-ahead {f.los.delta:.0f} m")
    for k, name in ((0, "leg 1 (north)"), (1, "leg 2 (east)")):
        leg = a[a[:, c["leg"]] == k]
        if len(leg) == 0:
            continue
        # settled: the leg's middle, from 150 m after its start to 100 m before its end
        (an, ae), (bn, be) = ROUTE[k], ROUTE[k + 1]
        L = math.hypot(bn - an, be - ae)
        s = ((leg[:, c["n"]] - an) * (bn - an) + (leg[:, c["e"]] - ae) * (be - ae)) / L
        mid = leg[(s > 150.0) & (s < L - 100.0)]
        if len(mid) == 0:
            continue
        chi = math.atan2(be - ae, bn - an)
        t_hat = np.array([math.cos(chi), math.sin(chi), 0.0])
        w_cross = float(np.cross(t_hat, W)[2])                      # + from the left of the track
        tri = D(math.asin(max(-1.0, min(1.0, -w_cross / f.a.airspeed))))
        crab_air = D(np.mean([math.remainder(x, 2 * math.pi) for x in mid[:, c["chi_air"]] - mid[:, c["chi"]]]))
        crab_nose = D(np.mean([math.remainder(x, 2 * math.pi) for x in mid[:, c["psi"]] - mid[:, c["chi"]]]))
        print(f"   {name}: cross-track rms {np.sqrt(np.mean(mid[:, c['e_los']] ** 2)):.2f} m (max {np.abs(mid[:, c['e_los']]).max():.2f}, "
              f"mean {mid[:, c['e_los']].mean():+.2f}); crosswind {w_cross:+.2f} m/s; the air velocity {crab_air:+.2f} deg off the "
              f"track (asin(-W_cross / V_a) {tri:+.2f}), the nose {crab_nose:+.2f}; ground speed {mid[:, c['Vg']].mean():.2f} m/s")
    turn = a[(a[:, c["n"]] > 450.0) & (a[:, c["e"]] < 150.0)]
    if len(turn):
        print(f"   the turn: past the corner {max(0.0, a[:, c['n']].max() - ROUTE[1][0]):.1f} m, off the route at most "
              f"{turn[:, c['cross_track']].max():.1f} m, bank up to {D(np.abs(turn[:, c['phi']]).max()):.1f} deg")
    print(f"   height {a[:, c['h']].min():.2f}..{a[:, c['h']].max():.2f} m, airspeed {a[:, c['Va']].min():.2f}..{a[:, c['Va']].max():.2f} m/s, "
          f"throttle {a[:, c['throttle']].min():.2f}..{a[:, c['throttle']].max():.2f}, motor {a[:, c['I_motor']].mean():.1f} A "
          f"(battery {a[:, c['I_battery']].mean():.1f} A) on average; guards {f.x8.guards}")


def telemetry(f):
    t0 = time.time()
    every = int(round(1.0 / f.x8.dt))
    D = math.degrees
    while not f.finished():
        f.step()
        if f.x8.steps % every == 0:
            r = f.rows[-1]
            print(f"t {r[0]:5.1f}  n {r[1]:7.1f} e {r[2]:7.1f} h {r[3]:6.2f}  Va {r[4]:5.2f} Vg {r[5]:5.2f}  course {D(r[6]):+7.2f} "
                  f"heading {D(r[7]):+7.2f}  bank {D(r[8]):+6.2f}  leg {r[13]:.0f} e {r[14]:+7.2f}  thr {r[16]:.2f}")
    print(f"({time.time() - t0:.1f} s wall)")
    summary(f)
    if f.a.csv:
        np.savetxt(f.a.csv, np.array(f.rows), delimiter=",", header=",".join(COLS), comments="", fmt="%.6g")
        print(f"[x8] wrote {f.a.csv}")


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
    line = tp.Line(g, m)
    line.frustum_culled = False
    return line, g


class Scene:
    def __init__(self, tp, f, w, h, headless):
        self.tp, self.f = tp, f
        a = f.a
        self.canvas = tp.Canvas("threepp - Skywalker X8", width=w, height=h, antialiasing=4, headless=headless, vsync=not headless)
        r = self.r = tp.GLRenderer(self.canvas)
        r.tone_mapping = tp.ToneMapping.ACESFilmic
        r.tone_mapping_exposure = 1.0
        s = self.scene = tp.Scene()
        s.background = 0x9eb8d6
        s.set_fog(tp.Color(0x9eb8d6), 300.0, 2200.0)
        s.add(tp.HemisphereLight(0xdce6f2, 0x4a5a38, 1.1))
        sun = tp.DirectionalLight(0xfff4e0, 2.2)
        sun.position.set(-300.0, 500.0, 200.0)
        s.add(sun)
        ground = tp.Mesh(tp.PlaneGeometry(5000.0, 5000.0), material(tp, 0x5b7240, 0.95))
        ground.rotation.x = -math.pi / 2
        s.add(ground)
        lake = tp.Mesh(tp.CircleGeometry(230.0, 96), material(tp, 0x3d5866, 0.25, 0.1))
        lake.rotation.x = -math.pi / 2
        lake.position.set(*ned_to_world(520.0, 150.0, -0.05))
        s.add(lake)
        rng = np.random.default_rng(7)
        tree = material(tp, 0x2f4526, 0.9)
        cone = tp.ConeGeometry(2.6, 10.0, 8, 1)
        for _ in range(420):
            n, e = rng.uniform(-400.0, 1100.0), rng.uniform(-500.0, 900.0)
            if math.hypot(n - 520.0, e - 150.0) < 250.0:
                continue
            k = rng.uniform(0.7, 1.4)
            t = tp.Mesh(cone, tree)
            t.scale.set(k, k, k)
            t.position.set(*ned_to_world(n, e, -5.0 * k))
            s.add(t)
        # the route at the commanded height, poles at its waypoints
        hgt = a.height
        pts = [ned_to_world(START[0], START[1], -hgt)] + [ned_to_world(n, e, -hgt) for n, e in ROUTE]
        line, _ = polyline(tp, pts, 0xffffff)
        s.add(line)
        pole = material(tp, 0xe8e8e8, 0.6)
        knob = material(tp, 0xffffff, 0.4)
        for n, e in ROUTE:
            p = tp.Mesh(tp.CylinderGeometry(0.25, 0.25, hgt, 8, 1), pole)
            p.position.set(*ned_to_world(n, e, -0.5 * hgt))
            s.add(p)
            b = tp.Mesh(tp.SphereGeometry(0.9, 16, 10), knob)
            b.position.set(*ned_to_world(n, e, -hgt))
            s.add(b)
        # the aim point, the line of sight to it, the flown track
        aim_mat = material(tp, 0xff7a1a, 0.5)
        aim_mat.emissive = tp.Color(1.0, 0.45, 0.1)
        aim_mat.emissive_intensity = 0.6
        self.aim = tp.Mesh(tp.SphereGeometry(1.1, 16, 10), aim_mat)
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
            sys.exit("x8.glb is not built: blender --background --factory-startup --python build_x8_blender.py")
        self.model = tp.GLTFLoader().load(mp).scene
        s.add(self.model)
        self.vis = Visual(self.model)
        self.cam = tp.PerspectiveCamera(42.0, w / h, 0.2, 8000.0)
        self.mode = a.cam
        self.eye = None

    def sync(self):
        """The aircraft, the aim point, the line of sight and the track from the flight's state."""
        f, x8 = self.f, self.f.x8
        self.vis.pose(x8, h0=0.0)
        p = ned_to_world(x8.n, x8.e, x8.d)
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
        tp, x8 = self.tp, self.f.x8
        p = np.array(ned_to_world(x8.n, x8.e, x8.d))
        vg = x8.ground_velocity()
        fwd = np.array(ned_to_world(vg[0], vg[1], 0.0))
        fwd /= max(np.linalg.norm(fwd), 1e-6)
        if self.mode == "chase":
            eye, at = p - 13.0 * fwd + np.array([0.0, 3.2, 0.0]), p + 6.0 * fwd
        elif self.mode == "high":
            eye, at = p - 70.0 * fwd + np.array([0.0, 45.0, 0.0]) + 25.0 * np.array([-fwd[2], 0.0, fwd[0]]), p + 40.0 * fwd
        else:
            eye, at = np.array(ned_to_world(560.0, -70.0, -2.0)), p
        # the lens: 42 deg, or narrower so that the aircraft spans about a sixth of the frame
        dist = float(np.linalg.norm(at - eye)) if self.mode == "ground" else 0.0
        fov = min(42.0, math.degrees(2.0 * math.atan(6.0 / max(dist, 1.0)))) if self.mode == "ground" else 42.0
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
        while f.x8.t < t - 1e-9 and not f.finished():
            f.advance(min(frame, t - f.x8.t))
            sc.sync()
        sc.camera()
        sc.render()
        sc.r.save_frame(path)
        print(f"[x8] t {f.x8.t:.2f} s -> {path}")
    summary(f)


def window(tp, f, w, h):
    sc = Scene(tp, f, w, h, headless=False)
    clock = tp.Clock()
    state = {"debt": 0.0, "c": False}
    modes = ("chase", "high", "ground")

    def animate():
        dt = min(clock.get_delta(), 0.1)
        state["debt"] += dt
        n = int(state["debt"] / f.x8.dt)
        state["debt"] -= n * f.x8.dt
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
            paths = [os.path.join(a.out_dir, f"x8_{a.cam}_{t:05.1f}s.png") for t in times]
        stills(tp, f, times, paths, w, h)
        return
    window(tp, f, w, h)


if __name__ == "__main__":
    main()
