"""The REMUS 100 running a route on its published model: a leg north at 4 m, a leg east down to
12 m, a leg south at 12 m, over a sand seabed at 20 m, with line-of-sight guidance and a
current you set.

The vehicle is ../remus_rig.py, which any scene can import: Remus (the 6-DOF model of T.
Prestero's thesis, MIT / WHOI 2001, with its numbers from remus100_spec.json), Autopilot and LOS
(ours: stern planes on pitch and depth, rudder on course, propeller on speed, and line-of-sight
guidance), and Visual (remus100.glb posed from the model). This file is the scene, the cameras and
the telemetry.

    python remus_dive.py                              # a window; C cycles the camera
    python remus_dive.py --current 0                  # still water (default: 0.3 m/s setting east)
    python remus_dive.py --current 0.5 --set 45       # 0.5 m/s setting north-east
    python remus_dive.py --hold heading               # hold the heading, not the track: it drifts
    python remus_dive.py --shot 40 --out remus.png    # headless still at t = 40 s
    python remus_dive.py --stills 20,95,120 --out-dir out
    python remus_dive.py --telemetry                  # no renderer: a line every 5 s and a summary
    python remus_dive.py --telemetry --csv remus.csv  # and every step to a file

Options: --current m/s (default 0.3), --set deg (the direction the current flows toward: 90, the
default, is east, across the first leg), --speed 1.75 (m/s through the water), --lookahead 8 (m),
--hold course|heading, --no-guards (the thesis's equations bare), --seconds (default: to the end
of the route), --size 1280x720, --cam chase|side|fixed (the stills' camera; fixed stands 8 m off the first corner at 9 m).

What is drawn: the route at its depths (white), a pole from the seabed at each waypoint, the aim
point the guidance steers at (orange) and the line of sight to it, the travelled track (cyan), and
the current as an arrow on the seabed at the start (4 m per m/s). Needs remus100.glb (build it once:
see build_remus_blender.py). The telemetry mode needs only numpy.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # python/ (threepp)
sys.path.insert(0, os.path.dirname(HERE))                    # examples/ (remus_rig)

from remus_rig import LOS, Autopilot, Remus, Visual, cross_track, model_path, ned_to_world  # noqa: E402

ROUTE = [(0.0, 0.0, 4.0), (200.0, 0.0, 4.0), (200.0, 120.0, 12.0), (40.0, 120.0, 12.0)]   # NED (north, east, depth), m
START = (-40.0, 0.0)                                         # trimmed here, at the first depth, on the first leg's course
SEABED = 20.0                                                # m below the surface (the NED origin)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", type=float, default=0.3, help="current speed, m/s")
    ap.add_argument("--set", dest="set_deg", type=float, default=90.0, help="the direction the current flows toward, deg (0 north, 90 east)")
    ap.add_argument("--speed", type=float, default=None, help="speed through the water, m/s (default: the spec's 1.75)")
    ap.add_argument("--lookahead", type=float, default=None)
    ap.add_argument("--hold", choices=("course", "heading"), default="course")
    ap.add_argument("--no-guards", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--shot", type=float, default=None)
    ap.add_argument("--stills", default=None, help="comma-separated times, s")
    ap.add_argument("--out", default="remus_dive.png")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--cam", choices=("chase", "side", "fixed"), default="chase")
    return ap.parse_args(argv)


def current_ned(speed, set_deg):
    """The water's velocity (NED) for a current of `speed` m/s flowing toward `set_deg`."""
    a = math.radians(set_deg)
    return (speed * math.cos(a), speed * math.sin(a), 0.0)


# --------------------------------------------------------------------------- #
#  The run
# --------------------------------------------------------------------------- #
class Dive:
    """The Remus, its autopilot and its guidance on ROUTE."""

    def __init__(self, a):
        self.a = a
        self.current = current_ned(a.current, a.set_deg)
        self.reset()

    def reset(self):
        a = self.a
        self.auv = Remus(current=self.current, guards=not a.no_guards)
        self.ap = Autopilot(self.auv, speed=a.speed, hold=a.hold)
        chi0 = math.atan2(ROUTE[1][1] - ROUTE[0][1], ROUTE[1][0] - ROUTE[0][0])
        self.auv.trim(self.ap.rpm, n=START[0], e=START[1], d=ROUTE[0][2], course=chi0 if a.hold == "course" else None, psi=chi0)
        self.ap.command(depth=ROUTE[0][2])
        self.los = LOS(ROUTE, lookahead=a.lookahead, auv=self.auv)
        self.rows = []

    def depth_cmd(self):
        return ROUTE[min(self.los.k + 1, len(ROUTE) - 1)][2]

    def step(self):
        auv, a = self.auv, self.a
        vg = auv.ground_velocity()
        chi = self.los.update(auv.n, auv.e, math.hypot(vg[0], vg[1]))
        if a.hold == "course":
            self.ap.command(course=chi, depth=self.depth_cmd())
        else:
            self.ap.command(heading=chi, depth=self.depth_cmd())
        self.ap.update()
        o = auv.out if auv._out_at == auv.steps else auv.evaluate()
        vw = auv.water_velocity_ned()
        self.rows.append((auv.t, auv.n, auv.e, auv.d, auv.u, auv.v, auv.w, o["U"], math.hypot(vg[0], vg[1]),
                          math.atan2(vg[1], vg[0]), auv.psi, auv.phi, auv.theta, math.atan2(vw[1], vw[0]),
                          self.los.k, self.los.e, cross_track(ROUTE, auv.n, auv.e), self.depth_cmd(),
                          auv.x[12], auv.x[13], auv.x[14], o["alpha_s"], o["alpha_r"]))
        auv.step()

    def advance(self, seconds):
        for _ in range(int(round(seconds / self.auv.dt))):
            if self.finished():
                return
            self.step()

    def finished(self):
        return self.los.done or (self.a.seconds is not None and self.auv.t >= self.a.seconds - 1e-9)


COLS = ("t", "n", "e", "d", "u", "v", "w", "U", "Vg", "chi", "psi", "phi", "theta", "chi_water", "leg", "e_los",
        "cross_track", "d_cmd", "stern", "rudder", "rpm", "alpha_stern", "alpha_rudder")


def summary(f):
    """The figures the current is about: per leg the crab angle against the velocity triangle,
    the track's error, the depth held, and what the vehicle did."""
    a = np.array(f.rows)
    c = {k: i for i, k in enumerate(COLS)}
    D = math.degrees
    C = np.asarray(f.current)
    print(f"[remus] {a[-1, c['t']]:.1f} s, current {f.a.current:.2f} m/s toward {f.a.set_deg:.0f} deg, {f.a.hold} hold, "
          f"look-ahead {f.los.delta:.0f} m, speed {f.ap.u_c:.2f} m/s ({f.ap.rpm:.0f} RPM at the trim), turn radius {f.los.radius:.1f} m")
    names = ("leg 1 (north, 4 m)", "leg 2 (east, down to 12 m)", "leg 3 (south, 12 m)")
    for k, name in enumerate(names):
        leg = a[a[:, c["leg"]] == k]
        if len(leg) == 0:
            continue
        (an, ae, _), (bn, be, bd) = ROUTE[k], ROUTE[k + 1]
        L = math.hypot(bn - an, be - ae)
        s = ((leg[:, c["n"]] - an) * (bn - an) + (leg[:, c["e"]] - ae) * (be - ae)) / L
        mid = leg[(s > 50.0) & (s < L - 30.0)]
        if len(mid) == 0:
            continue
        chi = math.atan2(be - ae, bn - an)
        t_hat = np.array([math.cos(chi), math.sin(chi), 0.0])
        c_cross = float(np.cross(t_hat, C)[2])                      # + from the left of the track
        U = mid[:, c["U"]].mean()
        tri = D(math.asin(max(-1.0, min(1.0, -c_cross / U))))
        crab_w = D(np.mean([math.remainder(x, 2 * math.pi) for x in mid[:, c["chi_water"]] - mid[:, c["chi"]]]))
        crab_n = D(np.mean([math.remainder(x, 2 * math.pi) for x in mid[:, c["psi"]] - mid[:, c["chi"]]]))
        print(f"   {name}: cross-track rms {np.sqrt(np.mean(mid[:, c['e_los']] ** 2)):.2f} m (max {np.abs(mid[:, c['e_los']]).max():.2f}, "
              f"mean {mid[:, c['e_los']].mean():+.2f}); cross current {c_cross:+.3f} m/s; the water velocity {crab_w:+.2f} deg off the "
              f"track (asin(-V_cross / U) {tri:+.2f}), the nose {crab_n:+.2f}; depth {mid[:, c['d']].mean():.2f} m "
              f"(cmd {bd:.0f}, max err {np.abs(mid[:, c['d']] - bd).max():.2f}); U {U:.3f} m/s, over ground {mid[:, c['Vg']].mean():.3f}")
    print(f"   depth {a[:, c['d']].min():.2f}..{a[:, c['d']].max():.2f} m, pitch {D(a[:, c['theta']].min()):+.1f}..{D(a[:, c['theta']].max()):+.1f} deg, "
          f"roll {D(a[:, c['phi']].min()):+.1f}..{D(a[:, c['phi']].max()):+.1f} deg, U {a[:, c['U']].min():.2f}..{a[:, c['U']].max():.2f} m/s, "
          f"propeller {a[:, c['rpm']].min():.0f}..{a[:, c['rpm']].max():.0f} RPM, stern {D(a[:, c['stern']].min()):+.1f}..{D(a[:, c['stern']].max()):+.1f} deg, "
          f"rudder {D(a[:, c['rudder']].min()):+.1f}..{D(a[:, c['rudder']].max()):+.1f} deg")
    print(f"   guards {f.auv.guards}, flags {f.auv.flags}")


def telemetry(f):
    t0 = time.time()
    every = int(round(5.0 / f.auv.dt))
    D = math.degrees
    while not f.finished():
        f.step()
        if f.auv.steps % every == 0:
            r = f.rows[-1]
            print(f"t {r[0]:6.1f}  n {r[1]:7.1f} e {r[2]:7.1f} d {r[3]:6.2f} (cmd {r[17]:4.1f})  U {r[7]:4.2f} Vg {r[8]:4.2f}  "
                  f"course {D(r[9]):+7.2f} heading {D(r[10]):+7.2f}  pitch {D(r[12]):+6.2f} roll {D(r[11]):+6.2f}  "
                  f"leg {r[14]:.0f} e {r[15]:+6.2f}  stern {D(r[18]):+6.2f} rudder {D(r[19]):+6.2f} rpm {r[20]:5.0f}")
    print(f"({time.time() - t0:.1f} s wall)")
    summary(f)
    if f.a.csv:
        np.savetxt(f.a.csv, np.array(f.rows), delimiter=",", header=",".join(COLS), comments="", fmt="%.9g")
        print(f"[remus] wrote {f.a.csv}")


# --------------------------------------------------------------------------- #
#  The scene
# --------------------------------------------------------------------------- #
WATER = 0x0e4a5c


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
        self.canvas = tp.Canvas("threepp - REMUS 100", width=w, height=h, antialiasing=4, headless=headless, vsync=not headless)
        r = self.r = tp.GLRenderer(self.canvas)
        r.tone_mapping = tp.ToneMapping.ACESFilmic
        r.tone_mapping_exposure = 1.1
        s = self.scene = tp.Scene()
        s.background = WATER
        s.set_fog_exp2(tp.Color(WATER), 0.028)
        s.add(tp.HemisphereLight(0x9fd0e0, 0x2a3a30, 1.4))
        sun = tp.DirectionalLight(0xe8f6ff, 2.0)
        sun.position.set(40.0, 200.0, -60.0)
        s.add(sun)
        # the seabed: sand, with scattered stones
        bed = tp.Mesh(tp.PlaneGeometry(1200.0, 1200.0), material(tp, 0x8c7b5c, 0.95))
        bed.rotation.x = -math.pi / 2
        bed.position.set(*ned_to_world(100.0, 60.0, SEABED))
        s.add(bed)
        rng = np.random.default_rng(11)
        stone = material(tp, 0x4f4a42, 0.9)
        weed = material(tp, 0x2f5a2c, 0.8)
        rock = tp.IcosahedronGeometry(1.0, 0)
        frond = tp.ConeGeometry(0.15, 1.0, 5, 1)
        for _ in range(700):
            n, e = rng.uniform(-120.0, 320.0), rng.uniform(-120.0, 240.0)
            k = rng.uniform(0.2, 1.4)
            m = tp.Mesh(rock, stone)
            m.scale.set(k, 0.55 * k, 0.8 * k)
            m.rotation.y = rng.uniform(0.0, 6.28)
            m.position.set(*ned_to_world(n, e, SEABED - 0.2 * k))
            s.add(m)
        for _ in range(900):
            n, e = rng.uniform(-120.0, 320.0), rng.uniform(-120.0, 240.0)
            k = rng.uniform(0.5, 2.2)
            m = tp.Mesh(frond, weed)
            m.scale.set(1.0, k, 1.0)
            m.position.set(*ned_to_world(n, e, SEABED - 0.5 * k))
            s.add(m)
        # the surface, seen from below
        top = tp.MeshBasicMaterial()
        top.color = tp.Color(0x6fb6c8)
        top.side = tp.Side.Double
        surf = tp.Mesh(tp.PlaneGeometry(1200.0, 1200.0), top)
        surf.rotation.x = -math.pi / 2
        surf.position.set(*ned_to_world(100.0, 60.0, 0.0))
        s.add(surf)
        # the route at its depths, poles from the seabed at its waypoints
        pts = [ned_to_world(START[0], START[1], ROUTE[0][2])] + [ned_to_world(n, e, d) for n, e, d in ROUTE]
        line, _ = polyline(tp, pts, 0xffffff)
        s.add(line)
        pole = material(tp, 0xe8e8e8, 0.6)
        knob = material(tp, 0xffd23a, 0.4)
        for n, e, d in ROUTE:
            p = tp.Mesh(tp.CylinderGeometry(0.06, 0.06, SEABED - d, 8, 1), pole)
            p.position.set(*ned_to_world(n, e, 0.5 * (SEABED + d)))
            s.add(p)
            b = tp.Mesh(tp.SphereGeometry(0.3, 16, 10), knob)
            b.position.set(*ned_to_world(n, e, d))
            s.add(b)
        # the aim point, the line of sight to it, the track
        aim_mat = material(tp, 0xff7a1a, 0.5)
        aim_mat.emissive = tp.Color(1.0, 0.45, 0.1)
        aim_mat.emissive_intensity = 0.6
        self.aim = tp.Mesh(tp.SphereGeometry(0.25, 16, 10), aim_mat)
        s.add(self.aim)
        self.sight, self.sight_g = polyline(tp, [(0, 0, 0), (0, 0, 0)], 0xff7a1a)
        s.add(self.sight)
        self.track_n = 0
        self.track, self.track_g = polyline(tp, [(0, 0, 0)], 0x2fd4ff, capacity=40000)
        self.track_buf = np.zeros((40000, 3), np.float32)
        s.add(self.track)
        # the current
        if a.current > 0.005:
            cv = np.asarray(f.current)
            d = np.array(ned_to_world(cv[0], cv[1], 0.0))
            d /= np.linalg.norm(d)
            arrow = tp.ArrowHelper(tp.Vector3(*map(float, d)), tp.Vector3(*ned_to_world(START[0] + 10.0, -8.0, SEABED - 0.5)),
                                   a.current * 4.0 + 1.0, tp.Color(0.95, 0.85, 0.2), 0.6, 0.4)
            s.add(arrow)
        # the vehicle
        mp = model_path()
        if mp is None:
            sys.exit("remus100.glb is not built: blender --background --factory-startup --python build_remus_blender.py")
        self.model = tp.GLTFLoader().load(mp).scene
        s.add(self.model)
        self.vis = Visual(self.model)
        self.cam = tp.PerspectiveCamera(50.0, w / h, 0.05, 400.0)
        self.mode = a.cam
        self.eye = None

    def sync(self):
        """The vehicle, the aim point, the line of sight and the track from the run's state."""
        f, auv = self.f, self.f.auv
        self.vis.pose(auv, h0=0.0)
        p = ned_to_world(auv.n, auv.e, auv.d)
        an, ae = f.los.aim
        q = ned_to_world(an, ae, f.depth_cmd())
        self.aim.position.set(*q)
        self.sight_g.update_attribute("position", np.array([p, q], np.float32))
        if self.track_n < len(self.track_buf):
            if self.track_n == 0 or np.linalg.norm(self.track_buf[self.track_n - 1] - p) > 0.2:
                self.track_buf[self.track_n] = p
                self.track_n += 1
                self.track_g.update_attribute("position", self.track_buf)
                self.track_g.set_draw_range(0, self.track_n)

    def camera(self, smooth=None):
        tp, auv = self.tp, self.f.auv
        p = np.array(ned_to_world(auv.n, auv.e, auv.d))
        R = auv.world_rotation()
        fwd = R[:, 0].copy()
        fwd[1] = 0.0
        fwd /= max(np.linalg.norm(fwd), 1e-6)
        side = np.array([-fwd[2], 0.0, fwd[0]])                    # starboard, horizontal
        if self.mode == "chase":
            eye, at = p - 2.6 * fwd + 0.5 * side + np.array([0.0, 0.6, 0.0]), p + 0.5 * fwd
        elif self.mode == "side":
            eye, at = p + 2.4 * side - 0.4 * fwd + np.array([0.0, 0.3, 0.0]), p
        else:
            eye, at = np.array(ned_to_world(192.0, -7.0, 9.0)), p
        # the fixed camera's lens: 50 deg, or narrower so that the vehicle spans about a quarter of the frame
        fov = min(50.0, math.degrees(2.0 * math.atan(2.6 / max(float(np.linalg.norm(at - eye)), 1.0)))) if self.mode == "fixed" else 50.0
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
        while f.auv.t < t - 1e-9 and not f.finished():
            f.advance(min(frame, t - f.auv.t))
            sc.sync()
        sc.camera()
        sc.render()
        sc.r.save_frame(path)
        print(f"[remus] t {f.auv.t:.2f} s -> {path}")
    summary(f)


def window(tp, f, w, h):
    sc = Scene(tp, f, w, h, headless=False)
    clock = tp.Clock()
    state = {"debt": 0.0, "c": False}
    modes = ("chase", "side", "fixed")

    def animate():
        dt = min(clock.get_delta(), 0.1)
        state["debt"] += dt
        n = int(state["debt"] / f.auv.dt)
        state["debt"] -= n * f.auv.dt
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
    f = Dive(a)
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
            paths = [os.path.join(a.out_dir, f"remus_{a.cam}_{t:05.1f}s.png") for t in times]
        stills(tp, f, times, paths, w, h)
        return
    window(tp, f, w, h)


if __name__ == "__main__":
    main()
