"""
Fly the Babyshark 260 VTOL by hand: the keyboard on babyshark_rig.Pilot, the mouse on the camera.

    python babyshark_fly.py                               # on the pad, nose into the wind
    python babyshark_fly.py --wind 0                      # still air
    python babyshark_fly.py --wind 8 --from 250 --turbulence 1
    python babyshark_fly.py --demo                        # the keys press themselves: a hand flight to watch
    python babyshark_fly.py --demo --stills 20,60,150 --out-dir out/babyshark_fly    # the same without a window, as stills

    W S  (or up, down)       hovering: forward and back; on the wing: faster and slower
    A D  (or left, right)    turn: the yaw hovering, the bank on the wing
    Q E                      sideways (hovering)
    SPACE, SHIFT             up and down; on the ground SPACE starts the rotors and lifts off,
                             and SHIFT held on the ground stops them
    T                        transition: hover to wing, wing to hover
    M                        the autopilot flies the circuit and lands on the pad
    H                        the autopilot flies home and lands
    V                        the hover's weathervane, on and off
    C                        camera: chase (turns with the nose), free, tower (from beside the pad)
    R                        back on the pad
    mouse                    drag to look around the aircraft, wheel to zoom

Any of W A S D Q E SPACE SHIFT takes the aircraft back from the autopilot (M, H).

What flies is rigs/babyshark_rig.py, the same model, autopilot and mission as babyshark_flight.py
(its checks say what is the thesis's and what is OURS or ASSUMED). The keys go through its Pilot:
they move the autopilot's setpoints, the way a VTOL's assisted modes take a pilot's sticks, and
the two transitions are the autopilot's. Two things to expect, both the model's: on the wing it
climbs slowly (the panel says what the pusher has to give at this airspeed; slower climbs better),
and hovering it cannot hold its tail across a fresh wind, so left alone it turns its nose into it.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rigs"))
sys.path.insert(0, HERE)
from babyshark_flight import Scene, circuit  # noqa: E402
from babyshark_rig import Autopilot, Babyshark, Pilot, Wind, load_spec, ned_to_world, wrap  # noqa: E402

DEG = 180.0 / math.pi
PANEL = 288                    # the panel's width, px
STEP = 1.0 / 30.0              # a frame of the window-less run, s

# The demo's hand flight: (from this second on, the keys held). H hands it to the autopilot, and
# the camera goes to the pad to watch it come in.
DEMO = ((1.0, "SPACE"), (15.0, ""), (17.0, "W"), (23.0, "W D"), (28.0, ""), (33.0, "Q"), (36.0, ""), (39.0, "A"), (42.0, ""),
        (45.0, "T"), (45.2, ""), (62.0, "A"), (72.0, ""), (78.0, "SPACE"), (92.0, ""), (94.0, "D"), (104.0, ""),
        (108.0, "SHIFT"), (114.0, ""), (120.0, "H"), (120.2, ""), (122.0, "C"), (122.2, ""), (122.4, "C"), (122.6, ""))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wind", type=float, default=4.0, help="wind speed 10 m up, m/s")
    ap.add_argument("--from", dest="from_deg", type=float, default=250.0, help="the direction the wind blows from, deg (0 north, 90 east)")
    ap.add_argument("--turbulence", type=float, default=1.0, help="scales the Dryden sigmas")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--demo", action="store_true", help="the keys press themselves")
    ap.add_argument("--stills", default=None, help="with --demo: no window, a still at each of these times, s (comma-separated)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--cam", choices=("chase", "free", "tower"), default="chase")
    ap.add_argument("--spec", default=None, help="another spec file in place of babyshark_spec.json")
    return ap.parse_args(argv)


class Session:
    """The aircraft on the pad with its autopilot and its pilot; reset() puts a new one there."""

    def __init__(self, a, spec):
        self.args, self.spec = a, spec
        self.legs, self.route = circuit(a.from_deg)
        self.reset()

    def reset(self):
        a = self.args
        wind = Wind(a.wind, a.from_deg, turbulence=a.turbulence, seed=a.seed, spec=self.spec) if a.wind > 0.0 else None
        self.aircraft = ac = Babyshark(self.spec, wind=wind)
        ac.place_on_ground(0.0, 0.0, yaw=math.radians(a.from_deg))
        self.ap = Autopilot(ac)
        self.ap.armed = False
        self.pilot = Pilot(ac, self.ap, self.spec)
        self.debt = 0.0

    def advance(self, dt):
        ac, ap, pilot = self.aircraft, self.ap, self.pilot
        self.debt += dt
        n = int(self.debt / ac.dt + 1e-9)
        self.debt -= n * ac.dt
        for _ in range(n):
            pilot.update()
            ap.update()
            ac.step()


class Script:
    """A keyboard that plays: [(from this second on, "the keys held"), ...]."""

    def __init__(self, steps):
        self.steps = sorted(steps)
        self.held = frozenset()
        self.end = self.steps[-1][0]

    def at(self, t):
        for t0, keys in self.steps:
            if t0 > t + 1e-9:
                break
            self.held = frozenset(keys.split())

    def is_key_down(self, key):
        return key in self.held


class Keyboard:
    """The window's keys and the script's, either holds a key down; hit() is the press."""

    def __init__(self, *sources):
        self.sources = sources
        self.was = {}

    def is_key_down(self, key):
        return any(s.is_key_down(key) for s in self.sources)

    def hit(self, key):
        down = self.is_key_down(key)
        hit = down and not self.was.get(key, False)
        self.was[key] = down
        return hit


class Follow:
    """The camera: around the aircraft on the mouse (tp.OrbitControls), carried along with it
    and, in "chase", turned with its nose; or a spectator's, standing `tower` (N, E) beside the
    pad: the aircraft high in the view, the horizon still in it while the lens can ("tower")."""

    MODES = ("chase", "free", "tower")
    FOV = 42.0
    FIELD = 24.0               # the tower's view is no narrower than this at the aircraft, m

    def __init__(self, tp, cam, canvas, mode, tower):
        self.tp, self.cam, self.mode = tp, cam, mode
        self.tower = np.array(ned_to_world(tower[0], tower[1], -2.0))
        self.controls = c = tp.OrbitControls(cam, canvas)
        c.enable_pan = False
        c.min_distance, c.max_distance = 2.0, 400.0
        self.p = None
        self.yaw = 0.0

    def next(self):
        self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]
        if self.mode == "chase":                           # from the tower: behind it again
            self.p = None
        self.cam.fov = self.FOV
        self.cam.update_projection_matrix()

    def update(self, ac, dt, mouse):
        tp, cam = self.tp, self.cam
        at = np.array(ac.world_position(), float)
        p = at + np.array([0.0, 0.7, 0.0])                 # what it looks at: a little over the aircraft
        psi = ac.x[11]
        if self.mode == "tower":
            eye = self.tower
            d = at - eye
            flat = max(math.hypot(d[0], d[2]), 1.0)
            up = math.atan2(d[1], flat) * DEG                  # the aircraft's elevation from here
            fov = min(self.FOV, max(2.0 * DEG * math.atan(0.5 * self.FIELD / math.hypot(flat, d[1])), up / 0.75))
            aim = max(up - 0.35 * fov, -0.25 * fov)            # the aircraft 15 % under the view's top edge
            cam.fov = fov
            cam.update_projection_matrix()
            cam.position.set(*map(float, eye))
            cam.look_at(tp.Vector3(float(at[0]), float(eye[1] + flat * math.tan(aim / DEG)), float(at[2])))
            self.controls.enabled = False
            return
        if self.p is None:                                 # behind, above and a little to the right
            fwd = np.array(ned_to_world(math.cos(psi), math.sin(psi), 0.0))
            off = -6.0 * fwd + 0.9 * np.array([0.0, 1.0, 0.0]) + 0.8 * np.array([-fwd[2], 0.0, fwd[0]])
            self.yaw = psi
        else:
            off = np.array([cam.position.x, cam.position.y, cam.position.z]) - self.p
            if self.mode == "chase":                       # round the vertical with the nose, eased
                d = wrap(psi - self.yaw) * (1.0 - math.exp(-dt / 0.35))
                self.yaw = wrap(self.yaw + d)
                c, s = math.cos(d), -math.sin(d)
                off = np.array([off[0] * c + off[2] * s, off[1], -off[0] * s + off[2] * c])
            else:
                self.yaw = psi
        self.p = p
        eye = p + off
        cam.position.set(*map(float, eye))
        self.controls.target.set(*map(float, p))
        self.controls.enabled = mouse
        self.controls.update()
        if cam.position.y < 0.4:                           # not under the field
            cam.position.set(cam.position.x, 0.4, cam.position.z)
            cam.look_at(tp.Vector3(*map(float, p)))


# --------------------------------------------------------------------------- #
#  The panel: one frame down the window's left edge, the view beside it
# --------------------------------------------------------------------------- #
INK, DIM, RULE = (0.93, 0.95, 0.97, 1.0), (0.56, 0.62, 0.68, 1.0), (1.0, 1.0, 1.0, 0.16)
CYAN, AMBER, RED, DARK = (0.38, 0.84, 1.0, 1.0), (1.0, 0.76, 0.28, 1.0), (1.0, 0.42, 0.36, 1.0), (0.05, 0.07, 0.09, 1.0)
CH, ROW = 7.0, 17.0            # the UI font's advance and the panel's line, px


def panel(ig, s, follow, keys, track, height):
    """Draws the panel with tp.imgui inside ImguiContext.render: what it is doing, the flight's
    numbers against their setpoints, the motors and the batteries, the keys (lit while held)
    and a chart of where it is. `track`: the flown track's world points, thinned."""
    ac, ap, pilot = s.aircraft, s.ap, s.pilot
    ig.set_next_window_pos(0.0, 0.0)
    ig.set_next_window_size(float(PANEL), float(height))
    if ig.begin("Babyshark 260 VTOL"):
        ig.dummy(PANEL - 16.0, height - 44.0)
        x0, y0, x1, y1 = ig.item_rect()
        x, o = ac.x, ac.out
        vg = ac.ground_velocity()
        wing = ap.mode in ("fw", "front")
        y = [y0 + 2.0]

        def rule():
            ig.draw_line(x0, y[0] + 3.0, x1, y[0] + 3.0, RULE, 1.0)
            y[0] += 10.0

        def row(label, value, note="", color=INK):
            ig.draw_text(x0, y[0], label, DIM)
            ig.draw_text(x0 + 13 * CH, y[0], value, color)
            if note:
                ig.draw_text(x0 + 25 * CH, y[0], note, DIM)
            y[0] += ROW

        def bar(label, fraction, color):
            ig.draw_text(x0, y[0], label, DIM)
            bx, bw = x0 + 13 * CH, 11 * CH
            f = min(max(fraction, 0.0), 1.0)
            ig.draw_rect(bx, y[0] + 2.0, bx + bw, y[0] + 11.0, RULE, 1.0, True)
            ig.draw_rect(bx, y[0] + 2.0, bx + bw * f, y[0] + 11.0, RED if f < 0.2 else color, 1.0, True)
            ig.draw_text(x0 + 25 * CH, y[0], f"{100.0 * f:3.0f} %", INK if f >= 0.2 else RED)
            y[0] += ROW

        def caps(names, text):
            cx = x0
            for name in names:
                held = keys.is_key_down(name)
                wd = CH * len(name) + 8.0
                ig.draw_rect(cx, y[0] - 2.0, cx + wd, y[0] + 14.0, CYAN if held else RULE, 1.0, held)
                if not held:
                    ig.draw_rect(cx, y[0] - 2.0, cx + wd, y[0] + 14.0, DIM, 1.0, False)
                ig.draw_text(cx + 4.0, y[0], name, DARK if held else INK)
                cx += wd + 4.0
            ig.draw_text(x0 + 14 * CH, y[0], text, DIM)
            y[0] += ROW + 3.0

        state = pilot.phase.upper()
        ig.draw_text(x0, y[0], state, {"down": RED, "auto": AMBER}.get(pilot.state, CYAN))
        y[0] += ROW
        hint = {"ground": "SPACE starts the rotors", "down": "it is on the ground: R resets", "landed": "the rotors stop",
                "auto": "any stick takes it back"}.get(pilot.state, "")
        ig.draw_text(x0, y[0], hint, DIM)
        y[0] += ROW
        rule()
        V = o.get("Va", 0.0)
        row("airspeed", f"{V:5.1f} m/s", f"set {ap.va_c:4.1f}" if wing else "")
        row("ground speed", f"{math.hypot(vg[0], vg[1]):5.1f} m/s")
        row("height", f"{ac.h:5.1f} m", f"set {ap.h_c if wing else ap.pos_c[2]:5.1f}" if pilot.state in ("fly", "auto") else "")
        row("climb", f"{-vg[2]:+5.1f} m/s", f"can {ap.fw_cfg['altitude']['climb_fraction'] * ap.climb_max * V:+4.1f}" if ap.mode == "fw" else "")
        row("heading", f"{x[11] * DEG % 360.0:5.0f} deg")
        row("bank, pitch", f"{x[9] * DEG:+4.0f} {x[10] * DEG:+4.0f} deg")
        wn, we, _ = ac.wind_ned()
        row("wind", f"{math.hypot(wn, we):5.1f} m/s", f"from {math.atan2(-we, -wn) * DEG % 360.0:3.0f}")
        row("the pad", f"{math.hypot(x[0], x[1]):5.0f} m", f"at {math.atan2(-x[1], -x[0]) * DEG % 360.0:3.0f}")
        rule()
        row("lift rev/s", " ".join(f"{v:3.0f}" for v in x[16:20]))
        bar("lift battery", ac.lift_fraction, CYAN)
        row("pusher rev/s", f"{x[15]:3.0f}")
        bar("cruise batt.", ac.cruise_fraction, CYAN)
        rule()
        caps(("W", "S"), "faster, slower" if wing else "forward, back")
        caps(("A", "D"), "bank" if wing else "turn")
        caps(("Q", "E"), "-" if wing else "sideways")
        caps(("SPACE", "SHIFT"), "up, down")
        caps(("T",), "to the hover" if wing else "to the wing")
        caps(("M",), "fly the circuit")
        caps(("H",), "home and land")
        caps(("V",), "weathervane " + ("on" if pilot.weathervane else "off"))
        caps(("C",), "camera: " + follow.mode)
        caps(("R",), "back on the pad")
        ig.draw_text(x0, y[0], "mouse", INK)
        ig.draw_text(x0 + 14 * CH, y[0], "look, wheel zooms", DIM)
        y[0] += ROW
        if y1 - y[0] > 90.0:
            rule()
            chart(ig, s, track, x0, y[0], x1, y1)
    ig.end()


def chart(ig, s, track, x0, y0, x1, y1):
    """North up: the pad, the circuit, the flown track and the aircraft, to a scale that keeps
    the aircraft on it."""
    ac = s.aircraft
    n, e, psi = ac.x[0], ac.x[1], ac.x[11]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * min(x1 - x0, y1 - y0) - 4.0
    reach = max([550.0, 1.15 * abs(n), 1.15 * abs(e) * (y1 - y0) / (x1 - x0)])
    k = half / reach

    def at(pn, pe):
        return (cx + k * pe, cy - k * pn)
    ig.draw_rect(x0, y0, x1, y1, RULE, 1.0, False)
    ig.draw_polyline([at(pn, pe) for pn, pe in [(0.0, 0.0)] + list(s.route) + [(0.0, 0.0)]], DIM, 1.0)
    if len(track) > 1:
        ig.draw_polyline([(cx + k * float(p[0]), cy + k * float(p[2])) for p in track], CYAN, 1.0)
    px, py = at(0.0, 0.0)
    ig.draw_circle(px, py, 3.0, INK, 1.0, True)
    ax, ay = at(n, e)
    fx, fy = math.sin(psi), -math.cos(psi)                   # the nose, on the chart
    ig.draw_polyline([(ax + 9.0 * fx, ay + 9.0 * fy), (ax - 5.0 * fx - 5.0 * fy, ay - 5.0 * fy + 5.0 * fx),
                      (ax - 5.0 * fx + 5.0 * fy, ay - 5.0 * fy - 5.0 * fx), (ax + 9.0 * fx, ay + 9.0 * fy)], AMBER, 1.5)
    ig.draw_text(x0 + 4.0, y0 + 3.0, "N up", DIM)
    label = f"{2.0 * reach / 1000.0:.1f} km"
    ig.draw_text(x1 - 4.0 - CH * len(label), y0 + 3.0, label, DIM)


# --------------------------------------------------------------------------- #
#  The window
# --------------------------------------------------------------------------- #
class Cockpit:
    def __init__(self, tp, a, spec, headless):
        self.tp, self.args = tp, a
        w, h = (int(v) for v in a.size.lower().split("x"))
        self.s = Session(a, spec)
        self.sc = sc = Scene(tp, self.s, a, w, h, headless)
        self.ui = tp.ImguiContext(sc.canvas, sc.r) if tp.HAS_IMGUI else None
        self.left = PANEL if self.ui else 0
        beam = math.radians(a.from_deg + 90.0)             # the spectator stands 60 m off the pad, across the wind
        self.follow = Follow(tp, sc.cam, sc.canvas, a.cam, (60.0 * math.cos(beam), 60.0 * math.sin(beam)))
        self.script = Script(DEMO) if a.demo else None
        sources = ([self.script] if self.script else []) + ([] if headless else [sc.canvas])
        self.keys = Keyboard(*sources)
        self.t = 0.0
        self.resize(w, h)
        sc.canvas.on_window_resize(self.resize)

    def resize(self, w, h):
        sc = self.sc
        self.size = (w, h)
        sc.r.set_size(w, h)
        sc.r.set_viewport(self.left, 0, max(w - self.left, 1), h)
        sc.cam.aspect = max(w - self.left, 1) / max(h, 1)
        sc.cam.update_projection_matrix()

    def tick(self, dt):
        """The keys, the flight and the camera over dt."""
        s, sc, k = self.s, self.sc, self.keys
        if self.script:
            self.script.at(self.t)
        if k.hit("R"):
            s.reset()
            sc.track_n = 0
            sc.track_g.set_draw_range(0, 0)
            self.follow.p = None
        if k.hit("T"):
            s.pilot.toggle()
        if k.hit("M"):
            s.pilot.fly(s.legs)
        if k.hit("H"):
            s.pilot.fly([dict(kind="land", pos=(0.0, 0.0))])
        if k.hit("V"):
            s.pilot.weathervane = not s.pilot.weathervane
        if k.hit("C"):
            self.follow.next()
        s.pilot.keys(k)
        s.advance(dt)
        self.t += dt
        sc.sync()
        self.follow.update(s.aircraft, dt, mouse=not (self.ui and self.ui.want_capture_mouse))

    def draw(self):
        sc = self.sc
        sc.render()
        if self.ui:
            n = sc.track_n
            track = sc.track_buf[:n:max(1, n // 400)]
            self.ui.render(lambda: panel(self.tp.imgui, self.s, self.follow, self.keys, track, self.size[1]))


def window(tp, a, spec):
    c = Cockpit(tp, a, spec, headless=False)
    if not c.ui:
        print(__doc__)
    clock = tp.Clock()

    def animate():
        c.tick(min(clock.get_delta(), 0.1))
        c.draw()
    c.sc.canvas.animate(animate)


def stills(tp, a, spec, times):
    """The demo without a window, a still at each time: the frames the window would draw."""
    c = Cockpit(tp, a, spec, headless=True)
    s = c.s
    os.makedirs(a.out_dir, exist_ok=True)
    for t in sorted(times):
        while c.t < t - 1e-9:
            c.tick(STEP)
            if abs(c.t / 5.0 - round(c.t / 5.0)) < 1e-6:
                ac = s.aircraft
                print(f"[fly] {c.t:5.0f} s  {s.pilot.phase:22s} {s.ap.mode:5s} N {ac.x[0]:7.1f} E {ac.x[1]:7.1f} h {ac.h:5.1f} "
                      f"Va {ac.out.get('Va', 0.0):4.1f}  keys {' '.join(sorted(c.script.held)) or '-'}")
        for _ in range(3):
            c.draw()
        path = os.path.join(a.out_dir, f"babyshark_fly_{t:05.1f}s.png")
        c.sc.r.save_frame(path)
        print(f"[fly] t {c.t:.1f} s, {s.pilot.phase}, {s.ap.mode}, camera {c.follow.mode} -> {path}")


def main(argv=None):
    a = parse(argv)
    if a.stills and not a.demo:
        sys.exit("--stills plays the demo's keys: add --demo")
    spec = load_spec(a.spec)
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))           # python/ (threepp)
    import threepp as tp
    if a.stills:
        stills(tp, a, spec, [float(t) for t in a.stills.split(",")])
    else:
        window(tp, a, spec)


if __name__ == "__main__":
    main()
