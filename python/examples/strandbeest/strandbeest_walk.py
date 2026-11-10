"""A Strandbeest walks downwind across sand, simulated by PhysX: nothing in the walk is posed.

    python strandbeest_walk.py                          # window: drag to orbit
    python strandbeest_walk.py --wind 9                 # a stronger wind (m/s; 6 only rocks it, 19 blows it over)
    python strandbeest_walk.py --telemetry --seconds 30 # no window: the run in numbers
    python strandbeest_walk.py --shot 12 --out walk.png # headless still at t = 12 s
    python strandbeest_walk.py --gl                     # the GLRenderer where Vulkan is available

It draws with the VulkanRenderer where this threepp has one, else with the GLRenderer.

The machine is ../rigs/strandbeest_rig.py: Theo Jansen's linkage, twenty legs on one crankshaft, as
one PhysX articulation whose loops are closed by joints between its own links. The wind pushes on
the sails; the feet's grip on the ground turns the legs and the crank. It surges in the gusts and
slows in the lulls, and where the wind is too weak to lift it over the gait's humps it rocks in
place. The sand is a height field with gentle swells, under a clear sky built in numpy that
lights the scene (an environment map) and is its backdrop.

Build the model once for the lashed canes (plain tubes are drawn without it):

    blender --background --factory-startup --python build_strandbeest_blender.py
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))            # python/ (threepp)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rigs"))       # examples/rigs/ (strandbeest_rig)
import strandbeest_rig as SB  # noqa: E402

HZ = 480                       # physics rate
FRAME = 1.0 / 60.0             # the sails and the picture
X0, X1, Z0, Z1, CELL = -40.0, 200.0, -50.0, 50.0, 0.5
SUN = (42.0, 39.0, 78.0)       # where the sun shines from (the light's position; it aims at the origin)
SKY_FILL = 0.7                 # scene.environment_intensity: the sky lights as much as the HemisphereLight it replaced


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wind", type=float, default=7.6, help="mean wind speed, m/s, blowing along +x")
    ap.add_argument("--seconds", type=float, default=30.0, help="length of a --telemetry run")
    ap.add_argument("--telemetry", action="store_true", help="no window: print the run")
    ap.add_argument("--shot", type=float, default=None, help="headless still at this time, s")
    ap.add_argument("--out", default="strandbeest_walk.png")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--gl", action="store_true", help="draw with the GLRenderer even where Vulkan is available")
    return ap.parse_args(argv)


def ground_height(x, z):
    """Gentle swells, a few centimetres over the machine's length."""
    return 0.10 * np.sin(0.11 * x + 0.3) * np.cos(0.07 * z) + 0.05 * np.sin(0.23 * z + 0.05 * x) + 0.03 * np.sin(0.41 * x - 0.17 * z)


def wind_at(t, mean):
    """(wind along +x, across along +z) m/s: gusts and lulls of some seconds, and a wandering cross part."""
    u = mean * (1.0 + 0.16 * math.sin(0.45 * t + 0.7) + 0.10 * math.sin(1.1 * t + 2.0) + 0.04 * math.sin(2.9 * t))
    w = mean * (0.07 * math.sin(0.31 * t + 1.9) + 0.04 * math.sin(0.83 * t))
    return np.array([u, 0.0, w])


class Walk:
    """The physics: the sand, the machine standing on it, the wind."""

    def __init__(self, tp, a):
        self.tp, self.a = tp, a
        self.world = tp.PhysxWorld(tp.Vector3(0, -9.81, 0), 1.0 / HZ, 64, 4, False, False, True)
        xs, zs = np.arange(X0, X1 + CELL / 2, CELL), np.arange(Z0, Z1 + CELL / 2, CELL)
        self.xs, self.zs = xs, zs
        self.H = ground_height(xs[None, :], zs[:, None]).astype(np.float32)          # rows are z
        self.world.add_static_heightfield(self.H, CELL, tp.Vector3(X0, 0.0, Z0), 0.5, "y")
        y = float(ground_height(0.0, 0.0)) - SB.lowest_foot(0.0) + 0.03
        self.machine = SB.Machine(tp, self.world, (0.0, y, 0.0))
        self.t = 0.0
        self.wind = np.zeros(3)
        self._o = self.machine.frame()[0]
        self.velocity = np.zeros(3)

    def step(self, dt):
        m = self.machine
        self.wind = wind_at(self.t, self.a.wind) if self.t >= 1.0 else np.zeros(3)      # a second to settle on its feet
        force, point = SB.sail_thrust(m, self.wind, self.velocity)
        m.push(force, point)
        self.world.step(dt)
        self.t += dt
        o = m.frame()[0]
        self.velocity, self._o = (o - self._o) / dt, o
        return force

    def fallen(self):
        o, R = self.machine.frame()
        return R[1, 1] < 0.5 or o[0] > X1 - 15.0 or abs(o[2]) > Z1 - 15.0


def telemetry(tp, a):
    w = Walk(tp, a)
    m = w.machine
    print(f"{len(m.names)} links, {len(m._joints)} loop joints, {m.mass:.0f} kg, {len(m.feet)} feet; wind {a.wind:.1f} m/s")
    speeds, gaps, downs = [], [], []
    th0 = m.crank_angle()
    turns = 0.0
    for k in range(int(a.seconds * 30)):
        w.step(1.0 / 30.0)
        o, R = m.frame()
        th = m.crank_angle()
        turns += ((th - th0 + math.pi) % (2 * math.pi) - math.pi) / (2 * math.pi)
        th0 = th
        f = m.foot_positions()
        down = int((f[:, 1] - ground_height(f[:, 0], f[:, 2]) < 0.02).sum())
        gap = m.loop_gap()
        if w.t > 2.0:
            speeds.append(float(w.velocity @ R[:, 0]))
            gaps.append(gap)
            downs.append(down)
        if k % 60 == 59:
            print(f"t {w.t:5.1f} s  x {o[0]:+7.2f} z {o[2]:+6.2f} m  speed {float(w.velocity @ R[:, 0]):+5.2f} m/s  crank {turns:+6.2f} turns  "
                  f"wind {w.wind[0]:4.1f} m/s  tilt {math.degrees(math.acos(min(1.0, R[1, 1]))):4.1f} deg  feet down {down:2d}  loop gap {gap * 1000:4.1f} mm")
        if w.fallen():
            print(f"t {w.t:5.1f} s: it is over on its side or off the sand")
            break
    if speeds:
        print(f"from t = 2 s: speed {min(speeds):+.2f}..{max(speeds):+.2f} m/s (mean {np.mean(speeds):.2f}), feet down {min(downs)}..{max(downs)}, "
              f"largest loop gap {max(gaps) * 1000:.1f} mm")


def sand_normal_map(tp, n=256, seed=4):
    """Wind ripples as a tileable normal map (filtered noise, crests across the wind)."""
    rng = np.random.default_rng(seed)
    fx, fz = np.fft.fftfreq(n, 1.0 / n)[None, :], np.fft.fftfreq(n, 1.0 / n)[:, None]
    h = np.fft.ifft2(np.fft.fft2(rng.normal(0, 1, (n, n))) * np.exp(-((np.abs(fx) - 9.0) / 4.0) ** 2 - (fz / 3.0) ** 2)).real
    h /= np.abs(h).max()
    gx, gz = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 2.2, (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 2.2
    nz = 1.0 / np.sqrt(1.0 + gx * gx + gz * gz)
    px = np.stack([0.5 - 0.5 * gx * nz, 0.5 - 0.5 * gz * nz, 0.5 + 0.5 * nz], -1)
    tex = tp.data_texture(np.ascontiguousarray(np.clip(px * 255.0 + 0.5, 0, 255).astype(np.uint8)), False)
    tex.wrap_s = tex.wrap_t = tp.TextureWrapping.Repeat
    return tex


def sky_texture(tp, h=256):
    """A clear sky over sand as an equirect map of linear radiance (row 0 is straight down).

    Tone mapped (ACES, exposure 1), the horizon is the fog's colour, so far sand fades into it, and
    the zenith a pale blue, with a soft warm aureole round the sun. Below the horizon, hazy far sand
    for the background past the sand's edge, then the near sand's bounce. The sun itself is the
    DirectionalLight, not in the map."""
    el = ((np.arange(h) + 0.5) / h - 0.5)[:, None] * np.pi
    az = ((np.arange(2 * h) + 0.5) / (2 * h) - 0.5)[None, :] * 2 * np.pi
    d = np.stack([np.cos(el) * np.cos(az), np.sin(el) * np.ones_like(az), np.cos(el) * np.sin(az)], -1)
    horizon, zenith = np.array([0.389, 0.511, 0.682]), np.array([0.128, 0.238, 0.515])
    far, near = np.array([0.446, 0.388, 0.305]), np.array([0.103, 0.059, 0.024])
    up = (np.sin(np.clip(el, 0.0, None)) ** 0.5)[..., None]
    haze = np.clip(-el / math.radians(2.0), 0.0, 1.0)[..., None]                       # the first 2 degrees down
    deep = np.clip((-el - math.radians(2.0)) / math.radians(13.0), 0.0, 1.0)[..., None]
    sky = horizon * (1.0 - up) + zenith * up
    ground = (horizon * (1.0 - haze) + far * haze) * (1.0 - deep) + near * deep
    s = np.array(SUN) / np.linalg.norm(SUN)
    angle = np.arccos(np.clip(d @ s, -1.0, 1.0))
    aureole = 0.45 * np.exp(-(angle / math.radians(12.0)) ** 2) * (el >= 0.0)
    L = np.where(el[..., None] >= 0.0, sky, ground) + aureole[..., None] * np.array([0.60, 0.50, 0.36])
    return tp.float_texture(np.ascontiguousarray(L, np.float32))


class Scene:
    def __init__(self, tp, walk, w, h, headless, vulkan):
        self.tp, self.walk, self.vulkan = tp, walk, vulkan
        # Vulkan presents without vsync (MAILBOX), as its other examples do. It traces the sun's
        # shadow and smooths edges with TAA, so the canvas's antialiasing, the shadow map's size and
        # bias and the shadow box follow() slides are read by GL only.
        self.canvas = tp.Canvas("threepp - Strandbeest", width=w, height=h, antialiasing=4, headless=headless,
                                vsync=not (headless or vulkan))
        r = self.r = tp.VulkanRenderer(self.canvas) if vulkan else tp.GLRenderer(self.canvas)
        r.shadow_map_enabled = True
        r.tone_mapping = tp.ToneMapping.ACESFilmic
        r.tone_mapping_exposure = 1.0
        s = self.scene = tp.Scene()
        s.background = s.environment = sky_texture(tp)       # the sky is the fill light and the backdrop
        s.environment_intensity = SKY_FILL
        s.set_fog(tp.Color(0xb9c6d3), 70.0, 240.0)
        self.sun = tp.DirectionalLight(0xffe2b8, 2.6)
        self.sun.cast_shadow = True
        self.sun.shadow.map_size.set(4096, 4096)
        self.sun.shadow.bias = -0.0004
        # the sun shines at the world's origin from far off; its shadow box is slid along with the machine
        self.sun.position.set(*SUN)
        z = np.array(SUN) / np.linalg.norm(SUN)
        self._sun_x = np.cross([0.0, 1.0, 0.0], z) / np.linalg.norm(np.cross([0.0, 1.0, 0.0], z))
        self._sun_y = np.cross(z, self._sun_x)
        s.add(self.sun)
        # the sand: the height field's own samples as a mesh
        xs, zs = walk.xs, walk.zs
        X, Z = np.meshgrid(xs, zs)
        P = np.stack([X, walk.H, Z], -1).reshape(-1, 3).astype(np.float32)
        nx, nz = len(xs), len(zs)
        i, j = np.meshgrid(np.arange(nx - 1), np.arange(nz - 1))
        a = (j * nx + i).ravel()
        idx = np.stack([a, a + nx, a + 1, a + 1, a + nx, a + nx + 1], -1).ravel().astype(np.uint32)
        geo = tp.BufferGeometry()
        geo.set_attribute("position", P)
        geo.set_attribute("uv", np.ascontiguousarray(np.stack([X / 2.5, Z / 2.5], -1).reshape(-1, 2), np.float32))
        geo.set_index(idx)
        geo.compute_vertex_normals()
        sand = tp.MeshStandardMaterial()
        sand.color = tp.Color(0.80, 0.62, 0.40)
        sand.roughness = 1.0
        sand.normal_map = sand_normal_map(tp)
        ground = tp.Mesh(geo, sand)
        ground.receive_shadow = True
        s.add(ground)
        # the machine
        path = SB.model_path()
        model = tp.GLTFLoader().load(path).scene if path else None
        if model is None:
            print("strandbeest.glb is not built (plain tubes are drawn): blender --background --factory-startup --python build_strandbeest_blender.py")
        self.visual = SB.Visual(tp, s, walk.machine, model)
        self.sails = SB.Sails(tp, s, self.visual.sail_material, walk.machine.frame())
        self.cam = tp.PerspectiveCamera(42.0, w / h, 0.1, 800.0)
        o = walk.machine.frame()[0]
        self.cam.position.set(float(o[0]) + 5.0, float(o[1]) - 1.6, float(o[2]) + 15.0)
        self.target = np.array(o) + [0.0, -0.3, 0.0]
        self.cam.look_at(*[float(v) for v in self.target])

    def follow(self):
        """Carry the camera and the sun's shadow box along with the machine."""
        o = self.walk.machine.frame()[0] + np.array([0.0, -0.3, 0.0])
        d = o - self.target
        self.target = o
        p = self.cam.position
        p.set(p.x + float(d[0]), p.y + float(d[1]), p.z + float(d[2]))
        u, v = float(o @ self._sun_x), float(o @ self._sun_y)
        self.sun.set_shadow_frustum(u - 14.0, u + 14.0, v + 14.0, v - 14.0)
        return d

    def advance(self, dt):
        w = self.walk
        w.step(dt)
        self.visual.update()
        self.sails.step(w.machine.frame(), dt, w.t, w.wind if w.t >= 1.0 else wind_at(w.t, w.a.wind))

    def render(self):
        self.r.render(self.scene, self.cam)


def still(tp, a, w, h, vulkan):
    from threepp import capture
    sc = Scene(tp, Walk(tp, a), w, h, headless=True, vulkan=vulkan)
    while sc.walk.t < a.shot - 1e-9:
        sc.advance(FRAME)
    sc.follow()
    sc.cam.look_at(*[float(v) for v in sc.target])
    if vulkan:
        sc.r.sim_time = sc.walk.t       # the walk's clock, not the wall's, for TAA and anything else timed
    for _ in range(16 if vulkan else 3):  # Vulkan: let TAA settle on the held pose
        sc.render()
    capture.write_png(a.out, np.ascontiguousarray(sc.r.read_pixels()))
    o, R = sc.walk.machine.frame()
    print(f"wrote {a.out} ({'Vulkan' if vulkan else 'GL'}): t = {sc.walk.t:.1f} s, x = {o[0]:+.2f} m, "
          f"loop gap {sc.walk.machine.loop_gap() * 1000:.1f} mm")


def window(tp, a, w, h, vulkan):
    state = {"sc": Scene(tp, Walk(tp, a), w, h, headless=False, vulkan=vulkan), "debt": 0.0}
    sc = state["sc"]
    controls = tp.OrbitControls(sc.cam, sc.canvas)
    controls.target.set(*[float(v) for v in sc.target])
    controls.update()
    sc.canvas.on_window_resize(lambda ww, hh: (setattr(sc.cam, "aspect", ww / max(hh, 1)), sc.cam.update_projection_matrix(), sc.r.set_size(ww, hh)))
    clock = tp.Clock()

    def animate():
        dt = min(clock.get_delta(), 0.1)
        state["debt"] += dt
        n = min(int(state["debt"] / FRAME), 3)                 # never more than three steps a frame: slow motion over a stall
        state["debt"] -= n * FRAME if n < 3 else state["debt"]
        if not sc.walk.fallen():
            for _ in range(n):
                sc.advance(FRAME)
        sc.follow()
        controls.target.set(*[float(v) for v in sc.target])
        controls.update()
        sc.render()
    sc.canvas.animate(animate)


def main(argv=None):
    a = parse(argv)
    import threepp as tp
    if a.telemetry:
        telemetry(tp, a)
        return
    w, h = (int(v) for v in a.size.lower().split("x"))
    vulkan = tp.HAS_VULKAN and not a.gl
    if a.shot is not None:
        still(tp, a, w, h, vulkan)
        return
    window(tp, a, w, h, vulkan)


if __name__ == "__main__":
    main()
