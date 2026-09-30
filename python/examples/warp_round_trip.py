"""Warp and threepp both ways round, with nothing copied through the CPU.

A curtain hangs in a gusty wind while a chrome ball sweeps through it, and a
nozzle sprays sparks into the same wind; they bounce off the cloth, the ball and
the floor. NVIDIA Warp simulates both, and writes them into buffers the Vulkan
renderer shares with it: the cloth's vertices (`enable_vertex_interop`) and the
sparks' particle field (`enable_particle_field_interop`). The ray tracer sees
the new shapes in the same frame, in the ball's reflection and in the shadows.
Then the renderer hands the finished frame back (`enable_frame_interop`), and a
third Warp kernel reads it where it lies and turns it into what an event camera
would see: a pixel fires when its brightness has changed by more than a
threshold since it last fired.

    python warp_round_trip.py                        # a window; drag to orbit, Esc quits
    python warp_round_trip.py --video 8              # headless: warp_round_trip.mp4, render | events
    python warp_round_trip.py --stream 1920x1080 --seconds 30 [--dots] [--preroll 1]
                                                     # raw frames on stdout, for the lesson film

`--dots` prints one dot per particle on the cloth, coloured by particle index:
one dot is one Warp thread. NVIDIA only: every direction imports Vulkan memory
into CUDA. The `# [name]` markers are where the lesson film (threepp-lessons, films/warp_threepp.py)
quotes this file.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from threepp.cuda_interop import VkInteropArray
from warp_common import Encoder, cli_arg, parse_size, standard_material

# --- the cloth, the sparks, the wind ------------------------------------------------

NX, NY = 63, 47                  # quad segments; (NX+1)*(NY+1) particles
WIDTH, HEIGHT = 3.2, 2.4         # metres
TOP_Y = 3.1                      # height of the rod
N = (NX + 1) * (NY + 1)
DT = 1.0 / 240.0
SUBSTEPS, ITERATIONS = 4, 24     # 4 substeps = one 60 Hz frame
BALL_R = 0.5
NS = 1 << 13                     # sparks in the pool
THRESHOLD = 0.12                 # an event: log brightness changed by this much

GRAVITY = wp.constant(wp.vec3(0.0, -9.81, 0.0))
WIND = wp.constant(wp.vec3(0.33, 0.0, 0.94))
COLS = wp.constant(NX + 1)
LAST_X, LAST_Y = wp.constant(NX), wp.constant(NY)
DAMPING = wp.constant(0.015)
BEND = wp.constant(0.4)          # the skip-one springs (stiffer diverges under this Jacobi relaxation)
EMIT = wp.constant(wp.vec3(-2.5, 0.9, 3.5))                   # where the sparks come from: just out of shot
NOZZLE = wp.constant(wp.normalize(wp.vec3(0.5, 0.42, -0.76)))
BALL_RADIUS = wp.constant(BALL_R)
TOUCH = wp.constant(0.07)        # how close a spark comes to a cloth particle before it bounces
SPARK_R = wp.constant(0.011)


@wp.func
def wind(p: wp.vec3, t: float) -> wp.vec3:
    """One gusty wind for the cloth and the sparks: gusts that roll across the room."""
    gust = wp.sin(1.7 * t - 1.9 * p[0] + 0.8 * p[1])
    return WIND * (2.5 + 4.5 * gust)


# [kernel]
@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
              pred: wp.array(dtype=wp.vec3), free: wp.array(dtype=float),
              t: float, dt: float):
    i = wp.tid()                                  # this thread's particle
    p = pos[i]
    v = (p - prev[i]) * (1.0 - DAMPING)
    prev[i] = p
    a = GRAVITY + wind(p, t)
    pred[i] = p + (v + a * dt * dt) * free[i]     # pinned particles: free = 0
# [/kernel]


@wp.func
def pull(p: wp.vec3, pos: wp.array(dtype=wp.vec3), ix: int, iy: int, rest: float, k: float) -> wp.vec3:
    if ix < 0 or ix > LAST_X or iy < 0 or iy > LAST_Y:
        return wp.vec3(0.0, 0.0, 0.0)
    d = pos[iy * COLS + ix] - p
    l = wp.length(d)
    if l < 1.0e-9:
        return wp.vec3(0.0, 0.0, 0.0)
    return d * (0.5 * k * (l - rest) / l)          # half: the neighbour moves the other half


@wp.kernel
def solve(p_in: wp.array(dtype=wp.vec3), p_out: wp.array(dtype=wp.vec3), free: wp.array(dtype=float),
          rx: float, ry: float, ball: wp.vec3):
    """One Jacobi pass over the stretch, shear and bend springs, then the ball and the floor."""
    i = wp.tid()
    p = p_in[i]
    if free[i] == 0.0:
        p_out[i] = p
        return
    ix = i % COLS
    iy = i // COLS
    rd = wp.sqrt(rx * rx + ry * ry)
    c = pull(p, p_in, ix - 1, iy, rx, 1.0) + pull(p, p_in, ix + 1, iy, rx, 1.0)
    c += pull(p, p_in, ix, iy - 1, ry, 1.0) + pull(p, p_in, ix, iy + 1, ry, 1.0)
    c += pull(p, p_in, ix - 1, iy - 1, rd, 0.85) + pull(p, p_in, ix + 1, iy - 1, rd, 0.85)
    c += pull(p, p_in, ix - 1, iy + 1, rd, 0.85) + pull(p, p_in, ix + 1, iy + 1, rd, 0.85)
    c += pull(p, p_in, ix - 2, iy, 2.0 * rx, BEND) + pull(p, p_in, ix + 2, iy, 2.0 * rx, BEND)
    c += pull(p, p_in, ix, iy - 2, 2.0 * ry, BEND) + pull(p, p_in, ix, iy + 2, 2.0 * ry, BEND)
    p = p + c * 0.3                               # everyone moves at once: under-relax
    d = p - ball
    l = wp.length(d)
    if l < BALL_RADIUS + 0.03:
        p = ball + d * ((BALL_RADIUS + 0.03) / wp.max(l, 1.0e-6))
    p_out[i] = wp.vec3(p[0], wp.max(p[1], 0.01), p[2])


@wp.kernel
def normals(pos: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    ix = i % COLS
    iy = i // COLS
    du = pos[iy * COLS + wp.min(ix + 1, LAST_X)] - pos[iy * COLS + wp.max(ix - 1, 0)]
    dv = pos[wp.min(iy + 1, LAST_Y) * COLS + ix] - pos[wp.max(iy - 1, 0) * COLS + ix]
    n = wp.cross(dv, du)                          # PlaneGeometry rows run top-down
    nrm[i] = n / wp.max(wp.length(n), 1.0e-9)


@wp.func
def bounce(vel: wp.vec3, n: wp.vec3, restitution: float) -> wp.vec3:
    """Reflect the part of vel going into a surface with normal n."""
    vn = wp.dot(vel, n)
    if vn < 0.0:
        return vel - n * ((1.0 + restitution) * vn)
    return vel


# [sparks]
@wp.kernel
def sparks(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), age: wp.array(dtype=float),
           life: wp.array(dtype=float), grid: wp.uint64, cloth: wp.array(dtype=wp.vec3),
           cloth_n: wp.array(dtype=wp.vec3), ball: wp.vec3, t: float, dt: float, seed: int):
    i = wp.tid()                                  # this thread's spark
    a = age[i] + dt
    if a >= life[i] or (a >= 0.0 and age[i] < 0.0):     # burnt out, or due: a new one leaves the nozzle
        s = wp.rand_init(seed, i)
        age[i] = 0.0
        life[i] = 0.8 + 1.4 * wp.randf(s)
        x[i] = EMIT
        v[i] = (NOZZLE + wp.vec3(wp.randn(s), wp.randn(s), wp.randn(s)) * 0.06) * (7.0 + 2.0 * wp.randf(s))
        return
    age[i] = a
    if a < 0.0:                                   # not lit yet
        return
    p = x[i]
    vel = v[i] + (GRAVITY + (wind(p, t) * 0.3 - v[i]) * 0.9) * dt      # fall, and drift with the wind
    q = p + vel * dt
    d = q - ball                                  # off the ball
    if wp.length(d) < BALL_RADIUS:
        n = wp.normalize(d)
        q = ball + n * BALL_RADIUS
        vel = bounce(vel, n, 0.5)
    best = TOUCH                                  # off the cloth: its nearest particle
    near = int(-1)
    query = wp.hash_grid_query(grid, q, TOUCH)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        dj = wp.length(q - cloth[j])
        if dj < best:
            best = dj
            near = j
    if near >= 0:
        n = cloth_n[near]
        if wp.dot(p - cloth[near], n) < 0.0:      # the side the spark came from
            n = -n
        depth = TOUCH - wp.dot(q - cloth[near], n)
        if depth > 0.0:
            q = q + n * depth
            vel = bounce(vel, n, 0.35) * 0.85
    if q[1] < SPARK_R:                            # off the floor
        q = wp.vec3(q[0], SPARK_R, q[2])
        vel = bounce(vel, wp.vec3(0.0, 1.0, 0.0), 0.35) * 0.7
    x[i] = q
    v[i] = vel
# [/sparks]


@wp.kernel
def spark_look(x: wp.array(dtype=wp.vec3), age: wp.array(dtype=float), life: wp.array(dtype=float),
               out_pos: wp.array(dtype=wp.vec4), out_col: wp.array(dtype=wp.vec4)):
    """Each spark as the renderer's particle field reads it: xyz and radius, and a colour
    that cools from white-hot through orange to a dull red."""
    i = wp.tid()
    a = age[i]
    if a < 0.0:
        out_pos[i] = wp.vec4(0.0, 0.0, 0.0, -1.0)     # w < 0: a dead slot
        return
    u = a / life[i]
    hot = wp.vec3(6.0, 4.0, 1.6)
    mid = wp.vec3(3.0, 0.8, 0.12)
    cold = wp.vec3(0.35, 0.03, 0.0)
    c = wp.lerp(hot, mid, wp.min(2.0 * u, 1.0))
    c = wp.lerp(c, cold, wp.max(2.0 * u - 1.0, 0.0))
    p = x[i]
    out_pos[i] = wp.vec4(p[0], p[1], p[2], SPARK_R * (1.0 - 0.5 * u))
    out_col[i] = wp.vec4(c[0], c[1], c[2], 1.0)


@wp.kernel
def spark_mesh(x: wp.array(dtype=wp.vec3), age: wp.array(dtype=float), life: wp.array(dtype=float),
               corner: wp.array(dtype=wp.vec3), out_pos: wp.array(dtype=wp.vec3), out_nrm: wp.array(dtype=wp.vec3)):
    """Each spark as a tiny glowing octahedron, for the ray tracer: the sprites are drawn after
    the traced frame, so without these the chrome would not reflect the sparks. A spark that is
    not alive collapses to a point."""
    i = wp.tid()
    a = age[i]
    r = float(0.0)
    if a >= 0.0:
        r = SPARK_R * wp.sqrt(wp.max(1.0 - a / life[i], 0.0))
    p = x[i]
    for k in range(24):
        c = corner[k]
        out_pos[i * 24 + k] = p + c * r
        out_nrm[i * 24 + k] = c


def octahedron_soup():
    """The 24 corners of an octahedron as a triangle soup, counter-clockwise from outside."""
    X, Y, Z = np.eye(3, dtype=np.float32)
    faces = [(X, Y, Z), (Y, -X, Z), (-X, -Y, Z), (-Y, X, Z), (Y, X, -Z), (-X, Y, -Z), (-Y, -X, -Z), (X, -Y, -Z)]
    return np.array([v for f in faces for v in f], np.float32)


@wp.func
def brightness(frame: wp.array3d(dtype=wp.uint8), y: int, x: int) -> float:
    """Log luminance of a BGRA pixel, as an event camera's photoreceptor sees it."""
    lum = 0.114 * float(frame[y, x, 0]) + 0.587 * float(frame[y, x, 1]) + 0.299 * float(frame[y, x, 2])
    return wp.log(1.0 + lum)


@wp.func
def paint(out: wp.array3d(dtype=wp.uint8), y: int, x: int, c: wp.vec3):
    out[y, x, 0] = wp.uint8(c[0])
    out[y, x, 1] = wp.uint8(c[1])
    out[y, x, 2] = wp.uint8(c[2])


# [events]
@wp.kernel
def events(frame: wp.array3d(dtype=wp.uint8), ref: wp.array2d(dtype=float),
           out: wp.array3d(dtype=wp.uint8), threshold: float):
    y, x = wp.tid()                               # one thread per pixel
    level = brightness(frame, y, x)
    d = level - ref[y, x]
    if d > threshold:                             # brighter: an ON event
        ref[y, x] = level
        paint(out, y, x, wp.vec3(255.0, 92.0, 80.0))
    elif d < -threshold:                          # darker: an OFF event
        ref[y, x] = level
        paint(out, y, x, wp.vec3(76.0, 201.0, 240.0))
    else:                                         # no event: what was shown fades
        paint(out, y, x, wp.vec3(float(out[y, x, 0]), float(out[y, x, 1]), float(out[y, x, 2])) * 0.84)
# [/events]


@wp.kernel
def seed(frame: wp.array3d(dtype=wp.uint8), ref: wp.array2d(dtype=float)):
    y, x = wp.tid()
    ref[y, x] = brightness(frame, y, x)


@wp.kernel
def swap_rb(frame: wp.array3d(dtype=wp.uint8)):
    """An RGBA export (the uncommon case) as BGRA, so `events` reads one layout."""
    y, x = wp.tid()
    r = frame[y, x, 0]
    frame[y, x, 0] = frame[y, x, 2]
    frame[y, x, 2] = r


def ball_at(t):
    return wp.vec3(1.0 * math.sin(0.55 * t), 1.55 + 0.35 * math.sin(0.9 * t), 0.4 + 0.5 * math.sin(0.37 * t))


def camera_at(t, cam):
    az = math.radians(24.0 + 16.0 * math.sin(0.11 * t))
    r = 6.0
    cam.position.set(r * math.sin(az), 1.8 + 0.25 * math.sin(0.07 * t), r * math.cos(az))
    cam.look_at(-0.25, 1.62, 0.3)


def studio_env(key_dir, w=1024, h=512):
    """A dark studio as a float equirect (same layout as warp_common.sky_env): a near-black
    room with a warm key softbox on the sun's line and a cool strip behind, so the chrome
    ball has something to reflect and the backdrop stays dark."""
    elev = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * math.pi
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * math.pi
    d = np.empty((h, w, 3), np.float32)
    d[..., 0] = np.cos(elev)[:, None] * np.cos(az)[None, :]
    d[..., 1] = np.sin(elev)[:, None]
    d[..., 2] = np.cos(elev)[:, None] * np.sin(az)[None, :]
    up = np.clip(d[..., 1], 0.0, 1.0)[..., None]
    col = np.float32([0.010, 0.012, 0.017]) + up * np.float32([0.012, 0.016, 0.026])

    def box(direction, radius_deg, power, tint):
        v = np.asarray(direction, np.float32)
        ang = np.degrees(np.arccos(np.clip(d @ (v / np.linalg.norm(v)), -1.0, 1.0)))
        return (np.clip((radius_deg - ang) / 3.0, 0.0, 1.0) * power)[..., None] * np.float32(tint)
    col += box(key_dir, 16.0, 7.0, (1.0, 0.93, 0.82))
    col += box((-0.7, 0.45, -0.55), 10.0, 5.0, (0.55, 0.75, 1.0))
    col += box((-0.9, 0.2, 0.4), 7.0, 2.5, (1.0, 0.55, 0.35))
    col += box((0.2, 0.15, 1.0), 38.0, 0.35, (0.8, 0.85, 1.0))          # a dim wall behind the camera
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = col
    return tp.float_texture(out)


def dot_texture():
    """One bright dot per cloth particle, coloured along the particle index: one dot, one thread."""
    cell = 24
    img = np.full(((NY + 1) * cell, (NX + 1) * cell, 3), 0.07, np.float32)
    yy, xx = np.mgrid[0:cell, 0:cell]
    disc = np.clip(1.6 - np.hypot(yy - cell / 2 + 0.5, xx - cell / 2 + 0.5) / 3.2, 0, 1)[..., None]
    for iy in range(NY + 1):
        for ix in range(NX + 1):
            u = (iy * (NX + 1) + ix) / (N - 1)
            c = np.array([0.5 + 0.5 * math.cos(2 * math.pi * (u + 0.0)), 0.5 + 0.5 * math.cos(2 * math.pi * (u + 0.33)),
                          0.5 + 0.5 * math.cos(2 * math.pi * (u + 0.67))]) * 0.85 + 0.15
            y0, x0 = iy * cell, ix * cell
            img[y0:y0 + cell, x0:x0 + cell] = img[y0:y0 + cell, x0:x0 + cell] * (1 - disc) + c * disc
    # the plane's UVs put vertex (ix, iy) at (ix / NX, 1 - iy / NY): stretch the cells so each dot sits on one
    h, w = img.shape[:2]
    ys = np.clip(((np.arange(NY * cell) + 0.5) / (NY * cell) * NY * cell + cell / 2).astype(int), 0, h - 1)
    xs = np.clip(((np.arange(NX * cell) + 0.5) / (NX * cell) * NX * cell + cell / 2).astype(int), 0, w - 1)
    img = img[ys][:, xs]
    return tp.data_texture((np.clip(img, 0, 1) * 255).astype(np.uint8), True)


class RoundTrip:
    def __init__(self, width, height, headless, dots=False):
        self.film_camera = headless                   # a window orbits with the mouse instead
        wp.init()
        self.device = dev = wp.get_device("cuda:0")
        self.w, self.h = width, height
        self.canvas = tp.Canvas("threepp x warp - round trip", width=width, height=height, vsync=False,
                                headless=headless)
        r = self.renderer = tp.VulkanRenderer(self.canvas)
        r.tone_mapping = tp.ToneMapping.ACESFilmic
        r.tone_mapping_exposure = 0.9
        r.shadow_map_enabled = True

        scene = self.scene = tp.Scene()
        sun_dir = np.array([0.45, 0.8, 0.4], np.float32)
        sun_dir /= np.linalg.norm(sun_dir)
        scene.environment = scene.background = studio_env(sun_dir)
        sun = tp.DirectionalLight(0xfff1dc, 2.4)
        sun.position.set(*(sun_dir * 20.0))
        sun.cast_shadow = True
        scene.add(sun)
        floor = tp.Mesh(tp.PlaneGeometry(40, 40), standard_material(0x0b0d12, 0.07, 0.0))
        floor.rotate_x(-math.pi / 2)
        floor.receive_shadow = True
        scene.add(floor)

        fabric = standard_material(0xffffff if dots else 0xe8502e, 0.75, 0.0, side=tp.Side.Double)
        if dots:
            fabric.map = dot_texture()
        self.cloth = tp.Mesh(tp.PlaneGeometry(WIDTH, HEIGHT, NX, NY), fabric)
        self.cloth.cast_shadow = self.cloth.receive_shadow = True
        self.cloth.frustum_culled = False
        scene.add(self.cloth)
        self.ball = tp.Mesh(tp.SphereGeometry(BALL_R, 64, 32), standard_material(0xffffff, 0.05, 1.0))
        self.ball.cast_shadow = True
        scene.add(self.ball)

        # the glow the (out of shot) source of the sparks throws on the scene
        e, nz = np.array(EMIT, np.float32), np.array(NOZZLE, np.float32)
        self.glow = tp.PointLight(0xff8a3a, 2.0, 0.0, 2.0)
        self.glow.position.set(*(e + nz * 0.5))
        scene.add(self.glow)
        # the sparks as geometry, which the ray tracer sees (reflections, emissive light)
        g = tp.BufferGeometry()
        g.set_attribute("position", np.zeros((NS * 24, 3), np.float32))
        g.set_attribute("normal", np.tile(octahedron_soup(), (NS, 1)))
        hot = standard_material(0x000000, 1.0, 0.0, emissive=tp.Color(1.0, 0.48, 0.14), emissive_intensity=8.0)
        self.proxies = tp.Mesh(g, hot)
        self.proxies.frustum_culled = False
        scene.add(self.proxies)
        self.corner = wp.array(octahedron_soup(), dtype=wp.vec3, device=dev)

        cfg = tp.ParticleField.Config()
        cfg.capacity = NS
        cfg.ownership = tp.ParticleField.Ownership.Interop
        cfg.w_semantic = tp.ParticleField.WSemantic.Radius
        cfg.uniform_radius = float(SPARK_R)
        cfg.attributes = True                         # per-spark colour, written by a kernel
        self.field = tp.ParticleField.create(cfg)
        self.field.frustum_culled = False
        self.field.set_billboard_repr(tp.Color(1, 1, 1), tp.Color(1, 1, 1), 1.0, 1.0)
        bb = self.field.billboard_repr
        bb.softness = 0.3
        bb.bright_jitter = 0.0
        bb.fade_power = 0.0
        bb.size_taper = 0.0
        bb.glow = 1.0
        scene.add(self.field)

        self.camera = tp.PerspectiveCamera(40, width / height, 0.1, 100)
        camera_at(0.0, self.camera)

        xs = np.linspace(-WIDTH / 2, WIDTH / 2, NX + 1, dtype=np.float32)
        ys = np.linspace(TOP_Y, TOP_Y - HEIGHT, NY + 1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        gz = np.random.default_rng(7).uniform(-1e-3, 1e-3, gx.shape).astype(np.float32)
        p0 = np.stack([gx, gy, gz], -1).reshape(-1, 3)
        free = np.ones(N, np.float32)
        free[:NX + 1:3] = 0.0                              # hung from a rod: every third top particle
        self.pos = wp.array(p0, dtype=wp.vec3, device=dev)
        self.prev = wp.array(p0, dtype=wp.vec3, device=dev)
        self.pred = wp.zeros(N, dtype=wp.vec3, device=dev)
        self.tmp = wp.zeros(N, dtype=wp.vec3, device=dev)
        self.nrm = wp.zeros(N, dtype=wp.vec3, device=dev)
        self.free = wp.array(free, dtype=float, device=dev)
        self.grid = wp.HashGrid(64, 64, 64, device=dev)
        rng = np.random.default_rng(11)
        life = rng.uniform(0.9, 2.2, NS).astype(np.float32)
        self.sx = wp.zeros(NS, dtype=wp.vec3, device=dev)
        self.sv = wp.zeros(NS, dtype=wp.vec3, device=dev)
        self.age = wp.array(-rng.uniform(0.0, 2.2, NS).astype(np.float32), dtype=float, device=dev)  # lit in turn
        self.life = wp.array(life, dtype=float, device=dev)
        self.t, self.k = 0.0, 0
        self.ref = wp.zeros((height, width), dtype=float, device=dev)
        self.ev = wp.zeros((height, width, 3), dtype=wp.uint8, device=dev)
        self.seeded = False
        self.vk_pos = self.vk_nrm = self.vk_spark_pos = self.vk_spark_col = self.frame = None
        self.vk_proxy_pos = self.vk_proxy_nrm = None
        self.cloth.geometry.update_attribute("position", p0)
        self._place()
        self.field.set_live_count(0)
        r.render(scene, self.camera)                  # meshes and fields get their GPU records on their first frame
        self._share()
        self.field.set_live_count(NS)

    def _share(self):
        r, dev = self.renderer, self.device
        # [in]
        h = r.enable_vertex_interop(self.cloth, self.on_frame)
        self.vk_pos = VkInteropArray(*h[0], wp.vec3, N, dev)      # buffers the renderer shares
        self.vk_nrm = VkInteropArray(*h[1], wp.vec3, N, dev)
        f = r.enable_particle_field_interop(self.field, self.on_sparks)
        self.vk_spark_pos = VkInteropArray(f[0], f[1], wp.vec4, NS, dev)
        self.vk_spark_col = VkInteropArray(f[2], f[3], wp.vec4, NS, dev)
        # [/in]
        g = r.enable_vertex_interop(self.proxies, self.on_proxies)     # the sparks again, as geometry
        self.vk_proxy_pos = VkInteropArray(*g[0], wp.vec3, NS * 24, dev)
        self.vk_proxy_nrm = VkInteropArray(*g[1], wp.vec3, NS * 24, dev)
        # [out]
        out = r.enable_frame_interop(0, ["color"])[0]
        self.frame = VkInteropArray(out["handle"], out["size_bytes"], wp.uint8,
                                    (out["height"], out["width"], 4), dev)
        # [/out]
        self.bgra = out["bgra"]
        if (out["height"], out["width"]) != (self.h, self.w):
            raise RuntimeError(f"the frame is {out['width']}x{out['height']}, not {self.w}x{self.h}")

    # [on_frame]
    def on_frame(self):                               # inside render(), before the frame is drawn
        wp.copy(self.vk_pos.array, self.pos)
        wp.copy(self.vk_nrm.array, self.nrm)
        wp.synchronize_device()

    def on_sparks(self):
        wp.launch(spark_look, dim=NS, inputs=[self.sx, self.age, self.life,
                                              self.vk_spark_pos.array, self.vk_spark_col.array])
        wp.synchronize_device()
    # [/on_frame]

    def on_proxies(self):
        wp.launch(spark_mesh, dim=NS, inputs=[self.sx, self.age, self.life, self.corner,
                                             self.vk_proxy_pos.array, self.vk_proxy_nrm.array])
        wp.synchronize_device()

    def _place(self):
        b = ball_at(self.t)
        self.ball.position.set(b[0], b[1], b[2])
        self.glow.intensity = 2.0 + 0.6 * math.sin(37.0 * self.t) * math.sin(23.0 * self.t)
        if self.film_camera:
            camera_at(self.t, self.camera)

    def step(self):
        """One 60 Hz frame of cloth and sparks."""
        dev = self.device
        for _ in range(SUBSTEPS):
            b = ball_at(self.t)
            wp.launch(integrate, dim=N, inputs=[self.pos, self.prev, self.pred, self.free, self.t, DT], device=dev)
            a, c = self.pred, self.tmp
            for _ in range(ITERATIONS):
                wp.launch(solve, dim=N, inputs=[a, c, self.free, WIDTH / NX, HEIGHT / NY, b], device=dev)
                a, c = c, a
            wp.copy(self.pos, a)
            wp.launch(normals, dim=N, inputs=[self.pos, self.nrm], device=dev)
            self.grid.build(self.pos, float(TOUCH))
            wp.launch(sparks, dim=NS, inputs=[self.sx, self.sv, self.age, self.life, self.grid.id, self.pos,
                                              self.nrm, b, self.t, DT, self.k], device=dev)
            self.t += DT
            self.k += 1
        self._place()

    def render(self):
        """Draw the frame, then run the event camera on it where it lies."""
        r = self.renderer
        r.render(self.scene, self.camera)
        # [see]
        r.sync_frame_interop()                        # the frame is finished
        if not self.bgra:
            wp.launch(swap_rb, dim=(self.h, self.w), inputs=[self.frame.array])
        if not self.seeded:
            wp.launch(seed, dim=(self.h, self.w), inputs=[self.frame.array, self.ref])
            self.seeded = True
        wp.launch(events, dim=(self.h, self.w), inputs=[self.frame.array, self.ref, self.ev, THRESHOLD])
        # [/see]
        wp.synchronize_device()

    def picture(self):
        """The render (RGB) and the events, both (h, w, 3) uint8; copies, for the film and the video."""
        f = self.frame.array.numpy()
        rgb = f[..., 2::-1] if self.bgra else f[..., :3]
        return np.ascontiguousarray(rgb), self.ev.numpy()

    def close(self):
        for a in (self.vk_pos, self.vk_nrm, self.vk_spark_pos, self.vk_spark_col, self.vk_proxy_pos,
                  self.vk_proxy_nrm, self.frame):
            if a is not None:
                a.close()
        self.renderer.disable_frame_interop(0)
        self.renderer.disable_vertex_interop(self.cloth)       # the particle field's export goes with the renderer
        self.renderer.disable_vertex_interop(self.proxies)


def stream_main():
    """Raw frames on stdout: a header line, then per frame the render and the events, (h, w, 3) each."""
    fd = os.dup(1)
    os.dup2(2, 1)                                     # everything else printed goes to stderr
    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(fd, os.O_BINARY)
    out = os.fdopen(fd, "wb")
    w, h = parse_size(cli_arg("--stream", "1920x1080", str))
    seconds = cli_arg("--seconds", 10.0, float)
    preroll = cli_arg("--preroll", 1.0, float)
    rt = RoundTrip(w, h, True, dots="--dots" in sys.argv)
    for _ in range(int(round(preroll * 60))):
        rt.step()
        rt.render()
    n = int(round(seconds * 60)) + 1
    out.write(f"FRAMES {n}\n".encode())
    out.flush()
    for _ in range(n):
        rt.step()
        rt.render()
        rgb, ev = rt.picture()
        out.write(rgb.tobytes())
        out.write(ev.tobytes())
        out.flush()
    out.close()
    rt.close()


def video_main():
    seconds = cli_arg("--video", 8.0, float)
    w, h = parse_size(cli_arg("--size", "960x540", str))
    rt = RoundTrip(w, h, True, dots="--dots" in sys.argv)
    enc = Encoder("warp_round_trip.mp4", 2 * w, h, 60, crf=16)
    for _ in range(60):
        rt.step()
        rt.render()
    for _ in range(int(round(seconds * 60))):
        rt.step()
        rt.render()
        rgb, ev = rt.picture()
        enc.send(np.concatenate([rgb, ev], axis=1))
    enc.close()
    rt.close()
    print("wrote warp_round_trip.mp4 (render | events)")


def window_main():
    rt = RoundTrip(1280, 720, False)
    controls = tp.OrbitControls(rt.camera, rt.canvas)
    controls.target.set(-0.25, 1.62, 0.3)
    count = [0]

    def animate():
        rt.step()
        rt.render()
        count[0] += 1
        if count[0] % 120 == 0:
            lit = float((rt.ev.numpy().max(axis=2) > 200).mean())
            print(f"t {rt.t:5.1f} s   pixels firing this frame {100 * lit:.2f} %")

    rt.canvas.animate(animate)
    rt.close()


if __name__ == "__main__":
    if "--stream" in sys.argv:
        stream_main()
    elif "--video" in sys.argv:
        video_main()
    else:
        window_main()
