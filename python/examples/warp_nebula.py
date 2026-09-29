"""A particle nebula -- NVIDIA Warp simulation, threepp rendering (GL or Vulkan).

A galaxy disk in a softened central gravity well, stirred by animated 4D curl
noise, with differential rotation shearing the turbulence into spiral arms.
Every NOVA_PERIOD seconds the core detonates: a decaying radial blast hurls
the whole disk outward, then gravity re-collects it. Particle colors are
computed in the same kernel from speed (cold blue -> white hot), so the
explosion ignites the palette by itself.

The entire simulation is ONE Warp kernel with no neighbor queries. The 4D curl
noise is sampled on a coarse grid rebuilt each frame and fetched trilinearly
per particle: the field is smooth by construction, so a few grid samples per
noise feature carry all the information a per-particle evaluation would, at a
small fraction of the cost.

    pip install warp-lang
    python warp_nebula.py                # GL: window; drag to orbit
    python warp_nebula.py --vulkan       # Vulkan: the same disk as a VOLUME

GL (default): 18M additive points
---------------------------------
On a CUDA device the renderer's vertex buffers are registered with CUDA
(GLRenderer.gl_buffer_id + wp.RegisteredGLBuffer) and the sim kernel writes
positions and colors STRAIGHT INTO the mapped buffers -- nothing crosses host
memory, Python never touches the bytes, and there is no separate copy pass.
--no-direct (or a CPU device) falls back to the host path: pinned host
mirrors, GPU/CPU pipelining, colours shipped less often than positions.

    python warp_nebula.py --n 3000000    # fewer particles (exposure auto-adjusts)
    python warp_nebula.py --shot 6.6     # headless PNG
    python warp_nebula.py --bench        # timed phase breakdown
    python warp_nebula.py --no-direct    # force the host-copy path
    python warp_nebula.py --dither       # draw a random half per frame at 2x
                                         # brightness: statistically the same
                                         # additive image, half the raster cost
    python warp_nebula.py --exact-noise  # per-particle curl noise

The GL picture is gorgeous -- as a PROJECTION. Additive blending commutes,
which is exactly why it needs no sort and exactly why it is flat: the image
carries zero occlusion information, so a filament in front of the core and the
same filament behind it produce identical pixels.

Vulkan (--vulkan): the light transport put back
-----------------------------------------------
The particles still carry the IMAGE (4 million coloured sprites, full
per-particle frequency); a coarse density volume built from the SAME particles
carries only the LIGHT TRANSPORT -- transmittance toward the sun, and toward
the camera. Lighting is low-frequency by nature, so a soft shadow survives a
160^3 lattice where a filament would not. Both factors depend only on a
sprite's own position, so the blend stays additive and therefore ORDERLESS:
still no sort.

    python warp_nebula.py --vulkan --shot 4.0         # headless PNG, the quiet disk
    python warp_nebula.py --vulkan --shot 4.0 --flat  # the SAME frame, knobs at 0
    python warp_nebula.py --vulkan --shot 6.05        # the nova instant
    python warp_nebula.py --vulkan --shot 6.6         # the shell, half a second later
    python warp_nebula.py --vulkan --n 8000000        # more particles
    python warp_nebula.py --vulkan --bench            # frame time, knobs on vs --flat

WHERE TO LOOK. t=4.0 is the clearest: a dark dust lane cuts across the bulge
where the near arm crosses it, the near half of the disk sits in its own
shadow, and the far edge carries a lit rim -- against a --flat frame that is
one evenly bright lens with no front and no back. t=6.05 is the nova instant,
where the shock front's near arc reads as a hard rim over a core the front's
own dust is dimming.

--flat zeroes volume_extinction and volume_shadow and changes nothing else, so
the pair at one timestamp is the whole acceptance test: dust lanes across the
core, a lit rim toward the key light, and a nova that reads as a ball rather
than as a poster.

TRANSPORT. 4M particles x 16 B of position + 16 B of colour does not go on the
bus twice a frame. Both allocations are EXPORTED by the renderer and imported
once into CUDA (threepp.cuda_interop.VkInteropArray), and the Warp kernel writes
them device to device inside render(). The colours are the reason
Config.attributes exists: an interop field has no closed form and therefore no
age, so without them its billboards could only ever be one flat colour.

DETERMINISM is forfeit on this leg and that is accepted, not overlooked: an
interop field's bytes are authored on the other side of the import (see the
trade recorded in ParticleField.hpp). --no-interop falls back to the host ring
at a fraction of the count.

Warp falls back to CPU if no CUDA device is present -- slower, same picture.
"""
import math
import os
import sys
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import arm_particle_interop, bench_loop, cli_arg, orbit_loop, shot_loop

VULKAN = "--vulkan" in sys.argv
if VULKAN and not tp.HAS_VULKAN:
    print("This build has no Vulkan backend (configure -DTHREEPP_WITH_VULKAN=ON).")
    sys.exit(0)

N = cli_arg("--n", 4_000_000 if VULKAN else 18_000_000, int)
BENCH = "--bench" in sys.argv
SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 6.6, float)

FPS = 60
DT = 1.0 / FPS
G = 0.18                 # central gravity
V_CAP = 1.4              # tames extreme nova ejecta; orbits stay far below this
COOL = 0.4               # 1/s damping of the RADIAL + VERTICAL velocity only.
                         # Blanket drag cold-collapses the disk into a ball and
                         # zero dissipation heat-deaths it into fog; damping only
                         # the non-tangential components conserves angular
                         # momentum, so each orbit re-circularises at its own
                         # radius and the disk lives forever.
TURB = 0.025             # curl-noise stir strength; must stay well below
                         # orbital acceleration or it bulk-advects the disk
FREQ = 5.0               # curl wavelength well under the disk radius
FLATTEN = 0.4            # disk-flattening pull toward the galactic plane
R_MAX = 2.4              # soft containment radius
V_HOT = 1.6              # speed for full white heat -- nova ejecta only
# Per-particle intensity. Additive sums clip at 1.0 and overlap grows with
# the particle count, so exposure follows a 1/sqrt(N) law. The Vulkan base is
# higher because a world-space sprite covers more pixels than a GL point.
BRIGHT = cli_arg("--bright", 0.010 if VULKAN else 0.0018, float) * math.sqrt(16_000_000 / N)
RECYCLE_R = 0.30         # particles below this radius may be reborn on the rim...
RECYCLE_P = 0.0015       # ...at this per-frame rate. Higher evacuates the core
                         # into a dark accretion-disk eye, lower keeps it a
                         # filled golden glow.
NOVA_T0 = 6.0            # first detonation
NOVA_PERIOD = 9.0
NOVA_AMP = 1.6           # impulse ~ escape speed at the core, a shudder outside
NOVA_TAU = 0.25

# --- curl-noise field ----------------------------------------------------------
# The turbulence is baked to an NG^3 grid once per frame and sampled trilinearly
# in the step kernel. Ejecta beyond the grid clamp to the edge sample; the soft
# containment is already pulling them back by then.
EXACT_NOISE = "--exact-noise" in sys.argv and not VULKAN
NG = cli_arg("--noise-grid", 96, int)
R_NOISE = 2.6            # covers R_MAX plus nova overshoot
NG_SCALE = (NG - 1) / (2.0 * R_NOISE)

# Temporal dither (GL): draw a random half of the particles each frame at double
# brightness. The additive accumulation of millions of subpixel points is
# statistical anyway, and the particle order is random by construction, so
# alternating halves is the same image in motion for half the vertex/fill cost.
DITHER = "--dither" in sys.argv and not VULKAN
if DITHER:
    BRIGHT *= 2.0


@wp.func
def sample_noise(f: wp.array3d(dtype=wp.vec3), p: wp.vec3) -> wp.vec3:
    # trilinear fetch from the baked curl field; out-of-grid clamps to the edge
    gx = wp.clamp((p[0] + R_NOISE) * NG_SCALE, 0.0, float(NG - 1) - 1.0e-3)
    gy = wp.clamp((p[1] + R_NOISE) * NG_SCALE, 0.0, float(NG - 1) - 1.0e-3)
    gz = wp.clamp((p[2] + R_NOISE) * NG_SCALE, 0.0, float(NG - 1) - 1.0e-3)
    i = int(wp.floor(gx))
    j = int(wp.floor(gy))
    k = int(wp.floor(gz))
    fx = gx - float(i)
    fy = gy - float(j)
    fz = gz - float(k)
    c00 = f[i, j, k] * (1.0 - fx) + f[i + 1, j, k] * fx
    c10 = f[i, j + 1, k] * (1.0 - fx) + f[i + 1, j + 1, k] * fx
    c01 = f[i, j, k + 1] * (1.0 - fx) + f[i + 1, j, k + 1] * fx
    c11 = f[i, j + 1, k + 1] * (1.0 - fx) + f[i + 1, j + 1, k + 1] * fx
    a = c00 * (1.0 - fy) + c10 * fy
    b = c01 * (1.0 - fy) + c11 * fy
    return a * (1.0 - fz) + b * fz


@wp.kernel
def bake_noise(t: float, f: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    s = wp.rand_init(7)
    x = float(i) / NG_SCALE - R_NOISE
    y = float(j) / NG_SCALE - R_NOISE
    z = float(k) / NG_SCALE - R_NOISE
    f[i, j, k] = wp.curlnoise(s, wp.vec4(x * FREQ, y * FREQ, z * FREQ, t * 0.15))


@wp.func
def integrate(p: wp.vec3, v: wp.vec3, i: int, turb: wp.vec3,
              dt: float, t: float, blast: float, frame: int, bright: float):
    r2 = wp.dot(p, p) + 0.04
    r = wp.sqrt(r2)
    # softened point-mass gravity + disk flattening
    a = p * (-G / (r2 * r))
    a += wp.vec3(0.0, -p[1] * FLATTEN, 0.0)
    # animated curl-noise turbulence, evaluated by the caller (grid or exact)
    a += turb
    # supernova: radial blast, strongest near the core
    if blast > 0.0:
        a += (p / r) * (blast / (0.3 + r))
    # soft containment so novae don't scatter the disk to infinity
    if r > R_MAX:
        a += p * ((R_MAX - r) * 4.0 / r)
    v = v + a * dt
    sp = wp.length(v)
    if sp > V_CAP:
        v = v * (V_CAP / sp)
    # thermostat: damp radial + vertical motion, keep the tangential component
    h2 = p[0] * p[0] + p[2] * p[2]
    if h2 > 1.0e-6:
        e_r = wp.vec3(p[0], 0.0, p[2]) * (1.0 / wp.sqrt(h2))
        v = v - e_r * (wp.dot(v, e_r) * COOL * dt)
        v = wp.vec3(v[0], v[1] * (1.0 - COOL * dt), v[2])
    p = p + v * dt
    # Stellar recycling: a sunken core particle is occasionally reborn on the
    # outer rim with a clean circular orbit, keeping the core from clipping
    # white and re-feeding the disk that drag slowly drains.
    if r < RECYCLE_R:
        st = wp.rand_init(4321, i + frame * 2654435)
        if wp.randf(st) < RECYCLE_P:
            ang = wp.randf(st) * 6.2831853
            rr = 0.95 + 0.3 * wp.randf(st)
            p = wp.vec3(rr * wp.cos(ang), (wp.randf(st) - 0.5) * 0.1, rr * wp.sin(ang))
            r2n = rr * rr + 0.04
            vc = wp.sqrt(G * rr * rr / (r2n * wp.sqrt(r2n)))
            v = wp.vec3(-wp.sin(ang), 0.0, wp.cos(ang)) * vc
            r = rr
    # palette: radius sets the base hue (golden core -> blue rim), speed
    # pushes toward white heat -- so the nova ignites everything on its own
    w = 1.0 - wp.min(r, 1.0)
    warmth = w * w
    base = wp.vec3(0.12, 0.20, 0.60) * (1.0 - warmth) + wp.vec3(1.0, 0.70, 0.30) * warmth
    h = wp.min(wp.length(v) / V_HOT, 1.0)
    c = (base * (1.0 - h) + wp.vec3(1.0, 0.85, 0.65) * h) * bright
    return p, v, c


# GL: out_pos/out_col are the DISPLAY destinations: in direct mode the
# renderer's own mapped VBOs (the kernel writes them in place), in host mode the
# state array itself plus the staging color buffer.
@wp.kernel
def step(pos: wp.array(dtype=wp.vec3),
         vel: wp.array(dtype=wp.vec3),
         out_pos: wp.array(dtype=wp.vec3),
         out_col: wp.array(dtype=wp.vec3),
         noise: wp.array3d(dtype=wp.vec3),
         dt: float, t: float, blast: float, frame: int, bright: float):
    i = wp.tid()
    p = pos[i]
    turb = sample_noise(noise, p) * TURB
    p, v, c = integrate(p, vel[i], i, turb, dt, t, blast, frame, bright)
    pos[i] = p
    vel[i] = v
    out_pos[i] = p
    out_col[i] = c


@wp.kernel
def step_exact(pos: wp.array(dtype=wp.vec3),
               vel: wp.array(dtype=wp.vec3),
               out_pos: wp.array(dtype=wp.vec3),
               out_col: wp.array(dtype=wp.vec3),
               dt: float, t: float, blast: float, frame: int, bright: float):
    # per-particle 4D curl noise instead of the baked grid
    i = wp.tid()
    p = pos[i]
    s = wp.rand_init(7)
    turb = wp.curlnoise(s, wp.vec4(p[0] * FREQ, p[1] * FREQ, p[2] * FREQ, t * 0.15)) * TURB
    p, v, c = integrate(p, vel[i], i, turb, dt, t, blast, frame, bright)
    pos[i] = p
    vel[i] = v
    out_pos[i] = p
    out_col[i] = c


# Vulkan: out_pos is the exported POSITION allocation (xyz + w = the per-particle
# world radius, WSemantic.Radius); out_col is the exported ATTRIBUTE allocation
# (rgb = linear HDR radiance, a reserved). Both are written in place, device to
# device.
@wp.kernel
def step_vk(pos: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3),
            out_pos: wp.array(dtype=wp.vec4),
            out_col: wp.array(dtype=wp.vec4),
            noise: wp.array3d(dtype=wp.vec3),
            dt: float, t: float, blast: float, frame: int,
            radius: float, bright: float):
    i = wp.tid()
    p = pos[i]
    turb = sample_noise(noise, p) * TURB
    p, v, c = integrate(p, vel[i], i, turb, dt, t, blast, frame, bright)
    pos[i] = p
    vel[i] = v
    # w IS the world radius (WSemantic.Radius) and w < 0 is the DEAD sentinel
    # every consumer tests -- nothing here ever dies, so it is always positive.
    out_pos[i] = wp.vec4(p[0], p[1], p[2], radius)
    out_col[i] = wp.vec4(c[0], c[1], c[2], 1.0)


# --- initial disk (numpy, once) ------------------------------------------------

rng = np.random.default_rng(5)
u = rng.random(N).astype(np.float32)
radius = 1.2 * np.sqrt(u)
bulge = rng.random(N) < 0.09                 # dense core
radius[bulge] *= 0.45
theta = rng.uniform(0, 2 * math.pi, N).astype(np.float32)
# Seed two logarithmic spiral arms instead of waiting for differential
# rotation to shear them into existence: pull each particle's azimuth
# toward the nearest arm, harder for the outer disk (the bulge stays round).
arm_phase = np.log(np.maximum(radius, 0.05)) / math.tan(math.radians(16))
arm = (theta - arm_phase) % math.pi          # two arms, pi apart
pull = np.clip((radius - 0.25) / 0.9, 0.0, 1.0) * 0.8
theta = (theta - (arm - math.pi / 2) * pull).astype(np.float32)
theta += rng.normal(0, 0.05, N).astype(np.float32)
height = rng.normal(0, 1, N).astype(np.float32) * 0.09 * (0.25 + radius)

p0 = np.stack([radius * np.cos(theta), height, radius * np.sin(theta)], axis=-1)
# Circular-orbit speed for the *softened* force law used in the kernel
# (a = G r / (r^2+eps)^1.5) -- mismatching these evacuates the core.
v_circ = (np.sqrt(G * radius ** 2 / (radius ** 2 + 0.04) ** 1.5)
          * rng.uniform(0.95, 1.05, N).astype(np.float32))
v0 = np.stack([-np.sin(theta) * v_circ, np.zeros(N, np.float32),
               np.cos(theta) * v_circ], axis=-1)
v0 += rng.normal(0, 0.02, (N, 3)).astype(np.float32)

wp.init()
device = wp.get_preferred_device()

pos = wp.array(p0.astype(np.float32), dtype=wp.vec3, device=device)
vel = wp.array(v0.astype(np.float32), dtype=wp.vec3, device=device)
noise = wp.zeros((NG, NG, NG), dtype=wp.vec3, device=device)
del v0, u, radius, theta, height, arm, arm_phase, pull, v_circ, bulge

sim_time = 0.0
frame_no = 0


def blast_now():
    ph = sim_time - NOVA_T0
    return NOVA_AMP * math.exp(-(ph % NOVA_PERIOD) / NOVA_TAU) if ph >= 0.0 else 0.0


if not VULKAN:
    # ── GL: additive points written straight into the renderer's VBOs ──────────
    print(f"nebula: {N:,} particles on {device}")

    col = None               # host path only -- direct mode writes colors straight
                             # into the mapped VBO, so no colour state is allocated

    DIRECT = ("--no-direct" not in sys.argv) and device.is_cuda

    def launch_step(out_pos, out_col, blast):
        """Bake this frame's noise field (grid mode), then one sim step."""
        if not EXACT_NOISE:
            wp.launch(bake_noise, dim=(NG, NG, NG), device=device,
                      inputs=[sim_time, noise])
            wp.launch(step, dim=N, device=device,
                      inputs=[pos, vel, out_pos, out_col, noise,
                              DT, sim_time, blast, frame_no, BRIGHT])
        else:
            wp.launch(step_exact, dim=N, device=device,
                      inputs=[pos, vel, out_pos, out_col,
                              DT, sim_time, blast, frame_no, BRIGHT])

    def apply_dither():
        # Alternate halves of the (randomly ordered) particle array. BRIGHT was
        # doubled at init, so the accumulated image matches the full draw.
        if not DITHER:
            return
        h = N // 2
        if frame_no & 1:
            geometry.set_draw_range(h, N - h)
        else:
            geometry.set_draw_range(0, h)

    # Host-path fallback: persistent pinned mirrors -- allocating fresh
    # multi-megabyte numpy arrays every frame hitches -- double-buffered so the
    # GPU simulates frame N+1 while the CPU uploads and renders frame N.
    pos_host = col_host = pos_view = col_view = None
    col_fresh = [True, True]
    COLOR_EVERY = 3          # colours drift slowly outside a nova, so they ship
                             # less often than positions

    def enable_host_path():
        global DIRECT, pos_host, col_host, pos_view, col_view, col
        DIRECT = False
        col = wp.zeros(N, dtype=wp.vec3, device=device)
        pos_host = [wp.zeros(N, dtype=wp.vec3, device="cpu", pinned=True) for _ in range(2)]
        col_host = [wp.zeros(N, dtype=wp.vec3, device="cpu", pinned=True) for _ in range(2)]
        pos_view = [a.numpy() for a in pos_host]
        col_view = [a.numpy() for a in col_host]

    cur = 0                  # host-buffer pair the CPU consumes this frame

    def gpu_advance(buf):
        """Queue one sim step + DMA into host pair `buf` -- all async."""
        global sim_time, frame_no
        blast = blast_now()
        # out_pos aliases the state array (writing p twice is harmless); colors
        # land in the staging buffer the DMA below ships.
        launch_step(pos, col, blast)
        wp.copy(pos_host[buf], pos)
        col_fresh[buf] = (frame_no % COLOR_EVERY == 0) or blast > 0.05
        if col_fresh[buf]:
            wp.copy(col_host[buf], col)
        sim_time += DT
        frame_no += 1

    def step_frame_host():
        """Consume the ready frame, queue the next; returns (gpu_s, update_s)."""
        global cur
        t0 = time.perf_counter()
        wp.synchronize_device(device)        # sim + DMA into `cur` are done
        t1 = time.perf_counter()
        gpu_advance(1 - cur)                 # GPU races ahead while the CPU uploads
        geometry.update_attribute("position", pos_view[cur])
        if col_fresh[cur]:
            geometry.update_attribute("color", col_view[cur])
        t2 = time.perf_counter()
        cur = 1 - cur
        apply_dither()
        return t1 - t0, t2 - t1

    reg_pos = reg_col = None

    def step_frame_direct():
        """Sim straight into the renderer's own mapped VBOs. Zero copies anywhere."""
        global sim_time, frame_no
        blast = blast_now()
        t0 = time.perf_counter()
        dst_pos = reg_pos.map(dtype=wp.vec3, shape=(N,))
        dst_col = reg_col.map(dtype=wp.vec3, shape=(N,))
        launch_step(dst_pos, dst_col, blast)
        reg_pos.unmap()
        reg_col.unmap()
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        sim_time += DT
        frame_no += 1
        apply_dither()
        return t1 - t0, 0.0

    canvas = tp.Canvas("threepp x warp - nebula", width=1280, height=800,
                       headless=SHOT or BENCH)
    renderer = tp.GLRenderer(canvas)

    scene = tp.Scene()
    scene.background = 0x000000

    camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.01, 100)
    camera.position.set(0.95, 1.05, 0.95)
    camera.look_at(0, 0, 0)

    geometry = tp.BufferGeometry()
    geometry.set_attribute("position", p0.astype(np.float32))
    geometry.set_attribute("color", np.full((N, 3), 0.1, dtype=np.float32))
    del p0

    mat = tp.PointsMaterial()
    mat.size = 0.0015
    mat.size_attenuation = True
    mat.vertex_colors = True
    mat.transparent = True
    mat.blending = tp.Blending.Additive
    mat.depth_write = False

    nebula = tp.Points(geometry, mat)
    nebula.frustum_culled = False
    scene.add(nebula)

    if DIRECT:
        renderer.render(scene, camera)       # first render creates + fills the VBOs
        pid = renderer.gl_buffer_id(geometry, "position")
        cid = renderer.gl_buffer_id(geometry, "color")
        try:
            if pid is None or cid is None:
                raise RuntimeError("attribute VBOs not uploaded")
            flags = wp.RegisteredGLBuffer.WRITE_DISCARD
            reg_pos = wp.RegisteredGLBuffer(int(pid), device, flags)
            reg_col = wp.RegisteredGLBuffer(int(cid), device, flags)
            print("direct mode: warp writes the renderer's VBOs in place")
        except Exception as exc:
            print(f"GL interop unavailable ({exc}); using the host-copy path")
            enable_host_path()

    if DIRECT:
        step_frame = step_frame_direct
    else:
        if pos_host is None:
            enable_host_path()
        gpu_advance(cur)                     # prime the host-path pipeline
        step_frame = step_frame_host

    if BENCH:
        bench_loop(step_frame, lambda: renderer.render(scene, camera),
                   ("gpu", "update"), warmup=60, timed=240,
                   label=f"{N:,} particles [{'direct' if DIRECT else 'host'}]")
        renderer.save_frame("warp_nebula.png")
    elif SHOT:
        shot_loop(renderer, scene, camera, step_frame, SHOT_TIME, "warp_nebula.png")
    else:
        def step_and_drift():
            step_frame()
            # slow parallax drift so a recording always has camera motion;
            # dragging to orbit composes on top of it
            nebula.rotation.y = 0.04 * sim_time

        orbit_loop(canvas, renderer, scene, camera, step_and_drift)

else:
    # ── Vulkan: sprites carry the image, a density volume the light transport ──
    del p0
    FLAT = "--flat" in sys.argv
    INTEROP = "--no-interop" not in sys.argv and device.is_cuda
    W, H = cli_arg("--width", 1280, int), cli_arg("--height", 800, int)

    # The density volume is LATCHED at these bounds and never refitted per frame:
    # a box that tracks its own matter re-phases the whole lattice every frame and
    # the volume visibly swims. The disk lives inside +/-(2.4, 0.5, 2.4) and nova
    # ejecta overshoot a little, so the box is sized once for the worst case.
    BOX_HALF = (2.6, 0.9, 2.6)
    BOX_RES = cli_arg("--vol-res", 160, int)
    SIGMA = cli_arg("--sigma", 0.050, float)   # sigma_t one particle contributes
    RADIUS = cli_arg("--radius", 0.0026, float)
    EXPOSURE = cli_arg("--exposure", 1.0, float)
    EXTINCTION = 0.0 if FLAT else cli_arg("--extinction", 1.0, float)
    SHADOW = 0.0 if FLAT else cli_arg("--shadow", 0.60, float)

    print(f"nebula: {N:,} particles on {device}"
          f"{'  [--flat: volumetrics OFF]' if FLAT else ''}")

    canvas = tp.Canvas("threepp x warp - volumetric nebula", width=W, height=H,
                       vsync=False, headless=SHOT or BENCH)
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    # PINNED: the billboards bypass auto-exposure entirely (they composite after
    # the post stack), so a drifting auto exposure would move the sprites and the
    # background against each other and make an A/B meaningless.
    renderer.tone_mapping_exposure = EXPOSURE

    scene = tp.Scene()
    scene.background = tp.Color(0.0, 0.0, 0.0)

    camera = tp.PerspectiveCamera(52, canvas.aspect(), 0.01, 100)
    # LOW, deliberately. A dust lane is an EXTINCTION path, and the length of that
    # path is what makes it visible: from overhead, every sight line leaves a thin
    # disk almost immediately and T_cam is ~1 whatever sigma is, so the extinction
    # knob has nothing to bite on. Twelve degrees above the plane puts the near arm
    # in front of the core with a metre of dust behind it, which is why every real
    # photograph of a dust lane is of an edge-on galaxy.
    camera.position.set(1.78, cli_arg("--cam-y", 0.20, float), 1.78)
    camera.look_at(0, 0.02, 0)

    # The KEY. One DirectionalLight is the scene's sun under the one-sun policy, and
    # it is what volume_shadow marches toward: the rim it lights is the whole point
    # of having a third dimension. Placed off to camera-left and slightly behind the
    # disk, so the near face is shadowed and the far rim flares.
    sun = tp.DirectionalLight(0xFFF0E0, 3.0)
    sun.position.set(-2.6, 0.9, -1.2)
    scene.add(sun)
    scene.add(tp.AmbientLight(0x101018, 1.0))

    cfg = tp.ParticleField.Config()
    cfg.capacity = N
    cfg.ownership = tp.ParticleField.Ownership.Interop if INTEROP \
        else tp.ParticleField.Ownership.HostRing
    cfg.w_semantic = tp.ParticleField.WSemantic.Radius
    cfg.uniform_radius = RADIUS
    cfg.attributes = True          # per-particle colour, straight out of the kernel
    field = tp.ParticleField.create(cfg)
    field.frustum_culled = False
    scene.add(field)

    # The DENSITY half: the same particles, scattered into one world box that only
    # the LIGHT TRANSPORT reads. Latched here, once -- center/half_extent are
    # per-frame writable and are deliberately never written.
    field.set_density_repr(tp.Vector3(0, 0, 0), tp.Vector3(*BOX_HALF), SIGMA, BOX_RES)
    # A NEARLY BLACK albedo, deliberately. The same volume is also sampled by the
    # deferred fog march, which in-scatters ambient and sun light through it — and
    # that term is low frequency by construction, so at any albedo worth seeing it
    # lays a smooth grey veil over exactly the filaments this demo exists to keep.
    # The volume's job here is TRANSPORT, not radiance: the sprites carry every
    # photon, and a hint of albedo is left only so the medium is not a black hole.
    field.density_repr.albedo = tp.Color(0.05, 0.055, 0.08)
    field.density_repr.anisotropy = 0.0

    # The IMAGE half.
    field.set_billboard_repr(tp.Color(1, 1, 1), tp.Color(1, 1, 1), 1.0, 1.0)
    bb = field.billboard_repr
    bb.softness = 0.62
    bb.bright_jitter = 0.0        # the sim authors the colour; do not hash over it
    bb.fade_power = 0.0           # no age exists on an interop field
    bb.size_taper = 0.0
    bb.lit_phase_g = 0.42         # forward-ish: the rim flares when you look through
    bb.volume_extinction = EXTINCTION
    bb.volume_shadow = SHADOW
    bb.volume_ambient = 0.30
    bb.volume_sun_gain = 0.60

    host_pos = host_col = None

    def launch(out_pos, out_col):
        wp.launch(bake_noise, dim=(NG, NG, NG), device=device, inputs=[sim_time, noise])
        wp.launch(step_vk, dim=N, device=device,
                  inputs=[pos, vel, out_pos, out_col, noise, DT, sim_time,
                          blast_now(), frame_no, RADIUS, BRIGHT])

    def step_frame_host():
        """The fallback: simulate into host mirrors and push them across the bus."""
        global sim_time, frame_no
        launch(host_pos, host_col)
        wp.synchronize_device(device)
        field.submit(host_pos.numpy(), DT)
        field.set_attributes(host_col.numpy())
        sim_time += DT
        frame_no += 1

    def step_frame_interop():
        """Nothing to do here: the sim IS the renderer's device_copy callback."""
        global sim_time, frame_no
        sim_time += DT
        frame_no += 1

    # ── ARM THE ZERO-COPY PATH ──────────────────────────────────────────────────
    _io = arm_particle_interop(renderer, scene, camera, field, launch, N, device,
                               INTEROP)
    INTEROP = _io.on
    host_pos, host_col = _io.host_pos, _io.host_col
    step_frame = step_frame_interop if INTEROP else step_frame_host

    print(f"       volume: {BOX_RES}^3 over {2 * BOX_HALF[0]:.1f} x "
          f"{2 * BOX_HALF[1]:.1f} x {2 * BOX_HALF[2]:.1f} m, sigma/particle {SIGMA:g}\n"
          f"       knobs:  extinction {EXTINCTION:g}, shadow {SHADOW:g}, "
          f"radius {RADIUS:g}, bright {BRIGHT:.5f}")

    def run_to(seconds):
        frames = int(round(seconds * FPS))
        t0 = time.perf_counter()
        for f in range(frames):
            step_frame()
            renderer.render(scene, camera)
            if f % 60 == 0:
                print(f"  t={sim_time:5.2f}", flush=True)
        return frames, time.perf_counter() - t0

    if BENCH:
        run_to(2.0)                                   # warm the pipeline
        n = 240
        t0 = time.perf_counter()
        for _ in range(n):
            step_frame()
            renderer.render(scene, camera)
        dt = (time.perf_counter() - t0) / n
        print(f"bench {N:,} particles [{'flat' if FLAT else 'volumetric'}]: "
              f"{1e3 * dt:.2f} ms/frame ({1.0 / dt:.0f} fps)")
    elif SHOT:
        frames, wall = run_to(SHOT_TIME)
        out = cli_arg("--out", "warp_nebula_vk_flat.png" if FLAT else "warp_nebula_vk.png", str)
        renderer.save_frame(scene, camera, out)
        print(f"simulated {SHOT_TIME:.1f} s ({frames} frames) in {wall:.1f}s, wrote {out}")
    else:
        controls = tp.OrbitControls(camera, canvas)
        controls.enable_damping = True

        def animate():
            step_frame()
            controls.update()
            renderer.render(scene, camera)

        canvas.animate(animate)
