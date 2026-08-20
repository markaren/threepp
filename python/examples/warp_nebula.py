"""A sixteen-million-particle nebula — NVIDIA Warp simulation, threepp rendering.

A galaxy disk in a softened central gravity well, stirred by animated 4D curl
noise, with differential rotation shearing the turbulence into spiral arms.
Every NOVA_PERIOD seconds the core detonates: a decaying radial blast hurls
the whole disk outward, then gravity re-collects it. Particle colors are
computed in the same kernel from speed (cold blue -> white hot), so the
explosion ignites the palette by itself. Rendered as additive points.

The entire simulation is ONE Warp kernel with no neighbor queries. On a CUDA
device the renderer's vertex buffers are registered with CUDA (GLRenderer.
gl_buffer_id + wp.RegisteredGLBuffer) and the sim kernel writes positions and
colors STRAIGHT INTO the mapped buffers — nothing crosses host memory, Python
never touches the bytes, and there is no separate copy pass. --no-direct (or a
CPU device) falls back to the tier-1 path: pinned host mirrors, GPU/CPU
pipelining, colors every 3rd frame.

The 4D curl noise is sampled on a coarse grid rebuilt each frame and fetched
trilinearly per particle: measured, the direct per-particle wp.curlnoise call
was 22 of the frame's 52 ms — ~70x more evaluations than the field, which is
smooth by construction, has information for. --exact-noise restores it as the
A/B baseline.

    pip install warp-lang
    python warp_nebula.py                # window, 16M particles; drag to orbit
    python warp_nebula.py --n 3000000    # fewer (exposure auto-adjusts)
    python warp_nebula.py --shot 6.6     # headless PNG (nova at t=6.0)
    python warp_nebula.py --bench        # timed phase breakdown
    python warp_nebula.py --no-direct    # force the host-copy path
    python warp_nebula.py --dither       # draw a random half per frame at 2x
                                         # brightness: statistically the same
                                         # additive image, half the raster cost
    python warp_nebula.py --exact-noise  # per-particle curlnoise (A/B)

Warp falls back to CPU if no CUDA device is present — slower, same picture.
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


def cli_arg(flag, default, cast):
    if flag in sys.argv:
        k = sys.argv.index(flag)
        if k + 1 < len(sys.argv) and not sys.argv[k + 1].startswith("--"):
            return cast(sys.argv[k + 1])
    return default


N = cli_arg("--n", 18_000_000, int)
BENCH = "--bench" in sys.argv
SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 6.6, float)

DT = 1.0 / 60.0
G = 0.18                 # central gravity; orbital period ~15 s at r=1
V_CAP = 1.4              # tames extreme nova ejecta; orbits stay far below this
COOL = 0.4               # 1/s damping of RADIAL + VERTICAL velocity only. The
                         # thermostat that keeps the disk alive forever: blanket
                         # drag cold-collapses it into a white ball, zero
                         # dissipation heat-deaths it into fog, and relaxing
                         # toward the local circular speed ratchets everything
                         # outward into a rim ring. Damping the non-tangential
                         # components conserves angular momentum, so each orbit
                         # re-circularizes at its own radius.
TURB = 0.025             # curl-noise stir strength; must stay well below
                         # orbital acceleration or it bulk-advects the disk
FREQ = 5.0               # curl wavelength well under the disk radius
FLAT = 0.4               # disk-flattening pull toward the galactic plane
R_MAX = 2.4              # soft containment radius
V_HOT = 1.6              # speed for full white heat — nova ejecta only
# Per-particle intensity. Additive sums clip at 1.0 and overlap grows with
# the particle count, so exposure follows an empirical 1/sqrt(N) law —
# anchored at the hand-tuned look for 16M (0.0018; 3M was 0.0048).
BRIGHT = 0.0018 * math.sqrt(16_000_000 / N)
RECYCLE_R = 0.30         # particles below this radius may be reborn on the rim...
RECYCLE_P = 0.0015       # ...at this per-frame rate. An aesthetic knob now that
                         # the thermostat holds orbits: ~0.004 evacuates the core
                         # into a dark accretion-disk eye, ~0.001 keeps it a
                         # filled golden glow, 0 lets it slowly brighten.
NOVA_T0 = 6.0            # first detonation
NOVA_PERIOD = 9.0
NOVA_AMP = 1.6           # impulse ~ escape speed at the core, a shudder outside
NOVA_TAU = 0.25

# --- curl-noise field ----------------------------------------------------------
# The turbulence is baked to an NG^3 grid once per frame and sampled trilinearly
# in the step kernel. At FREQ=5 a noise feature spans ~0.2 world units; 96 cells
# over [-R_NOISE, R_NOISE] give ~3.7 samples per feature — enough for a stir
# whose job is aesthetic. 885k curlnoise evals instead of 18M (~1 ms vs 22).
# Ejecta beyond the grid clamp to the edge sample; the soft containment is
# already pulling them back by then.
EXACT_NOISE = "--exact-noise" in sys.argv
NG = cli_arg("--noise-grid", 96, int)
R_NOISE = 2.6            # covers R_MAX plus nova overshoot
NG_SCALE = (NG - 1) / (2.0 * R_NOISE)

# Temporal dither: draw a random half of the particles each frame at double
# brightness. The additive accumulation of millions of subpixel points is
# statistical anyway (BRIGHT already follows a 1/sqrt(N) law), and the particle
# order is random by construction, so alternating halves is the same image in
# motion for half the vertex/fill cost.
DITHER = "--dither" in sys.argv
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
              dt: float, t: float, blast: float, frame: int):
    r2 = wp.dot(p, p) + 0.04
    r = wp.sqrt(r2)
    # softened point-mass gravity + disk flattening
    a = p * (-G / (r2 * r))
    a += wp.vec3(0.0, -p[1] * FLAT, 0.0)
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
    # pushes toward white heat — so the nova ignites everything on its own
    w = 1.0 - wp.min(r, 1.0)
    warmth = w * w
    base = wp.vec3(0.12, 0.20, 0.60) * (1.0 - warmth) + wp.vec3(1.0, 0.70, 0.30) * warmth
    h = wp.min(wp.length(v) / V_HOT, 1.0)
    c = (base * (1.0 - h) + wp.vec3(1.0, 0.85, 0.65) * h) * BRIGHT
    return p, v, c


# out_pos/out_col are the DISPLAY destinations: in direct mode the renderer's
# own mapped VBOs (the kernel writes them in place — the copy pass this
# replaces was a measured 5.6 ms of pure device bandwidth), in host mode the
# state array itself plus the staging color buffer.
@wp.kernel
def step(pos: wp.array(dtype=wp.vec3),
         vel: wp.array(dtype=wp.vec3),
         out_pos: wp.array(dtype=wp.vec3),
         out_col: wp.array(dtype=wp.vec3),
         noise: wp.array3d(dtype=wp.vec3),
         dt: float, t: float, blast: float, frame: int):
    i = wp.tid()
    p = pos[i]
    turb = sample_noise(noise, p) * TURB
    p, v, c = integrate(p, vel[i], i, turb, dt, t, blast, frame)
    pos[i] = p
    vel[i] = v
    out_pos[i] = p
    out_col[i] = c


@wp.kernel
def step_exact(pos: wp.array(dtype=wp.vec3),
               vel: wp.array(dtype=wp.vec3),
               out_pos: wp.array(dtype=wp.vec3),
               out_col: wp.array(dtype=wp.vec3),
               dt: float, t: float, blast: float, frame: int):
    # the A/B baseline: per-particle 4D curlnoise, ~70x the noise evals
    i = wp.tid()
    p = pos[i]
    s = wp.rand_init(7)
    turb = wp.curlnoise(s, wp.vec4(p[0] * FREQ, p[1] * FREQ, p[2] * FREQ, t * 0.15)) * TURB
    p, v, c = integrate(p, vel[i], i, turb, dt, t, blast, frame)
    pos[i] = p
    vel[i] = v
    out_pos[i] = p
    out_col[i] = c


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
# (a = G r / (r^2+eps)^1.5) — mismatching these evacuates the core.
v_circ = (np.sqrt(G * radius ** 2 / (radius ** 2 + 0.04) ** 1.5)
          * rng.uniform(0.95, 1.05, N).astype(np.float32))
v0 = np.stack([-np.sin(theta) * v_circ, np.zeros(N, np.float32),
               np.cos(theta) * v_circ], axis=-1)
v0 += rng.normal(0, 0.02, (N, 3)).astype(np.float32)

wp.init()
device = wp.get_preferred_device()
print(f"nebula: {N:,} particles on {device}")

pos = wp.array(p0.astype(np.float32), dtype=wp.vec3, device=device)
vel = wp.array(v0.astype(np.float32), dtype=wp.vec3, device=device)
noise = wp.zeros((NG, NG, NG), dtype=wp.vec3, device=device)
col = None               # host path only — direct mode writes colors straight
                         # into the mapped VBO, so the 216 MB state array (at
                         # 18M) is never allocated there

DIRECT = ("--no-direct" not in sys.argv) and device.is_cuda


def launch_step(out_pos, out_col, blast):
    """Bake this frame's noise field (grid mode), then one sim step."""
    if not EXACT_NOISE:
        wp.launch(bake_noise, dim=(NG, NG, NG), device=device,
                  inputs=[sim_time, noise])
        wp.launch(step, dim=N, device=device,
                  inputs=[pos, vel, out_pos, out_col, noise,
                          DT, sim_time, blast, frame_no])
    else:
        wp.launch(step_exact, dim=N, device=device,
                  inputs=[pos, vel, out_pos, out_col,
                          DT, sim_time, blast, frame_no])


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

# Host-path fallback (tier 1): persistent pinned mirrors — allocating fresh
# 24 MB numpy arrays every frame makes the OS hitch — double-buffered so the
# GPU simulates frame N+1 while the CPU uploads and renders frame N.
pos_host = col_host = pos_view = col_view = None
col_fresh = [True, True]
COLOR_EVERY = 3          # colors drift slowly outside a nova; ship every 3rd frame


def enable_host_path():
    global DIRECT, pos_host, col_host, pos_view, col_view, col
    DIRECT = False
    col = wp.zeros(N, dtype=wp.vec3, device=device)
    pos_host = [wp.zeros(N, dtype=wp.vec3, device="cpu", pinned=True) for _ in range(2)]
    col_host = [wp.zeros(N, dtype=wp.vec3, device="cpu", pinned=True) for _ in range(2)]
    pos_view = [a.numpy() for a in pos_host]
    col_view = [a.numpy() for a in col_host]

sim_time = 0.0
frame_no = 0
cur = 0                  # host-buffer pair the CPU consumes this frame


def gpu_advance(buf):
    """Queue one sim step + DMA into host pair `buf` — all async."""
    global sim_time, frame_no
    ph = sim_time - NOVA_T0
    blast = NOVA_AMP * math.exp(-(ph % NOVA_PERIOD) / NOVA_TAU) if ph >= 0.0 else 0.0
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
    ph = sim_time - NOVA_T0
    blast = NOVA_AMP * math.exp(-(ph % NOVA_PERIOD) / NOVA_TAU) if ph >= 0.0 else 0.0
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


# --- threepp scene ----------------------------------------------------------------

canvas = tp.Canvas("threepp x warp — nebula", width=1280, height=800,
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
    WARMUP, TIMED = 60, 240
    for _ in range(WARMUP):
        step_frame()
        renderer.render(scene, camera)
    wait = upd = rend = 0.0
    for _ in range(TIMED):
        w, u_ = step_frame()
        t0 = time.perf_counter()
        renderer.render(scene, camera)
        wait += w
        upd += u_
        rend += time.perf_counter() - t0
    ms = 1000.0 / TIMED
    total = (wait + upd + rend) * ms
    print(f"bench {N:,} particles [{'direct' if DIRECT else 'host'}]: "
          f"gpu {wait * ms:.2f} | update {upd * ms:.2f} | render {rend * ms:.2f} "
          f"= {total:.2f} ms/frame ({1000.0 / total:.0f} fps)")
    renderer.save_frame("warp_nebula.png")
elif SHOT:
    for _ in range(int(round(SHOT_TIME * 60))):
        step_frame()
    renderer.render(scene, camera)
    renderer.save_frame("warp_nebula.png")
    print(f"simulated {SHOT_TIME:.1f} s, wrote warp_nebula.png")
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)

    canvas.on_window_resize(on_resize)

    def animate():
        step_frame()
        # slow parallax drift so a recording always has camera motion;
        # dragging to orbit composes on top of it
        nebula.rotation.y = 0.04 * sim_time
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)
