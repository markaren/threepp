"""A squishy pressurized ball simulated with NVIDIA Warp, rendered by threepp.

A soft-body "balloon" model in raw Warp kernels: an icosphere shell with XPBD
distance constraints (edges + bending pairs) and a single global volume
constraint that acts as internal pressure. The ball drops onto a ramp,
squashes, tumbles off the edge, and bounces on the floor. Integration,
constraint projection, collision + friction, the volume reduction, and
smooth vertex normals all run on the GPU; threepp gets positions/normals as
numpy arrays through update_attribute each frame.

    pip install warp-lang
    python warp_squishy_ball.py            # window; drag to orbit, Esc quits
    python warp_squishy_ball.py --shot 4   # headless: sim 4 s, write warp_squishy_ball.png

Warp falls back to CPU if no CUDA device is present -- slower, same picture.
"""
import math
import os
import sys

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (accum_normals, cli_arg, csr_from_pairs, grad_sumsq, ground_plane,
                         icosphere, integrate, orbit_loop, scatter_soup, shell_pairs,
                         shot_loop, signed_volume, standard_material, studio_lights,
                         volume_apply, volume_grad)

SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 4.0, float)

# --- tunables ---------------------------------------------------------------

SUBDIV = 4                   # icosphere subdivisions
RADIUS = 0.55
SPAWN = (-1.7, 3.0, 0.03)
DT = 1.0 / 240.0
SUBSTEPS = 4
ITERATIONS = 10
DAMPING = 0.015
GRAVITY = wp.vec3(0.0, -9.81, 0.0)
RELAX = 0.35                 # Jacobi under-relaxation for the edge solve
OVERPRESSURE = 1.05          # target volume as a fraction of rest volume
STIFF_EDGE = 1.0
STIFF_BEND = 0.3
MU_FLOOR = 0.35
MU_RAMP = 0.15

RAMP_DEG = 20.0
RAMP_Q = (-1.2, 1.35, 0.0)   # a point on the ramp's top surface
RAMP_X0, RAMP_X1 = -2.19, -0.21
RAMP_ZW = 0.9

_s, _c = math.sin(math.radians(RAMP_DEG)), math.cos(math.radians(RAMP_DEG))
RAMP_N = (_s, _c, 0.0)       # normal tips +x, so the downhill direction is +x


# --- warp kernels -----------------------------------------------------------

@wp.func
def collide(p: wp.vec3, ramp_q: wp.vec3, ramp_n: wp.vec3,
            x0: float, x1: float, zw: float) -> wp.vec3:
    if p[1] < 0.0:
        p = wp.vec3(p[0], 0.0, p[2])
    if p[0] > x0 and p[0] < x1 and wp.abs(p[2]) < zw:
        d = wp.dot(p - ramp_q, ramp_n)
        if d < 0.0:
            p = p - ramp_n * d
    return p


@wp.kernel
def solve(p_in: wp.array(dtype=wp.vec3),
          p_out: wp.array(dtype=wp.vec3),
          offsets: wp.array(dtype=int),
          indices: wp.array(dtype=int),
          rests: wp.array(dtype=float),
          stiffs: wp.array(dtype=float),
          ramp_q: wp.vec3, ramp_n: wp.vec3, x0: float, x1: float, zw: float):
    i = wp.tid()
    p = p_in[i]
    c = wp.vec3(0.0, 0.0, 0.0)
    for k in range(offsets[i], offsets[i + 1]):
        d = p_in[indices[k]] - p
        l = wp.length(d)
        if l > 1.0e-9:
            # Half the correction: the neighbour computes the other half.
            c += d * (0.5 * stiffs[k] * (l - rests[k]) / l)
    p_out[i] = collide(p + c * RELAX, ramp_q, ramp_n, x0, x1, zw)


@wp.kernel
def contacts(pos: wp.array(dtype=wp.vec3),
             prev: wp.array(dtype=wp.vec3),
             ramp_q: wp.vec3, ramp_n: wp.vec3, x0: float, x1: float, zw: float):
    i = wp.tid()
    p = collide(pos[i], ramp_q, ramp_n, x0, x1, zw)
    # Friction: on contact, take back a fraction of this substep's
    # tangential motion. Contact verts lag, the rest of the ball keeps
    # going -- which is what makes it tumble instead of skate.
    if p[1] < 1.0e-4:
        t = p - prev[i]
        p = p - wp.vec3(t[0], 0.0, t[2]) * MU_FLOOR
    if p[0] > x0 and p[0] < x1 and wp.abs(p[2]) < zw:
        if wp.dot(p - ramp_q, ramp_n) < 1.0e-4:
            t = p - prev[i]
            t = t - ramp_n * wp.dot(t, ramp_n)
            p = p - t * MU_RAMP
    pos[i] = p


# --- mesh construction (numpy, once) ----------------------------------------

verts0, faces = icosphere(SUBDIV)
p0 = verts0 * RADIUS
V0 = signed_volume(p0, faces)

# Constraints: mesh edges (stretch) and, per interior edge, the pair of
# opposite vertices of its two triangles (bending, as a weak distance
# constraint). Both go into one CSR adjacency the solve kernel gathers over.
pairs = shell_pairs(faces, STIFF_EDGE, STIFF_BEND)
n_verts = len(p0)
offsets_np, idx_np, rest_np, stiff_np, _ = csr_from_pairs(n_verts, pairs, p0)

rng = np.random.default_rng(11)
p0 = p0 + np.array(SPAWN, dtype=np.float32) \
        + rng.uniform(-1e-3, 1e-3, p0.shape).astype(np.float32)

wp.init()
device = wp.get_preferred_device()
print(f"squishy ball: {n_verts} particles, {len(faces)} faces, "
      f"{len(pairs)} constraints on {device}")

x = wp.array(p0, dtype=wp.vec3, device=device)
prev = wp.array(p0, dtype=wp.vec3, device=device)
pred = wp.zeros(n_verts, dtype=wp.vec3, device=device)
scratch = wp.zeros(n_verts, dtype=wp.vec3, device=device)
nrm = wp.zeros(n_verts, dtype=wp.vec3, device=device)
grad = wp.zeros(n_verts, dtype=wp.vec3, device=device)
vol = wp.zeros(1, dtype=float, device=device)
sumsq = wp.zeros(1, dtype=float, device=device)
v_target = wp.array(np.float32([OVERPRESSURE * V0]), dtype=float, device=device)
offsets = wp.array(offsets_np, dtype=int, device=device)
indices = wp.array(idx_np, dtype=int, device=device)
rests = wp.array(rest_np, dtype=float, device=device)
stiffs = wp.array(stiff_np, dtype=float, device=device)
tris = wp.array(faces.reshape(-1), dtype=int, device=device)
n_corners = faces.size
soup_pos = wp.zeros(n_corners, dtype=wp.vec3, device=device)
soup_nrm = wp.zeros(n_corners, dtype=wp.vec3, device=device)

RQ = wp.vec3(*RAMP_Q)
RN = wp.vec3(*RAMP_N)


def step_frame():
    """Advance one 60 fps frame of simulation and refresh the geometry."""
    for _ in range(SUBSTEPS):
        wp.launch(integrate, dim=n_verts, device=device,
                  inputs=[x, prev, pred, DT, DAMPING, GRAVITY])
        pa, pb = pred, scratch
        for _ in range(ITERATIONS):
            wp.launch(solve, dim=n_verts, device=device,
                      inputs=[pa, pb, offsets, indices, rests, stiffs,
                              RQ, RN, RAMP_X0, RAMP_X1, RAMP_ZW])
            pa, pb = pb, pa
        vol.zero_()
        sumsq.zero_()
        grad.zero_()
        wp.launch(volume_grad, dim=len(faces), device=device,
                  inputs=[pa, tris, len(faces), vol, grad])
        wp.launch(grad_sumsq, dim=n_verts, device=device, inputs=[grad, n_verts, sumsq])
        wp.launch(volume_apply, dim=n_verts, device=device,
                  inputs=[pa, grad, vol, sumsq, v_target, n_verts])
        wp.launch(contacts, dim=n_verts, device=device,
                  inputs=[pa, prev, RQ, RN, RAMP_X0, RAMP_X1, RAMP_ZW])
        wp.copy(x, pa)
    nrm.zero_()
    wp.launch(accum_normals, dim=len(faces), device=device, inputs=[x, tris, nrm])
    wp.launch(scatter_soup, dim=n_corners, device=device,
              inputs=[x, nrm, tris, soup_pos, soup_nrm])
    # Device -> host -> geometry: one copy per attribute per frame.
    geometry.update_attribute("position", soup_pos.numpy())
    geometry.update_attribute("normal", soup_nrm.numpy())


# --- threepp scene ----------------------------------------------------------

canvas = tp.Canvas("threepp x warp - squishy ball", width=1280, height=800,
                   antialiasing=4, headless=SHOT)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True

scene = tp.Scene()
scene.background = 0x202830

camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 100)
camera.position.set(1.2, 2.1, 5.2)
camera.look_at(-0.4, 1.2, 0)

studio_lights(scene)

# Non-indexed triangle soup; the scatter kernel writes per-corner positions
# and (smooth, accumulated) normals in face order.
geometry = tp.BufferGeometry()
geometry.set_attribute("position", p0[faces.reshape(-1)])
geometry.set_attribute("normal", verts0[faces.reshape(-1)])
ball = tp.Mesh(geometry, standard_material(0xe0713a, 0.45))
ball.cast_shadow = True
ball.frustum_culled = False     # positions change under the renderer's feet
scene.add(ball)

ramp = tp.Mesh(tp.BoxGeometry(2.1, 0.08, 2 * RAMP_ZW), standard_material(0x5a6472, 0.6))
ramp.rotation.z = -math.radians(RAMP_DEG)
ramp.position.set(RAMP_Q[0] - 0.04 * RAMP_N[0], RAMP_Q[1] - 0.04 * RAMP_N[1], 0)
ramp.cast_shadow = True
ramp.receive_shadow = True
scene.add(ramp)

ground_plane(scene)

if SHOT:
    shot_loop(renderer, scene, camera, step_frame, SHOT_TIME, "warp_squishy_ball.png")
else:
    orbit_loop(canvas, renderer, scene, camera, step_frame)
