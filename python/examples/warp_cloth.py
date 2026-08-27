"""NVIDIA Warp cloth simulated on the GPU, rendered live by threepp.

A curtain pinned at three points flaps in gusty wind while a sphere sweeps
through it. The whole simulation -- Verlet integration, XPBD-style constraint
projection, sphere/floor collision, and per-vertex normals -- runs as Warp
kernels on the GPU. Each frame the positions and normals are copied back to
numpy and pushed into a threepp PlaneGeometry with update_attribute (in
place, partial upload, no buffer churn).

    pip install warp-lang
    python warp_cloth.py            # opens a window; drag to orbit, Esc quits
    python warp_cloth.py --shot 6   # headless: simulate 6 s, write warp_cloth.png

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
from warp_common import (cli_arg, ground_plane, orbit_loop, shot_loop, standard_material,
                         studio_lights)

SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 6.0, float)

# --- cloth setup -----------------------------------------------------------

NX, NY = 95, 95              # quad segments; (NX+1)*(NY+1) particles
WIDTH, HEIGHT = 3.0, 2.2     # metres
TOP_Y = 2.9                  # world height of the pinned edge
REST_X = WIDTH / NX
REST_Y = HEIGHT / NY
V = (NX + 1) * (NY + 1)

DT = 1.0 / 240.0             # sim substep
SUBSTEPS = 4                 # SUBSTEPS * DT = one 60 fps frame of sim time
ITERATIONS = 8               # constraint projections per substep
DAMPING = 0.015
GRAVITY = wp.vec3(0.0, -9.81, 0.0)
WIND_DIR = wp.vec3(0.33, 0.0, 0.94)
WIND_BASE, WIND_GUST = 2.5, 4.5
SPHERE_R = 0.45


@wp.func
def grid_index(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix


@wp.func
def spring(p: wp.vec3, pos: wp.array(dtype=wp.vec3),
           ix: int, iy: int, nx: int, ny: int,
           rest: float, stiffness: float) -> wp.vec3:
    # A neighbour outside the grid contributes nothing.
    if ix < 0 or ix > nx or iy < 0 or iy > ny:
        return wp.vec3(0.0, 0.0, 0.0)
    d = pos[grid_index(ix, iy, nx)] - p
    l = wp.length(d)
    if l < 1.0e-9:
        return wp.vec3(0.0, 0.0, 0.0)
    # Half the correction: the neighbour computes the other half for itself.
    return d * (0.5 * stiffness * (l - rest) / l)


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3),
              prev: wp.array(dtype=wp.vec3),
              pred: wp.array(dtype=wp.vec3),
              inv_mass: wp.array(dtype=float),
              nx: int, dt: float, t: float,
              wind_base: float, wind_gust: float):
    i = wp.tid()
    p = pos[i]
    prev_p = prev[i]
    prev[i] = p
    if inv_mass[i] == 0.0:
        pred[i] = p
        return
    ix = i % (nx + 1)
    iy = i // (nx + 1)
    # Gusts vary over both time and the cloth surface so folds travel
    # instead of the whole sheet pumping in unison.
    phase = 0.31 * float(ix) + 0.17 * float(iy)
    gust = wp.sin(1.9 * t + phase) * (0.5 + 0.5 * wp.sin(0.7 * t + 0.13 * float(iy)))
    acc = GRAVITY + WIND_DIR * (wind_base + wind_gust * gust)
    vel = (p - prev_p) * (1.0 - DAMPING)
    pred[i] = p + vel + acc * dt * dt


@wp.kernel
def solve(p_in: wp.array(dtype=wp.vec3),
          p_out: wp.array(dtype=wp.vec3),
          inv_mass: wp.array(dtype=float),
          nx: int, ny: int, rx: float, ry: float,
          sphere_c: wp.vec3, sphere_r: float):
    i = wp.tid()
    p = p_in[i]
    if inv_mass[i] == 0.0:
        p_out[i] = p
        return
    ix = i % (nx + 1)
    iy = i // (nx + 1)
    rd = wp.sqrt(rx * rx + ry * ry)
    c = wp.vec3(0.0, 0.0, 0.0)
    # structural
    c += spring(p, p_in, ix - 1, iy, nx, ny, rx, 1.0)
    c += spring(p, p_in, ix + 1, iy, nx, ny, rx, 1.0)
    c += spring(p, p_in, ix, iy - 1, nx, ny, ry, 1.0)
    c += spring(p, p_in, ix, iy + 1, nx, ny, ry, 1.0)
    # shear
    c += spring(p, p_in, ix - 1, iy - 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, ix + 1, iy - 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, ix - 1, iy + 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, ix + 1, iy + 1, nx, ny, rd, 0.85)
    # bending
    c += spring(p, p_in, ix - 2, iy, nx, ny, 2.0 * rx, 0.4)
    c += spring(p, p_in, ix + 2, iy, nx, ny, 2.0 * rx, 0.4)
    c += spring(p, p_in, ix, iy - 2, nx, ny, 2.0 * ry, 0.4)
    c += spring(p, p_in, ix, iy + 2, nx, ny, 2.0 * ry, 0.4)
    # Jacobi projection: everyone moves at once, so under-relax to converge.
    p = p + c * 0.3
    # sphere pushout (radius padded for cloth thickness)
    d = p - sphere_c
    l = wp.length(d)
    if l < sphere_r:
        p = sphere_c + d * (sphere_r / wp.max(l, 1.0e-6))
    # floor
    p = wp.vec3(p[0], wp.max(p[1], 0.01), p[2])
    p_out[i] = p


@wp.kernel
def compute_normals(pos: wp.array(dtype=wp.vec3),
                    nrm: wp.array(dtype=wp.vec3),
                    nx: int, ny: int):
    i = wp.tid()
    ix = i % (nx + 1)
    iy = i // (nx + 1)
    xm = pos[grid_index(wp.max(ix - 1, 0), iy, nx)]
    xp = pos[grid_index(wp.min(ix + 1, nx), iy, nx)]
    ym = pos[grid_index(ix, wp.max(iy - 1, 0), nx)]
    yp = pos[grid_index(ix, wp.min(iy + 1, ny), nx)]
    # Rows run top-down in PlaneGeometry, so cross(dv, du) faces +z at rest,
    # matching the plane's front face.
    n = wp.cross(yp - ym, xp - xm)
    nrm[i] = n / wp.max(wp.length(n), 1.0e-9)


wp.init()
device = wp.get_preferred_device()
print(f"cloth: {V} particles on {device}, "
      f"{SUBSTEPS * ITERATIONS} constraint launches per frame")

# Initial positions in PlaneGeometry vertex order: rows top-down, columns
# left-to-right. Tiny z jitter breaks the perfectly flat start.
xs = np.linspace(-WIDTH / 2, WIDTH / 2, NX + 1, dtype=np.float32)
ys = np.linspace(TOP_Y, TOP_Y - HEIGHT, NY + 1, dtype=np.float32)
gx, gy = np.meshgrid(xs, ys)
rng = np.random.default_rng(7)
gz = rng.uniform(-1e-3, 1e-3, gx.shape).astype(np.float32)
p0 = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

inv_mass_np = np.ones(V, dtype=np.float32)
# The pinned row is iy=0, so grid indices are just ix: two corners + top centre.
inv_mass_np[[0, NX // 2, NX]] = 0.0

pos = wp.array(p0, dtype=wp.vec3, device=device)
prev = wp.array(p0, dtype=wp.vec3, device=device)
pred = wp.zeros(V, dtype=wp.vec3, device=device)
scratch = wp.zeros(V, dtype=wp.vec3, device=device)
nrm = wp.zeros(V, dtype=wp.vec3, device=device)
inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

sim_time = 0.0


def sphere_center(t):
    return (0.9 * math.sin(0.5 * t),
            1.6 + 0.35 * math.sin(0.9 * t),
            0.35 + 0.45 * math.sin(0.36 * t))


def step_frame():
    """Advance one 60 fps frame of simulation and refresh the geometry."""
    global sim_time
    for _ in range(SUBSTEPS):
        cx, cy, cz = sphere_center(sim_time)
        c = wp.vec3(cx, cy, cz)
        wp.launch(integrate, dim=V, device=device,
                  inputs=[pos, prev, pred, inv_mass, NX, DT, sim_time, WIND_BASE, WIND_GUST])
        a, b = pred, scratch
        for _ in range(ITERATIONS):
            wp.launch(solve, dim=V, device=device,
                      inputs=[a, b, inv_mass, NX, NY, REST_X, REST_Y, c, SPHERE_R + 0.03])
            a, b = b, a
        wp.copy(pos, a)
        sim_time += DT
    wp.launch(compute_normals, dim=V, device=device, inputs=[pos, nrm, NX, NY])
    # Device -> host -> geometry: one copy per attribute per frame.
    geometry.update_attribute("position", pos.numpy())
    geometry.update_attribute("normal", nrm.numpy())
    sx, sy, sz = sphere_center(sim_time)
    sphere.position.set(sx, sy, sz)


# --- threepp scene ----------------------------------------------------------

canvas = tp.Canvas("threepp x warp - cloth", width=1280, height=800,
                   antialiasing=4, headless=SHOT)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True

scene = tp.Scene()
scene.background = 0x202830

camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 100)
camera.position.set(2.6, 2.4, 4.8)
camera.look_at(0, 1.6, 0)

studio_lights(scene)

geometry = tp.PlaneGeometry(WIDTH, HEIGHT, NX, NY)
cloth = tp.Mesh(geometry, standard_material(0xb23a3a, 0.85, side=tp.Side.Double))
cloth.cast_shadow = True
scene.add(cloth)

sphere = tp.Mesh(tp.SphereGeometry(SPHERE_R), standard_material(0x7788aa, 0.3, 0.6))
sphere.cast_shadow = True
scene.add(sphere)

ground_plane(scene)

if SHOT:
    shot_loop(renderer, scene, camera, step_frame, SHOT_TIME, "warp_cloth.png")
else:
    orbit_loop(canvas, renderer, scene, camera, step_frame)
