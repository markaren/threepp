"""Deformable ground -- the same boots and the same rolling ball, in mud and in snow.

Two plots of soft terrain side by side, each a `threepp.terrain_deform.DeformableTerrain`:
a height grid that the colliders push down into, that banks the volume they displace and
shoves it onto the neighbouring cells that are not themselves under a boot. The model is
Sumner/O'Brien/Hodgins 1999, run as batched Warp kernels; this demo is the K=1 case of the
same class an RL env runs at K=1024.

The materials differ in one number that does most of the work -- `compression`, the
fraction of the displaced volume that compacts instead of moving:

    mud   compression 0.15, repose tan 1.4  -> nothing packs, so everything the boot
                                               displaces piles up beside it, and the
                                               trench walls stand
    snow  compression 0.70, repose tan 0.9  -> most of the volume is air being squeezed
                                               out: prints are deeper, rims are low

Watch the left plot grow berms along the boot track and a raised bow-wave ahead of the
ball, and the right plot take deep stamped prints with almost nothing thrown out of them.

    pip install warp-lang
    python warp_mudsnow.py             # window; drag to orbit, Esc quits
    python warp_mudsnow.py --shot 6    # headless: sim 6 s, write warp_mudsnow.png
    python warp_mudsnow.py --bench     # per-phase timings

Warp falls back to CPU if no CUDA device is present -- slower, same picture.
"""
import math
import os
import sys
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import threepp as tp
from threepp.terrain_deform import DeformableTerrain, GridMeshView
from warp_common import (bench_loop, cli_arg, ground_plane, orbit_loop, shot_loop,
                         standard_material, studio_lights)

SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 6.0, float)
BENCH = "--bench" in sys.argv

# --- the two plots ----------------------------------------------------------

CELL = 0.025
PLOT_W = 4.0                    # across (world x)
PLOT_L = 7.0                    # along  (world -z)
NX = int(round(PLOT_W / CELL)) + 1
NY = int(round(PLOT_L / CELL)) + 1
GAP = 0.6
MUD_X0 = -(GAP * 0.5 + PLOT_W)  # left plot's lower x corner
SNOW_X0 = GAP * 0.5
Y0 = -PLOT_L * 0.5              # grid y runs from here; world z = -y

MUD_COLOR, MUD_ROUGH = 0x574026, 0.35      # wet clay reads darker and glossier
SNOW_COLOR, SNOW_ROUGH = 0xeef2f6, 0.95

# --- the actors -------------------------------------------------------------

DT = 1.0 / 60.0
SUB = 2                         # deformation substeps per rendered frame

R_BOOT = 0.07
STRIDE = 0.45                   # per foot per half cycle
CADENCE = 1.0                   # gait cycles per second
LIFT = 0.20
TRACK = 0.13                    # half the distance between the two boots
BOOT_Y0 = 6.4                   # start this far along the plot, then walk toward the camera

R_BALL = 0.30
BALL_SPEED = 0.85
BALL_Y0 = 6.7
BALL_SWEEP = 0.85               # amplitude of the S-curve
BALL_WAVE = 3.2                 # metres per S

# how deep the actors are pressed below the datum while in contact
SINK = {"mud": (0.040, 0.060), "snow": (0.065, 0.100)}   # (boot, ball)


def actors(t, plot_x0, sink):
    """Sphere centres [3, 3] for (boot L, boot R, ball) at time t, in grid coordinates.

    Everything is prescribed: the boots plant for half of each cycle and swing clear for
    the other half, the ball rolls a slow S. The reaction forces `deform` returns are what
    an RL env would feed back into its physics; this demo just drives the kinematics.
    """
    boot_sink, ball_sink = sink
    out = np.zeros((3, 3), dtype=np.float32)
    cx_boot = plot_x0 + 1.10
    for f in range(2):
        phase = t * CADENCE + 0.5 * f
        s = math.floor(phase)
        u = phase - s
        swing = min(max((u - 0.5) * 2.0, 0.0), 1.0)
        swing = swing * swing * (3.0 - 2.0 * swing)          # smoothstep
        travelled = 2.0 * STRIDE * s + f * STRIDE + 2.0 * STRIDE * swing
        planted = u < 0.5
        bottom = -boot_sink if planted else LIFT * math.sin(math.pi * swing)
        out[f] = (cx_boot + (2 * f - 1) * TRACK,
                  (BOOT_Y0 - travelled) % PLOT_L + Y0,
                  bottom + R_BOOT)
    d = BALL_SPEED * t
    out[2] = (plot_x0 + 2.70 + BALL_SWEEP * math.sin(2.0 * math.pi * d / BALL_WAVE),
              (BALL_Y0 - d) % PLOT_L + Y0,
              -ball_sink + R_BALL)
    return out


class Plot:
    """One material: its terrain, its mesh view, its three actor meshes."""

    def __init__(self, scene, x0, material, color, roughness, device):
        self.terrain = DeformableTerrain((x0, Y0), CELL, (NX, NY), K=1,
                                         material=material, device=device)
        self.view = GridMeshView(self.terrain, standard_material(color, roughness), up="y")
        scene.add(self.view.mesh)
        self.x0 = x0
        self.sink = SINK[material]
        self.radii = torch.tensor([R_BOOT, R_BOOT, R_BALL], device=device)
        self.centers = torch.zeros(1, 3, 3, device=device)
        self.vel = torch.zeros(1, 3, 3, device=device)
        self.prev = actors(0.0, x0, self.sink)

        boot_mat = standard_material(0x24272c, 0.65)
        ball_mat = standard_material(0x6b6f78, 0.35, 0.55)
        self.meshes = [tp.Mesh(tp.SphereGeometry(R_BOOT, 24, 16), boot_mat),
                       tp.Mesh(tp.SphereGeometry(R_BOOT, 24, 16), boot_mat),
                       tp.Mesh(tp.SphereGeometry(R_BALL, 40, 24), ball_mat)]
        for m in self.meshes:
            m.cast_shadow = True
            scene.add(m)

    def advance(self, t):
        """Move the actors to time t and displace the ground under them."""
        c = actors(t, self.x0, self.sink)
        self.centers[0].copy_(torch.from_numpy(c))
        self.vel[0].copy_(torch.from_numpy((c - self.prev) / (DT / SUB)))
        self.prev = c
        self.terrain.deform(self.centers, self.radii, self.vel)
        for m, p in zip(self.meshes, c):
            # grid (x, y, height) -> the y-up scene, the same rotation GridMeshView uses
            m.position.set(float(p[0]), float(p[2]), -float(p[1]))


# --- scene ------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

canvas = tp.Canvas("threepp x warp - mud and snow", width=1280, height=800,
                   antialiasing=4, headless=SHOT or BENCH)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True

scene = tp.Scene()
scene.background = 0x222a33

camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.1, 100)
camera.position.set(0.0, 3.5, 7.6)
camera.look_at(0.0, 0.0, 0.6)

# A low sun, across the tracks rather than along them: the ruts are centimetres deep and
# the berms millimetres tall, and grazing light is the only thing that turns that into
# contrast -- lit from the left, along the direction the actors walk, the mud berms lose
# their shading almost entirely. The sun is also turned up, because the hemisphere fill is
# what flattens relief this shallow.
sun = studio_lights(scene, sun_pos=(6.0, 2.6, 4.5))
sun.intensity = 4.0
ground_plane(scene, size=40.0, y=-0.06, color=0x343a41)

mud = Plot(scene, MUD_X0, "mud", MUD_COLOR, MUD_ROUGH, device)
snow = Plot(scene, SNOW_X0, "snow", SNOW_COLOR, SNOW_ROUGH, device)
plots = (mud, snow)
print(f"mud/snow: 2 x {NX}x{NY} cells at {CELL * 100:.1f} cm on {device}")

sim_t = 0.0


def step_frame():
    """One 60 fps frame: SUB deformation substeps, then repose relaxation, then publish."""
    global sim_t
    for _ in range(SUB):
        sim_t += DT / SUB
        for p in plots:
            p.advance(sim_t)
    for p in plots:
        p.terrain.relax(2)
    for p in plots:
        p.view.update()


def step_timed():
    """The same frame, split into the three phases the bench reports."""
    global sim_t
    t0 = time.perf_counter()
    for _ in range(SUB):
        sim_t += DT / SUB
        for p in plots:
            p.advance(sim_t)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    for p in plots:
        p.terrain.relax(2)
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    for p in plots:
        p.view.update()
    t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2


if BENCH:
    bench_loop(step_timed, lambda: renderer.render(scene, camera),
               ("deform", "relax", "view"), 20, 120, f"{NX}x{NY} x2")
elif SHOT:
    shot_loop(renderer, scene, camera, step_frame, SHOT_TIME, "warp_mudsnow.png")
else:
    orbit_loop(canvas, renderer, scene, camera, step_frame, target=(0.0, 0.0, 0.4))
