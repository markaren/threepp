"""Rigid-body physics in Python — a pile of boxes tumbling onto the floor.

    python physics_demo.py

Author the scene with the usual threepp API, add meshes to a PhysxWorld, and
call world.step(dt) each frame — the bound meshes' transforms follow the sim.
Needs a build with the omniverse-physx-sdk (threepp.HAS_PHYSX) and a display.

For *synthetic data*, swap GLRenderer for VulkanRenderer and grab AOVs each step
(rgb / segmentation / depth / motion) to get temporally-coherent labeled video —
physics gives you the moving scenes; the G-buffer gives you the labels for free.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

if not tp.HAS_PHYSX:
    print("This build has no PhysX backend. Configure the module with the vcpkg "
          "toolchain (the omniverse-physx-sdk) to enable it.")
    sys.exit(0)

random.seed(7)
PALETTE = [0xE54B4B, 0x3CA0E5, 0x49C66A, 0xE5C04B, 0xA64BE5, 0xE5814B, 0xD9A05B]

canvas = tp.Canvas("threepp — physics")
renderer = tp.GLRenderer(canvas)
renderer.set_clear_color(0x9fb8d4)

scene = tp.Scene()
scene.background = 0x9fb8d4

camera = tp.PerspectiveCamera(58, canvas.aspect(), 0.1, 1000)
camera.position.set(14, 11, 18)
controls = tp.OrbitControls(camera, canvas)
controls.target = tp.Vector3(0, 2, 0)
controls.enable_damping = True

scene.add(tp.HemisphereLight(0xffffff, 0x444a55, 1.0))
sun = tp.DirectionalLight(0xffffff, 2.2)
sun.position.set(10, 20, 8)
scene.add(sun)

world = tp.PhysxWorld()

# Floor: a visual box + a matching static collider (top face at y=0).
floor_mat = tp.MeshStandardMaterial()
floor_mat.color = 0x6b7a55
floor = tp.Mesh(tp.BoxGeometry(60, 1, 60), floor_mat)
floor.position.y = -0.5
scene.add(floor)
world.add_static(floor)


def spawn_box():
    s = random.uniform(0.6, 1.2)
    mat = tp.MeshStandardMaterial()
    mat.color = random.choice(PALETTE)
    mat.roughness = 0.7
    m = tp.Mesh(tp.BoxGeometry(s, s, s), mat)
    m.position.set(random.uniform(-3, 3), random.uniform(8, 22), random.uniform(-3, 3))
    m.rotate_x(random.uniform(0, math.pi))
    m.rotate_z(random.uniform(0, math.pi))
    scene.add(m)
    world.add(m, density=200)


for _ in range(35):
    spawn_box()


def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)

clock = tp.Clock()
elapsed = {"t": 0.0}


def animate():
    dt = clock.get_delta()
    elapsed["t"] += dt
    # keep raining a few boxes so the pile stays lively
    if elapsed["t"] > 1.2:
        elapsed["t"] = 0.0
        spawn_box()
    world.step(dt)
    controls.update()
    renderer.render(scene, camera)


canvas.animate(animate)
