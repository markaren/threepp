"""Mouse picking on an InstancedMesh: input callbacks, Raycaster and set_matrices.

    python mouse_picking.py              # interactive window
    python mouse_picking.py --headless   # 3 frames to mouse_picking.png, no window

A 20x20 grid of cubes rides a gentle wave. Every frame their matrices are built
as one numpy (N, 4, 4) array and handed over with InstancedMesh.set_matrices (no
per-instance Python loop). Canvas.on_mouse_move keeps the cursor; each frame a
Raycaster aimed through it with set_from_camera finds the hovered cube
(Intersection.instance_id), which is tinted with set_color_at.

Mouse callbacks report window pixels, origin top-left. Raycaster.set_from_camera
wants normalized device coordinates in [-1, 1], y up:

    nx = 2 * x / width - 1
    ny = 1 - 2 * y / height

Left click (press and release without dragging) toggles a cube's colour; left
drag orbits, right drag pans, the wheel zooms. R resets the colours, SPACE
pauses the wave, ESC quits.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

HEADLESS = "--headless" in sys.argv
GRID = 20
SPACING = 0.6
N = GRID * GRID

BASE = tp.Color(0x5577aa)
HOVER = tp.Color(0xffcc33)
PICKED = tp.Color(0xe0443a)

canvas = tp.Canvas("Mouse picking", width=960, height=600, antialiasing=4, headless=HEADLESS)
renderer = tp.GLRenderer(canvas)

scene = tp.Scene()
scene.background = 0x1c2128

camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 200)
camera.position.set(0, 9, 11)
camera.look_at(0, 0, 0)

orbit = tp.OrbitControls(camera, canvas)

scene.add(tp.HemisphereLight(0xffffff, 0x404048, 1.0))
sun = tp.DirectionalLight(0xffffff, 2.0)
sun.position.set(5, 10, 6)
scene.add(sun)

floor_mat = tp.MeshStandardMaterial()
floor_mat.color = 0x3a3f47
floor = tp.Mesh(tp.PlaneGeometry(30, 30), floor_mat)
floor.rotate_x(-math.pi / 2)
scene.add(floor)

cube_mat = tp.MeshStandardMaterial()
cube_mat.color = 0xffffff  # instance colours multiply this
cube_mat.roughness = 0.5
cubes = tp.InstancedMesh(tp.BoxGeometry(0.45, 0.45, 0.45), cube_mat, N)
scene.add(cubes)

# Grid layout and the (N, 4, 4) row-major matrices: identity with the
# translation in the last column, as M[row, col].
ix, iz = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
xs = ((ix.ravel() - (GRID - 1) / 2) * SPACING).astype(np.float32)
zs = ((iz.ravel() - (GRID - 1) / 2) * SPACING).astype(np.float32)
matrices = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
matrices[:, 0, 3] = xs
matrices[:, 2, 3] = zs

picked = set()
state = {"ndc": None, "hovered": None, "paused": False, "t": 0.0, "down_at": None}
raycaster = tp.Raycaster()


def paint():
    """Every instance's colour from the picked set and the hovered instance."""
    for i in range(N):
        cubes.set_color_at(i, PICKED if i in picked else BASE)
    if state["hovered"] is not None:
        cubes.set_color_at(state["hovered"], HOVER)
    cubes.instance_color_needs_update()


def to_ndc(x, y):
    w, h = canvas.size()
    return 2 * x / w - 1, 1 - 2 * y / h


def on_mouse_move(x, y):
    state["ndc"] = to_ndc(x, y)


def on_mouse_down(button, x, y):
    if button == 0:
        state["down_at"] = (x, y)


def on_mouse_up(button, x, y):
    # A press and release within a few pixels is a click; anything longer was an orbit drag.
    start = state["down_at"]
    state["down_at"] = None
    if button != 0 or start is None or math.hypot(x - start[0], y - start[1]) > 4:
        return
    if state["hovered"] is not None:
        picked.symmetric_difference_update({state["hovered"]})
        paint()


def on_key_pressed(key, mods):
    if key == "R":
        picked.clear()
        paint()
    elif key == "SPACE":
        state["paused"] = not state["paused"]
    elif key == "ESCAPE":
        canvas.close()


def on_resize(w, h):
    camera.aspect = w / h
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_mouse_move(on_mouse_move)
canvas.on_mouse_down(on_mouse_down)
canvas.on_mouse_up(on_mouse_up)
canvas.on_key_pressed(on_key_pressed)
canvas.on_window_resize(on_resize)


def frame(dt):
    if not state["paused"]:
        state["t"] += dt
    t = state["t"]
    matrices[:, 1, 3] = 0.35 + 0.25 * np.sin(1.6 * t + 0.9 * xs) * np.cos(1.1 * t + 0.7 * zs)
    cubes.set_matrices(matrices)
    cubes.update_matrix_world()

    hovered = None
    if state["ndc"] is not None:
        camera.update_matrix_world()
        raycaster.set_from_camera(state["ndc"], camera)
        hits = raycaster.intersect_object(cubes)
        if hits:
            hovered = hits[0].instance_id
    if hovered != state["hovered"]:
        state["hovered"] = hovered
        paint()

    renderer.render(scene, camera)


paint()

if HEADLESS:
    # No cursor without a window: aim just right of the screen centre (x = 0
    # runs down the gap between two cube columns) and pre-pick a diagonal so
    # the still shows all three colours.
    picked.update(i * GRID + i for i in range(0, GRID, 3))
    paint()
    state["ndc"] = (0.04, 0.0)
    for _ in range(3):
        frame(1 / 60)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mouse_picking.png")
    renderer.save_frame(out)
    print(f"hovered instance {state['hovered']}; wrote {out}")
else:
    print("Hover to highlight, left click toggles a cube, drag orbits, R resets, SPACE pauses, ESC quits.")
    clock = tp.Clock()
    canvas.animate(lambda: frame(clock.get_delta()))
