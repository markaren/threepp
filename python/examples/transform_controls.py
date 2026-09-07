"""Demonstrates TransformControls — grab the gizmo to translate, rotate, or scale a mesh.

    python transform_controls.py

Keyboard shortcuts:
  W       translate mode
  E       rotate mode
  R       scale mode
  Q       toggle local / world space
  X/Y/Z   toggle individual gizmo axes
  SPACE   toggle controls on/off
  SHIFT   hold for snapping (1 m / 15 deg / 0.25 scale)
  ESC     quit
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

canvas = tp.Canvas("TransformControls", antialiasing=4)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True

scene = tp.Scene()
scene.background = 0x202830

camera = tp.PerspectiveCamera(60, canvas.aspect(), 0.1, 1000)
camera.position.set(0, 5, 8)

orbit = tp.OrbitControls(camera, canvas)
orbit.enable_damping = True

# ---- lights ------------------------------------------------------------------
scene.add(tp.AmbientLight(0xaaaaaa, 0.8))
key = tp.DirectionalLight(0xffffff, 2.0)
key.position.set(5, 10, 7)
key.cast_shadow = True
scene.add(key)

# ---- objects -----------------------------------------------------------------
mat = tp.MeshStandardMaterial()
mat.color = 0x4488ff
mat.roughness = 0.4
mat.metalness = 0.1
mesh = tp.Mesh(tp.BoxGeometry(1.0, 1.0, 1.0), mat)
mesh.position.y = 0.5
mesh.cast_shadow = True
scene.add(mesh)

ground = tp.Mesh(tp.PlaneGeometry(20, 20), tp.MeshStandardMaterial())
ground.receive_shadow = True
ground.rotate_x(-math.pi / 2)
scene.add(ground)
scene.add(tp.GridHelper(20, 20))

# ---- transform controls ------------------------------------------------------
tc = tp.TransformControls(camera, canvas)
tc.attach(mesh)
scene.add(tc)           # the gizmo is itself an Object3D

# ---- resize ------------------------------------------------------------------
def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)

canvas.on_window_resize(on_resize)

# ---- one-shot key edge detection ---------------------------------------------
_prev_keys = {}

def pressed(key):
    now = canvas.is_key_down(key)
    fired = now and not _prev_keys.get(key, False)
    _prev_keys[key] = now
    return fired

# ---- animate -----------------------------------------------------------------
def animate():
    if pressed("W"):
        tc.set_mode("translate")
    if pressed("E"):
        tc.set_mode("rotate")
    if pressed("R"):
        tc.set_mode("scale")
    if pressed("Q"):
        tc.set_space("local" if tc.get_space() == "world" else "world")
    if pressed("X"):
        tc.show_x = not tc.show_x
    if pressed("Y"):
        tc.show_y = not tc.show_y
    if pressed("Z"):
        tc.show_z = not tc.show_z
    if pressed("SPACE"):
        tc.enabled = not tc.enabled

    if canvas.is_key_down("SHIFT"):
        tc.set_translation_snap(1.0)
        tc.set_rotation_snap(math.radians(15))
        tc.set_scale_snap(0.25)
    else:
        tc.set_translation_snap(None)
        tc.set_rotation_snap(None)
        tc.set_scale_snap(None)

    orbit.enabled = not tc.dragging
    orbit.update()
    renderer.render(scene, camera)

canvas.animate(animate)
