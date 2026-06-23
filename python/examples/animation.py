"""Keyframe animation, driven by an AnimationMixer.

Two paths, mirroring three.js:

  * Procedural (default): build AnimationClips from KeyframeTracks in Python and
    play them on a mixer — no asset files needed. A cube bounces while spinning.

  * glTF: pass a .glb/.gltf with embedded animations and the first clip is played
    instead:  python animation.py path/to/Soldier.glb

Either way the loop is the same three.js pattern:

    mixer  = tp.AnimationMixer(root)
    mixer.clip_action(clip).play()
    ...
    mixer.update(clock.get_delta())   # per frame, before render

Opens a window and plays the animation live until the window is closed.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp


def build_procedural():
    """A cube with .position keyed to bounce and .quaternion keyed to spin."""
    cube = tp.Mesh(tp.BoxGeometry(1, 1, 1), tp.MeshStandardMaterial())
    cube.material.color = 0xff7733

    # Bounce y: 0 -> 2 -> 0 over 2s. Tracks are flat [x,y,z, x,y,z, ...] arrays.
    bounce = tp.VectorKeyframeTrack(
        ".position", [0.0, 1.0, 2.0], [0, 0, 0, 0, 2, 0, 0, 0, 0])
    # Spin a full turn about Y over 2s, as 3 quaternion keyframes (xyzw each).
    s = math.sin(math.pi / 2)
    c = math.cos(math.pi / 2)
    spin = tp.QuaternionKeyframeTrack(
        ".quaternion", [0.0, 1.0, 2.0], [0, 0, 0, 1, 0, s, 0, c, 0, 0, 0, -1])
    clip = tp.AnimationClip("bounce_spin", 2.0, [bounce, spin])
    return cube, clip


def build_from_gltf(path):
    result = tp.GLTFLoader().load(path)
    if not result.animations:
        raise SystemExit(f"{path} contains no animations")
    print(f"loaded {len(result.animations)} clip(s): "
          + ", ".join(c.name for c in result.animations))
    model = result.scene
    model.traverse(lambda o: setattr(o, "cast_shadow", True) if isinstance(o, tp.Mesh) else None)
    return model, result.animations[0]


# --- scene -----------------------------------------------------------------
scene = tp.Scene()
scene.background = 0x202830
scene.add(tp.HemisphereLight(0xffffff, 0x404048, 1.2))
sun = tp.DirectionalLight(0xffffff, 2.0)
sun.position.set(3, 5, 2)
scene.add(sun)

if len(sys.argv) > 1:
    root, clip = build_from_gltf(sys.argv[1])
else:
    root, clip = build_procedural()
scene.add(root)

# Frame the animated object from its bounding box.
bbox = tp.Box3().set_from_object(root)
center = bbox.get_center()
size = bbox.get_size()
radius = max(size.x, size.y, size.z) or 1.0

canvas = tp.Canvas("animation")
renderer = tp.GLRenderer(canvas)

camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.01, 100 * radius)
camera.position.set(center.x + radius * 3, center.y + radius * 2, center.z + radius * 4)
camera.look_at(center.x, center.y, center.z)

# --- animation -------------------------------------------------------------
mixer = tp.AnimationMixer(root)
mixer.clip_action(clip).play()

def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)

controls = tp.OrbitControls(camera, canvas)

clock = tp.Clock()

def animate():
    mixer.update(clock.get_delta())
    renderer.render(scene, camera)


canvas.animate(animate)
