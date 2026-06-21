"""Drive the physics hexapod around with WASD — no training, just the CPG gait.

    python spider_demo.py

  W / S : walk forward / back   A / D : turn left / right   R : reset   Esc : quit

A statically-stable tripod gait (examples/spider/hexapod.py) maps your velocity
command to joint motors; PhysX does the rest. This is the controllable spider —
RL later makes the gait robust/adaptive, but it already walks. Needs a PhysX
build and a display.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # python/ (for `threepp`)
sys.path.insert(0, _HERE)                                    # for `hexapod`

import threepp as tp
from arena import build_arena, reset_props
from hexapod import Hexapod

if not tp.HAS_PHYSX:
    print("This build has no PhysX backend (configure with the vcpkg toolchain).")
    sys.exit(0)

canvas = tp.Canvas("threepp — drive the spider (WASD)", width=1100, height=720)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True

scene = tp.Scene()
camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 200)
camera.position.set(-3.5, 2.6, 0.0)

world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
props = build_arena(scene, world)  # lit ground arena: grid, perimeter, rocks, knock-around props

spider = Hexapod(world, position=(0, 0.40, 0))
spider.add_to_scene(scene)
for m in spider.meshes:
    m.cast_shadow = True


def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)

ui = tp.ImguiContext(canvas) if tp.HAS_IMGUI else None
clock = tp.Clock()
cam = [-3.5, 2.6, 0.0]  # smoothed chase-cam position
key_r = [False]         # edge-detect the reset key


def do_reset():
    spider.reset()
    reset_props(props)
    p, f = spider.position, spider.forward
    fl = math.hypot(f.x, f.z) or 1.0
    cam[:] = [p.x - 3.4 * f.x / fl, p.y + 2.3, p.z - 3.4 * f.z / fl]  # snap cam behind


def draw_ui():
    tp.imgui.set_next_window_pos(10, 10)
    tp.imgui.set_next_window_size(250, 0)
    tp.imgui.begin("Spider")
    tp.imgui.text("W/S walk   A/D turn   R reset   Esc quit")
    f, t = spider.cmd
    tp.imgui.text("command:  fwd %+.0f  turn %+.0f" % (f, t))
    p = spider.position
    tp.imgui.text("position: (%+.1f, %+.1f)" % (p.x, p.z))
    tp.imgui.text("%.0f fps" % tp.imgui.get_framerate())
    tp.imgui.end()


def animate():
    dt = min(clock.get_delta(), 1 / 30)
    if canvas.is_key_down("ESCAPE"):
        canvas.close()
        return
    r = canvas.is_key_down("R")
    if r and not key_r[0]:
        do_reset()
    key_r[0] = r

    fwd = (1.0 if canvas.is_key_down("W") else 0.0) - (1.0 if canvas.is_key_down("S") else 0.0)
    turn = (1.0 if canvas.is_key_down("A") else 0.0) - (1.0 if canvas.is_key_down("D") else 0.0)
    spider.set_command(fwd, turn)
    spider.update(dt)
    world.step(dt)

    # Chase cam: trail behind the spider's actual heading, smoothed.
    p = spider.position
    f = spider.forward
    fl = math.hypot(f.x, f.z) or 1.0
    want = (p.x - 3.4 * f.x / fl, p.y + 2.3, p.z - 3.4 * f.z / fl)
    for i in range(3):
        cam[i] += (want[i] - cam[i]) * 0.07
    camera.position.set(*cam)
    camera.look_at(p.x, p.y + 0.1, p.z)

    renderer.render(scene, camera)
    if ui:
        ui.render(draw_ui)


print(__doc__)
canvas.animate(animate)
