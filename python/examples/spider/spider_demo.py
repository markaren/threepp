"""Drive the physics hexapod around with WASD — no training, just the CPG gait.

    python spider_demo.py

  W / S : walk forward / back     A / D : turn left / right     Esc : quit

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
from hexapod import Hexapod

if not tp.HAS_PHYSX:
    print("This build has no PhysX backend (configure with the vcpkg toolchain).")
    sys.exit(0)

canvas = tp.Canvas("threepp — drive the spider (WASD)", width=1100, height=720)
renderer = tp.GLRenderer(canvas)
renderer.set_clear_color(0x222a32)

scene = tp.Scene()
scene.background = 0x222a32
camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 200)
camera.position.set(-3.5, 2.6, 0.0)

scene.add(tp.HemisphereLight(0xffffff, 0x334433, 1.0))
sun = tp.DirectionalLight(0xffffff, 2.0)
sun.position.set(6, 12, 5)
scene.add(sun)

world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))

ground_mat = tp.MeshStandardMaterial()
ground_mat.color = 0x55624f
ground = tp.Mesh(tp.BoxGeometry(80, 1, 80), ground_mat)
ground.position.y = -0.5
scene.add(ground)
world.add_static(ground)

# A ring of light props so motion is visible and the spider can shove them.
for i in range(10):
    a = i / 10 * math.tau
    m = tp.MeshStandardMaterial()
    m.color = (0xE5C04B, 0x3CA0E5, 0xE5814B, 0x49C66A)[i % 4]
    prop = tp.Mesh(tp.BoxGeometry(0.3, 0.3, 0.3), m)
    prop.position.set(2.6 * math.cos(a), 0.16, 2.6 * math.sin(a))
    scene.add(prop)
    world.add(prop, density=60)

spider = Hexapod(world, position=(0, 0.40, 0))
spider.add_to_scene(scene)


def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)

ui = tp.ImguiContext(canvas) if tp.HAS_IMGUI else None
clock = tp.Clock()
cam = [-3.5, 2.6, 0.0]  # smoothed chase-cam position


def draw_ui():
    tp.imgui.set_next_window_pos(10, 10)
    tp.imgui.set_next_window_size(250, 0)
    tp.imgui.begin("Spider")
    tp.imgui.text("W/S walk   A/D turn   Esc quit")
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
