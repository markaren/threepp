"""Watch the GPU-trained cart-pole SWING UP from hanging straight down, then balance.

    python play_cartpole.py

The policy (train_cartpole.py, ~1.5 min on the GPU) starts the pole hanging dead-down and
pumps the cart to swing it up and balance it at the top — unfakeable real control. Mouse:
orbit / zoom. Keys: R = drop & swing up again, P = shove, Esc = quit.
"""
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from cartpole import CartPole
from cartpole_env import CONFIG as ENV_CONFIG
from cartpole_env import make_obs   # the ONE observation function, shared with training
from threepp.rl import load_policy

MODEL = os.path.join(_HERE, "cartpole_swingup.pt")
if not tp.HAS_PHYSX:
    print("This build has no PhysX backend."); sys.exit(0)
if not os.path.exists(MODEL):
    print("No policy yet — run: python train_cartpole.py"); sys.exit(0)

ac, norm, meta = load_policy(MODEL, device="cpu")
# The deploy contract comes from the policy's meta (what it was TRAINED with), not hand-typed
# constants. Guard against this module having drifted from the trained config since.
for k in ("force_scale", "rail", "dt", "max_substeps", "v_scale", "w_scale"):
    if k in meta and k in ENV_CONFIG and abs(float(meta[k]) - float(ENV_CONFIG[k])) > 1e-9:
        raise SystemExit(f"cartpole_env.{k}={ENV_CONFIG[k]} has drifted from the trained policy "
                         f"(meta {k}={meta[k]}) — retrain or revert before deploying.")
FORCE_SCALE = float(meta["force_scale"])
RAIL = float(meta["rail"])
DT = float(meta["dt"])
MAX_SUBSTEPS = int(meta["max_substeps"])
DOWN = np.array([0.0, math.pi], np.float32)


def std_mat(color, rough=0.5, metal=0.3, emissive=0x000000):
    m = tp.MeshStandardMaterial()
    m.color = color; m.roughness = rough; m.metalness = metal; m.emissive = emissive
    return m


def lerp_color(c0, c1, t):
    t = max(0.0, min(1.0, t))
    out = 0
    for sh in (16, 8, 0):
        a, b = (c0 >> sh) & 255, (c1 >> sh) & 255
        out |= int(a + (b - a) * t) << sh
    return out


# ---- scene / camera / lighting (dark studio, RLtools-style) --------------------
canvas = tp.Canvas("threepp — cart-pole swing-up (GPU-trained)", width=1200, height=760)
renderer = tp.GLRenderer(canvas)
scene = tp.Scene()
scene.background = tp.Color(0x222a33)
camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.1, 200)
camera.position.set(3.6, 2.7, 7.2)
controls = tp.OrbitControls(camera, canvas)
controls.target.set(0.0, 0.95, 0.0)
controls.enable_damping = True
controls.update()

scene.add(tp.HemisphereLight(0xbfd4ff, 0x202830, 1.15))
sun = tp.DirectionalLight(0xffffff, 1.6); sun.position.set(4, 8, 5)
scene.add(sun)

# floor: dark slab + subtle grid
floor = tp.Mesh(tp.BoxGeometry(40, 0.04, 40), std_mat(0x283038, rough=1.0, metal=0.0))
floor.position.set(0, -0.02, 0); floor.receive_shadow = True
scene.add(floor)
grid = tp.Group()
gmat = std_mat(0x33404d, rough=1.0, metal=0.0)
HALF, STEP = 8.0, 0.8
for i in range(int(2 * HALF / STEP) + 1):
    p = -HALF + i * STEP
    lx = tp.Mesh(tp.BoxGeometry(2 * HALF, 0.006, 0.012), gmat); lx.position.set(0, 0.005, p); grid.add(lx)
    lz = tp.Mesh(tp.BoxGeometry(0.012, 0.006, 2 * HALF), gmat); lz.position.set(p, 0.005, 0); grid.add(lz)
scene.add(grid)

# ---- physics (box colliders); visuals are built separately and follow the links
world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), fixed_timestep=DT, max_substeps=MAX_SUBSTEPS)
cp = CartPole(world, x0=0.0)
cp.art.set_joint_positions(DOWN)
world.step(DT)
RAIL_Y, CART_TOP, L = cp.rail_y, cp.cart_top, cp.l
ZAX = tp.Vector3(0, 0, 1)

# gantry frame so the rail reads as a real apparatus (static, set back in Z)
frame_mat = std_mat(0x46505d, rough=0.6, metal=0.35)
ZB = -0.22
beam = tp.Mesh(tp.BoxGeometry(2 * RAIL + 0.7, 0.07, 0.07), frame_mat); beam.position.set(0, RAIL_Y, ZB)
scene.add(beam)
for sx in (-(RAIL + 0.35), (RAIL + 0.35)):
    post = tp.Mesh(tp.BoxGeometry(0.1, RAIL_Y, 0.1), frame_mat); post.position.set(sx, RAIL_Y * 0.5, ZB)
    scene.add(post)

# moving visuals
cart_vis = tp.Mesh(tp.BoxGeometry(0.34, 0.2, 0.24), std_mat(0x2f6fb0, rough=0.35, metal=0.55))
scene.add(cart_vis)
pivot = tp.Group()                                   # at the cart-top hinge; rotates with the pole
pivot.add(tp.Mesh(tp.SphereGeometry(0.05, 20, 14), std_mat(0x9aa6b4, rough=0.4, metal=0.7)))
arm = tp.Mesh(tp.CylinderGeometry(0.022, 0.022, L, 16), std_mat(0xced6e0, rough=0.4, metal=0.4))
arm.position.set(0, L * 0.5, 0); pivot.add(arm)
bob_mat = std_mat(0xff5a3c, rough=0.35, metal=0.1)
bob = tp.Mesh(tp.SphereGeometry(0.085, 24, 16), bob_mat); bob.position.set(0, L, 0)
pivot.add(bob)
scene.add(pivot)


def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)
ui = tp.ImguiContext(canvas) if tp.HAS_IMGUI else None
clock = tp.Clock()
st = {"accum": 0.0, "key_p": False, "key_r": False, "up": -1.0, "rng": np.random.default_rng(0)}


def observation():
    # the SAME make_obs the env trains with, fed this single robot's CPU joint readings
    return make_obs(torch.tensor([cp.cart.joint_position]), torch.tensor([cp.cart.joint_velocity]),
                    torch.tensor([cp.pole.joint_position]), torch.tensor([cp.pole.joint_velocity]))


@torch.no_grad()
def control_step():
    a = float(ac.act_mean(norm.norm(observation())).squeeze().item())
    cp.cart.add_force(tp.Vector3(max(-1.0, min(1.0, a)) * FORCE_SCALE, 0, 0))
    world.step(DT)


def sync_visuals():
    cx = cp.cart.position.x
    th = cp.pole.joint_position
    cart_vis.position.set(cx, RAIL_Y, 0.0)
    pivot.position.set(cx, CART_TOP, 0.0)
    pivot.quaternion.set_from_axis_angle(ZAX, th)
    st["up"] = math.cos(th)
    t = 0.5 * (st["up"] + 1.0)
    bob_mat.color = lerp_color(0xff5a3c, 0x3ad36a, t)        # orange (down) -> green (up)
    bob_mat.emissive = lerp_color(0x2a0d06, 0x10401e, t)


def draw_ui():
    tp.imgui.set_next_window_pos(12, 12)
    tp.imgui.set_next_window_size(300, 0)
    tp.imgui.begin("Cart-pole swing-up")
    tp.imgui.text("GPU-trained  |  PhysX direct-GPU PPO (~1.5 min)")
    tp.imgui.text("uprightness: %+.2f   (-1 down  ->  +1 up)" % st["up"])
    tp.imgui.text("mouse: orbit / zoom")
    tp.imgui.text("R = drop & swing up    P = shove    Esc = quit")
    tp.imgui.end()


def animate():
    if canvas.is_key_down("ESCAPE"):
        canvas.close(); return
    r = canvas.is_key_down("R")
    if r and not st["key_r"]:
        cp.art.set_joint_positions(DOWN)
    st["key_r"] = r
    p = canvas.is_key_down("P")
    if p and not st["key_p"]:
        cp.pole.add_impulse(tp.Vector3(float(st["rng"].choice([-1.0, 1.0])) * 0.5, 0, 0))
    st["key_p"] = p

    st["accum"] += min(clock.get_delta(), 0.1)
    while st["accum"] >= DT:
        st["accum"] -= DT
        control_step()
    sync_visuals()

    if ui:
        controls.enabled = not ui.want_capture_mouse
    controls.update()
    renderer.render(scene, camera)
    if ui:
        ui.render(draw_ui)


print(__doc__)
canvas.animate(animate)
