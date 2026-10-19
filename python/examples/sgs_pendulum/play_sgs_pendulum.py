"""Watch a trained pendulum-pool policy work a grid of starts in the threepp viewer.

    python play_sgs_pendulum.py --policy runs/sgs_K1024_s0.pt

Columns are the torque limits of the pool (weakest on the left), rows are start states: hanging
still, horizontal left and right, and nearly upright. Every pendulum runs the SAME policy; the
env is the training env itself (sim-less VecTask on the CPU), so what you see is what the
evaluation scores. Each episode restarts from its cell when the time limit hits. The bob turns
green while the pendulum counts as upright. Keys: R = restart all, Esc = quit.
"""
import argparse
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import threepp as tp                                                           # noqa: E402
from demo_common import resize_handler                                         # noqa: E402
from pendulum_env import (EPISODE_S, L, N_TH, N_W, TAUS, UP_COS, UP_W, PendulumPoolEnv)  # noqa: E402
from threepp.rl import load_policy                                             # noqa: E402

ROWS = ((math.pi, "hanging"), (-math.pi / 2, "left"), (math.pi / 2, "right"), (0.35, "near top"))
SPACING = 2.6


def cell_of(theta0, w0, tau_index):
    ti = int(((theta0 + math.pi) / (2 * math.pi)) * N_TH) % N_TH
    wi = min(N_W - 1, max(0, int((w0 + 4.0) / 8.0 * N_W)))
    return (ti * N_W + wi) * len(TAUS) + tau_index


def std_mat(color, rough=0.5, metal=0.3):
    m = tp.MeshStandardMaterial()
    m.color = color; m.roughness = rough; m.metalness = metal
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(_HERE, "sgs_pendulum.pt"))
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-quit after this long (self-test)")
    args = ap.parse_args()
    if not os.path.exists(args.policy):
        raise SystemExit(f"no policy at {args.policy} — train one with train_sgs_pendulum.py --out")

    ac, norm, meta = load_policy(args.policy, device="cpu")
    cells = [cell_of(th, 0.0, k) for th, _ in ROWS for k in range(len(TAUS))]
    env = PendulumPoolEnv(len(cells), device="cpu", seed=0, cells=cells)
    env.d_th = env.d_w = 0.0                  # the viewer starts exactly at the cell centres
    obs = env.reset()
    dt = env.dt

    canvas = tp.Canvas("threepp — pendulum pool (success-guided sampling)", width=1280, height=800)
    renderer = tp.GLRenderer(canvas)
    scene = tp.Scene(); scene.background = tp.Color(0x222a33)
    camera = tp.PerspectiveCamera(45, canvas.aspect(), 0.1, 200)
    cx, cz = (len(TAUS) - 1) * SPACING * 0.5, (len(ROWS) - 1) * SPACING * 0.5
    camera.position.set(cx, 9.0, cz + 14.0)
    controls = tp.OrbitControls(camera, canvas); controls.target.set(cx, 1.0, cz); controls.update()
    scene.add(tp.HemisphereLight(0xbfd4ff, 0x202830, 1.15))
    sun = tp.DirectionalLight(0xffffff, 1.4); sun.position.set(6, 10, 8); scene.add(sun)
    floor = tp.Mesh(tp.BoxGeometry(40, 0.04, 40), std_mat(0x283038, rough=1.0, metal=0.0))
    floor.position.set(cx, -0.02, cz); scene.add(floor)

    arm_mat = std_mat(0xced6e0, rough=0.4, metal=0.4)
    post_mat = std_mat(0x46505d, rough=0.6, metal=0.35)
    pivots, bob_mats = [], []
    for r, (th0, _) in enumerate(ROWS):
        for k in range(len(TAUS)):
            x, z, y = k * SPACING, r * SPACING, 1.3
            post = tp.Mesh(tp.BoxGeometry(0.08, y, 0.08), post_mat); post.position.set(x, y * 0.5, z - 0.18)
            scene.add(post)
            pivot = tp.Group(); pivot.position.set(x, y, z)
            pivot.add(tp.Mesh(tp.SphereGeometry(0.06, 16, 12), post_mat))
            arm = tp.Mesh(tp.CylinderGeometry(0.025, 0.025, L, 12), arm_mat); arm.position.set(0, L * 0.5, 0)
            pivot.add(arm)
            bm = std_mat(0xff5a3c, rough=0.35, metal=0.1)
            bob = tp.Mesh(tp.SphereGeometry(0.11, 20, 14), bm); bob.position.set(0, L, 0)
            pivot.add(bob)
            scene.add(pivot); pivots.append(pivot); bob_mats.append(bm)

    canvas.on_window_resize(resize_handler(camera, renderer))
    ui = tp.ImguiContext(canvas, renderer) if tp.HAS_IMGUI else None
    clock = tp.Clock()
    st = {"obs": obs, "accum": 0.0, "key_r": False, "t": 0.0, "succ": [0] * len(cells), "eps": [0] * len(cells)}
    zax = tp.Vector3(0, 0, 1)

    @torch.no_grad()
    def control_step():
        o = st["obs"] if norm is None else norm.norm(st["obs"])
        obs, _, done, _, _ = env.step(ac.act_mean(o))
        st["obs"] = obs
        for i in torch.nonzero(done).flatten().tolist():
            st["eps"][i] += 1; st["succ"][i] += int(env.last_success[i])

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12); tp.imgui.set_next_window_size(360, 0)
        tp.imgui.begin("Pendulum pool")
        tp.imgui.text(f"policy: {os.path.basename(args.policy)}")
        tp.imgui.text("columns: torque limit  " + "  ".join(f"{t:g}" for t in TAUS))
        tp.imgui.text("rows (front to back): " + ", ".join(n for _, n in ROWS))
        tp.imgui.text(f"episode {EPISODE_S:.0f} s, success = upright through the last 1.5 s")
        for r, (_, name) in enumerate(ROWS):
            row = [f"{st['succ'][r * len(TAUS) + k]}/{st['eps'][r * len(TAUS) + k]}" for k in range(len(TAUS))]
            tp.imgui.text(f"{name:>9}: " + "  ".join(row))
        tp.imgui.text("R = restart all    Esc = quit")
        tp.imgui.end()

    def animate():
        if canvas.is_key_down("ESCAPE") or (args.seconds and st["t"] > args.seconds):
            canvas.close(); return
        r = canvas.is_key_down("R")
        if r and not st["key_r"]:
            st["obs"] = env.reset()
        st["key_r"] = r
        d = min(clock.get_delta(), 0.1)
        st["t"] += d; st["accum"] += d
        while st["accum"] >= dt:
            st["accum"] -= dt
            control_step()
        th = env.theta.tolist(); w = env.w.tolist()
        for i, pv in enumerate(pivots):
            pv.quaternion.set_from_axis_angle(zax, th[i])
            up = math.cos(th[i]) > UP_COS and abs(w[i]) < UP_W
            bob_mats[i].color = 0x3ad36a if up else 0xff5a3c
        if ui:
            controls.enabled = not ui.want_capture_mouse
        controls.update()
        renderer.render(scene, camera)
        if ui:
            ui.render(draw_ui)

    print(__doc__)
    canvas.animate(animate)


if __name__ == "__main__":
    main()
