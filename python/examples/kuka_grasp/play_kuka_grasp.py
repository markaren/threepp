"""play_kuka_grasp.py — watch a trained policy reach, grasp by friction and lift a cube.

    python play_kuka_grasp.py --policy kuka_grasp.pt                     # three arms, one per start family
    python play_kuka_grasp.py --policy kuka_grasp.pt --layout reaching   # one arm, Reaching starts
    python play_kuka_grasp.py --policy kuka_grasp.pt --shot strip.png    # headless: frames of the first 0.8 s

The env is the training env (pool mode, GPU PhysX) built with the URDF's visual meshes; each
frame the meshes take the GPU link and cube poses, so what is drawn is what the evaluation
scores. The policy's action semantics come from its checkpoint (delta for the pool arms, the
June absolute mapping for a scaffold policy). In the "families" layout the arms start, left to
right, from a held-out Reaching cell, a Stable Grasp cell and a Near-Goal cell; when an episode
ends, that arm moves on to the next cell of its family. The cube turns green while it counts
toward success (lifted and between the fingers). Keys: R = new cells for every arm, Esc = quit.
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_EXAMPLES = os.path.dirname(_HERE)
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)
_PYTHON_DIR = os.path.dirname(_EXAMPLES)
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import threepp as tp  # noqa: E402
from threepp.rl import load_policy  # noqa: E402

import kuka_grasp_contract as C  # noqa: E402
from kuka_grasp_env import KukaGraspEnv  # noqa: E402
from kuka_grasp_pool import FAMILIES, NEAR_GOAL, REACHING, STABLE_GRASP, load_pool  # noqa: E402

LAYOUTS = {"families": (REACHING, STABLE_GRASP, NEAR_GOAL), "reaching": (REACHING,)}
CUBE_IDLE, CUBE_HELD = 0xff8c42, 0x3ad36a


def viewer_pool(device):
    """The held-out Reaching cells followed by the training pool's held cells."""
    ev, tr = load_pool(device=device, split="eval"), load_pool(device=device)
    keep = tr["family"] != REACHING
    out = {k: torch.cat([ev[k], tr[k][keep]]) for k in ("q9", "hold", "cube", "family", "sub", "on_table")}
    out["n"] = int(out["q9"].shape[0])
    return out


class ViewerEnv(KukaGraspEnv):
    """Each arm walks through its family's cells, one per episode."""

    def __init__(self, families, pool, seed=0, **kw):
        self.queues = []
        g = np.random.default_rng(seed)
        fam = pool["family"].cpu().numpy()
        for f in families:
            q = np.flatnonzero(fam == f)
            g.shuffle(q)
            self.queues.append(list(q))
        self.pos = [0] * len(families)
        cells = torch.tensor([q[0] for q in self.queues], device="cuda")
        super().__init__(len(families), mode="pool", pool=pool, cells=cells, render_visuals=True, **kw)
        self.episodes = [0] * len(families)
        self.successes = [0] * len(families)

    def advance(self, i):
        self.pos[i] = (self.pos[i] + 1) % len(self.queues[i])
        self.fixed_cells[i] = int(self.queues[i][self.pos[i]])

    def on_done(self, idx):
        super().on_done(idx)
        succ = self.ep_success[idx].tolist()
        for i, s in zip(idx.tolist(), succ):
            self.episodes[i] += 1
            self.successes[i] += int(s)
            self.advance(i)


def build_scene(env):
    scene = tp.Scene()
    scene.background = tp.Color(0x1a2029)
    scene.add(tp.HemisphereLight(0xbfd8ff, 0x223044, 1.0))
    sun = tp.DirectionalLight(0xffffff, 2.0)
    sun.position.set(3.0, -3.0, 5.0)
    scene.add(sun)
    sim = env.sim
    scene.add(sim.ground)
    for t in sim.tables:
        scene.add(t)
    for meshes in sim.arm_meshes:
        for m in meshes:
            scene.add(m)
    for c in sim.cube_meshes:
        c.material.color = CUBE_IDLE
        scene.add(c)
    return scene


def sync_meshes(env):
    """GPU poses -> the visual meshes (link poses are at the collider centres, which is where
    the loader parents each link's visual)."""
    sim = env.sim
    lp = sim.link_pose.cpu().numpy()
    cp = sim.cube_pose.cpu().numpy()
    held = (env.hold > 0).cpu().numpy()
    for i, meshes in enumerate(sim.arm_meshes):
        for j, m in enumerate(meshes):
            q, p = lp[i, j, 0:4], lp[i, j, 4:7]
            m.position.set(float(p[0]), float(p[1]), float(p[2]))
            m.quaternion.set(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        c = sim.cube_meshes[i]
        c.position.set(float(cp[i, 4]), float(cp[i, 5]), float(cp[i, 6]))
        c.quaternion.set(float(cp[i, 0]), float(cp[i, 1]), float(cp[i, 2]), float(cp[i, 3]))
        c.material.color = CUBE_HELD if held[i] else CUBE_IDLE


def make_camera(n, aspect):
    """The interactive camera: all n arms in view."""
    cam = tp.PerspectiveCamera(40, aspect, 0.05, 100.0)
    cam.up.set(0, 0, 1)
    cx = (n - 1) * C.SPACING * 0.5 + C.TABLE_CX * 0.6
    if n == 1:
        cam.position.set(1.45, -1.2, 1.0)
    else:
        cam.position.set(cx, -1.2 - 1.15 * n, 1.35)
    target = (cx, 0.0, 0.55)
    cam.look_at(*target)
    return cam, target


def arm_camera(i, aspect):
    """A close camera on arm i, for the --shot tiles."""
    cam = tp.PerspectiveCamera(40, aspect, 0.05, 50.0)
    cam.up.set(0, 0, 1)
    x0 = i * C.SPACING
    cam.position.set(x0 + 1.35, -1.05, 0.95)
    cam.look_at(x0 + 0.42, 0.0, 0.55)
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(_HERE, "kuka_grasp.pt"))
    ap.add_argument("--layout", choices=tuple(LAYOUTS), default="families")
    ap.add_argument("--seed", type=int, default=0, help="order in which each arm visits its cells")
    ap.add_argument("--shot", metavar="PNG", help="headless: write one episode as a strip of frames")
    ap.add_argument("--frames", type=int, default=5, help="frames in the --shot strip")
    ap.add_argument("--span", type=float, default=0.8,
                    help="seconds the --shot frames spread over from the episode start (a success "
                         "ends the episode, so a longer span shows the arm's next cell)")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-quit after this long (self-test)")
    args = ap.parse_args()
    if not os.path.exists(args.policy):
        raise SystemExit(f"no policy at {args.policy}; train one with train_kuka_grasp.py")

    ac, norm, meta = load_policy(args.policy, device="cuda")
    families = LAYOUTS[args.layout]
    env = ViewerEnv(families, viewer_pool("cuda"), seed=args.seed)
    action = meta.get("action", "absolute")
    env.delta = action == "delta"
    env.vel_terminal.fill_(action == "delta")
    st = {"obs": env.reset(), "accum": 0.0, "key_r": False, "t": 0.0}

    @torch.no_grad()
    def control_step():
        o = st["obs"] if norm is None else norm.norm(st["obs"])
        st["obs"] = env.step(ac.act_mean(o))[0]

    scene = build_scene(env)
    print(f"[play] {os.path.basename(args.policy)} ({meta.get('sampler', '?')}, {action} actions, "
          f"iteration {meta.get('iter', '?')}); arms: {', '.join(FAMILIES[f] for f in families)}")

    if args.shot:
        from threepp import capture
        w, h = 480, 400
        renderer = capture.make_renderer("gl", (w, h))
        cams = [arm_camera(i, w / h) for i in range(len(families))]
        last = min(env.max_steps, int(round(args.span / env.dt)))
        at = np.linspace(0, last - 1, args.frames).round().astype(int)
        rows = [[] for _ in families]           # one row per arm, time left to right
        for k in range(last):
            control_step()
            if k in at:
                sync_meshes(env)
                for i, cam in enumerate(cams):
                    rows[i].append(capture.render(scene, cam, backend=renderer).image)
        strip = np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)
        capture.write_png(args.shot, strip)
        print(f"[play] wrote {args.shot} ({strip.shape[1]}x{strip.shape[0]}): frames at "
              f"{', '.join(f'{(k + 1) * env.dt:.2f} s' for k in at)}; success per arm "
              f"{env.successes} of {env.episodes} finished episodes")
        return

    canvas = tp.Canvas("threepp - KUKA friction grasp", width=1280, height=760, antialiasing=4)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
    camera, target = make_camera(len(families), canvas.aspect())
    controls = tp.OrbitControls(camera, canvas)
    controls.target.set(*target)
    controls.update()

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)
    canvas.on_window_resize(on_resize)
    ui = tp.ImguiContext(canvas, renderer) if tp.HAS_IMGUI else None
    clock = tp.Clock()

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(380, 0)
        tp.imgui.begin("KUKA friction grasp")
        tp.imgui.text(f"policy: {os.path.basename(args.policy)} ({meta.get('sampler', '?')})")
        tp.imgui.text("success: cube 0.12 m up, between the fingers, for 0.5 s")
        for i, f in enumerate(families):
            tp.imgui.text(f"{FAMILIES[f]:>13}: {env.successes[i]} of {env.episodes[i]} episodes")
        tp.imgui.text("R = new cells    Esc = quit")
        tp.imgui.end()

    def animate():
        if canvas.is_key_down("ESCAPE") or (args.seconds and st["t"] > args.seconds):
            canvas.close()
            return
        r = canvas.is_key_down("R")
        if r and not st["key_r"]:
            for i in range(env.K):
                env.advance(i)
            st["obs"] = env.reset()
        st["key_r"] = r
        d = min(clock.get_delta(), 0.1)
        st["t"] += d
        st["accum"] += d
        n = 0
        while st["accum"] >= env.dt and n < 4:
            st["accum"] -= env.dt
            control_step()
            n += 1
        sync_meshes(env)
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
