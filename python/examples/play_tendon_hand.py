"""Run a trained hold policy on the CPU hand, with its REAL cables.

This is the deploy path and the sim-to-sim check in one. Training runs on the batched torch
cable because PhysX rejects addForce under the direct-GPU API; the film runs on the C++
`TendonCable`, which is the reference implementation of the same law. Nothing about the policy
changes between the two -- the same 25 tensions go in -- so if the port is right the posture
here is the posture the policy was trained into, and if it is not, this is where it shows.

The observation is rebuilt from the CPU state in the same order `TendonHandEnv.observe` builds
it, and every constant it needs (T_MAX, the filter time constant, the palm target, the pull
schedule) is read from the checkpoint's meta rather than re-declared, so the two cannot drift.

Run:  python play_tendon_hand.py tendon_hand_hold.pt --view
      python play_tendon_hand.py tendon_hand_hold.pt --object sphere --seconds 6
      python play_tendon_hand.py tendon_hand_hold.pt --shots out/ --shot-times 1.0,3.5,6.0

--shots renders headless stills out of the SAME run that prints the hold numbers, on the same
GL headless path tendon_hand.py's visual() uses. It is deliberately not a separate replay:
a still taken from a second, differently-seeded run would be a picture of a different grasp
than the one the numbers describe.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import threepp as tp
from tendon_hand import DT, Forearm, Hand, RopeView, VIEWS, _scene
from tendon_hand_env import OBJ_TYPES, object_mesh
from threepp.rl import load_policy


# ---- the still/film camera and scene -------------------------------------------------------
#
# SCREEN UP IS WORLD -Y, and that is not a style choice. Gravity in this task is +Y: the hand
# is fixed-base with its pad along -Y, so "palm up" is achieved by turning gravity round rather
# than by rotating the hand. World +Y is therefore DOWN. Leave the camera's default up of +Y
# and the render comes out upside down -- the object falls upward out of frame and the floor is
# in the sky. Every framing constant below is written against that.
SHOT_UP = (0.0, -1.0, 0.0)
SHOT_TARGET = (0.062, -0.022, 0.002)    # the palm, with the object sitting on it
SHOT_DIR = (0.45, -0.78, -0.46)         # target -> camera: distal, above the palm, thumb side
SHOT_DIST = 0.29
SHOT_FOV = 38.0
GROUND_Y = 0.075                        # the floor, BELOW the hand in the gravity sense
CABLE_XRAY = 0.85                       # the cables read as being inside the fingers


def shot_scene(width, height, headless=True):
    """A lit scene with a floor, for a picture someone who has not seen this hand can read.

    tendon_hand.py's own `_scene` is a dark studio void with a hard key: right for a cutaway
    poster of the mechanism, wrong here, where the subject is a grasp and the viewer needs a
    ground to judge the hand against and enough fill to see finger volume.
    """
    canvas = tp.Canvas("tendon hand", width=width, height=height, antialiasing=4,
                       headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    scene = tp.Scene()
    scene.background = tp.Background(0xDFE3E8)

    cam = tp.PerspectiveCamera(SHOT_FOV, width / height, 0.01, 10)
    cam.up.set(*SHOT_UP)
    d = np.array(SHOT_DIR, dtype=float)
    d /= np.linalg.norm(d)
    cam.position.set(*(np.array(SHOT_TARGET) + d * SHOT_DIST))
    cam.look_at(*SHOT_TARGET)

    scene.add(tp.HemisphereLight(0xFFFFFF, 0x9AA4B0, 0.75))
    key = tp.DirectionalLight(0xFFF6EC, 2.1)
    # Far out along the same direction, and that distance is the whole reason a shadow appears
    # at all. A directional light's shadow camera has a default near plane of 0.5, and this
    # scene is 0.2 m across: with the light at its natural 0.47 m the entire hand sat in front
    # of the near plane, the shadow map was empty, and the hand floated on a flat grey field.
    key.position.set(0.40, -1.80, 0.55)          # above the palm: it throws onto the floor
    key.cast_shadow = True
    # Wide enough for the SHADOW, not just the hand. At +-0.18 the hand fitted and its shadow
    # did not: the contact patch fell outside the ortho box and came back torn into fragments
    # in the corner of the frame, which reads as a rendering fault rather than as a shadow.
    key.set_shadow_frustum(-0.40, 0.40, 0.40, -0.40)
    key.set_shadow_bias(-0.0003)
    scene.add(key)
    # A volar fill from the camera side so the finger undersides are not black, and a low rim
    # from behind to separate the silhouette from the floor.
    fill = tp.DirectionalLight(0xE6EEFF, 1.0)
    fill.position.set(-0.10, -0.26, -0.30)
    scene.add(fill)
    rim = tp.DirectionalLight(0xFFE4CC, 0.8)
    rim.position.set(-0.30, 0.10, 0.26)
    scene.add(rim)

    floor_mat = tp.MeshStandardMaterial()
    floor_mat.color = tp.Color(0xA8AFB8)
    floor_mat.roughness = 0.95
    floor_mat.metalness = 0.0
    floor_mat.side = tp.Side.Double
    floor = tp.Mesh(tp.PlaneGeometry(2.0, 2.0), floor_mat)
    floor.rotate_x(math.pi / 2)                  # normal along -Y, i.e. facing the hand
    floor.position.y = GROUND_Y
    floor.receive_shadow = True
    scene.add(floor)
    return canvas, renderer, scene, cam


def dress(hand, obj_mesh):
    """Turn the cutaway hand into a readable one, without touching how it is built.

    Everything here is a material property on meshes that already exist, set after Hand has
    constructed them -- Hand, the cables and the env are untouched. Three changes:
      the SKIN goes to 0.85 with depth write ON. At 0.28 with depth write off, every capsule
        blends with every capsule behind it and a finger reads as a brown cloud rather than as
        three segments.
      the CABLES become an x-ray overlay: depth test off, drawn after the skin. Without this
        they are occluded in the places that matter and look like rods lying on top of the
        hand; with it they read as running inside the fingers, which is what they do.
      the OBJECT goes opaque and saturated, so it is obviously a held thing and not more skin.
    """
    for m in hand.meshes:
        # OPAQUE, not 0.85. Two reasons, both found by rendering: a transparent skin does not
        # land in the shadow map, so the hand floated with no contact shadow at all; and at any
        # alpha the capsules still tint each other where they overlap, which is exactly the
        # blob the cutaway look produces. The cables stay visible because they are an x-ray
        # overlay drawn with depth test off, so nothing is lost by closing the skin.
        m.material.transparent = False
        m.material.opacity = 1.0
        m.material.depth_write = True
        m.material.roughness = 0.65
        m.cast_shadow = True
        m.receive_shadow = True
    obj_mesh.material.transparent = False
    obj_mesh.material.opacity = 1.0
    obj_mesh.material.depth_write = True
    obj_mesh.material.roughness = 0.45
    obj_mesh.material.metalness = 0.05
    obj_mesh.cast_shadow = True


def xray(view):
    """The cable materials, after RopeView has made them. Shared per cable, so setting them
    here covers every segment the pool will ever create."""
    for m in view.mats.values():
        m.transparent = True
        m.opacity = CABLE_XRAY
        m.depth_test = False
        m.roughness = 0.4
        m.metalness = 0.0


CONTACT_TIMES = (0.2, 1.0, 2.5, 4.0, 5.5, 6.0)


def contact_sheet(film_dir, obj, hz, n_steps, cols=3, cell=(640, 360)):
    """Six frames of the episode in one image, so the MOTION is visible in a still.

    A 3x2 grid rather than one row of six: at six across, each cell is 426 px wide and the
    cables stop being separable, which is the whole thing the sheet is meant to show.
    """
    from PIL import Image, ImageDraw
    rows = (len(CONTACT_TIMES) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell[0], rows * cell[1]), (198, 204, 212))
    draw = ImageDraw.Draw(sheet)
    for k, t in enumerate(CONTACT_TIMES):
        fr = min(n_steps, max(1, int(round(t * hz))))
        fn = os.path.join(film_dir, f"frame_{fr:04d}.png")
        if not os.path.exists(fn):
            continue
        im = Image.open(fn).convert("RGB").resize(cell, Image.LANCZOS)
        x, y = (k % cols) * cell[0], (k // cols) * cell[1]
        sheet.paste(im, (x, y))
        draw.text((x + 12, y + 10), f"t = {t:.1f} s", fill=(26, 31, 40))
    out = os.path.join(film_dir, "contact.png")
    sheet.save(out)
    print(f"    wrote {out}  ({cols}x{rows} of {cell[0]}x{cell[1]})")
    return out


def build(meta, kind, size_a, size_b, seed):
    g = np.array(meta["gravity"], dtype=float)
    world = tp.PhysxWorld(gravity=tp.Vector3(*g), fixed_timestep=DT, max_substeps=1,
                          tgs_pcm=True)
    pad = world.create_material(1.2, 1.1, 0.0, friction_combine="min")
    hand = Hand(world, material=pad).finalize()
    hand.route()                                    # the REAL cables, this time
    rng = np.random.default_rng(seed)

    target = np.array(meta["palm_target"], dtype=float)
    g_hat = g / np.linalg.norm(g)
    mesh = object_mesh(kind, size_a, size_b)
    jitter = np.array([(rng.random() * 2 - 1) * 0.015, 0.0, (rng.random() * 2 - 1) * 0.015])
    mesh.position.set(*(target + jitter - g_hat * (0.03 + 0.03 * rng.random())))
    art = world.create_articulation(fixed_base=False)
    art.add_link(mesh, density=meta["obj_density"], material=pad)
    art.finalize()
    return world, hand, art, mesh, target, g_hat


def observe(meta, hand, art, mesh, target, tension, pull):
    """The same 80 numbers, in the same order, that TendonHandEnv.observe stacks."""
    q = hand.pose()
    qd = np.array([hand.links[n].joint_velocity for n in hand.dof_names])
    rv = art.root_velocity()
    pos = np.array([mesh.position.x, mesh.position.y, mesh.position.z])
    desc = meta["_desc"]
    return np.concatenate([q, qd * 0.1, tension / meta["t_max"], pos - target,
                           rv[0:3], rv[3:6] * 0.1, desc, pull / 10.0]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--object", default="sphere", choices=list(OBJ_TYPES))
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="window, with every cable drawn")
    ap.add_argument("--shots", default=None, metavar="DIR",
                    help="render headless PNG stills into DIR, no window")
    ap.add_argument("--shot-times", default="1.0,3.5,6.0",
                    help="seconds at which --shots writes a still")
    ap.add_argument("--film", default=None, metavar="DIR",
                    help="60 fps PNG sequence of the whole episode at 1280x720, plus a "
                         "contact sheet; encode it with the ffmpeg line in tendon_hand.md")
    ap.add_argument("--size", default="1280x800")
    a = ap.parse_args()

    ac, norm, meta = load_policy(a.checkpoint, "cpu")
    hz, dt = meta["control_hz"], 1.0 / meta["control_hz"]
    sub = meta["substeps"]
    alpha = dt / (meta["tau_filter"] + dt)
    rng = np.random.default_rng(a.seed)
    if a.object == "sphere":
        sa, sb = rng.uniform(0.018, 0.025), 0.0
    elif a.object == "capsule":
        sa, sb = rng.uniform(0.015, 0.020), rng.uniform(0.060, 0.080)
    else:
        sa, sb = rng.uniform(0.030, 0.045), 0.0
    meta["_desc"] = np.array([sa * 20.0, sb * 20.0, OBJ_TYPES.index(a.object) / 2.0])

    world, hand, art, mesh, target, g_hat = build(meta, a.object, sa, sb, a.seed)
    # The cable order the 25 actions are in. The checkpoint carries it; the hand's own routing
    # is the same list by construction, so it is both the fallback for an older checkpoint and
    # a check that the two have not drifted.
    names = list(hand.routing)
    if "cable_names" in meta and list(meta["cable_names"]) != names:
        raise RuntimeError("the checkpoint's cable order is not this hand's:\n  "
                           f"checkpoint {list(meta['cable_names'])}\n  hand {names}")
    tension = np.zeros(len(names))
    pull = np.zeros(3)
    pull_dir = np.zeros(3)
    body = art.link(0)
    shot_t = [float(s) for s in a.shot_times.split(",") if s.strip()] if a.shots else []
    # Run at least long enough to reach the last still asked for, so --shots 8.0 does not
    # silently write nothing.
    n_steps = int(round(max(a.seconds, max(shot_t, default=0.0)) * hz))
    shot_frame = {int(round(t * hz)): t for t in shot_t}
    every = max(1, int(round(meta["pull_resample_s"] * hz)))

    canvas = renderer = scene = cam = view = arm = None
    arrow = hud = None
    render_w, render_h = (int(v) for v in a.size.split("x"))
    if a.film:
        render_w, render_h = 1280, 720
    if a.view or a.shots or a.film:
        if a.view:
            canvas, renderer, scene, cam = _scene(render_w, render_h, False)
        else:
            canvas, renderer, scene, cam = shot_scene(render_w, render_h)
        for m in hand.meshes:
            scene.add(m)
        scene.add(mesh)
        # The Forearm belongs in the window and NOT in a still. It extends every cable
        # backwards to a motor bank 130 mm proximal, and at this framing those 25 straight rods
        # cross the entire frame and sit on top of the palm -- measured on the first render,
        # they were the most prominent thing in the picture and the grasp was behind them. The
        # extension is decoration either way: the physics cable begins at a via point on the
        # palm, which is the fixed root.
        arm = Forearm(scene, hand) if a.view else None
        view = RopeView(scene, hand)
        if a.view:
            pos, tgt = VIEWS["3q"]
            cam.position.set(*pos)
            cam.look_at(*tgt)
            controls = tp.OrbitControls(cam, canvas)
            controls.target = tp.Vector3(*tgt)
            controls.enable_damping = True
        else:
            dress(hand, mesh)
            xray(view)
            # The pull is the whole point of the second half of the episode and it is invisible
            # -- an 8 N force on a 30 g object is a picture of nothing. 1 cm per newton makes it
            # the same order of size as the hand, so a still shows what is being resisted.
            arrow = tp.ArrowHelper(tp.Vector3(1, 0, 0), tp.Vector3(*target), 0.001,
                                   tp.Color(0xD81028), 0.022, 0.016)
            scene.add(arrow)
            hud = tp.TextSprite(tp.FontLoader().default_font())
            hud.set_color(tp.Color(0x1A1F28))
            hud.set_world_scale(0.0075)
            hud.position.set(0.02, -0.105, -0.01)
            scene.add(hud)
    for d in (a.shots, a.film):
        if d:
            os.makedirs(d, exist_ok=True)

    state = {"i": 0}

    def control():
        i = state["i"]
        nonlocal pull_dir, pull, tension
        if i % every == 0:
            d = rng.normal(size=3)
            pull_dir = d / max(np.linalg.norm(d), 1e-6)
        t = i * dt
        k = min(1.0, max(0.0, (t - meta["pull_start_s"])
                         / (meta["episode_s"] - meta["pull_start_s"])))
        pull = pull_dir * (meta["pull_max"] * k)

        obs = torch.from_numpy(observe(meta, hand, art, mesh, target, tension, pull))[None]
        if norm is not None:
            obs = norm.norm(obs)
        act = ac.act_mean(obs)[0].clamp(-1.0, 1.0).numpy()
        cmd = meta["t_max"] * 0.5 * (act + 1.0)
        tension = (1.0 - alpha) * tension + alpha * cmd
        for n, T in zip(names, tension):
            hand.cables[n].set_tension(float(T))
        for _ in range(sub):
            body.add_force(tp.Vector3(*pull))        # re-applied: PhysX clears it every step
            world.step(dt / sub)
        state["i"] = i + 1
        return np.linalg.norm(np.array([mesh.position.x, mesh.position.y,
                                        mesh.position.z]) - target)

    if a.view:
        def loop():
            d = control()
            view.update(arm.update())
            controls.update()
            renderer.render(scene, cam)
            if state["i"] % 30 == 0:
                print(f"  t {state['i']*dt:4.2f} s   |d| {d*1000:6.1f} mm   "
                      f"pull {np.linalg.norm(pull):4.2f} N   "
                      f"mean tension {tension.mean():5.2f} N")
        canvas.animate(loop)
        return

    print(f"policy {a.checkpoint} on the CPU hand with its real cables: {a.object} "
          f"r={sa*1000:.1f} mm, {a.seconds:.1f} s at {hz} Hz")
    worst = 0.0
    for i in range(n_steps):
        d = control()
        worst = max(worst, d)
        if (i + 1) % 30 == 0:
            print(f"  t {(i+1)*dt:4.2f} s   |d| {d*1000:6.1f} mm   "
                  f"pull {np.linalg.norm(pull):4.2f} N   "
                  f"mean tension {tension.mean():5.2f} N")
        if scene is not None:
            # Rebuilt from the LIVE link poses every step, exactly as the CPU demo's rope view
            # does, so what is in the still is what is pulling -- not a cable drawn from the
            # routing and hoping the two agree.
            view.update(arm.update() if arm is not None else None)
            t = (i + 1) * dt
            if arrow is not None:
                p = np.array([mesh.position.x, mesh.position.y, mesh.position.z])
                f = float(np.linalg.norm(pull))
                arrow.position.set(*p)
                if f > 1e-6:
                    arrow.set_direction(tp.Vector3(*(pull / f)))
                arrow.set_length(max(1e-4, 0.01 * f), 0.022, 0.016)   # 1 cm per newton
                arrow.visible = f > 1e-6
                hud.set_text(f"t {t:4.2f} s    pull {f:4.1f} N    "
                             f"{'held' if d < meta['drop_dist'] else 'DROPPED'}")
            if (i + 1) in shot_frame:
                renderer.render(scene, cam)
                fn = os.path.join(a.shots, f"{a.object}_t{shot_frame[i+1]:.1f}.png")
                renderer.save_frame(fn)
                print(f"    wrote {fn}   |d| {d*1000:.1f} mm   "
                      f"pull {np.linalg.norm(pull):.2f} N")
            if a.film:
                renderer.render(scene, cam)
                renderer.save_frame(os.path.join(a.film, f"frame_{i+1:04d}.png"))
    if a.film:
        contact_sheet(a.film, a.object, hz, n_steps)
    held = worst < meta["drop_dist"]
    print(f"  worst offset {worst*1000:.1f} mm against a {meta['drop_dist']*1000:.0f} mm "
          f"limit -> {'HELD' if held else 'DROPPED'}")


if __name__ == "__main__":
    main()
