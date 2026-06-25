"""Watch the trained vision turret defend a base — third-person, dusk air-defense scene.

    python play_turret.py                 # live window
    python play_turret.py --shot 18       # headless: save a montage to turret_shot.png

The turret model swivels and fires laser bolts to shoot down incoming colliders, all driven by the
trained pixel policy. Two renders per frame: a CLEAN one (world hidden, threats on black, downsized)
feeds the policy exactly what it trained on; the DISPLAY one is the full lit scene with the turret,
ground grid, structures, beams and explosions. Reuses turret_env's constants so it can't drift.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from threepp.rl import load_policy
from turret_env import (COOLDOWN, DEFENSE_R, FLIGHT_HI, FLIGHT_LO, FOV, FRAME_STACK, GRAVITY, H,
                        HIT_CONE, LAUNCH_HI, LAUNCH_LO, N_PROJ, PITCH_HI, PITCH_LO, PITCH_RATE,
                        SIZE_HI, SIZE_LO, SPAWN_CONE_P, SPAWN_CONE_Y, TARGET_SCATTER, W, YAW_RATE,
                        _SHAPES, _THREAT_COLOR, aim_dir)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(_HERE, "turret_policy.pt"))
ap.add_argument("--shot", type=int, default=0)
args = ap.parse_args()
if not os.path.exists(args.model):
    print("No policy — run train_turret.py first."); sys.exit(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ac, _, meta = load_policy(args.model, device=dev)
DT = float(meta["dt"])
DISP_W, DISP_H = 960, 600
SKY = 0x14213d            # dusk sky (display clear color); the policy's feed is cleared BLACK
HEAD_Y, BARREL_L = 1.7, 2.5
YAW_CLAMP = 1.0          # keep the engagement in a frontal arc the fixed camera frames

# ---- single-turret state (reusing the env's exact spawn/step math) ----
g = torch.Generator(device=dev).manual_seed(1)
yaw = torch.zeros(1, device=dev); pitch = torch.full((1,), 0.4, device=dev)
ppos = torch.zeros(1, N_PROJ, 3, device=dev); pvel = torch.zeros(1, N_PROJ, 3, device=dev)
psize = torch.ones(1, N_PROJ, device=dev); cooldown = torch.zeros(1, dtype=torch.long, device=dev)
frames = torch.zeros(1, FRAME_STACK, 3, H, W, dtype=torch.uint8, device=dev)


def rnd(n, lo, hi):
    return torch.rand(n, device=dev, generator=g) * (hi - lo) + lo


def spawn(slots):
    if not slots:
        return
    sl = torch.tensor(slots, device=dev); n = len(slots)
    syaw = (yaw + rnd(n, -SPAWN_CONE_Y, SPAWN_CONE_Y)).clamp(-YAW_CLAMP, YAW_CLAMP)
    spit = (pitch + rnd(n, -SPAWN_CONE_P, SPAWN_CONE_P)).clamp(0.08, 1.2)
    p = rnd(n, LAUNCH_LO, LAUNCH_HI)[:, None] * aim_dir(syaw, spit)      # hurled from far
    target = torch.stack([rnd(n, -1, 1), rnd(n, 0.0, 0.6), rnd(n, -1, 1)], -1) * TARGET_SCATTER
    t = rnd(n, FLIGHT_LO, FLIGHT_HI)[:, None]
    v = (target - p) / t; v[:, 1] = v[:, 1] + 0.5 * GRAVITY * t[:, 0]    # ballistic, lands on base
    ppos[0, sl] = p
    pvel[0, sl] = v
    psize[0, sl] = rnd(n, SIZE_LO, SIZE_HI)


# ---- scene ----
canvas = tp.Canvas("turret — air defense", width=DISP_W, height=DISP_H, headless=bool(args.shot))
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True
renderer.tone_mapping_exposure = 1.3
scene = tp.Scene()


def std(color, emissive=0x000000, rough=0.7, metal=0.2):
    m = tp.MeshStandardMaterial(); m.color = color; m.emissive = emissive
    m.roughness = rough; m.metalness = metal
    return m


# lights: cool dusk hemisphere + a warm low sun (long shadows) — bright enough that the threats
# look the same in the clean feed as in training.
scene.add(tp.HemisphereLight(0xaac4ff, 0x222a3a, 1.35))
sun = tp.DirectionalLight(0xfff4ea, 2.0); sun.position.set(-6, 7, 5); sun.cast_shadow = True
scene.add(sun)
key = tp.PointLight(0x66ddff, 0.8); key.position.set(0, 3, -3); scene.add(key)

world = []   # everything hidden during the clean policy render


def add_world(mesh):
    scene.add(mesh); world.append(mesh); return mesh


# ground + tech grid
_FLAT = lambda mesh: mesh.quaternion.set_from_axis_angle(tp.Vector3(1, 0, 0), -math.pi / 2)
ground = add_world(tp.Mesh(tp.PlaneGeometry(120, 120), std(0x2b3550, rough=1.0, metal=0.0)))
_FLAT(ground); ground.receive_shadow = True
gm = tp.MeshBasicMaterial(); gm.color = 0x2f7fff; gm.wireframe = True
grid = add_world(tp.Mesh(tp.PlaneGeometry(120, 120), gm))
_FLAT(grid); grid.position.y = 0.02
# scattered structures (a base to defend)
_rg = np.random.default_rng(7)
for _ in range(22):
    a = _rg.uniform(0, 2 * math.pi); r = _rg.uniform(9, 42)
    hh = _rg.uniform(1.0, 5.5); bw = _rg.uniform(1.2, 3.0)
    b = add_world(tp.Mesh(tp.BoxGeometry(bw, hh, bw), std(0x141a26, 0x0a0f1a, rough=0.8)))
    b.position.set(math.cos(a) * r, hh / 2, math.sin(a) * r); b.cast_shadow = True; b.receive_shadow = True
    wm = tp.MeshBasicMaterial(); wm.color = 0x335577 if _rg.random() < 0.5 else 0x886622
    win = add_world(tp.Mesh(tp.BoxGeometry(bw * 1.01, hh * 0.7, bw * 1.01), wm))
    win.position.set(math.cos(a) * r, hh * 0.55, math.sin(a) * r)

# turret: base + head + barrel (positioned analytically each frame from yaw/pitch)
base = add_world(tp.Mesh(tp.CylinderGeometry(1.25, 1.6, 1.1), std(0x2a313d, rough=0.45, metal=0.8)))
base.position.set(0, 0.55, 0); base.cast_shadow = True
collar = add_world(tp.Mesh(tp.CylinderGeometry(0.85, 1.05, 0.8), std(0x222934, 0x07222e, rough=0.45, metal=0.8)))
collar.position.set(0, 1.25, 0); collar.cast_shadow = True
head = add_world(tp.Mesh(tp.BoxGeometry(1.4, 1.1, 1.7), std(0x2c3543, 0x06283a, rough=0.35, metal=0.85)))
head.cast_shadow = True
barrel = add_world(tp.Mesh(tp.CylinderGeometry(0.18, 0.22, BARREL_L), std(0x1a1f28, 0x0a3550, rough=0.35, metal=0.9)))
muzzle = add_world(tp.Mesh(tp.CylinderGeometry(0.26, 0.26, 0.4), std(0x0e1014, 0x33ddff)))

# threats (emissive glow, varied shapes)
_geos = {"sphere": lambda s: tp.SphereGeometry(s), "box": lambda s: tp.BoxGeometry(2 * s, 2 * s, 2 * s),
         "cone": lambda s: tp.ConeGeometry(s, 2.4 * s), "cyl": lambda s: tp.CylinderGeometry(s, s, 2.2 * s)}
pmesh = []
for j in range(N_PROJ):
    m = tp.Mesh(_geos[_SHAPES[j % len(_SHAPES)]](1.0), std(_THREAT_COLOR, _THREAT_COLOR, rough=0.4))
    scene.add(m); pmesh.append(m)         # NOT in `world` -> visible in the clean policy feed

# beam + flash + debris pools (in `world` -> only in the display render)
NB = 6
beams, flashes, blife, flife = [], [], [0] * NB, [0] * NB
for _ in range(NB):
    b = add_world(tp.Mesh(tp.CylinderGeometry(0.06, 0.06, 1.0), std(0x66ffff, 0x66ffff))); b.visible = False; beams.append(b)
    fm = std(0xffb347, 0xff8800); fm.transparent = True; fm.opacity = 0.62
    fl = add_world(tp.Mesh(tp.SphereGeometry(1.0), fm)); fl.visible = False; flashes.append(fl)
NF = 24
deb, dvel, dlife = [], np.zeros((NF, 3), np.float32), [0] * NF
for _ in range(NF):
    d = add_world(tp.Mesh(tp.BoxGeometry(0.12, 0.12, 0.12), std(0xff8844, 0xff6622))); d.visible = False; deb.append(d)
_bn = _dn = 0

cam = tp.PerspectiveCamera(55, DISP_W / DISP_H, 0.1, 220)
cam.position.set(-25.0, 12, -60.0); cam.look_at(0.0, 5.5, 12.0)   # low 3/4 behind turret, looking UP the arc


def orient_y_to(mesh, d):
    """Point a +Y-aligned mesh along unit direction d (np[3])."""
    ax = np.cross([0, 1, 0], d); a = float(np.linalg.norm(ax))
    if a > 1e-6:
        mesh.quaternion.set_from_axis_angle(tp.Vector3(*(ax / a).tolist()),
                                            math.acos(max(-1.0, min(1.0, float(d[1])))))


def pose_turret():
    d = aim_dir(yaw, pitch)[0].cpu().numpy()
    head.position.set(0, HEAD_Y, 0)
    head.quaternion.set_from_axis_angle(tp.Vector3(0, 1, 0), float(yaw))
    bc = np.array([0, HEAD_Y, 0]) + d * (BARREL_L * 0.5)
    barrel.position.set(*bc.tolist()); orient_y_to(barrel, d)
    mz = np.array([0, HEAD_Y, 0]) + d * BARREL_L
    muzzle.position.set(*mz.tolist()); orient_y_to(muzzle, d)
    return np.array([0, HEAD_Y, 0]) + d * BARREL_L     # muzzle world pos


def fx_hit(muzzle_w, target_w):
    global _bn, _dn
    i = _bn % NB; _bn += 1
    seg = target_w - muzzle_w; L = float(np.linalg.norm(seg)) or 1e-3
    bm = beams[i]; bm.position.set(*(muzzle_w + seg * 0.5).tolist()); bm.scale.set(1, L, 1)
    orient_y_to(bm, seg / L); bm.visible = True; blife[i] = 3
    fl = flashes[i]; fl.position.set(*target_w.tolist()); fl.scale.set(0.3, 0.3, 0.3); fl.visible = True; flife[i] = 6
    for _ in range(6):
        k = _dn % NF; _dn += 1
        deb[k].position.set(*target_w.tolist()); deb[k].visible = True; dlife[k] = 8
        dvel[k] = (np.random.rand(3) - 0.5) * 9.0; dvel[k][1] = abs(dvel[k][1]) + 2.0


def age_fx():
    for i in range(NB):
        if blife[i] > 0:
            blife[i] -= 1; beams[i].visible = blife[i] > 0
        if flife[i] > 0:
            flife[i] -= 1; s = 0.3 + (6 - flife[i]) * 0.12; flashes[i].scale.set(s, s, s); flashes[i].visible = flife[i] > 0
    for k in range(NF):
        if dlife[k] > 0:
            dlife[k] -= 1
            p = np.array([deb[k].position.x, deb[k].position.y, deb[k].position.z]) + dvel[k] * DT
            dvel[k][1] -= 14 * DT; deb[k].position.set(*p.tolist()); deb[k].visible = dlife[k] > 0


def set_world(v):
    for m in world:
        if v and m in beams + flashes + deb:
            continue                       # FX manage their own visibility
        m.visible = v


@torch.no_grad()
def policy_obs():
    set_world(False)                       # hide everything but the threats
    renderer.set_clear_color(0x000000)
    renderer.render(scene, fp_cam())
    px = torch.from_numpy(renderer.read_pixels(True)).to(dev).permute(2, 0, 1).float()[None]
    small = F.interpolate(px, size=(H, W), mode="area").to(torch.uint8)
    global frames
    frames = torch.roll(frames, -1, dims=1); frames[:, -1] = small
    return frames.reshape(1, 3 * FRAME_STACK, H, W)


_fp = tp.PerspectiveCamera(FOV, W / H, 0.1, 70.0)   # past the launch range so far throws render


def fp_cam():
    d = aim_dir(yaw, pitch)[0].cpu().numpy()
    _fp.position.set(0, 1.0, 0); _fp.look_at(float(d[0]), 1.0 + float(d[1]), float(d[2]))
    return _fp


def update_threats():
    pos = ppos[0].cpu().numpy(); sc = psize[0].cpu().numpy()
    for j, m in enumerate(pmesh):
        m.position.set(float(pos[j, 0]), float(pos[j, 1]), float(pos[j, 2]))
        m.scale.set(float(sc[j]), float(sc[j]), float(sc[j]))


@torch.no_grad()
def tick():
    global yaw, pitch, ppos, cooldown
    update_threats()
    obs = policy_obs()
    a = ac.act_mean(obs).clamp(-1, 1)[0]
    yaw = (yaw + a[0] * YAW_RATE).clamp(-YAW_CLAMP, YAW_CLAMP)
    pitch = (pitch + a[1] * PITCH_RATE).clamp(PITCH_LO, PITCH_HI)
    pvel[:, :, 1] = pvel[:, :, 1] - GRAVITY * DT   # ballistic arc
    ppos = ppos + pvel * DT
    cooldown = (cooldown - 1).clamp_min(0)
    mz = pose_turret()

    d = aim_dir(yaw, pitch); rel = F.normalize(ppos, dim=-1)
    ang = torch.arccos((d[:, None, :] * rel).sum(-1).clamp(-1, 1))[0]
    na, ni = ang.min(0)
    resp = []
    if (a[2] > 0) and int(cooldown) == 0:
        cooldown[0] = COOLDOWN
        if float(na) < HIT_CONE:
            fx_hit(mz, ppos[0, int(ni)].cpu().numpy()); resp.append(int(ni))
    for j in range(N_PROJ):
        if float(ppos[0, j].norm()) < DEFENSE_R and j not in resp:
            resp.append(j)
    spawn(resp)

    age_fx(); set_world(True); update_threats()
    renderer.set_clear_color(SKY)
    renderer.render(scene, cam)


spawn(list(range(N_PROJ))); pose_turret(); update_threats()

if args.shot:
    cols = []
    for k in range(args.shot * COOLDOWN):
        tick()
        if k % COOLDOWN == 0:
            cols.append(renderer.read_pixels(True))
    rows = [np.concatenate(cols[r:r + 6], axis=1) for r in range(0, min(len(cols), 12), 6)]
    strip = np.concatenate(rows, axis=0)
    out = os.path.join(_HERE, "turret_shot.png")
    import imageio.v2 as imageio
    imageio.imwrite(out, strip); print("saved montage ->", out, strip.shape)
else:
    def on_resize(w, h):
        cam.aspect = w / max(h, 1)
        cam.update_projection_matrix()
        renderer.set_size(w, h)

    canvas.on_window_resize(on_resize)

    controls = tp.OrbitControls(cam, canvas)

    print(__doc__); canvas.animate(tick)
