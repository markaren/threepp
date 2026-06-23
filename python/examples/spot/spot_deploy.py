"""Deploy an Isaac Lab locomotion policy on Boston Dynamics Spot in threepp.

Loads a Spot URDF + a TorchScript policy exported from Isaac Lab's velocity
locomotion task, builds Spot as a PhysX reduced-coordinate articulation, and runs
the policy closed-loop. Drive it with the arrow keys or the numpad; the camera
follows. Needs a PhysX-enabled threepp build (`tp.HAS_PHYSX`) + torch.

    forward   UP / NUM 8      backward   DOWN / NUM 2
    strafe L  LEFT / NUM 4    strafe R   RIGHT / NUM 6
    turn L    N / NUM 7       turn R     M / NUM 9

    python spot_deploy.py                  # interactive — assets auto-download on first run
    python spot_deploy.py --shot out.png

The Spot policy (+ params + URDF) download once to ~/.cache/threepp/spot; pass
--assets <folder> to use your own copy (it just needs `spot_policy.pt`). The
policy is the Isaac Lab Spot velocity task (48-d obs, 12 joint-position actions,
action_scale 0.2, PD 60/1.5, the standing default pose below).

Sim-to-sim note: the URDF carries no inertials or collision, so masses are
approximated and each link gets a Box/Capsule collider; the knee's remotized
actuator is treated as a plain PD. It still transfers because threepp and Isaac
Lab both run PhysX 5 — the obs/action contract just has to be exact.
"""
import argparse
import math
import os
import pathlib
import shutil
import struct
import sys
import tempfile
import urllib.request
import zipfile
import zlib

import numpy as np
import torch

# --------------------------------------------------------------------------- #
#  On-demand assets (cached under ~/.cache/threepp/spot; downloaded once)
# --------------------------------------------------------------------------- #
_ISAAC = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
          "Assets/Isaac/5.1/Isaac/Samples/Policies/Spot_Policies/")
ASSET_URLS = {"spot_policy.pt": _ISAAC + "spot_policy.pt",
              "spot_env.yaml": _ISAAC + "spot_env.yaml"}
URDF_ZIP_URL = "https://raw.githubusercontent.com/boston-dynamics/spot-sdk/master/files/spot_base_urdf.zip"


def _download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  downloading {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "threepp-spot-demo"})
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)
    return dest


def fetch_assets():
    """Download the Spot policy (+ params + URDF) on demand; return the cache folder."""
    cache = pathlib.Path.home() / ".cache" / "threepp" / "spot"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"[spot] assets dir: {cache}")
    if any(not (cache / f).exists() for f in ("spot_policy.pt", "spot_env.yaml", "model.urdf")):
        print("[spot] some assets missing — downloading (one-time; cached for later runs)")
    _download(ASSET_URLS["spot_policy.pt"], cache / "spot_policy.pt")    # required by the demo
    for name in ("spot_env.yaml",):                                      # for reference / the importer
        try:
            _download(ASSET_URLS[name], cache / name)
        except Exception as e:                                          # noqa: BLE001
            print(f"  (skipped {name}: {e})")
    if not (cache / "model.urdf").exists():                             # URDF zip -> extract + flatten
        try:
            zp = _download(URDF_ZIP_URL, cache / "spot_base_urdf.zip")
            with zipfile.ZipFile(zp) as z:
                z.extractall(cache)
            found = next(cache.rglob("model.urdf"), None)
            if found and found.parent != cache:
                for item in list(found.parent.iterdir()):
                    shutil.move(str(item), str(cache / item.name))
        except Exception as e:                                          # noqa: BLE001
            print(f"  (skipped URDF: {e})")
    return str(cache)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import threepp as tp

# --------------------------------------------------------------------------- #
#  Isaac Lab contract (Spot velocity task)
# --------------------------------------------------------------------------- #
LEGS = ["fl", "fr", "hl", "hr"]
SIGN = {"fl": (+1, +1), "fr": (+1, -1), "hl": (-1, +1), "hr": (-1, -1)}  # (front, left)
DEFAULT = {}
for L in LEGS:
    sx, sy = SIGN[L]
    DEFAULT[L + "_hx"] = 0.1 if sy > 0 else -0.1     # hips: left +0.1, right -0.1
    DEFAULT[L + "_hy"] = 0.9 if sx > 0 else 1.1       # thighs: front 0.9, hind 1.1
    DEFAULT[L + "_kn"] = -1.5                          # knees -1.5
ISAAC = [L + "_" + j for j in ("hx", "hy", "kn") for L in LEGS]   # policy joint order (type-grouped)
ADD = [L + "_" + j for L in LEGS for j in ("hx", "hy", "kn")]      # our add order (per-leg)
default_q = np.array([DEFAULT[n] for n in ISAAC], np.float32)
isaac_to_add = np.array([ADD.index(n) for n in ISAAC])
add_to_isaac = np.array([ISAAC.index(n) for n in ADD])
ACTION_SCALE = 0.2

# URDF kinematics + Isaac PD gains (stiffness, damping, effort)
HIP_X, HIP_Y, HY_Y = 0.29785, 0.055, 0.1108
KN = np.array([0.025, 0.0, -0.32]); FOOT = np.array([0.0, 0.0, -0.34])
LIM = {"hx": (-0.7854, 0.7854), "hy": (-0.8988, 2.295), "kn": (-2.7929, -0.247)}
GAINS = {"hx": (60., 1.5, 45.), "hy": (60., 1.5, 45.), "kn": (60., 1.5, 115.)}
MASS = {"base": 13.0, "hip": 1.2, "uleg": 2.0, "lleg": 0.55}
Z0 = 0.72   # build height (zero-config straight legs start just above ground)


def _capsule(length, radius, center, direction, color):
    m = tp.Mesh(tp.CapsuleGeometry(radius, length), tp.MeshStandardMaterial())
    m.material.color = color
    m.position.set(float(center[0]), float(center[1]), float(center[2]))
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    axis = np.cross([0, 1, 0], d); s = np.linalg.norm(axis); c = float(np.dot([0, 1, 0], d))
    if s < 1e-8:
        if c < 0: m.rotation.z = math.pi
    else:
        axis /= s; m.quaternion.set_from_axis_angle(tp.Vector3(*axis), math.atan2(s, c))
    vol = math.pi * radius * radius * length + 4 / 3 * math.pi * radius ** 3
    return m, vol


def build_spot(world, assets=None, base_xy=(0.0, 0.0), gains=None, foot_material=None):
    """Build Spot as a PhysX articulation; returns (articulation, render meshes).

    Physics uses tuned Box/Capsule colliders (the URDF has no collision/inertial). If `assets`
    is given, each link's URDF visual mesh (link_models/*.obj) is parented under its collider so
    Spot renders as the real robot while the primitives stay hidden but still drive the sim.
    `base_xy` offsets the whole robot in the ground plane (for GpuSim per-env grids).
    `gains` overrides the PD gains dict {hx,hy,kn: (stiffness, damping, max_force)} — default is the
    Isaac spec (stiffness 60), but 60 is too soft for our ~28 kg model (the legs sag ~12-34 deg under
    load and the body crouches), so the FROM-SCRATCH gait passes a stiffer set for a rigid stance.
    `foot_material` (from world.create_material) sets the shin/foot contact friction+restitution —
    grippy restitution-0 feet (vs the bouncy 0.5/0.5/0.2 default) for clean push-off; pass a per-env
    material for friction domain randomization."""
    gn = gains if gains is not None else GAINS
    ox, oy = float(base_xy[0]), float(base_xy[1])
    art = world.create_articulation(fixed_base=False, solver_position_iterations=12,
                                    disable_self_collision=True)
    bm = tp.Mesh(tp.BoxGeometry(0.70, 0.18, 0.19), tp.MeshStandardMaterial())
    bm.material.color = 0xffc24d
    bm.position.set(ox, oy, Z0)
    base = art.add_link(bm, parent=None, density=MASS["base"] / (0.70 * 0.18 * 0.19))
    if assets:
        _attach_obj(bm, (ox, oy, Z0), "base", 0xffc24d, assets)
    meshes = [bm]
    for L in LEGS:
        sx, sy = SIGN[L]
        Jhx = np.array([ox + sx * HIP_X, oy + sy * HIP_Y, Z0]); Jhy = Jhx + [0, sy * HY_Y, 0]
        Jkn = Jhy + KN; Jft = Jkn + FOOT
        hm, hv = _capsule(0.06, 0.045, (Jhx + Jhy) / 2, Jhy - Jhx, 0x303030)
        hip = art.add_link(hm, parent=base, density=MASS["hip"] / hv, axis=(1, 0, 0), anchor=tuple(Jhx),
                           lower=LIM["hx"][0], upper=LIM["hx"][1], stiffness=gn["hx"][0],
                           damping=gn["hx"][1], max_force=gn["hx"][2], drive_target=DEFAULT[L + "_hx"])
        um, uv = _capsule(0.30, 0.045, (Jhy + Jkn) / 2, Jkn - Jhy, 0xffc24d)
        uleg = art.add_link(um, parent=hip, density=MASS["uleg"] / uv, axis=(0, 1, 0), anchor=tuple(Jhy),
                            lower=LIM["hy"][0], upper=LIM["hy"][1], stiffness=gn["hy"][0],
                            damping=gn["hy"][1], max_force=gn["hy"][2], drive_target=DEFAULT[L + "_hy"])
        lm, lv = _capsule(0.30, 0.028, (Jkn + Jft) / 2, Jft - Jkn, 0x303030)
        art.add_link(lm, parent=uleg, density=MASS["lleg"] / lv, axis=(0, 1, 0), anchor=tuple(Jkn),
                     lower=LIM["kn"][0], upper=LIM["kn"][1], stiffness=gn["kn"][0],
                     damping=gn["kn"][1], max_force=gn["kn"][2], drive_target=DEFAULT[L + "_kn"],
                     material=foot_material)            # grippy restitution-0 (or per-env) foot contact
        if assets:
            _attach_obj(hm, Jhx, L + ".hip", 0x303030, assets)
            _attach_obj(um, Jhy, L + ".uleg", 0xffc24d, assets)
            _attach_obj(lm, Jkn, L + ".lleg", 0x303030, assets)
        meshes += [hm, um, lm]
    if not assets:                       # no visuals: let the primitive colliders cast shadows
        for m in meshes:
            m.cast_shadow = True
    art.finalize()
    return art, meshes


def _quat_to_R(q):  # q = [qx,qy,qz,qw], body->world
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)


def _quat_from_R(R):  # rotation matrix -> [qx,qy,qz,qw]
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def _compose(pos, quat):  # 4x4 from translation + quaternion
    M = np.eye(4); M[:3, :3] = _quat_to_R(quat); M[:3, 3] = pos
    return M


def _attach_obj(collider, link_pos, name, color, assets):
    """Parent a URDF visual mesh (link_models/<name>.obj, authored in the link frame) under
    its bound collider so it tracks the physics, and hide the collider's own primitive via its
    material (the object stays visible so the child still renders). Spot's visual origins are
    all identity, so the link frame here is (link_pos, no rotation)."""
    path = os.path.join(assets, "link_models", name + ".obj")
    try:
        grp = tp.OBJLoader().load(path, False)
    except Exception as e:                       # missing/odd OBJ -> keep the primitive
        print(f"[spot] visual {name}.obj not loaded ({e}); showing collider"); return None
    cw = _compose([collider.position.x, collider.position.y, collider.position.z],
                  [collider.quaternion.x, collider.quaternion.y, collider.quaternion.z, collider.quaternion.w])
    loc = np.linalg.inv(cw) @ _compose(list(link_pos), [0.0, 0.0, 0.0, 1.0])
    grp.position.set(*(float(v) for v in loc[:3, 3]))
    grp.quaternion.set(*(float(v) for v in _quat_from_R(loc[:3, :3])))

    def paint(o):
        try:                                     # Mesh children get Spot's colours; Groups skip
            mat = tp.MeshStandardMaterial(); mat.color = color; mat.roughness = 0.5; mat.metalness = 0.0
            o.set_material(mat); o.cast_shadow = True
        except Exception:
            pass
    grp.traverse(paint)
    collider.add(grp)
    collider.material.visible = False
    return grp


class SpotController:
    """Builds the 48-d Isaac observation, runs the policy, applies joint targets."""
    def __init__(self, art, policy):
        self.art, self.policy = art, policy
        self.last_action = np.zeros(12, np.float32)

    def obs(self, cmd):
        a = self.art
        rs, rv = a.root_state(), a.root_velocity()
        Rt = _quat_to_R(rs[3:7]).T
        lin_b, ang_b = Rt @ rv[0:3], Rt @ rv[3:6]
        proj_g = Rt @ np.array([0, 0, -1.0])
        qpos = a.joint_positions()[isaac_to_add] - default_q
        qvel = a.joint_velocities()[isaac_to_add]
        return np.concatenate([lin_b, ang_b, proj_g, cmd, qpos, qvel, self.last_action]).astype(np.float32)

    def step(self, world, cmd):
        with torch.no_grad():
            a = self.policy(torch.from_numpy(self.obs(cmd))[None]).numpy()[0]
        self.last_action = a
        self.art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
        world.step(0.02)   # 10 x 0.002 substeps, matching Isaac's decimation

    def hold(self, world, n):  # settle into a stand
        for _ in range(n):
            self.art.set_drive_targets(default_q[add_to_isaac].astype(np.float32))
            world.step(0.02)


def _write_png(path, rgb):
    """Minimal truecolor PNG writer (no Pillow dependency)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(typ, data):
        body = typ + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))  # 8-bit RGB
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def grid_texture(repeat):
    """One power-of-two grid cell (blue fill + white top/left lines), tiled `repeat`x."""
    px, line = 256, 5
    img = np.empty((px, px, 3), np.uint8)
    img[:] = np.array([47, 113, 168], np.uint8)            # blue fill
    img[:line, :] = img[:, :line] = np.array([226, 237, 248], np.uint8)  # white grid lines
    path = os.path.join(tempfile.gettempdir(), "threepp_spot_grid.png")
    _write_png(path, img)
    tex = tp.TextureLoader().load(path, tp.ColorSpace.SRGB)
    tex.wrap_s = tex.wrap_t = tp.TextureWrapping.Repeat
    tex.repeat = tp.Vector2(repeat, repeat)
    tex.anisotropy = 16
    return tex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", help="folder with spot_policy.pt "
                                     "(default: download on demand to ~/.cache/threepp/spot)")
    ap.add_argument("--shot", metavar="PNG", help="render headless after a short walk and save")
    args = ap.parse_args()
    assert tp.HAS_PHYSX, "needs a PhysX-enabled threepp build"
    assets = args.assets or fetch_assets()
    policy = torch.jit.load(os.path.join(assets, "spot_policy.pt"), map_location="cpu").eval()

    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002, max_substeps=20)
    ground = tp.Mesh(tp.BoxGeometry(80, 80, 1.0), tp.MeshStandardMaterial())
    ground.position.set(0, 0, -0.5)
    world.add_static(ground)
    art, meshes = build_spot(world, assets)   # assets -> render the URDF's link_models/*.obj
    ctrl = SpotController(art, policy)
    ctrl.hold(world, 150)   # stand up

    headless = bool(args.shot)
    canvas = tp.Canvas("threepp - Spot (Isaac policy)", width=1100, height=640,
                       antialiasing=4, headless=headless)
    rend = tp.GLRenderer(canvas)
    rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic
    rend.tone_mapping_exposure = 1.1

    FOG = 0x7ea9d0
    SIZE, CELL = 120.0, 0.6
    scene = tp.Scene()
    scene.background = tp.Background(FOG)
    scene.set_fog(FOG, 4.0, 26.0)     # the grid dissolves into a light-blue haze (tight = visible)
    scene.add(tp.HemisphereLight(0xd0e4f7, 0x4a5a6a, 1.15))
    key = tp.DirectionalLight(0xffffff, 2.8); key.position.set(6, -5, 11); key.cast_shadow = True
    scene.add(key)

    floor = tp.Mesh(tp.PlaneGeometry(SIZE, SIZE), tp.MeshStandardMaterial())
    floor.material.map = grid_texture(SIZE / CELL)   # tiled blue grid
    floor.material.color = 0xffffff                  # let the texture show through
    floor.material.roughness = 0.85
    floor.material.metalness = 0.0
    floor.receive_shadow = True
    scene.add(floor)
    for m in meshes:
        scene.add(m)   # each collider carries its hidden primitive + the visible OBJ (which casts)

    camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.01, 300)
    camera.up.set(0, 0, 1)   # Z-up world

    # controls HUD — screen-space TextSprites. GL composites them on top through an
    # internal ortho pass (anchor 0,0 = bottom-left, position = pixel offset), with
    # resize handled by the renderer — no overlay scene/camera needed.
    def add_controls_hud():
        font = tp.FontLoader().default_font()
        lines = [
            (0xffd34d, "Drive Spot"),
            (0xeef4ff, "arrows / numpad 8 2 4 6   move + strafe"),
            (0xeef4ff, "N / M  / numpad 7 9       turn"),
        ]
        for i, (color, txt) in enumerate(lines):
            s = tp.TextSprite(font, world_scale=26 if i == 0 else 20)
            s.set_text(txt)
            s.set_color(color)
            s.set_horizontal_alignment(tp.HorizontalAlignment.Left)
            s.screen_space = True
            s.screen_anchor = tp.Vector2(0, 0)
            s.position.set(18, 18 + (len(lines) - 1 - i) * 30, 0)
            scene.add(s)

    if headless:
        for _ in range(120):
            ctrl.step(world, np.array([1.0, 0, 0], np.float32))   # walk forward ~2.4 s
        bx = art.root_state()[0]
        camera.position.set(bx + 1.7, -2.5, 1.35); camera.look_at(bx - 0.2, 0, 0.35)   # close 3/4 hero view
        rend.render(scene, camera)        # clean still (the controls HUD is interactive-only)
        rend.save_frame(args.shot)
        print(f"saved {args.shot}  (walked to x={bx:.2f} m)")
        return

    # interactive: arrow keys / numpad drive the command; a chase cam trails Spot
    add_controls_hud()                   # screen-space controls panel (bottom-left)
    camera.position.set(-3.0, 0.0, 1.4)
    camera.look_at(0, 0, 0.3)
    BACK, HEIGHT, LAG = 2.8, 1.5, 0.08   # chase-cam distance / height / smoothing

    def on_resize(w, h):
        camera.aspect = w / max(h, 1); camera.update_projection_matrix(); rend.set_size(w, h)
    canvas.on_window_resize(on_resize)

    def down(*keys):
        return any(canvas.is_key_down(k) for k in keys)

    heading_lock = [None]   # yaw to hold while the user isn't turning

    def frame():
        # velocity command [vx, vy, wz] in Spot's body frame (+x fwd, +y left)
        vx = (1.5 if down("UP", "KP8") else 0.0) - (1.0 if down("DOWN", "KP2") else 0.0)
        vy = (1.0 if down("LEFT", "KP4") else 0.0) - (1.0 if down("RIGHT", "KP6") else 0.0)
        turn = (1.5 if down("N", "KP7") else 0.0) - (1.5 if down("M", "KP9") else 0.0)
        # Hold heading when not actively turning: the policy only regulates yaw
        # *rate* to 0, so any bias slowly spirals. A light P-controller on the yaw
        # error keeps it pointing straight (and walking straight) when idle.
        R = _quat_to_R(art.root_state()[3:7])
        yaw = math.atan2(R[1, 0], R[0, 0])
        if turn != 0.0:
            wz, heading_lock[0] = turn, yaw
        else:
            if heading_lock[0] is None:
                heading_lock[0] = yaw
            err = (yaw - heading_lock[0] + math.pi) % (2 * math.pi) - math.pi
            wz = float(np.clip(-2.0 * err, -1.0, 1.0))
        ctrl.step(world, np.array([vx, vy, wz], np.float32))

        # chase cam: sit BACK metres behind Spot's heading at HEIGHT, look at the body.
        # The desired pose is smoothed (lerp) so it trails as Spot turns.
        rs = art.root_state()
        p = np.array(rs[0:3], float)
        fwd = _quat_to_R(rs[3:7])[:, 0]               # body +x in world
        fwd = np.array([fwd[0], fwd[1], 0.0])          # keep the cam level
        nrm = np.linalg.norm(fwd)
        fwd = fwd / nrm if nrm > 1e-6 else np.array([1.0, 0.0, 0.0])
        desired = p - fwd * BACK + np.array([0.0, 0.0, HEIGHT])
        camera.position.lerp(tp.Vector3(float(desired[0]), float(desired[1]), float(desired[2])), LAG)
        camera.look_at(p[0] + fwd[0] * 0.4, p[1] + fwd[1] * 0.4, p[2] + 0.15)
        rend.render(scene, camera)

    canvas.animate(frame)


if __name__ == "__main__":
    main()
