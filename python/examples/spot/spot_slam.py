"""Spot SLAM — procedural terrain + trees + depth camera + live map reconstruction.

Spot walks on TerrainGenerator terrain while a body-mounted forward depth camera
accumulates a 3-D point cloud. Every ~3 s a background thread runs marching cubes
to reconstruct the growing SLAM surface (semi-transparent blue) over the ground truth.

    python spot_slam.py
    python spot_slam.py --seed 7 --amplitude 0.20
    python spot_slam.py --shot out.png

Controls: UP/DOWN/LEFT/RIGHT + N/M = drive  |  SPACE = toggle auto-forward  |  R = reset
"""
import argparse, math, os, sys, threading, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import threepp as tp
from threepp.rl import load_policy
from spot_deploy import (build_spot, fetch_assets,
                         _quat_to_R, _quat_from_R,
                         default_q, isaac_to_add, add_to_isaac, ACTION_SCALE, Z0)
from spot_depth_scan import ForwardDepthScanner
from spot_terrain_env import VX_HI, VY_HI, WZ_HI

GRAV = np.array([0.0, 0.0, -1.0])

# ── constants ──────────────────────────────────────────────────────────────────
WORLD_SZ   = 80.0
AMPLITUDE  = 10.8
TREE_COUNT = 60
CLEAR_R    = 7.0     # no trees within this radius of spawn
SENSOR_W   = 160
SENSOR_H   = 120
SENSOR_FAR = 8.0
SCAN_EVERY = 3       # depth scan every N frames; result cached for policy (~17 Hz)
MC_FRAMES  = 90      # trigger SLAM rebuild every N rendered frames


# ── terrain ────────────────────────────────────────────────────────────────────
def build_terrain_zup(field, world_size, amplitude):
    """Vectorised Z-up triangle soup (XY ground plane, Z = height) with UVs."""
    dim  = field.shape[0]
    half = world_size / 2.0
    lin  = np.linspace(-half, half, dim, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin, indexing='ij')
    # field is stored field_[iz][ix] (Z outer, X inner) — transpose so first index → world X
    Z    = (field.T * amplitude).astype(np.float32)
    U, V = np.meshgrid(np.linspace(0, 1, dim, dtype=np.float32),
                       np.linspace(0, 1, dim, dtype=np.float32), indexing='ij')

    ii, jj = np.meshgrid(np.arange(dim - 1), np.arange(dim - 1), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()

    def corner(a, b):
        return np.stack([X[a, b], Y[a, b], Z[a, b]], 1)
    def uvcorner(a, b):
        return np.stack([U[a, b], V[a, b]], 1)

    v00, v10, v11, v01 = corner(ii, jj), corner(ii+1, jj), corner(ii+1, jj+1), corner(ii, jj+1)
    u00, u10, u11, u01 = uvcorner(ii, jj), uvcorner(ii+1, jj), uvcorner(ii+1, jj+1), uvcorner(ii, jj+1)

    verts = np.vstack([np.stack([v00, v10, v11], 1).reshape(-1, 3),
                       np.stack([v00, v11, v01], 1).reshape(-1, 3)])
    uvs   = np.vstack([np.stack([u00, u10, u11], 1).reshape(-1, 2),
                       np.stack([u00, u11, u01], 1).reshape(-1, 2)])

    g = tp.BufferGeometry()
    g.set_attribute("position", np.ascontiguousarray(verts))
    g.set_attribute("uv",       np.ascontiguousarray(uvs))
    g.compute_vertex_normals()
    return g


# ── trees ──────────────────────────────────────────────────────────────────────
def scatter_trees(scene, gen, params, n=TREE_COUNT, seed=0):
    rng  = np.random.default_rng(seed)
    half = params.world_size / 2.0 - 4.0

    bark_alb, _ = tp.make_bark_textures(128, seed, [0.34, 0.22, 0.12])
    trunk_mat = tp.MeshStandardMaterial()
    trunk_mat.map       = bark_alb
    trunk_mat.roughness = 0.85

    leaf_mat = tp.MeshStandardMaterial()
    leaf_mat.color     = 0x3a7c2a
    leaf_mat.roughness = 0.9
    leaf_mat.side      = tp.Side.Double

    tpar = tp.TreeParams()
    tp.apply_tree_preset(0, tpar)     # Oak baseline
    tpar.max_iterations = 120         # cheaper for many trees
    tpar.trunk_height   = 4.0
    tpar.crown_radius_x = 2.5
    tpar.crown_radius_z = 2.5
    tpar.crown_height   = 3.0

    tgen = tp.TreeGenerator(0)        # reseeded per tree
    placed = 0
    attempts = 0
    while placed < n and attempts < n * 15:
        attempts += 1
        px, py = float(rng.uniform(-half, half)), float(rng.uniform(-half, half))
        if math.hypot(px, py) < CLEAR_R:
            continue
        tree_seed = int(rng.integers(0, 100_000))
        tgen.reseed(tree_seed)
        tgen.build_skeleton(tpar)
        trunk_geo = tgen.make_trunk_geometry(tpar)
        leaf_geo  = tgen.make_leaf_geometry(tpar)
        hz    = float(gen.height_at(px, py, params))
        scale = float(rng.uniform(0.7, 1.3))
        for geo, mat in ((trunk_geo, trunk_mat), (leaf_geo, leaf_mat)):
            m = tp.Mesh(geo, mat)
            m.rotation.x = math.pi / 2   # Y-up tree → Z-up world
            m.position.set(px, py, hz)
            m.scale.set(scale, scale, scale)
            m.cast_shadow = True
            scene.add(m)
        placed += 1
    print(f"[trees] placed {placed}/{n}")


# ── policy observation (94-d: mirrors SpotStepsEnv) ───────────────────────────
def v2_obs(art, last_act, cmd, ahead, h_here):
    rs, rv = art.root_state(), art.root_velocity()
    R = _quat_to_R(rs[3:7]); Rt = R.T
    lin_b = Rt @ rv[0:3]; ang_b = Rt @ rv[3:6]; proj_g = Rt @ GRAV
    jp_isaac = art.joint_positions()[isaac_to_add]
    jv_isaac = art.joint_velocities()[isaac_to_add]
    qpos = jp_isaac - default_q
    return np.concatenate([lin_b, ang_b, proj_g, cmd, qpos, jv_isaac, last_act,
                           [float(rs[2]) - h_here], ahead]).astype(np.float32)


def _scanner_pts(scanner):
    """Convert the accumulated elevation map H → world-space 3-D points for the SLAM mapper."""
    valid = ~np.isnan(scanner.H)
    if not valid.any():
        return np.empty((0, 3), np.float32)
    ix = np.where(valid)[0].astype(np.float32)
    iy = np.where(valid)[1].astype(np.float32)
    return np.stack([ix * scanner.cell + scanner.x0,
                     iy * scanner.cell + scanner.y0,
                     scanner.H[valid]], axis=1).astype(np.float32)


# ── SLAM mapper ────────────────────────────────────────────────────────────────
class SlamMapper:
    """Accumulates scan hits → VoxelGrid → marching-cubes surface (background thread)."""

    VOXEL = 0.12
    CELL  = 0.14
    RAD   = 0.22   # tight: surface hugs scan hits rather than ballooning above them
    ISO   = 0.55

    def __init__(self, scene):
        self.scene   = scene
        self.grid    = tp.VoxelGrid(self.VOXEL, max_points_per_voxel=3, min_spacing=0.12)
        self._surf   = [None]
        self._pending= [None]
        self._busy   = [False]
        self._lock   = threading.Lock()

    def insert(self, pts):
        if pts.shape[0] > 0:
            self.grid.insert_array(pts)

    def trigger_rebuild(self):
        with self._lock:
            if self._busy[0] or self.grid.voxel_count < 30:
                return
            pts = self.grid.collect()
            self._busy[0] = True
        threading.Thread(target=self._worker, args=(pts,), daemon=True).start()

    def _worker(self, pts):
        try:
            field = tp.splat_points_to_field(pts, self.CELL, self.RAD, max_nodes=6_000_000)
            iso   = tp.marching_cubes(field, self.ISO)
            with self._lock:
                self._pending[0] = iso if not iso.empty else None
        finally:
            with self._lock:
                self._busy[0] = False

    def apply_pending(self):
        with self._lock:
            iso = self._pending[0]; self._pending[0] = None
        if iso is None:
            return
        geo = tp.iso_mesh_to_geometry(iso)
        mat = tp.MeshStandardMaterial()
        mat.color       = 0x44aaff
        mat.roughness   = 0.35
        mat.metalness   = 0.10
        mat.side        = tp.Side.Double
        mat.transparent = True
        mat.opacity     = 0.55
        mesh = tp.Mesh(geo, mat); mesh.frustum_culled = False
        if self._surf[0] is not None:
            self.scene.remove(self._surf[0])
        self._surf[0] = mesh
        self.scene.add(mesh)

    def clear(self):
        """Remove the displayed surface and discard all accumulated data."""
        with self._lock:
            self._pending[0] = None
        self.grid.clear()
        if self._surf[0] is not None:
            self.scene.remove(self._surf[0])
            self._surf[0] = None

    @property
    def busy(self):  return self._busy[0]
    @property
    def voxels(self): return self.grid.voxel_count


# ── path trail ────────────────────────────────────────────────────────────────
class PathTrail:
    MAX = 3000

    def __init__(self, scene):
        g = tp.BufferGeometry()
        g.set_attribute("position", np.zeros((self.MAX, 3), np.float32))
        g.set_draw_range(0, 0)
        mat = tp.LineBasicMaterial(); mat.color = 0xffffff
        self.line = tp.Line(g, mat); self.line.frustum_culled = False
        self._g   = g; self._pts = []; self._prev = None
        scene.add(self.line)

    def update(self, rs):
        p = (float(rs[0]), float(rs[1]), float(rs[2]) + 0.06)
        if self._prev and math.dist(p, self._prev) < 0.15:
            return
        self._pts.append(p)
        if len(self._pts) > self.MAX:
            self._pts.pop(0)
        arr = np.array(self._pts, np.float32)
        buf = np.zeros((self.MAX, 3), np.float32)
        buf[:len(arr)] = arr
        self._g.update_attribute("position", buf)
        self._g.set_draw_range(0, len(self._pts))
        self._prev = p

    def clear(self):
        self._pts.clear(); self._prev = None
        self._g.update_attribute("position", np.zeros((self.MAX, 3), np.float32))
        self._g.set_draw_range(0, 0)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--amplitude", type=float, default=AMPLITUDE)
    ap.add_argument("--shot",      metavar="PNG")
    args = ap.parse_args()
    assert tp.HAS_PHYSX, "needs a PhysX-enabled threepp build"
    headless = bool(args.shot)

    # ── assets ────────────────────────────────────────────────────────────────
    assets = fetch_assets()
    model_path = os.path.join(_HERE, "spot_steps.pt")
    ac, _norm, _meta = load_policy(model_path, device="cpu")
    ac.eval()
    print(f"[policy] {os.path.basename(model_path)}")

    # ── terrain ───────────────────────────────────────────────────────────────
    print("[terrain] generating ...")
    tparams = tp.TerrainParams()
    tp.apply_terrain_preset(1, tparams)          # Rolling Hills base shape
    tparams.resolution    = 192                  # 192² cells — fast physics trimesh
    tparams.world_size    = WORLD_SZ
    tparams.amplitude     = float(args.amplitude)# physical height in metres (height_at must agree)
    tparams.feature_scale = 45.0                 # preset is 430 m for 1600 m world; scale to 80 m
    tparams.octaves       = 7                    # more detail layers
    tparams.warp          = 0.40                 # organic domain warp
    tparams.falloff       = tp.TerrainFalloff.Off# preset's radial bowl looks wrong at 80 m
    tparams.erosion       = tp.ErosionType.Off   # thermal grooves are too fine at this scale
    tparams.ao_strength   = 8.0

    gen  = tp.TerrainGenerator(args.seed)
    gen.build_field(tparams)
    field    = gen.get_field()
    terr_tex = gen.bake_splat_texture(tparams)
    terr_geo = build_terrain_zup(field, WORLD_SZ, float(args.amplitude))
    # Sample terrain under the four feet + base to find the highest point in the
    # footprint — prevents spawning any foot inside the mesh on sloped ground.
    _FEET = ((0.30, 0.17), (0.30, -0.17), (-0.30, 0.17), (-0.30, -0.17), (0.0, 0.0))
    h0 = max(float(gen.height_at(dx, dy, tparams)) for dx, dy in _FEET)
    print(f"[terrain] done  h_footprint={h0:.3f} m")

    # ── physics ───────────────────────────────────────────────────────────────
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002, max_substeps=20)
    base  = tp.Mesh(tp.BoxGeometry(WORLD_SZ + 10, WORLD_SZ + 10, 0.4), tp.MeshStandardMaterial())
    base.position.set(0, 0, -0.2)
    world.add_static(base)
    terr_phys = tp.Mesh(terr_geo, tp.MeshStandardMaterial())
    world.add_static_trimesh(terr_phys)

    art, meshes = build_spot(world, assets)

    def settle(n=80):
        for _ in range(n):
            art.set_drive_targets(default_q[add_to_isaac].astype(np.float32))
            world.step(0.02)

    art.reset(tp.Vector3(0.0, 0.0, Z0 + h0 + 0.02))
    settle()
    print("[spot] standing")

    # ── canvas + renderer ─────────────────────────────────────────────────────
    canvas = tp.Canvas("threepp · Spot SLAM", width=1200, height=720,
                       antialiasing=4, headless=headless)
    rend = tp.GLRenderer(canvas)
    rend.shadow_map_enabled       = True
    rend.tone_mapping             = tp.ToneMapping.ACESFilmic
    rend.tone_mapping_exposure    = 1.1

    # ── scene ─────────────────────────────────────────────────────────────────
    scene = tp.Scene()
    scene.background = tp.Background(0x8ab4d4)
    scene.set_fog(0x8ab4d4, 25.0, 80.0)
    scene.add(tp.HemisphereLight(0xd0e8ff, 0x3a4820, 0.9))
    sun = tp.DirectionalLight(0xfff8e0, 2.8)
    sun.position.set(15, -10, 20)
    sun.cast_shadow = True
    sun.set_shadow_frustum(-18, 18, 18, -18)  # fog hides >25 m; no need for full world frustum
    sun.set_shadow_bias(-0.0005)
    scene.add(sun)

    terr_vis = tp.Mesh(terr_geo, tp.MeshStandardMaterial())
    terr_vis.material.map       = terr_tex
    terr_vis.material.roughness = 0.9
    terr_vis.receive_shadow = True
    scene.add(terr_vis)

    for m in meshes:
        m.cast_shadow = True
        scene.add(m)

    print("[trees] scattering ...")
    scatter_trees(scene, gen, tparams, seed=args.seed)

    # ── camera (Z-up chase cam) ───────────────────────────────────────────────
    w, h = canvas.size()
    camera = tp.PerspectiveCamera(50, w / h, 0.05, 200)
    camera.up.set(0, 0, 1)
    spawn_z = Z0 + h0
    camera.position.set(-3.0, -3.0, spawn_z + 1.8)
    camera.look_at(0.0, 0.0, spawn_z + 0.3)
    BACK, HEIGHT, LAG = 3.5, 1.8, 0.08

    # ── SLAM objects ──────────────────────────────────────────────────────────
    half = WORLD_SZ / 2.0
    scanner = ForwardDepthScanner(rend, scene, meshes,
                                  bounds=(-half, half, -half, half),
                                  cell=0.15, far=SENSOR_FAR)
    scanner.prewarm(art.root_state())
    slam    = SlamMapper(scene)
    trail   = PathTrail(scene)
    last_act    = np.zeros(12, np.float32)
    ahead_cache = [np.zeros(45, np.float32)]   # last sensor reading; reused on skipped frames
    h_here_cache= [h0]

    # ── state ─────────────────────────────────────────────────────────────────
    fc        = [0]
    auto_fwd  = [True]
    hdg_lock  = [None]
    r_held    = [False]
    space_held= [False]

    def reset():
        rh = max(float(gen.height_at(dx, dy, tparams)) for dx, dy in _FEET)
        art.reset(tp.Vector3(0.0, 0.0, Z0 + rh + 0.02))
        last_act[:] = 0.0
        hdg_lock[0] = None
        settle(40)
        scanner.clear_map(); scanner.prewarm(art.root_state())
        slam.clear()
        trail.clear()

    # ── headless ──────────────────────────────────────────────────────────────
    if headless:
        cmd = np.array([1.0, 0.0, 0.0], np.float32)
        for _ in range(150):
            rs = art.root_state()
            ahead, h_here = scanner.scan(rs)
            obs = v2_obs(art, last_act, cmd, ahead, h_here)
            with torch.no_grad():
                a = ac.act_mean(torch.from_numpy(obs)[None])[0].numpy()
            last_act[:] = a
            art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
            world.step(0.02)
        rs = art.root_state()
        for _ in range(20):
            trail.line.visible = False
            ahead, h_here = scanner.scan(rs)
            trail.line.visible = True
        slam.insert(_scanner_pts(scanner))
        slam.trigger_rebuild()
        time.sleep(1.2)
        slam.apply_pending()
        p   = np.array(rs[:3], float)
        fwd = _quat_to_R(rs[3:7])[:, 0]; fwd[2] = 0
        fwd /= max(np.linalg.norm(fwd), 1e-6)
        eye = p - fwd * 4.5 + np.array([0.0, 0.0, 2.2])
        camera.position.set(float(eye[0]), float(eye[1]), float(eye[2]))
        camera.look_at(float(p[0]), float(p[1]), float(p[2]) + 0.3)
        rend.render(scene, camera)
        rend.save_frame(args.shot)
        print(f"saved {args.shot}")
        return

    # ── interactive ───────────────────────────────────────────────────────────
    ui = tp.ImguiContext(canvas, rend) if tp.HAS_IMGUI else None

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        rend.set_size(w, h)
    canvas.on_window_resize(on_resize)

    def down(*keys):
        return any(canvas.is_key_down(k) for k in keys)

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(300, 0)
        tp.imgui.begin("Spot SLAM")

        rs = art.root_state()
        tp.imgui.text(f"pos  x={rs[0]:+.1f}  y={rs[1]:+.1f}  z={rs[2]:.2f} m")
        _, auto_fwd[0] = tp.imgui.checkbox("auto-forward (SPACE)", auto_fwd[0])
        tp.imgui.separator()
        tp.imgui.text("Depth camera")
        chg, v = tp.imgui.slider_float("range noise (m)", scanner.sensor.range_noise, 0.0, 0.10)
        if chg: scanner.sensor.range_noise = v
        if scanner.cloud is not None:
            chg, v = tp.imgui.slider_float("point size", scanner.cloud.material.size, 0.02, 0.12)
            if chg: scanner.cloud.material.size = v
        _, scanner.show_cloud = tp.imgui.checkbox("show cloud", scanner.show_cloud)
        _, scanner.show_grid  = tp.imgui.checkbox("show scan grid", scanner.show_grid)
        tp.imgui.separator()
        tp.imgui.text(f"SLAM  voxels: {slam.voxels}  {'[rebuilding]' if slam.busy else ''}")
        if tp.imgui.button("rebuild surface now") and not slam.busy:
            slam.trigger_rebuild()
        if tp.imgui.button("reset (R)"):
            reset()
        tp.imgui.separator()
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   |   UP/DN/LT/RT + N/M")
        tp.imgui.end()

    def frame():
        fc[0] += 1

        # SPACE toggles auto-forward
        if down("SPACE"):
            if not space_held[0]: auto_fwd[0] = not auto_fwd[0]
            space_held[0] = True
        else:
            space_held[0] = False

        # keyboard command
        vx = (1.5 if down("UP", "KP8") else 0.0) - (1.0 if down("DOWN", "KP2") else 0.0)
        vy = (1.0 if down("LEFT", "KP4") else 0.0) - (1.0 if down("RIGHT", "KP6") else 0.0)
        wz_key = (1.5 if down("N", "KP7") else 0.0) - (1.5 if down("M", "KP9") else 0.0)
        if auto_fwd[0] and vx == 0.0 and vy == 0.0 and wz_key == 0.0:
            vx = 1.0

        # heading hold
        rs  = art.root_state()
        R   = _quat_to_R(rs[3:7])
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
        if wz_key != 0.0:
            wz = wz_key; hdg_lock[0] = yaw
        else:
            if hdg_lock[0] is None: hdg_lock[0] = yaw
            err = (yaw - hdg_lock[0] + math.pi) % (2 * math.pi) - math.pi
            wz  = float(np.clip(-2.0 * err, -1.0, 1.0))

        # scan → obs → policy → step
        # Re-render the depth sensor only every SCAN_EVERY frames; reuse cached obs otherwise.
        rs = art.root_state()
        if fc[0] % SCAN_EVERY == 0:
            _extra = [o for o in (slam._surf[0], trail.line) if o is not None]
            for o in _extra: o.visible = False
            ahead_cache[0], h_here_cache[0] = scanner.scan(rs)
            for o in _extra: o.visible = True
            slam.insert(_scanner_pts(scanner))

        cmd   = np.array([vx, vy, wz], np.float32)
        obs   = v2_obs(art, last_act, cmd, ahead_cache[0], h_here_cache[0])
        with torch.no_grad():
            a = ac.act_mean(torch.from_numpy(obs)[None])[0].numpy()
        last_act[:] = a
        art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
        world.step(0.02)
        rs = art.root_state()
        if fc[0] % MC_FRAMES == 0:
            slam.trigger_rebuild()
        slam.apply_pending()
        trail.update(rs)

        # R = reset
        if down("R"):
            if not r_held[0]: reset(); print("[reset]")
            r_held[0] = True
        else:
            r_held[0] = False

        # chase cam
        p   = np.array(rs[:3], float)
        fwd = R[:, 0].copy(); fwd[2] = 0.0
        nrm = np.linalg.norm(fwd); fwd = fwd / nrm if nrm > 1e-6 else np.array([1.0, 0.0, 0.0])
        des = p - fwd * BACK + np.array([0.0, 0.0, HEIGHT])
        camera.position.lerp(tp.Vector3(float(des[0]), float(des[1]), float(des[2])), LAG)
        camera.look_at(float(p[0] + fwd[0] * 0.5), float(p[1] + fwd[1] * 0.5), float(p[2]) + 0.2)

        rend.render(scene, camera)
        if ui:
            ui.render(draw_ui)

    print(__doc__)
    canvas.animate(frame)


if __name__ == "__main__":
    main()
