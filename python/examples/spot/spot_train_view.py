"""A live window onto a SpotStepsEnv while it trains: N of its robots, each on its own lane's terrain, drawn as the real
Spot (the URDF visuals) from the GPU sim's link poses every control tick.

The selected lanes sit thousands of metres apart in the training world (one lane per env, SPACING apart in y), so the
view lays them side by side: slot s shows lane i shifted by s * SPACING - i * SPACING in y, terrain and robot alike.
Lanes are picked across the course families, evenly within each family's block, so the window shows every kind of
terrain at once.

Used by train_spot_steps.py --view N, which builds the env with keep_meshes=True and calls frame() after every env
step. The window updates at the training rate (a few ticks of simulated time per second of wall time at K = 2048), and
holds still while PPO updates. Closing it stops the drawing; training carries on.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
import spot_course as sc
import spot_deploy as sd
from spot_terrain_env import SPACING

# GPU link order (threepp.rl.sim: root, hips 1-4, upper legs 5-8, lower legs 9-12, each fl/fr/hl/hr) for each of
# build_spot's meshes in its add order (base, then per leg hip, upper, lower).
_GPU_LINK = [0] + [j for leg in range(4) for j in (1 + leg, 5 + leg, 9 + leg)]

FAMILY_COLOR = {sc.FLAT: 0x8a9a7a, sc.STAIRS: 0xb8b2a6, sc.HILLS: 0x7f9a62, sc.CROSS: 0x9a9a62, sc.ROUGH: 0x8c7a62,
                sc.PYRAMID: 0xc9b48f, sc.PIT: 0xa89a86, sc.BLOCKS: 0x8f99a6}


def _material(color):
    m = tp.MeshStandardMaterial()
    m.color = color
    m.roughness = 0.95
    return m


class _VisualOnly:
    """A PhysxWorld stand-in: build_spot lays out Spot's link meshes and their URDF visuals, nothing is simulated."""

    def create_articulation(self, **_):
        return self

    def add_link(self, mesh, **_):
        return mesh

    def finalize(self):
        pass


def pick_lanes(codes, n):
    """n lane indices spread over the families present, evenly inside each family's contiguous block."""
    fams = [c for c in range(len(sc.FAMILIES)) if (codes == c).any()]
    per = [n // len(fams) + (1 if k < n % len(fams) else 0) for k in range(len(fams))]
    out = []
    for c, m in zip(fams, per):
        idx = np.flatnonzero(codes == c)
        if m > 0:
            out += [int(idx[int(round(q))]) for q in np.linspace(0, len(idx) - 1, m)]
    return out


class TrainView:
    def __init__(self, env, n=24, width=1600, height=900, assets=None, shots=None, shot_every=50):
        if getattr(env, "terrain_meshes", None) is None:
            raise ValueError("TrainView needs SpotStepsEnv(keep_meshes=True) with the raycast height source")
        fam = getattr(env, "_fam_k", None)
        codes = fam.cpu().numpy() if fam is not None else np.zeros(env.K, np.int64)
        self.env = env
        self.lanes = pick_lanes(codes, min(n, env.K))
        self.n = len(self.lanes)
        self.dy = np.array([(s - i) * SPACING for s, i in enumerate(self.lanes)], np.float64)

        self.canvas = tp.Canvas("Spot training (live)", width=width, height=height)
        self.renderer = tp.GLRenderer(self.canvas)       # no shadows: a map over the whole layout costs every tick
        self.renderer.tone_mapping = tp.ToneMapping.ACESFilmic
        self.renderer.tone_mapping_exposure = 1.0
        self.scene = tp.Scene()
        self.scene.background = tp.Background(0xa7bdd3)
        self.scene.add(tp.HemisphereLight(0xdce8f6, 0x55606c, 1.2))
        width_y = self.n * SPACING
        x_lo, x_hi = env.ground_extent
        sun = tp.DirectionalLight(0xffffff, 2.4)
        sun.position.set(-25.0, -35.0, 50.0)                 # a direction: the light aims at the origin
        self.scene.add(sun)

        ground = tp.Mesh(tp.PlaneGeometry(x_hi - x_lo, width_y + 6.0), _material(0x8a9a7a))
        ground.position.set(0.5 * (x_lo + x_hi), 0.5 * width_y - 0.5 * SPACING, 0.0)
        self.scene.add(ground)
        self._add_terrain(codes)
        if assets is None:
            try:
                assets = sd.fetch_assets()                   # the play scripts' cache of the BD URDF visuals
            except Exception as e:                           # noqa: BLE001 - primitives still show the gait
                print(f"[view] no URDF visuals ({e}); drawing the colliders")
        self._robots = [self._robot(assets) for _ in range(self.n)]

        self.camera = tp.PerspectiveCamera(45, width / height, 0.1, 1000.0)
        self.camera.up.set(0, 0, 1)
        # from behind the start of the lanes, looking down them: the first bands, where most robots train, up close
        cy = 0.5 * (width_y - SPACING)
        self.camera.position.set(-6.0, cy - 0.35 * width_y, 0.22 * width_y + 4.0)
        self.controls = tp.OrbitControls(self.camera, self.canvas)
        self.controls.target.set(10.0, cy, 0.0)
        self.controls.update()
        self.canvas.on_window_resize(self._resize)
        self.open = True
        self.shots, self.shot_every, self.frames = shots, int(shot_every), 0
        if shots:
            os.makedirs(shots, exist_ok=True)
        names = [sc.FAMILIES[int(codes[i])] for i in self.lanes]
        print("[view] slots (left to right): " + ", ".join(f"{s}:{f}" for s, f in enumerate(names)))

    def _resize(self, size):
        self.camera.aspect = size.aspect()
        self.camera.update_projection_matrix()
        self.renderer.set_size(size)

    def _add_terrain(self, codes):
        # Without shadows, rings of one colour run together and a foot on a lower ring reads as sinking into the one
        # behind it: shade every box by its top height, alternating light and dark per 5 cm, so steps separate.
        mats = {}

        def mat(code, top):
            k = (code, int(round(top / 0.05)) % 2 if top > 0.01 else -1)
            if k not in mats:
                col = tp.Color(FAMILY_COLOR[code])
                f = 1.0 if k[1] < 1 else 0.82
                mats[k] = _material(tp.Color(col.r * f, col.g * f, col.b * f))
            return mats[k]
        lane_y = np.array(self.lanes, np.float64) * SPACING
        half = 0.5 * SPACING - 1e-3
        added = 0
        for m in self.env.terrain_meshes:
            y = float(m.position.y)
            s = np.flatnonzero(np.abs(lane_y - y) < half)
            if s.size == 0:
                continue
            pos = m.geometry.get_attribute("position")
            if pos is not None and len(pos) and (np.ptp(np.asarray(pos)[:, 1]) * float(m.scale.y)) > SPACING + 0.5:
                continue                                     # the shared ground box spans every lane
            top = float(np.asarray(pos)[:, 2].max() * m.scale.z + m.position.z) if pos is not None and len(pos) else 0.0
            for slot in s:
                v = tp.Mesh(m.geometry, mat(int(codes[self.lanes[slot]]), top))
                v.position.set(float(m.position.x), y + float(self.dy[slot]), float(m.position.z))
                q = m.quaternion
                v.quaternion.set(q.x, q.y, q.z, q.w)
                v.scale.set(m.scale.x, m.scale.y, m.scale.z)
                self.scene.add(v)
                added += 1
        print(f"[view] {added} terrain meshes for {self.n} lanes")

    def _robot(self, assets):
        _, meshes = sd.build_spot(_VisualOnly(), assets=assets)
        for m in meshes:
            self.scene.add(m)
        return meshes

    def frame(self):
        """Pose every shown robot from the sim's link poses and draw one frame. False once the window is closed."""
        if not self.open:
            return False
        try:
            return self._frame()
        except Exception as e:                                       # noqa: BLE001 - a view must never end a run
            print(f"[view] stopped drawing: {type(e).__name__}: {e}")
            self.open = False
            return False

    def _frame(self):
        lp = self.env.sim.link_pose[self.lanes].cpu().numpy()        # [n, links, 7] = qx, qy, qz, qw, px, py, pz
        for s, meshes in enumerate(self._robots):
            for k, m in enumerate(meshes):
                q = lp[s, _GPU_LINK[k]]
                m.position.set(float(q[4]), float(q[5] + self.dy[s]), float(q[6]))
                m.quaternion.set(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        self.controls.update()
        self.frames += 1
        shoot = bool(self.shots) and self.frames % self.shot_every == 0

        def draw():
            self.renderer.render(self.scene, self.camera)
            if shoot:
                from PIL import Image
                Image.fromarray(self.renderer.read_pixels()).save(
                    os.path.join(self.shots, f"view_{self.frames:06d}.png"))
        self.open = bool(self.canvas.animate_once(draw))
        return self.open
