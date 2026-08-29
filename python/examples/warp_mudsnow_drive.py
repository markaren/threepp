"""A Range Rover driven over Bekker-Wong ground: mud, snow and a packed clay track.

WASD drives a PxVehicle2 Range Rover Evoque down three lanes of deformable
terrain (threepp.terrain_deform, the Sumner/O'Brien/Hodgins displacement grid
with a Bekker-Wong bearing and Janosi-Hanamoto shear). The lanes are 48 m long:
mud on one side, packed snow on the other, a firm clay strip down the middle to
spawn on and to compare against. The wheels cut ruts, the ruts stay, and the
soil decides how deep the car sits and how much of the throttle reaches the
ground.

    pip install warp-lang
    python warp_mudsnow_drive.py                  # Vulkan if available
    python warp_mudsnow_drive.py --gl             # OpenGL
    python warp_mudsnow_drive.py --vsync          # cap to the display
    python warp_mudsnow_drive.py --interop        # zero-copy lane meshes (see Strip)
    python warp_mudsnow_drive.py --bench          # ms/frame, vsync off, headless
    python warp_mudsnow_drive.py --shot --script mud     # one acceptance frame
    python warp_mudsnow_drive.py --cell 0.08 --width 6   # cheaper ground

    W / S     throttle / brake        R    gear (forward <-> reverse)
    A / D     steer                   SPACE handbrake
    V         driver POV              C    cinematic orbit
    BACKSPACE respawn                 T    reset the ground
    F         save a frame

`--script` drives a fixed input sequence headless and saves the frame the
comparison needs: `mud` and `snow` are the same line at the same throttle down
either soft lane, `spin_mud` / `spin_clay` the same full-throttle launch, `lap`
a circle in mud twice. Repeatable is the point -- a hand-driven pass is not the
same line twice, and the whole claim is that only the SOIL differs.

How the ground and the vehicle are coupled
------------------------------------------
PhysX finds ground with a raycast, and the deformable terrain does not live in
the PhysX scene. Mirroring the grid into a collider every frame is hopeless, and
holding the car up on terrain reaction forces alone makes a boat -- no sticky
tires, no suspension. So the split is:

  * PhysX keeps the WHOLE vehicle: suspension, tire model, sticky tires,
    substeps, load transfer. Untouched.
  * The road its suspension sees is Bekker's. Per wheel, per frame, we invert
    the Bekker bearing integral for the current wheel load W and hand the
    suspension a plane at `grade - z_eq(W)` (PhysxVehicle.set_road_override).
    The wheel rides IN the ground by exactly the sinkage the load dictates.
  * The friction the tire model gets is Mohr-Coulomb's:
    mu = (c*A(z_eq) + tan_phi*W) / W -- about 0.37 in mud, 0.40 in snow and
    0.75 on clay, against the 2.0 of the C++ demo's asphalt. That difference IS
    the driving experience.
  * The terrain module carves the ruts: the wheels are its collider spheres, fed
    the contact-patch slip velocity, so the prints follow the wheels and the
    module's own Janosi state is live for the panel.

Double-count audit -- what the soil model applies and what it only reports:

    bearing (module)   NOT applied. PhysX's suspension carries the car; the
                       module's Bekker force would be the same load a second
                       time. It is displayed instead, next to the PhysX
                       suspension load, and the two agreeing to a few percent
                       is the point of the middle panel.
    shear (module)     NOT applied. The tire model does traction, under the
                       Mohr-Coulomb mu ceiling we hand it -- which saturates at
                       the same c*A + tan_phi*W the module's Janosi law does.
                       Displayed as traction utilisation.
    bulldozing         APPLIED, via add_force_at_pos. The first-order soil wedge
                       a wheel pushes ahead of itself. PhysX has no notion of
                       it, so there is nothing to double-count: it is the whole
                       reason deep snow slows you down.
    suction, grade     module-internal; they shape the grid the wheels then read.

Keep that split. Applying the module's bearing or shear on top of PhysX's would
double the load the car already carries.
"""
import atexit
import math
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import warp as wp

import threepp as tp
from threepp.terrain_deform import MATERIALS, DeformableTerrain
from warp_common import cli_arg, parse_size, standard_material

try:
    from threepp.cuda_interop import VkInteropArray
except ImportError:                                 # build without the interop helper
    VkInteropArray = None

GL = "--gl" in sys.argv
INTEROP = "--interop" in sys.argv    # zero-copy publish; see the note on Strip
VSYNC = "--vsync" in sys.argv
SHOT = "--shot" in sys.argv
BENCH = "--bench" in sys.argv
SHOT_TIME = cli_arg("--shot", 8.0, float)
WIDTH, HEIGHT = parse_size(cli_arg("--size", "1600x900", str))
CELL = cli_arg("--cell", 0.05, float)
LANE_LEN = cli_arg("--length", 48.0, float)
LANE_W = cli_arg("--width", 8.0, float)
OUT_DIR = cli_arg("--out-dir", ".", str)
AA = cli_arg("--aa", 4, int)
WARM = cli_arg("--warm", 60, int)
NOSAN = "--no-sanitize" in sys.argv

DT = 1.0 / 60.0
R_WHEEL = 0.40                    # PhysX wheel radius, and the collider sphere
CHASSIS_MASS = 1500.0
REST_LOAD = CHASSIS_MASS * 9.81 / 4.0

# --- the yard -----------------------------------------------------------------
# 48 m of lane along x, three strips across z. The terrain module works in its
# own (x, y, height) frame and publishes y-up meshes as (x, y, z) -> (x, z, -y),
# so a lane's module-y is the NEGATIVE of its world z.

X0, X1 = -0.5 * LANE_LEN, 0.5 * LANE_LEN
NX = int(round((X1 - X0) / CELL)) + 1

CLAY_W = 4.0
LANES = ("mud", "clay", "snow")
LANE_Z = {"mud": (0.5 * CLAY_W, 0.5 * CLAY_W + LANE_W),
          "clay": (-0.5 * CLAY_W, 0.5 * CLAY_W),
          "snow": (-0.5 * CLAY_W - LANE_W, -0.5 * CLAY_W)}

# The presets are calibrated to a 5 cm Spot foot; the bearing numbers carry
# straight over to a 40 cm wheel (a 3.7 kN corner sits 22 mm into clay, 55 mm
# into mud, 64 mm into snow, which is what those materials do to a light 4x4).
# The one thing that does not carry over is the grade relaxation: at 0.15 /s a
# rut forgets it is a rut in about seven seconds, and a car comes back around
# for a second lap long before that. Slowed to 0.02 /s, the mud and snow ruts
# hold their datum for a minute, which is what makes the second pass ride in
# its own tracks instead of digging a fresh sinkage below them.
MATS = {"mud": replace(MATERIALS["mud"], grade_rate=0.02),
        "snow": replace(MATERIALS["snow"], grade_rate=0.02),
        "clay": MATERIALS["clay"]}

SPAWN_POS = tp.Vector3(-20.0, 1.05, 0.0)
SPAWN_ROT = tp.Quaternion().set_from_axis_angle(tp.Vector3(0, 1, 0), math.pi / 2)

# Bulldozing coefficient: the fraction of (frontal patch area x bearing
# pressure) that ends up as drag. 0.5 is the by-feel value -- around 460 N per
# wheel in mud and 830 N in snow at cruise, so the snow lane costs you about
# 2 m/s^2 and you feel the lane change without the car stopping dead.
C_BULLDOZE = 0.5

RIGID_Y = -0.03                   # the packed ground the lanes sit on
FALLBACK_MU = 1.1                 # see "handling" below


# --- Bekker inversion ----------------------------------------------------------


def bekker_table(mat, radius=R_WHEEL, cell=CELL, z_max=0.30, n=241):
    """W(z) and contact area A(z) for a sphere pressed z into flat ground.

    The SAME discrete sum the terrain module's imprint kernel evaluates -- one
    cell of the grid at a time, the chord half-width b floored at a cell, the
    pressure (k_c/b + k_phi) * z^n over the cell area -- so inverting this table
    lands the wheel at exactly the depth where the module's own bearing balances
    the load, and the panel's Bekker-vs-PhysX column agrees by construction
    rather than by fitting. Closed forms for the same integral do not: the
    b-floor and the grid quantisation are both worth several percent at these
    depths.
    """
    zs = np.linspace(0.0, z_max, n)
    m = int(math.ceil(radius / cell)) + 2
    g = np.arange(-m, m + 1) * cell
    dx, dy = np.meshgrid(g, g, indexing="ij")
    d2 = dx * dx + dy * dy
    load = np.zeros(n)
    area = np.zeros(n)
    for i, z in enumerate(zs):
        # Height of the sphere's lower surface over each cell, +BIG outside it.
        bottom = np.where(d2 < radius * radius,
                          (radius - z) - np.sqrt(np.maximum(radius * radius - d2, 0.0)),
                          1.0e9)
        zc = np.maximum(-bottom, 0.0)
        b = np.sqrt(np.maximum(zc * (2.0 * radius - zc), cell * cell))
        p = (mat.k_c / b + mat.k_phi) * np.power(zc, mat.n)
        load[i] = float((p * cell * cell).sum())
        area[i] = float((zc > 0.0).sum()) * cell * cell
    return zs, load, area


LUT = {k: bekker_table(m) for k, m in MATS.items()}


def sinkage(lane, w):
    """(z_eq, mu) for wheel load `w` on `lane`."""
    zs, load, area = LUT[lane]
    m = MATS[lane]
    z = float(np.interp(w, load, zs))
    a = float(np.interp(z, zs, area))
    return z, (m.cohesion * a + m.tan_phi * w) / max(w, 1.0)


def bulldoze(lane, z):
    """First-order bulldozing drag (N) for a wheel ploughing at sinkage z.

    The wedge of soil ahead of the wheel is 2b wide and z deep, and it is being
    pushed at the bearing pressure the wheel is already generating.
    """
    if z <= 1.0e-4:
        return 0.0
    m = MATS[lane]
    b = math.sqrt(max(z * (2.0 * R_WHEEL - z), CELL * CELL))
    p = (m.k_c / b + m.k_phi) * z ** m.n
    return C_BULLDOZE * z * 2.0 * b * p


# --- the ground ----------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("no CUDA device -- the terrain will run on the CPU and this will crawl")

terrain = {}
for name in LANES:
    z0, z1 = LANE_Z[name]
    ny = int(round((z1 - z0) / CELL)) + 1
    terrain[name] = DeformableTerrain((X0, -z1), CELL, (NX, ny), K=1,
                                      material=MATS[name], device=device, c_max=4)
    print(f"  {name:5s} {terrain[name]}")
cells = sum(t.nx * t.ny for t in terrain.values())
print(f"  {cells:,} cells total")


def bilinear(t, flat, x, y):
    """Bilinear lookup into one of a terrain's [K, nx, ny] grids, [K, P] in/out.

    DeformableTerrain.heights() is this against `h`; the coupling wants the same
    gather against `grade`, because the road we hand the suspension is measured
    from the bearing datum, not from whatever the last pass left on the surface.
    """
    fx = ((x - t.origin_t[:, 0:1]) / t.cell).clamp(0.0, t.nx - 1.0001)
    fy = ((y - t.origin_t[:, 1:2]) / t.cell).clamp(0.0, t.ny - 1.0001)
    ix, iy = fx.long(), fy.long()
    tx, ty = fx - ix, fy - iy
    gat = lambda a, b: flat.gather(1, a * t.ny + b)                     # noqa: E731
    return ((1 - tx) * (1 - ty) * gat(ix, iy) + tx * (1 - ty) * gat(ix + 1, iy)
            + (1 - tx) * ty * gat(ix, iy + 1) + tx * ty * gat(ix + 1, iy + 1))


def lane_at(z):
    for name in LANES:
        lo, hi = LANE_Z[name]
        if lo <= z <= hi:
            return name
    return None


# --- the vehicle ---------------------------------------------------------------

world = tp.PhysxWorld()

# The rigid ground the lanes are bedded into: what a wheel that leaves the
# terrain falls back onto (the scene query still runs under every wheel, and is
# what the road override stands in for). 3 cm below the lane datum, so driving
# off the end of a lane is a kerb rather than a cliff.
rigid = tp.Mesh(tp.BoxGeometry(400.0, 0.4, 400.0),
                standard_material(0x2a2a26, 0.95))
rigid.position.y = RIGID_Y - 0.2
rigid.receive_shadow = True
world.add_static(rigid)

# tire_friction is the ceiling on the RIGID fallback and on anything the road
# override does not cover. The C++ demo's 2.0 out-grips the geometry -- a 1.65 m
# track under a CoM ~0.85 m up rolls at about 1 g, and 2.0 of mu reaches 2 --
# so this demo runs 1.1 everywhere. On the lanes it does not matter (the
# Mohr-Coulomb override is 0.37-0.75 and slides long before the car tips); on
# the clay strip and the fallback it is the difference between a scrub and a
# roll. Measured: full lock at 50 km/h on clay peaks at 2.2 degrees of tilt.
vehicle = tp.PhysxVehicle(world,
                          chassis_width=1.95, chassis_height=1.4, chassis_length=4.4,
                          chassis_mass=CHASSIS_MASS,
                          wheelbase=2.66, track_width=1.65, wheel_radius=R_WHEEL,
                          driven_wheels=[True, True, True, True],
                          max_throttle_torque=1500.0,
                          tire_friction=FALLBACK_MU,
                          longitudinal_stiffness=100_000.0,
                          suspension_travel=0.3, suspension_stiffness=35_000.0,
                          suspension_damping=4500.0, suspension_attachment_y=-0.4,
                          wheel_damping_rate=1.5,
                          position=SPAWN_POS, rotation=SPAWN_ROT)


def qrot(q, v):
    """Rotate the numpy vec3 `v` by a threepp Quaternion."""
    u = np.array([q.x, q.y, q.z])
    t = 2.0 * np.cross(u, v)
    return v + q.w * t + np.cross(u, t)


# --- scene ---------------------------------------------------------------------

if not GL and not tp.vulkan_available():
    print("vulkan not available on this machine; falling back to OpenGL")
    GL = True

canvas = tp.Canvas("threepp x warp - mud & snow drive", width=WIDTH, height=HEIGHT,
                   antialiasing=AA, vsync=VSYNC, headless=SHOT or BENCH)
renderer = tp.GLRenderer(canvas) if GL else tp.VulkanRenderer(canvas)
if GL:
    renderer.shadow_map_enabled = True
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 1.1

scene = tp.Scene()
scene.background = 0x9fb0c4
# A graded backdrop rather than a horizon line: the lanes run 48 m and the far
# end has to fall away into the sky, or the yard reads as a table top. Global
# fog, not a per-material one -- some Vulkan paths ignore those.
scene.set_fog(0x9fb0c4, 45.0, 190.0)

# Low raking key across the lanes. Berms are centimetres tall; under a high sun
# they vanish, and under a low one every rut wall throws a shadow as long as it
# is deep. The hemisphere is the cold sky fill that keeps the snow from going
# grey in its own shadows.
scene.add(tp.HemisphereLight(0xb9cfe8, 0x4a4438, 0.75))
sun = tp.DirectionalLight(0xffd9a8, 4.0)
sun.position.set(-26.0, 7.0, -34.0)
sun.cast_shadow = True
sun.set_shadow_frustum(-40.0, 40.0, 40.0, -40.0)
sun.set_shadow_bias(-0.0008)
scene.add(sun)

# Mud eats the sunlight: rough AND weak-specular, so the raking key gives it a
# wet sheen in patches instead of a plastic gloss. Snow is the opposite -- bright
# and near-lambertian, so what you read on it is pure shape.
mud_mat = tp.MeshPhysicalMaterial()
mud_mat.color = 0x4a3320
mud_mat.roughness = 0.5
mud_mat.specular_intensity = 0.3
snow_mat = standard_material(0xeef4fb, 0.95)
clay_mat = standard_material(0x6d6152, 0.92)
LANE_MAT = {"mud": mud_mat, "snow": snow_mat, "clay": clay_mat}


# A lane is drawn as a row of STRIPS, not one mesh, and the ground is published
# through the plain attribute path by default. Both are worked around a
# VulkanRenderer defect rather than chosen:
#
#   * enable_vertex_interop loses the device (vkQueueSubmit2 ->
#     VK_ERROR_DEVICE_LOST, on the first frame after arming) on any geometry
#     with more than 65536 vertices. Bisected on this scene: 58,201 vertices
#     arms and renders, 67,821 arms and kills the device, and the same
#     96,681-vertex geometry renders fine on the host attribute route. 65536 is
#     exactly where VulkanCoreGeometry stops packing indices to uint16.
#   * Even under that ceiling the interop route dies once the SNOW lane has been
#     driven on -- mud and clay are stable indefinitely at the same 51,681
#     vertices a strip, and the identical snow drive is fine on the host route.
#     Not with validate=False, not at antialiasing=1, not with fewer frames.
#     Root cause not found inside this phase's budget.
#
# So `--interop` is opt-in and the default is a host copy of only the strips a
# wheel touched (Strip.publish). Reported as renderer work, not worked around in
# C++ -- no renderer changes in this phase. Strips also keep every mesh under
# the 65536 ceiling for whoever fixes the first bug, share their boundary column
# so there is no seam, and take their normals from the FULL grid so the shading
# crosses the joins too.
MAX_INTEROP_VERTS = 60_000


@wp.kernel
def strip_surface(h: wp.array3d(dtype=float),
                  i0: int, ox: float, oy: float, cell: float, z0: float,
                  pos: wp.array(dtype=wp.vec3),
                  nrm: wp.array(dtype=wp.vec3)):
    """terrain_deform's `surface` kernel over one column range of the grid.

    Same y-up map ((x, y, height) -> (x, height, -y), a proper rotation, so the
    index buffer's winding still faces out) and the same central-difference
    normal -- but the difference reads the whole grid while the write is
    strip-local, which is what makes the seam between two strips invisible.
    """
    s, j = wp.tid()
    i = i0 + s
    nx = h.shape[1]
    ny = h.shape[2]
    ia = wp.min(i + 1, nx - 1)
    ib = wp.max(i - 1, 0)
    ja = wp.min(j + 1, ny - 1)
    jb = wp.max(j - 1, 0)
    dhx = (h[0, ia, j] - h[0, ib, j]) / (float(ia - ib) * cell)
    dhy = (h[0, i, ja] - h[0, i, jb]) / (float(ja - jb) * cell)
    n = wp.normalize(wp.vec3(-dhx, -dhy, 1.0))
    t = s * ny + j
    pos[t] = wp.vec3(ox + float(i) * cell, z0 + h[0, i, j], -(oy + float(j) * cell))
    nrm[t] = wp.vec3(n[0], n[2], -n[1])


class Strip:
    """One column range of a lane as a threepp mesh, published device-side."""

    def __init__(self, terrain, material, i0, ncol):
        self.t = terrain
        self.i0 = i0
        self.ncol = ncol
        ny = terrain.ny
        self.n = ncol * ny
        self.ox, self.oy = (float(v) for v in terrain.origin_np[0])
        gi, gj = np.meshgrid(np.arange(i0, i0 + ncol), np.arange(ny), indexing="ij")
        p0 = np.stack([self.ox + gi * terrain.cell,
                       np.full_like(gi, terrain.z0, dtype=np.float32),
                       -(self.oy + gj * terrain.cell)], axis=-1)
        a = ((gi[:-1, :-1] - i0) * ny + gj[:-1, :-1]).ravel()
        faces = np.stack([a, a + ny, a + ny + 1, a, a + ny + 1, a + 1],
                         axis=1).astype(np.uint32)
        self.geometry = tp.BufferGeometry()
        self.geometry.set_attribute("position",
                                    np.ascontiguousarray(p0.reshape(-1, 3), np.float32))
        self.geometry.set_attribute("normal",
                                    np.tile(np.float32([0, 1, 0]), (self.n, 1)))
        self.geometry.set_index(np.ascontiguousarray(faces.reshape(-1), np.uint32))
        self.mesh = tp.Mesh(self.geometry, material)
        self.mesh.frustum_culled = False      # the vertices move under the renderer
        self.mesh.receive_shadow = True
        self.vk = None
        self._host = None
        self.dirty = True
        self.x0 = self.ox + i0 * terrain.cell
        self.x1 = self.ox + (i0 + ncol - 1) * terrain.cell

    def launch(self, pos, nrm):
        wp.launch(strip_surface, dim=(self.ncol, self.t.ny), device=self.t._wp_device,
                  inputs=[self.t.h, self.i0, self.ox, self.oy, self.t.cell, self.t.z0],
                  outputs=[pos, nrm])

    def arm(self, renderer):
        if not INTEROP or VkInteropArray is None                 or not hasattr(renderer, "enable_vertex_interop"):
            return False
        # Grid vertices keep their identity frame to frame, so the renderer's
        # per-vertex motion vectors are real motion: leave stable_correspondence
        # alone (a re-triangulating producer is the case that must opt out).
        h = renderer.enable_vertex_interop(self.mesh, self._on_frame,
                                          validate=not NOSAN)
        if h is None:
            return False
        (pos_h, pos_b), (nrm_h, nrm_b) = h
        try:
            self.vk = (VkInteropArray(pos_h, pos_b, wp.vec3, self.n, device),
                       VkInteropArray(nrm_h, nrm_b, wp.vec3, self.n, device))
        except Exception as e:                          # noqa: BLE001 - fall back
            print(f"  note: CUDA import failed ({e}) -- host route")
            renderer.disable_vertex_interop(self.mesh)
            self.vk = None
            return False
        self._on_frame()
        # Ordered teardown: the CUDA mappings go before the renderer frees the
        # memory they point at.
        atexit.register(self._release, renderer)
        return True

    def _on_frame(self):
        """Inside render(), post-fence and pre-record. The synchronize is
        MANDATORY: host ordering is all that sequences this write against the
        Vulkan frame that reads it."""
        self.launch(self.vk[0].array, self.vk[1].array)
        wp.synchronize_device(device)

    def publish(self):
        """Host route: device -> host -> geometry, only for a dirty strip.

        A height grid only changes where a wheel is, and a strip is 16 m of
        lane: republishing all eight every frame is 32 ms of copy for 4 m of
        change. Marked dirty under a wheel, plus one strip a frame round-robin
        so a slumping berm the car has already left still catches up.
        """
        if self.vk is not None or not self.dirty:
            return
        self.dirty = False
        if self._host is None:
            self._host = (wp.zeros(self.n, dtype=wp.vec3, device=self.t._wp_device),
                          wp.zeros(self.n, dtype=wp.vec3, device=self.t._wp_device))
        self.launch(*self._host)
        self.geometry.update_attribute("position", self._host[0].numpy())
        self.geometry.update_attribute("normal", self._host[1].numpy())

    def _release(self, renderer):
        if self.vk is None:
            return
        pair, self.vk = self.vk, None
        for a in pair:
            a.close()
        renderer.disable_vertex_interop(self.mesh)


def make_strips(name):
    t = terrain[name]
    per = max(2, min(t.nx, MAX_INTEROP_VERTS // t.ny))
    n_strip = int(math.ceil((t.nx - 1) / (per - 1)))
    per = int(math.ceil((t.nx - 1) / n_strip)) + 1
    out = []
    for k in range(n_strip):
        i0 = k * (per - 1)
        out.append(Strip(t, LANE_MAT[name], i0, min(per, t.nx - i0)))
    return out


strips = [s for name in LANES for s in make_strips(name)]
for s in strips:
    scene.add(s.mesh)
print(f"  {len(strips)} lane strips, {max(s.n for s in strips):,} vertices each at most")
_rr = 0


def mark_dirty(hub):
    """Whichever strips a wheel is standing in, plus one more, need a republish."""
    global _rr
    for s in strips:
        for i in range(4):
            if s.x0 - 0.6 <= hub[i, 0] <= s.x1 + 0.6:
                s.dirty = True
                break
    strips[_rr % len(strips)].dirty = True
    _rr += 1

# Speed perception: without something standing still beside the lane, 70 km/h
# over a flat grid reads as 20. Posts along both shoulders and a scatter of
# boulders on the verges, placed off the driving line but close enough to flick
# past. Built once -- adding or removing scene entries mid-drive rebuilds the
# Vulkan descriptor set and drops the TAA history.
rng = np.random.default_rng(7)
post_mat = standard_material(0x2e2b28, 0.85)
rock_mat = standard_material(0x53514c, 0.9)
props = tp.Group()
for x in np.arange(X0 + 2.0, X1, 6.0):
    for z in (LANE_Z["mud"][1] + 0.9, LANE_Z["snow"][0] - 0.9):
        post = tp.Mesh(tp.CylinderGeometry(0.07, 0.09, 1.5, 8), post_mat)
        post.position.set(float(x), 0.7, float(z))
        post.cast_shadow = True
        props.add(post)
for _ in range(46):
    r = float(rng.uniform(0.22, 0.7))
    rock = tp.Mesh(tp.SphereGeometry(r, 7, 5), rock_mat)
    side = rng.integers(0, 2)
    z = float(rng.uniform(13.0, 20.0)) * (1.0 if side else -1.0)
    rock.position.set(float(rng.uniform(X0 - 8.0, X1 + 8.0)), r * 0.35 + RIGID_Y, z)
    rock.rotation.x = float(rng.uniform(0.0, 3.0))
    rock.rotation.y = float(rng.uniform(0.0, 3.0))
    rock.cast_shadow = True
    rock.receive_shadow = True
    props.add(rock)
scene.add(props)
scene.add(rigid)

# --- the rover -----------------------------------------------------------------


def data_dir():
    """threepp_data's checkout. THREEPP_DATA_DIR wins; otherwise the usual
    places -- note the checkout is commonly named with a hyphen."""
    env = os.environ.get("THREEPP_DATA_DIR")
    if env and os.path.isdir(env):
        return env
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in ("threepp-data", "threepp_data"):
        cand = os.path.join(os.path.dirname(repo), name)
        if os.path.isdir(cand):
            return cand
    return ""


chassis = tp.Group()
scene.add(chassis)

# PhysX index -> the model's own wheel tag. The model labels its wheels from
# inside the car, so its "FL" sits at +x, which is where PhysX puts wheel 0
# (front-RIGHT, +x +z). The tags therefore look swapped and are not.
WHEEL_TAG = ("WheelFL", "WheelFR", "WheelBL", "WheelBR")
wheel_rigs = [tp.Group() for _ in range(4)]
for rig in wheel_rigs:
    chassis.add(rig)

brake_mat = reverse_mat = None
model_path = os.path.join(data_dir(), "models", "gltf",
                          "2015_land-rover_range_rover_evoque_coupe", "scene.gltf")
if os.path.isfile(model_path):
    body = tp.ModelLoader().load(model_path)
    body.scale.set(100.0, 100.0, 100.0)     # the gltf bakes a 0.01 at its root
    # The model's contact patches sit at its own y = 0; the PhysX chassis centre
    # rides wheel_radius + |attachment_y| + travel - jounce = 0.996 m above the
    # road. Drop the shell by a metre and the wheel wells land on the rigs.
    body.position.y = -1.0
    chassis.add(body)

    parts = [[] for _ in range(4)]
    lamps = {"lights_position_back": [], "lights_reverse": []}

    def _sort(o):
        if type(o).__name__ != "Mesh":
            return
        # The .gltf packs AO and metal-roughness into ONE texture (R = AO,
        # G = rough, B = metal) and never authored R, so an aoMap read is a
        # constant zero and the whole body goes black. Drop it.
        try:
            o.material.ao_map = None
        except Exception:                               # noqa: BLE001 - not all have one
            pass
        for i, tag in enumerate(WHEEL_TAG):
            if tag in o.name:
                parts[i].append(o)
                return
        for key in lamps:
            if key in o.name:
                lamps[key].append(o)

    body.traverse(_sort)

    for i, group in enumerate(parts):
        if not group:
            continue
        # The per-wheel "pivot" nodes are identity transforms -- the wheel
        # positions are baked into the geometry -- so the hub is the centre of
        # the combined vertex AABB, and a clone recentred on it spins about its
        # own axle under a rig driven by wheel_local_pose(). The wheel is
        # axisymmetric about that axle, which is why the model's own axis
        # convention does not need undoing here.
        lo = np.full(3, 1.0e30)
        hi = np.full(3, -1.0e30)
        for part in group:
            a = part.geometry.get_attribute("position")
            lo = np.minimum(lo, a.min(0))
            hi = np.maximum(hi, a.max(0))
        hub = 0.5 * (lo + hi)
        for part in group:
            clone = part.clone()                # clone BEFORE hiding: copy() takes `visible`
            part.visible = False
            clone.visible = True
            clone.position.set(float(-hub[0]), float(-hub[1]), float(-hub[2]))
            wheel_rigs[i].add(clone)

    # Brake and reverse lamps. The model gives them their own meshes but ONE
    # shared emissive material, so they get a fresh material each and can flare
    # independently.
    for key, colour in (("lights_position_back", 0xff2a12), ("lights_reverse", 0xfff2e0)):
        if not lamps[key]:
            continue
        mat = standard_material(0x201a18, 0.6, emissive=colour)
        mat.emissive_intensity = 1.0
        for m in lamps[key]:
            m.set_material(mat)
        if key == "lights_position_back":
            brake_mat = mat
        else:
            reverse_mat = mat
else:
    print(f"  note: no Evoque at {model_path} -- driving a box "
          f"(set THREEPP_DATA_DIR)")
    shell = tp.Mesh(tp.BoxGeometry(1.95, 1.2, 4.4), standard_material(0x7a2f24, 0.5, 0.3))
    shell.cast_shadow = True
    chassis.add(shell)
    for rig in wheel_rigs:
        w = tp.Mesh(tp.CylinderGeometry(R_WHEEL, R_WHEEL, 0.3, 18), standard_material(0x1c1c1e, 0.8))
        w.rotate_z(math.pi / 2)
        w.cast_shadow = True
        rig.add(w)

# --- cameras -------------------------------------------------------------------

camera = tp.PerspectiveCamera(58, canvas.aspect(), 0.1, 400)
camera.position.set(-30.0, 4.0, 0.0)
pov = tp.PerspectiveCamera(72, canvas.aspect(), 0.05, 400)
pov.position.set(0.4, 0.45, -0.15)
pov.rotation.y = math.pi           # camera looks down -z; chassis forward is +z
chassis.add(pov)

CHASE_DIST, CHASE_PITCH = 9.5, 0.30
cam_pos = np.array([-30.0, 4.0, 0.0])
cam_tgt = np.array([-20.0, 1.0, 0.0])
view_mode = 0                      # 0 chase, 1 POV, 2 cinematic orbit
cine_t = 0.0


def resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    pov.aspect = camera.aspect
    pov.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(resize)

# --- input ---------------------------------------------------------------------

steer_cmd = 0.0
throttle_cmd = 0.0
brake_cmd = 0.0
edge = {}


def pressed(key):
    """True on the frame `key` goes down (canvas.is_key_down is a level)."""
    down = canvas.is_key_down(key)
    was = edge.get(key, False)
    edge[key] = down
    return down and not was


# --- coupling state ------------------------------------------------------------

w_ema = np.full(4, REST_LOAD)
prev_hub_y = np.zeros(4)
radii = torch.full((4,), R_WHEEL, device=device)
centers = {n: torch.zeros(1, 4, 3, device=device) for n in LANES}
vels = {n: torch.zeros(1, 4, 3, device=device) for n in LANES}
c_host = {n: np.zeros((1, 4, 3), np.float32) for n in LANES}
v_host = {n: np.zeros((1, 4, 3), np.float32) for n in LANES}
hud = dict(lane=[None] * 4, z=np.zeros(4), mu=np.zeros(4), w=np.zeros(4),
           bek=np.zeros(4), slip=np.zeros(4), util=np.zeros(4), j=np.zeros(4),
           over=[False] * 4)
_readback = 0


def coupled_step(throttle, steer, brake, relax_iters=2):
    """One 1/60 step: ground under the wheels, then PhysX over the top."""
    global w_ema, prev_hub_y, _readback

    # 1. chassis pose + wheel world centres
    p, q = vehicle.position, vehicle.quaternion
    origin = np.array([p.x, p.y, p.z])
    hub = np.empty((4, 3))
    for i in range(4):
        lp, _ = vehicle.wheel_local_pose(i)
        hub[i] = origin + qrot(q, np.array([lp.x, lp.y, lp.z]))
    fwd = qrot(q, np.array([0.0, 0.0, 1.0]))
    right = qrot(q, np.array([1.0, 0.0, 0.0]))
    v_fwd = vehicle.forward_speed

    # 2. the GRADE under each wheel -- the bearing datum, not the surface. One
    #    gather per lane and one transfer for all of them.
    lane_of = [lane_at(hub[i, 2]) for i in range(4)]
    grade = np.zeros(4)
    gathered = []
    for name in LANES:
        idx = [i for i in range(4) if lane_of[i] == name]
        if not idx:
            continue
        t = terrain[name]
        xs = torch.tensor([[hub[i, 0] for i in idx]], device=device, dtype=torch.float32)
        ys = torch.tensor([[-hub[i, 2] for i in idx]], device=device, dtype=torch.float32)
        gathered.append((idx, bilinear(t, t.grade_torch.view(1, t.nx * t.ny), xs, ys)))
    if gathered:
        flat = torch.cat([g[1][0] for g in gathered]).cpu().numpy()
        k = 0
        for idx, _ in gathered:
            for i in idx:
                grade[i] = flat[k]
                k += 1

    # 3. wheel load, smoothed. The road-height/load loop has a frame of lag in
    #    it and rings at the suspension frequency without this.
    load = np.array([vehicle.suspension_force(i) for i in range(4)])
    w_ema += (load - w_ema) * (DT / (0.15 + DT))

    # 4/5. invert Bekker for the sinkage, Mohr-Coulomb for the friction, and
    #      hand the suspension a road at grade - z_eq.
    z_eq = np.zeros(4)
    mu = np.zeros(4)
    for i in range(4):
        if lane_of[i] is None:
            vehicle.clear_road_override(i)
            continue
        z_eq[i], mu[i] = sinkage(lane_of[i], w_ema[i])
        vehicle.set_road_override(i, float(grade[i] - z_eq[i]), float(mu[i]))

    # 6. carve. The wheels are the module's collider spheres; their bottoms are
    #    already at grade - z_eq, so the imprint cuts the rut to exactly the
    #    Bekker sinkage and the berm is whatever the material would not compact.
    #    Velocity is the CONTACT PATCH slip, not the hub velocity: that is what
    #    the Janosi integral is measured along.
    for name in LANES:
        c, v = c_host[name], v_host[name]
        for i in range(4):
            if lane_of[i] != name:
                c[0, i] = (0.0, -LANE_Z[name][1], 50.0)     # parked well clear
                v[0, i] = (0.0, 0.0, 0.0)
                continue
            c[0, i] = (hub[i, 0], -hub[i, 2], hub[i, 1])
            slip_long = vehicle.wheel_angular_speed(i) * R_WHEEL - v_fwd
            slip_lat = vehicle.tire_lateral_slip(i) * abs(v_fwd)
            s = slip_long * fwd + slip_lat * right
            v[0, i] = (s[0], -s[2], (hub[i, 1] - prev_hub_y[i]) / DT)
        centers[name].copy_(torch.from_numpy(c))
        vels[name].copy_(torch.from_numpy(v))
        terrain[name].deform(centers[name], radii, vels[name], DT)
        terrain[name].relax(relax_iters)
    prev_hub_y = hub[:, 1].copy()

    # 7. bulldozing drag -- the one soil force actually applied (see the audit
    #    at the top of the file). Opposes the wheel's horizontal travel.
    speed_h = math.hypot(fwd[0], fwd[2]) * abs(v_fwd)
    for i in range(4):
        if lane_of[i] is None:
            continue
        f = bulldoze(lane_of[i], z_eq[i])
        if f <= 0.0 or speed_h < 0.15:
            continue
        d = np.array([fwd[0], 0.0, fwd[2]]) * (1.0 if v_fwd >= 0 else -1.0)
        d /= max(np.linalg.norm(d), 1e-6)
        vehicle.add_force_at_pos(tp.Vector3(float(-d[0] * f), 0.0, float(-d[2] * f)),
                                 tp.Vector3(*hub[i]))

    # 8. and PhysX runs the car.
    vehicle.set_throttle(throttle)
    vehicle.set_steer(steer)
    vehicle.set_brake(brake)
    world.step(DT)

    # Panel telemetry. One small transfer every third frame: the module's own
    # Bekker bearing and Janosi shear next to what PhysX's suspension is
    # carrying. Nothing in the loop reads these back.
    _readback += 1
    if _readback % 3 == 0:
        pack = torch.stack([torch.cat([terrain[n].forces[0, :, 0:3].reshape(-1),
                                       terrain[n].slip_j[0]]) for n in LANES]).cpu().numpy()
        for i in range(4):
            n = lane_of[i]
            hud["bek"][i] = 0.0 if n is None else pack[LANES.index(n)][i * 3 + 2]
            hud["j"][i] = 0.0 if n is None else pack[LANES.index(n)][12 + i]
            if n is not None:
                ft = math.hypot(pack[LANES.index(n)][i * 3], pack[LANES.index(n)][i * 3 + 1])
                hud["util"][i] = ft / max(mu[i] * w_ema[i], 1.0)
    hud["lane"] = lane_of
    hud["z"] = z_eq
    hud["mu"] = mu
    hud["w"] = load
    hud["slip"] = np.array([vehicle.tire_longitudinal_slip(i) for i in range(4)])
    hud["over"] = [vehicle.road_override_active(i) for i in range(4)]
    return hub


# --- UI ------------------------------------------------------------------------

ui = tp.ImguiContext(canvas, renderer) if not (SHOT or BENCH) else None
fps_ema = 60.0
shot_no = 0
WHEEL_NAME = ("FR", "FL", "RR", "RL")


def bar(v, n=14):
    v = max(0.0, min(1.0, v))
    k = int(round(v * n))
    return "[" + "#" * k + "." * (n - k) + "]"


def draw_ui():
    tp.imgui.set_next_window_pos(8, 8)
    tp.imgui.set_next_window_size(376, 0)
    tp.imgui.begin("Bekker-Wong drive")
    tp.imgui.text(f"{vehicle.forward_speed * 3.6:6.1f} km/h   "
                  f"{'REVERSE' if vehicle.gear == tp.PhysxVehicle.Gear.REVERSE else 'FORWARD'}"
                  f"   {fps_ema:5.1f} fps")
    tp.imgui.text(f"throttle {bar(throttle_cmd)}  brake {bar(brake_cmd)}")
    tp.imgui.text(f"steer    {bar(0.5 + 0.5 * steer_cmd)}")
    tp.imgui.separator()
    tp.imgui.text("wh lane  sink   load   mu   slip  util   j")
    for i in range(4):
        n = hud["lane"][i]
        tp.imgui.text(f"{WHEEL_NAME[i]} {(n or 'rigid'):5s} "
                      f"{hud['z'][i] * 1000:5.1f} {hud['w'][i] / 1000:6.2f} "
                      f"{(hud['mu'][i] if n else FALLBACK_MU):5.2f} "
                      f"{hud['slip'][i]:6.2f} {hud['util'][i]:5.2f} "
                      f"{hud['j'][i] * 1000:5.1f}")
    tp.imgui.text("           mm     kN               -    mm")
    tp.imgui.separator()
    tp.imgui.text("soil bearing vs PhysX suspension load")
    for i in range(4):
        if hud["lane"][i] is None:
            tp.imgui.text(f"{WHEEL_NAME[i]}  -- off the lanes (rigid ground) --")
            continue
        err = 100.0 * (hud["bek"][i] / max(hud["w"][i], 1.0) - 1.0)
        tp.imgui.text(f"{WHEEL_NAME[i]} Bekker {hud['bek'][i]:7.0f} N   "
                      f"PhysX {hud['w'][i]:7.0f} N   {err:+5.1f} %")
    tp.imgui.separator()
    tp.imgui.text("W/S drive  A/D steer  R gear  SPACE handbrake")
    tp.imgui.text("V pov  C cinematic  BACKSPACE respawn  T reset  F frame")
    tp.imgui.end()


def save_shot(path):
    if GL:
        renderer.render(scene, camera)
        renderer.save_frame(path)
    else:
        renderer.save_frame(scene, active_camera(), path)
    print(f"  wrote {path}")


def active_camera():
    return pov if view_mode == 1 else camera


# --- frame ---------------------------------------------------------------------


def drive_inputs(dt):
    """Keyboard -> commands, with main.cpp's speed-sensitive steer and slew."""
    global steer_cmd, throttle_cmd, brake_cmd, view_mode, cine_t, shot_no
    left = canvas.is_key_down("A") or canvas.is_key_down("LEFT")
    rightk = canvas.is_key_down("D") or canvas.is_key_down("RIGHT")
    steer_in = (1.0 if left else 0.0) - (1.0 if rightk else 0.0)
    scale = 1.0 / (1.0 + abs(vehicle.forward_speed) * 3.6 * 0.015)
    steer_cmd += (steer_in * scale - steer_cmd) * min(1.0, dt * 2.0)
    throttle_cmd = 1.0 if (canvas.is_key_down("W") or canvas.is_key_down("UP")) else 0.0
    brake_cmd = 1.0 if (canvas.is_key_down("S") or canvas.is_key_down("DOWN")
                        or canvas.is_key_down("SPACE")) else 0.0

    if pressed("R"):
        vehicle.gear = (tp.PhysxVehicle.Gear.REVERSE
                        if vehicle.gear == tp.PhysxVehicle.Gear.FORWARD
                        else tp.PhysxVehicle.Gear.FORWARD)
    if pressed("V"):
        view_mode = 0 if view_mode == 1 else 1
    if pressed("C"):
        view_mode = 0 if view_mode == 2 else 2
        cine_t = 0.0
    if pressed("BACKSPACE"):
        vehicle.respawn(SPAWN_POS, SPAWN_ROT)
        vehicle.gear = tp.PhysxVehicle.Gear.FORWARD
        steer_cmd = 0.0
        w_ema[:] = REST_LOAD
    if pressed("T"):
        for t in terrain.values():
            t.reset()
    if pressed("F"):
        shot_no += 1
        save_shot(os.path.join(OUT_DIR, f"warp_mudsnow_drive_{shot_no:03d}.png"))


def update_camera(dt):
    """main.cpp's chase: exponential lerp toward a chassis-relative offset, read
    off the interpolated visual rather than the raw actor pose."""
    global cam_pos, cam_tgt, cine_t
    p, q = vehicle.position, vehicle.quaternion
    c = np.array([p.x, p.y, p.z])
    if view_mode == 2:
        cine_t += dt
        a = cine_t * 0.16
        want = c + np.array([math.cos(a) * 13.0, 3.6 + 1.2 * math.sin(a * 0.7),
                             math.sin(a) * 13.0])
        want_t = c + np.array([0.0, 0.4, 0.0])
    else:
        off = qrot(q, np.array([0.0, CHASE_DIST * math.sin(CHASE_PITCH),
                                -CHASE_DIST * math.cos(CHASE_PITCH)]))
        want = c + off
        want_t = c + qrot(q, np.array([0.0, 0.9, 3.0]))
    k = 1.0 - math.exp(-(2.0 if view_mode == 2 else 5.0) * dt)
    cam_pos += (want - cam_pos) * k
    cam_tgt += (want_t - cam_tgt) * k
    camera.position.set(*cam_pos)
    camera.look_at(tp.Vector3(*cam_tgt))


brake_was = reverse_was = False


def frame():
    global fps_ema, brake_was, reverse_was
    t0 = time.perf_counter()
    drive_inputs(DT)
    mark_dirty(coupled_step(throttle_cmd, steer_cmd, brake_cmd))

    p, q = vehicle.position, vehicle.quaternion
    chassis.position.set(p.x, p.y, p.z)
    chassis.quaternion.set(q.x, q.y, q.z, q.w)
    for i in range(4):
        lp, lq = vehicle.wheel_local_pose(i)
        wheel_rigs[i].position.set(lp.x, lp.y, lp.z)
        wheel_rigs[i].quaternion.set(lq.x, lq.y, lq.z, lq.w)

    # Lamps: only touch the material when the state flips -- needs_update()
    # re-uploads it.
    on = brake_cmd > 0.05
    if brake_mat is not None and on != brake_was:
        brake_was = on
        brake_mat.emissive_intensity = 6.0 if on else 1.0
        brake_mat.needs_update()
    rev = vehicle.gear == tp.PhysxVehicle.Gear.REVERSE
    if reverse_mat is not None and rev != reverse_was:
        reverse_was = rev
        reverse_mat.emissive_intensity = 6.0 if rev else 1.0
        reverse_mat.needs_update()

    update_camera(DT)
    for s in strips:
        s.publish()
    renderer.render(scene, active_camera())
    if ui is not None:
        ui.render(draw_ui)
    fps_ema += (1.0 / max(time.perf_counter() - t0, 1e-4) - fps_ema) * 0.05


# The first render is what creates the records the vertex exports come from.
renderer.render(scene, camera)
if not GL:
    armed = sum(s.arm(renderer) for s in strips)
    print(f"lane mesh route: "
          f"{'zero-copy CUDA -> Vulkan' if armed == len(strips) else 'mixed / host copy'}"
          f" ({armed}/{len(strips)} armed)")
for s in strips:
    s.publish()

def pose_visuals():
    p, q = vehicle.position, vehicle.quaternion
    chassis.position.set(p.x, p.y, p.z)
    chassis.quaternion.set(q.x, q.y, q.z, q.w)
    for k in range(4):
        lp, lq = vehicle.wheel_local_pose(k)
        wheel_rigs[k].position.set(lp.x, lp.y, lp.z)
        wheel_rigs[k].quaternion.set(lq.x, lq.y, lq.z, lq.w)


def scripted(spawn_z, beats, eye, look, path, note=""):
    """Drive a fixed input script, park a camera, save the frame.

    The shots the acceptance asks for have to be repeatable and identical
    between lanes -- the same line at the same speed is the whole comparison --
    which a hand-driven pass is not. Each beat is (seconds, throttle, steer).
    """
    vehicle.respawn(tp.Vector3(-20.0, 1.05, spawn_z), SPAWN_ROT)
    vehicle.gear = tp.PhysxVehicle.Gear.FORWARD
    w_ema[:] = REST_LOAD
    for _ in range(60):
        coupled_step(0.0, 0.0, 0.0)
    for secs, thr, st in beats:
        for _ in range(int(round(secs * 60.0))):
            mark_dirty(coupled_step(thr, st, 0.0))
    pose_visuals()
    for s in strips:
        s.dirty = True
        s.publish()
    camera.position.set(*eye)
    camera.look_at(tp.Vector3(*look))
    # The Vulkan pipeline is temporal (probe GI, denoisers, the upscaler): one
    # render after the sim would capture frame ONE of all of them.
    for _ in range(0 if GL else WARM):
        renderer.render(scene, camera)
    save_shot(path)
    print(f"  {os.path.basename(path)}: v={vehicle.forward_speed * 3.6:5.1f} km/h  "
          f"lane={hud['lane'][0]}  sink={np.round(hud['z'] * 1000, 1)} mm  "
          f"mu={np.round(hud['mu'], 2)}  slip={np.round(hud['slip'], 2)}  "
          f"util={np.round(hud['util'], 2)}  "
          f"Bekker/PhysX={np.round(100 * (hud['bek'] / np.maximum(hud['w'], 1.0) - 1.0), 1)} % "
          f"{note}")


if BENCH:
    # Honest wall clock: the whole frame the interactive loop runs, vsync off.
    for _ in range(60):
        mark_dirty(coupled_step(0.5, 0.15, 0.0))
        pose_visuals()
        for s in strips:
            s.publish()
        renderer.render(scene, camera)
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(240):
        mark_dirty(coupled_step(0.5, 0.15, 0.0))
        pose_visuals()
        for s in strips:
            s.publish()
        renderer.render(scene, camera)
    wp.synchronize_device(device)
    ms = (time.perf_counter() - t0) * 1000.0 / 240.0
    t1 = time.perf_counter()
    for _ in range(240):
        coupled_step(0.5, 0.15, 0.0)
    wp.synchronize_device(device)
    sim_ms = (time.perf_counter() - t1) * 1000.0 / 240.0
    print(f"bench [{'opengl' if GL else 'vulkan'}, "
          f"{'zero-copy interop' if INTEROP else 'host copy, dirty strips'}]: "
          f"{ms:.2f} ms/frame ({1000.0 / ms:.0f} fps), of which sim+coupling "
          f"{sim_ms:.2f} ms; {cells:,} cells, {len(strips)} strips")
elif SHOT:
    which = cli_arg("--script", "lanes", str)
    z_mud = 0.5 * (LANE_Z["mud"][0] + LANE_Z["mud"][1])
    z_snow = 0.5 * (LANE_Z["snow"][0] + LANE_Z["snow"][1])
    out = lambda n: os.path.join(OUT_DIR, f"warp_mudsnow_drive_{n}.png")   # noqa: E731

    # One script per process: VulkanRenderer.save_frame is a render + readback
    # and taking two of them in one run loses the device on the second.
    if which in ("mud", "snow"):
        # Same line, same throttle, both soft lanes: mud walls its trench,
        # snow swallows the wheel and barely rims.
        z = z_mud if which == "mud" else z_snow
        scripted(z, [(3.2, 0.45, 0.0)],
                 (-13.5, 1.15, z + (5.2 if z < 0 else -5.2)), (-15.5, -0.1, z),
                 out(f"1_{which}_rut"))
    if which in ("spin_mud", "spin_clay"):
        # Acceptance 2: the same full-throttle launch, mud vs the clay strip.
        z = z_mud if which == "spin_mud" else 0.0
        scripted(z, [(2.0, 1.0, 0.0)],
                 (-17.0, 1.5, z + (5.0 if z <= 0 else -5.0)), (-18.5, 0.2, z),
                 out(f"2_{which}"))
    if which == "lap":
        # Acceptance 3: a circle in mud, twice. Full lock at a walking pace
        # turns inside the lane; the readout is the depth of the wheel below
        # the LOCAL surface, which is a fresh sinkage on lap 1 and almost
        # nothing on lap 2 because the grade under the rut has not moved.
        def rut_depth():
            p = vehicle.position
            lp, _ = vehicle.wheel_local_pose(3)
            hubw = np.array([p.x, p.y, p.z]) + qrot(vehicle.quaternion,
                                                    np.array([lp.x, lp.y, lp.z]))
            t = terrain["mud"]
            x = torch.tensor([[hubw[0]]], device=device, dtype=torch.float32)
            y = torch.tensor([[-hubw[2]]], device=device, dtype=torch.float32)
            surf = float(t.heights(x, y)[0, 0])
            return (surf - (hubw[1] - R_WHEEL)) * 1000.0

        vehicle.respawn(tp.Vector3(-14.0, 1.05, z_mud), SPAWN_ROT)
        w_ema[:] = REST_LOAD
        for _ in range(60):
            coupled_step(0.0, 0.0, 0.0)
        lap = []
        for k in range(int(24.0 * 60)):
            mark_dirty(coupled_step(0.30, 1.0, 0.0))
            if k % 30 == 0:
                lap.append((k / 60.0, rut_depth(), float(hud["z"][3] * 1000.0)))
        pose_visuals()
        for s in strips:
            s.dirty = True
            s.publish()
        camera.position.set(-14.0, 6.0, z_mud - 6.5)
        camera.look_at(tp.Vector3(-15.5, -0.1, z_mud))
        for _ in range(0 if GL else WARM):
            renderer.render(scene, camera)
        save_shot(out("3_second_lap"))
        print("  circle in mud: t, wheel below LOCAL surface (mm), Bekker z_eq (mm)")
        for t_s, d, z in lap[::4]:
            print(f"    {t_s:5.1f}s  {d:6.1f}  {z:6.1f}")
else:
    canvas.animate(frame)
