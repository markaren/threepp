"""How does a robot see in 3D? From depth pixels to a map.  Part 2 of the threepp lessons.

    python depth_map.py --out D:/dev/lesson_out/depth_map.mp4       # the film (1080p60)
    python depth_map.py --stills 10,25,60 --outdir D:/dev/lesson_out  # individual frames
    python depth_map.py --sheet --outdir D:/dev/lesson_out            # contact sheet
    python depth_map.py --preview --out preview.mp4                   # 960x540 @ 30 fps

A depth camera on the FR3's hand scans five objects on a tray. Everything shown is
computed by threepp while the film is made: the depth images come from
`tp.DepthSensor` (with its range-noise model), the camera is placed by `tp.IkSolver`
(the camera pose is forward kinematics, which is the link back to Part 1), the map is
a `tp.VoxelGrid`, and the surface is `tp.marching_cubes` over the fused cloud. The
objects are primitives on purpose: their exact distance functions let the film
measure the reconstruction error against the true shapes instead of guessing it.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from lesson import (Cloud, Film, Hud, Keys, Segments, Stage, Timeline, Tube3D,  # noqa: E402
                    data_dir, ease_out, ease_out_back, envelope, remap, shrink, smooth, smoother, standard, tp,
                    turbo, turbo_hex, write_png)

FPS = 60
DOF = 7
FINGERS = 0.035
Q_READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
Q_AWAY = [2.75, -0.7, 0.0, -2.6, 0.0, 1.9, 0.785]      # arm parked behind its base for the map beats

C_TEXT = 0xf2f5fa
C_DIM = 0x9fb2cc
C_CAM = 0x4cc9f0
C_ACC = 0xffb347
C_OK = 0x5ee27a

# ── the set ───────────────────────────────────────────────────────────────────
C = np.array([0.50, 0.0, 0.0])     # tray centre (robot frame, z up)
TRAY_R, TRAY_H = 0.21, 0.03
Z0 = TRAY_H
OBJECTS = [   # kind, centre, params, colour
    ("sphere", (0.44, 0.08, Z0 + 0.05), dict(r=0.05), 0xf78c6b),
    ("box", (0.57, 0.07, Z0 + 0.04), dict(h=0.04, yaw=0.45), 0x5b8cff),
    ("can", (0.53, -0.09, Z0 + 0.06), dict(r=0.035, hh=0.06), 0x2ec4b6),
    ("torus", (0.41, -0.07, Z0 + 0.017), dict(R=0.05, r=0.017), 0xef476f),
    ("capsule", (0.64, -0.01, Z0 + 0.025), dict(r=0.025, hl=0.04), 0xf4d35e),
]
ROI_R = 0.30                       # what the film shows of each scan
MAP_R = TRAY_R - 0.004             # what goes into the map: the tray and what is on it

# ── the sensor ────────────────────────────────────────────────────────────────
SW, SH, FOV = 128, 96, 45.0
NEAR, FAR = 0.08, 1.5
NOISE = 0.003
CAM_OFF = np.array([0.0, 0.065, -0.045])   # camera on the hand's flank, in the tool frame
D_IMG = 0.16                                # where the film draws the image plane
FY = (SH / 2) / math.tan(math.radians(FOV) / 2)
FX = FY
CX, CY = SW / 2, SH / 2
ZMIN, ZMAX = 0.30, 0.72                     # colour range of the depth image
VOXEL = 0.01
ERR_MAX = 0.005                             # error colour scale
ERR_T0 = 0.16                               # error colours start at bright blue, not black
AIM = C + np.array([0.0, 0.0, 0.06])

# (azimuth deg about the tray, radius, height): the first is the single view
VIEWS = [(240, 0.33, 0.40), (262, 0.33, 0.36), (282, 0.30, 0.34), (300, 0.16, 0.50), (0, 0.06, 0.52),
         (60, 0.16, 0.50), (80, 0.30, 0.34), (100, 0.33, 0.36), (122, 0.33, 0.40), (140, 0.30, 0.42)]

# ── the script ────────────────────────────────────────────────────────────────
TL = Timeline()
TL.add("open", 0.0, 7.0)
TL.add("sensor", 7.0, 20.0)
TL.add("back", 20.0, 34.0)
TL.add("one", 34.0, 45.0)
TL.add("scan", 45.0, 68.0)
TL.add("voxel", 68.0, 80.0)
TL.add("surface", 80.0, 95.0)
TL.add("outro", 95.0, 103.0)

T_SCAN0 = 9.2          # the first scan fires
T_FLY = 23.6           # pixels leave the image plane
FLY_STAGGER, FLY_DUR = 3.0, 1.3
SEG0, SEG_DUR, MOVE = 45.6, 2.3, 1.55
T_RETREAT = (67.6, 70.6)
T_VOX = (69.0, 75.5)
T_SURF = (80.4, 83.4)
T_HEAT = (86.2, 89.4)


# ── kinematics ────────────────────────────────────────────────────────────────
def full(q7):
    return list(map(float, q7[:DOF])) + [FINGERS, FINGERS]


def mat4(M):
    m = tp.Matrix4()
    m.set(*[float(v) for v in np.asarray(M, float).reshape(-1)])
    return m


def view_point(az, rho, h):
    a = math.radians(az)
    return C + np.array([rho * math.cos(a), rho * math.sin(a), h])


def world_to_robot(P):
    P = np.asarray(P, np.float64)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], axis=-1)


# ── ground truth ──────────────────────────────────────────────────────────────
def sdf(P):
    """Exact signed distance to the tray and the five objects (robot frame, metres)."""
    P = np.asarray(P, np.float64)
    ds = []

    def cyl(Q, r, hh):
        dr = np.linalg.norm(Q[:, :2], axis=1) - r
        dz = np.abs(Q[:, 2]) - hh
        return np.minimum(np.maximum(dr, dz), 0) + np.hypot(np.maximum(dr, 0), np.maximum(dz, 0))

    for kind, c, p, _ in OBJECTS:
        Q = P - np.asarray(c)
        if kind == "sphere":
            ds.append(np.linalg.norm(Q, axis=1) - p["r"])
        elif kind == "box":
            cs, sn = math.cos(-p["yaw"]), math.sin(-p["yaw"])
            Q = np.stack([cs * Q[:, 0] - sn * Q[:, 1], sn * Q[:, 0] + cs * Q[:, 1], Q[:, 2]], 1)
            qd = np.abs(Q) - p["h"]
            ds.append(np.linalg.norm(np.maximum(qd, 0), axis=1) + np.minimum(qd.max(1), 0))
        elif kind == "can":
            ds.append(cyl(Q, p["r"], p["hh"]))
        elif kind == "torus":
            ds.append(np.hypot(np.linalg.norm(Q[:, :2], axis=1) - p["R"], Q[:, 2]) - p["r"])
        elif kind == "capsule":
            Q = Q.copy()
            Q[:, 0] -= np.clip(Q[:, 0], -p["hl"], p["hl"])
            ds.append(np.linalg.norm(Q, axis=1) - p["r"])
    ds.append(cyl(P - np.array([C[0], C[1], TRAY_H / 2]), TRAY_R, TRAY_H / 2))
    return np.min(np.stack(ds, 1), 1)


# ── fusion ────────────────────────────────────────────────────────────────────
TSDF_CELL = 0.003
TSDF_TRUNC = 0.012
TSDF_ROI_PAD = 0.001
TSDF_BLUR = True


def tsdf_fuse(scans):
    """Truncated signed distance fusion (KinectFusion style) of the scans' depth images.

    Every voxel is projected into each depth image; the measured depth minus the
    voxel's depth, clipped to +-TSDF_TRUNC, is one vote, and the votes are averaged.
    Returns (F as (nz, ny, nx), origin).
    """
    # the lowest node sits half a cell below the floor, so closing the grid's bottom
    # puts that face at z = 0, where the tray really ends
    lo = np.array([C[0] - TRAY_R - 0.01, C[1] - TRAY_R - 0.01, -0.5 * TSDF_CELL])
    hi = np.array([C[0] + TRAY_R + 0.01, C[1] + TRAY_R + 0.01, 0.17])
    n = np.ceil((hi - lo) / TSDF_CELL).astype(int) + 1
    zs, ys, xs = np.meshgrid(np.arange(n[2]), np.arange(n[1]), np.arange(n[0]), indexing="ij")
    P = lo + TSDF_CELL * np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    F = np.ones(len(P), np.float32)
    Wt = np.zeros(len(P), np.float32)
    inside_roi = np.linalg.norm(P[:, :2] - C[:2], axis=1) < TRAY_R + TSDF_ROI_PAD
    for s in scans:
        D = np.full((SH, SW), np.inf, np.float32)
        np.minimum.at(D, (s.vi, s.ui), s.depth)
        Pc = (np.linalg.inv(s.pose) @ np.c_[P, np.ones(len(P))].T).T
        z = -Pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.floor(CX + FX * Pc[:, 0] / z).astype(np.int64)
            v = np.floor(CY - FY * Pc[:, 1] / z).astype(np.int64)
        ok = (z > NEAR) & (u >= 0) & (u < SW) & (v >= 0) & (v < SH) & inside_roi
        d = np.full(len(P), np.inf, np.float32)
        d[ok] = D[v[ok], u[ok]]
        sdf = d - z
        ok &= np.isfinite(sdf) & (sdf > -TSDF_TRUNC)
        f = np.clip(sdf[ok] / TSDF_TRUNC, -1.0, 1.0)
        F[ok] = (F[ok] * Wt[ok] + f) / (Wt[ok] + 1.0)
        Wt[ok] += 1.0
    # Space no view saw is treated as occupied inside the tray region (it is what
    # lies behind the surfaces) and as empty outside it. Marking it empty instead
    # grows a second sheet one truncation distance behind every surface.
    F[(Wt == 0) & inside_roi] = -1.0
    F[(Wt == 0) & ~inside_roi] = 1.0
    F = F.reshape(n[2], n[1], n[0])
    # a light [1 2 1] blur along each axis takes the sensor's grain off the surface
    for ax in range(3 if TSDF_BLUR else 0):
        a = np.moveaxis(F, ax, 0)
        b = a.copy()
        b[1:-1] = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
        F = np.moveaxis(b, 0, ax)
    F[0] = 1.0            # close the bottom of the grid (it lies under the floor)
    return np.ascontiguousarray(F, np.float32), lo


# ── build ─────────────────────────────────────────────────────────────────────
def build(width, height):
    st = Stage(width, height, renderer="gl")
    robot = tp.URDFLoader().load(os.path.join(data_dir(), "urdf", "franka", "fr3.urdf"))
    robot.show_colliders(False)
    robot.traverse(lambda o: setattr(o, "cast_shadow", True))
    st.zup.add(robot)

    objs = tp.Group()
    st.zup.add(objs)
    tray = tp.Mesh(tp.CylinderGeometry(TRAY_R, TRAY_R, TRAY_H, 128), standard(0x252a33, 0.55))
    tray.rotate_x(math.pi / 2)
    tray.position.set(C[0], C[1], TRAY_H / 2)
    meshes = [tray]
    for kind, c, p, col in OBJECTS:
        mat = standard(col, 0.32)
        if kind == "sphere":
            m = tp.Mesh(tp.SphereGeometry(p["r"], 64, 40), mat)
        elif kind == "box":
            m = tp.Mesh(tp.BoxGeometry(2 * p["h"], 2 * p["h"], 2 * p["h"]), mat)
            m.rotate_z(p["yaw"])
        elif kind == "can":
            m = tp.Mesh(tp.CylinderGeometry(p["r"], p["r"], 2 * p["hh"], 64), mat)
            m.rotate_x(math.pi / 2)
        elif kind == "torus":
            m = tp.Mesh(tp.TorusGeometry(p["R"], p["r"], 32, 96), mat)
        else:
            m = tp.Mesh(tp.CapsuleGeometry(p["r"], 2 * p["hl"], 16, 32), mat)
            m.rotate_z(math.pi / 2)
        m.position.set(*c)
        meshes.append(m)
    for m in meshes:
        m.cast_shadow = True
        m.receive_shadow = True
        objs.add(m)
    for m in meshes:
        m.material.transparent = True   # GL keys the program on this: set it once, fade by opacity

    tcp = robot.get_object_by_name("fr3_hand_tcp")
    # the camera's body: a dark bar with a glowing lens ring, facing along the tool axis
    body = tp.Group()
    bar = tp.Mesh(tp.BoxGeometry(0.09, 0.026, 0.03), standard(0x1a1e25, 0.5))
    body.add(bar)
    for dx in (-0.025, 0.025):
        lens = tp.Mesh(tp.CylinderGeometry(0.008, 0.008, 0.004, 32), standard(0x05070a, 0.2))
        lens.rotate_x(math.pi / 2)
        lens.position.set(dx, 0, 0.016)
        body.add(lens)
        ring = tp.Mesh(tp.TorusGeometry(0.009, 0.0012, 12, 48), standard(C_CAM, 0.3, emissive=C_CAM,
                                                                           emissive_intensity=2.0))
        ring.position.set(dx, 0, 0.0175)
        body.add(ring)
    body.position.set(*(CAM_OFF - np.array([0, 0, 0.012])))
    tcp.add(body)

    sensor = tp.DepthSensor(fov_y=FOV, width=SW, height=SH, near=NEAR, far=FAR)
    sensor.range_noise = NOISE
    sensor.position.set(*CAM_OFF)
    sensor.rotate_x(math.pi)           # the sensor looks down its -Z, the tool down its +Z
    tcp.add(sensor)
    robot.set_end_effector("fr3_hand_tcp")
    return st, robot, objs, meshes, sensor


# ── choreography ──────────────────────────────────────────────────────────────
class Scan:
    pass


class Record:
    def __init__(self, n):
        self.n = n
        self.q = np.zeros((n, DOF))
        self.meta = {}


def choreograph(st, robot, sensor, verbose=True):
    n = int(round(TL.duration * FPS)) + 1
    rec = Record(n)
    T = lambda i: i / FPS  # noqa: E731
    idx = lambda t: int(round(t * FPS))  # noqa: E731

    off = np.eye(4)
    off[:3, 3] = CAM_OFF
    opts = tp.IkOptions()
    opts.task = tp.IkTask.AxisAlign        # aim the view axis; the camera's roll is free
    opts.max_iterations = 200
    opts.position_tolerance = 1e-4
    opts.orientation_tolerance = 1e-3
    opts.tool_offset = mat4(off)           # solve for the CAMERA, not the fingertips
    opts.tool_axis = tp.Vector3(0, 0, 1)
    opts.rest_pose = full(Q_READY)
    opts.rest_pose_gain = 0.05

    def aim(q, p, iters=1):
        d = AIM - p
        d /= np.linalg.norm(d)
        o = opts
        o.target_axis = tp.Vector3(*map(float, d))
        s = tp.IkSolver(robot, o)
        r = None
        for _ in range(iters):
            q, r = s.solve(q, tp.Vector3(*map(float, p)))
        return q, r

    # the first view
    q0, r0 = aim(full(Q_READY), view_point(*VIEWS[0]), iters=6)
    worst = r0.position_error

    # open: READY -> first view (joint space), then hold
    for i in range(0, idx(SEG0)):
        u = smoother(remap(T(i), 1.2, 6.4))
        rec.q[i] = np.array(Q_READY) * (1 - u) + np.array(q0[:DOF]) * u

    # scan: tracked moves between views, a scan at each arrival
    scan_times = [T_SCAN0]
    q = q0
    for k in range(1, len(VIEWS)):
        a = SEG0 + (k - 1) * SEG_DUR
        for i in range(idx(a), idx(a + SEG_DUR)):
            u = smoother(remap(T(i), a, a + MOVE))
            v = [VIEWS[k - 1][j] * (1 - u) + VIEWS[k][j] * u for j in range(3)]
            q, r = aim(q, view_point(*v), iters=1 if u < 1 else 3)
            rec.q[i] = q[:DOF]
            if u >= 1:
                worst = max(worst, r.position_error)
        scan_times.append(a + MOVE + 0.05)
    t_last = SEG0 + (len(VIEWS) - 1) * SEG_DUR
    q_last = np.array(q[:DOF])
    for i in range(idx(t_last), n):
        u = smoother(remap(T(i), *T_RETREAT))
        rec.q[i] = q_last * (1 - u) + np.array(Q_AWAY) * u
    rec.meta["scan_times"] = scan_times
    if verbose:
        print(f"[ik] camera placed on all {len(VIEWS)} views, worst position error {worst * 1000:.2f} mm")

    # ---- the scans, taken for real
    zinv = None
    sensor.noise.seed = 7
    sensor.reset_noise()
    scans = []
    for k, ts in enumerate(scan_times):
        robot.set_joint_values(full(rec.q[idx(ts)]))
        st.scene.update_matrix_world()
        if zinv is None:
            zinv = np.linalg.inv(st.zup.matrix_world.to_numpy())
        robot.visible = False                        # the arm does not scan itself
        pw, cols = sensor.scan_rgbd(st.r, st.scene)
        robot.visible = True
        s = Scan()
        s.t = ts
        s.pose = zinv @ sensor.matrix_world.to_numpy()          # sensor frame -> robot frame
        s.origin = s.pose[:3, 3].copy()
        s.pts = world_to_robot(pw).astype(np.float32)
        s.rgb = np.asarray(cols, np.float32)
        # pixel coordinates: project each hit back through the pinhole model
        Pc = (np.linalg.inv(s.pose) @ np.c_[s.pts, np.ones(len(s.pts))].T).T[:, :3]
        z = -Pc[:, 2]
        s.depth = z.astype(np.float32)
        u = CX + FX * Pc[:, 0] / z
        v = CY - FY * Pc[:, 1] / z
        s.ui = np.clip(np.floor(u).astype(int), 0, SW - 1)
        s.vi = np.clip(np.floor(v).astype(int), 0, SH - 1)
        s.heat = turbo(1.0 - (z - ZMIN) / (ZMAX - ZMIN)).astype(np.float32)   # near = warm
        # where each point sits on the (drawn) image plane, in the robot frame
        pc = np.c_[(s.ui + 0.5 - CX) / FX * D_IMG, -(s.vi + 0.5 - CY) / FY * D_IMG,
                   -np.full(len(z), D_IMG), np.ones(len(z))]
        s.on_plane = (s.pose @ pc.T).T[:, :3].astype(np.float32)
        r_h = np.linalg.norm(s.pts[:, :2] - C[:2], axis=1)
        s.roi = (r_h < ROI_R) & (s.pts[:, 2] > -0.01)
        s.mapped = (r_h < MAP_R) & (s.pts[:, 2] > 0.005)
        img = np.zeros((SH, SW, 4), np.uint8)
        img[s.vi, s.ui, :3] = (s.heat * 255).astype(np.uint8)
        img[s.vi, s.ui, 3] = 255
        s.image = img
        scans.append(s)
    rec.scans = scans
    s0 = scans[0]
    rec.meta["pixels"] = SW * SH
    rec.meta["hits0"] = len(s0.pts)
    if verbose:
        print(f"[scan] {len(scans)} scans, first {len(s0.pts)} points of {SW * SH} pixels")

    # the highlighted pixel: the one that lands nearest the top of the sphere's lit side
    target = np.array(OBJECTS[0][1]) + np.array([-0.02, -0.03, 0.03])
    h = int(np.argmin(np.linalg.norm(s0.pts - target, axis=1)))
    rec.meta["hi"] = h

    # ---- the map
    M = np.concatenate([s.pts[s.mapped] for s in scans])
    Mrgb = np.concatenate([s.rgb[s.mapped] for s in scans])
    rec.map_pts, rec.map_rgb = M, Mrgb
    grid = tp.VoxelGrid(VOXEL, max_points_per_voxel=3, min_spacing=0.004)
    grid.insert_array(M)
    centers = np.asarray(grid.collect_voxel_centers(), np.float64).reshape(-1, 3)
    # colour each voxel with the mean colour of the points that fell in it
    key = np.floor(M / VOXEL).astype(np.int64)
    ckey = np.floor(centers / VOXEL).astype(np.int64)
    kv = {}
    for kk, c in zip(map(tuple, key), Mrgb):
        acc = kv.get(kk)
        if acc is None:
            kv[kk] = [c.copy(), 1]
        else:
            acc[0] += c
            acc[1] += 1
    vcol = np.array([kv[tuple(k)][0] / kv[tuple(k)][1] if tuple(k) in kv else (0.6, 0.6, 0.6) for k in ckey],
                    np.float32)
    rec.vox_c, rec.vox_col = centers.astype(np.float32), vcol
    rec.meta["voxels"] = grid.voxel_count

    F, origin = tsdf_fuse(scans)
    # marching cubes wants "inside" high: the TSDF is negative behind the surface
    field = tp.ScalarField.from_numpy(-F, tp.Vector3(*map(float, origin)), TSDF_CELL)
    iso = tp.marching_cubes(field, 0.0)
    rec.iso = iso
    V = np.asarray(iso.positions, np.float32).reshape(-1, 3)
    err = np.abs(sdf(V))
    rec.surf_err = err.astype(np.float32)
    rec.meta["err_mean"] = float(err.mean())
    rec.meta["err_p95"] = float(np.percentile(err, 95))
    rec.meta["cloud_err"] = float(np.mean(np.abs(sdf(M))))
    rec.meta["tris"] = len(V) // 3
    rec.meta["map_points"] = len(M)
    if verbose:
        print(f"[map] {len(M)} points on the tray -> {grid.voxel_count} voxels ({VOXEL * 100:.0f} cm) -> "
              f"{len(V) // 3} triangles; cloud error {rec.meta['cloud_err'] * 1000:.2f} mm, "
              f"surface error mean {err.mean() * 1000:.2f} mm, p95 {np.percentile(err, 95) * 1000:.2f} mm")
    return rec


# ── the picture ───────────────────────────────────────────────────────────────
UNIT_CUBE = None


def cube_arrays():
    """24 vertices (flat faces), normals and 36 indices of a unit cube centred at 0."""
    faces = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
             ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]
    pos, nrm, idx = [], [], []
    for n, a, b in faces:
        n, a, b = np.array(n, float), np.array(a, float), np.array(b, float)
        k = len(pos)
        for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pos.append(0.5 * (n + sa * a + sb * b))
            nrm.append(n)
        idx += [k, k + 1, k + 2, k, k + 2, k + 3]
    return np.array(pos, np.float32), np.array(nrm, np.float32), np.array(idx, np.uint32)


def frustum_edges(depth):
    hw = (SW / 2) / FX * depth
    hh = (SH / 2) / FY * depth
    corners = [np.array([sx * hw, sy * hh, -depth]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    o = np.zeros(3)
    return [[o, c] for c in corners] + [[corners[i], corners[(i + 1) % 4]] for i in range(4)]


class Film3D:
    def __init__(self, st, robot, objs, meshes, sensor, rec):
        self.st, self.robot, self.objs, self.meshes, self.sensor, self.rec = st, robot, objs, meshes, sensor, rec
        root = st.zup
        # frustum + image plane, riding on the sensor
        self.frustum = [Tube3D(C_CAM, radius=0.0011, parent=sensor, emissive=1.6) for _ in range(8)]
        for tb, (a, b) in zip(self.frustum, frustum_edges(D_IMG)):
            tb.set_points([a, b], opacity=0.0)
        s0 = rec.scans[0]
        self.img = s0.image.copy()
        self.tex = tp.data_texture(np.flipud(self.img).copy(), True)
        for t in (self.tex,):
            t.mag_filter = tp.Filter.Nearest
            t.min_filter = tp.Filter.Nearest
            t.generate_mipmaps = False
        pm = tp.MeshBasicMaterial()
        pm.map = self.tex
        pm.transparent = True
        pm.side = tp.Side.Double
        pm.depth_write = False
        pm.tone_mapped = False
        self.plane_mat = pm
        wpl, hpl = SW / FX * D_IMG, SH / FY * D_IMG
        self.plane = tp.Mesh(tp.PlaneGeometry(wpl, hpl), pm)
        self.plane.position.set(0, 0, -D_IMG)
        sensor.add(self.plane)
        self.hi_mark = tp.Mesh(tp.PlaneGeometry(wpl / SW * 1.6, hpl / SH * 1.6),
                               standard(0xffffff, 0.3, emissive=0xffffff, emissive_intensity=2.0))
        self.hi_mark.material.side = tp.Side.Double
        sensor.add(self.hi_mark)
        self.hi_ray = Tube3D(0xffffff, radius=0.0012, parent=root, emissive=2.0)

        # rays, clouds
        self.rays = Segments(SW * SH, parent=root)
        self.cloud0 = Cloud(SW * SH, size=0.0042, parent=root)
        total = sum(len(s.pts) for s in rec.scans)
        self.cloud = Cloud(total, size=0.0036, parent=root)
        # frustum ghosts at every scan pose
        self.ghosts = []
        for s in rec.scans:
            g = tp.Group()
            tubes = [Tube3D(C_CAM, radius=0.0008, parent=g, emissive=1.2) for _ in range(8)]
            for tb, (a, b) in zip(tubes, frustum_edges(0.06)):
                tb.set_points([a, b], opacity=1.0)
                tb.mat.transparent = True
                tb.mesh.visible = False
            g.matrix_auto_update = False
            m = tp.Matrix4()
            m.set(*s.pose.reshape(-1))
            g.position.set(*s.pose[:3, 3])
            R = s.pose[:3, :3]
            w = math.sqrt(max(0.0, 1.0 + np.trace(R))) / 2
            g.matrix_auto_update = True
            g.quaternion.set((R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                             (R[1, 0] - R[0, 1]) / (4 * w), w)
            root.add(g)
            self.ghosts.append((g, tubes))

        # voxels: one mesh of cubes, rebuilt while they pop in
        pos, nrm, idx = cube_arrays()
        self.cube_pos, nV = pos, len(rec.vox_c)
        self.vox_geom = tp.BufferGeometry()
        self.vox_geom.set_attribute("position", np.zeros((nV * 24, 3), np.float32))
        self.vox_geom.set_attribute("normal", np.tile(nrm, (nV, 1)))
        self.vox_geom.set_attribute("color", np.repeat(rec.vox_col, 24, axis=0))
        self.vox_geom.set_index((idx[None, :] + 24 * np.arange(nV, dtype=np.uint32)[:, None]).reshape(-1))
        vm = standard(0xffffff, 0.55)
        vm.vertex_colors = True
        self.vox_mat = vm
        self.voxels = tp.Mesh(self.vox_geom, vm)
        self.voxels.frustum_culled = False
        self.voxels.cast_shadow = True
        self.voxels.receive_shadow = True
        root.add(self.voxels)
        z = rec.vox_c[:, 2]
        rng = np.random.default_rng(3)
        order = (z - z.min()) / max(np.ptp(z), 1e-6) + rng.uniform(0, 0.12, len(z))
        self.vox_start = T_VOX[0] + (T_VOX[1] - T_VOX[0] - 0.8) * order / order.max()
        self._vox_state = None

        # the surface
        geom = tp.iso_mesh_to_geometry(rec.iso)
        V = np.asarray(rec.iso.positions, np.float32).reshape(-1, 3)
        self.surf_z = V[:, 2]
        self.surf_base = np.tile(np.array([0.80, 0.85, 0.93], np.float32), (len(V), 1))
        self.surf_heat = turbo(ERR_T0 + (1 - ERR_T0) * np.clip(rec.surf_err / ERR_MAX, 0, 1)).astype(np.float32)
        geom.set_attribute("color", self.surf_base.copy())
        self.surf_geom = geom
        sm = standard(0xffffff, 0.5)
        sm.vertex_colors = True
        sm.side = tp.Side.Double
        sm.transparent = True
        self.surf_mat = sm
        self.surface = tp.Mesh(geom, sm)
        self.surface.cast_shadow = True
        self.surface.receive_shadow = True
        root.add(self.surface)
        self._heat_k = None

    def pose(self, q7):
        self.robot.set_joint_values(full(q7))
        self.st.scene.update_matrix_world()


# camera: (time, azimuth deg, elevation deg, distance m, look point in the ROBOT frame, fov)
CAM = [
    (0.0, -70, 18, 2.45, (-0.34, -0.22, 0.26), 30),
    (5.6, -62, 20, 2.30, (-0.26, -0.18, 0.24), 30),
    (8.5, -58, 20, 1.30, (0.42, -0.10, 0.22), 30),
    (19.5, -66, 20, 1.20, (0.43, -0.10, 0.20), 30),
    (22.0, -72, 18, 1.05, (0.44, -0.08, 0.18), 30),
    (33.5, -80, 18, 1.00, (0.46, -0.06, 0.14), 30),
    (38.5, 10, 30, 1.05, (0.50, 0.0, 0.08), 30),
    (44.5, 30, 32, 1.05, (0.50, 0.0, 0.08), 30),
    (47.5, -40, 32, 1.95, (0.40, 0.0, 0.22), 30),
    (67.0, -62, 30, 1.90, (0.42, 0.0, 0.20), 30),
    (70.0, -122, 50, 1.02, (0.56, 0.0, 0.0), 30),
    (80.0, -114, 55, 1.00, (0.56, 0.0, -0.01), 30),
    (94.5, -100, 55, 0.98, (0.56, 0.0, -0.01), 30),
    (103.0, -96, 50, 1.25, (0.56, 0.0, 0.05), 30),
]


def camera_at(t):
    az, el, dist, fov = (float(v) for v in Keys([(k[0], [k[1], k[2], k[3], k[5]]) for k in CAM])(t))
    look = Keys([(k[0], k[4]) for k in CAM])(t)
    az += 0.8 * math.sin(0.23 * t)
    el += 0.4 * math.sin(0.17 * t + 1.0)
    a, e = math.radians(az), math.radians(el)
    eye = look + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    zw = lambda p: np.array([p[0], p[2], -p[1]])  # noqa: E731
    return zw(eye), zw(look), fov


CAPTIONS = [
    (7.4, 13.4, "A depth camera measures one thing per pixel: how far away the surface is."),
    (13.8, 19.6, None),   # pixel count, filled in
    (20.4, 27.0, "Each pixel is a ray through the lens. Its depth says how far along that ray the surface is."),
    (27.4, 33.6, None),   # points from one image
    (34.4, 39.8, "But one view only sees the surfaces facing the camera."),
    (40.2, 44.6, "Behind every object, there is simply no data."),
    (45.4, 52.0, "So move the camera and scan again. The arm knows exactly where the camera is: forward kinematics, from Part 1."),
    (52.4, 59.2, "Every scan lands in the same frame, and the missing sides fill in."),
    (59.6, 67.4, None),   # total points
    (68.4, 74.4, "A raw cloud is huge and redundant. Bin it into a voxel grid: one cube per occupied centimetre."),
    (74.8, 79.6, None),   # points -> voxels
    (80.4, 86.0, "Fuse the depth images into a signed-distance grid, then pull out the surface with marching cubes."),
    (86.4, 94.6, None),   # error
]


class Painter:
    def __init__(self, st, f3: Film3D, rec, ov: Hud):
        self.st, self.f3, self.rec, self.ov = st, f3, rec, ov
        m = rec.meta
        fill = {
            1: f"Here {SW} \u00d7 {SH} pixels, each one a distance. Near is warm, far is cool.",
            3: f"Pixel plus depth gives a 3D point. One image: {m['hits0']:,} points.",
            8: f"{len(rec.scans)} views, {m['map_points']:,} points on the tray.",
            10: f"{m['map_points']:,} points become {m['voxels']:,} voxels.",
            12: f"Checked against the true shapes: {m['err_mean'] * 1000:.1f} mm off on average, from a sensor with "
                f"{NOISE * 1000:.0f} mm of noise. Fusing many views averages it away.",
        }
        self.captions = [(a, b, fill.get(k, txt)) for k, (a, b, txt) in enumerate(CAPTIONS)]
        # accumulated cloud order: scan by scan, ROI only
        self.acc_pts = [s.pts[s.roi] for s in rec.scans]
        self.acc_rgb = [s.rgb[s.roi] for s in rec.scans]
        self.acc_heat = [s.heat[s.roi] for s in rec.scans]
        self.acc_plane = [s.on_plane[s.roi] for s in rec.scans]
        self.acc_origin = [s.origin for s in rec.scans]
        self.idx = lambda t: min(int(round(t * FPS)), rec.n - 1)  # noqa: E731

    # ---- 3D ----------------------------------------------------------------
    def set3d(self, t, i):
        rec, f3 = self.rec, self.f3
        f3.pose(rec.q[i])
        eye, look, fov = camera_at(t)
        self.st.look(eye, look, fov)
        s0 = rec.scans[0]
        stimes = rec.meta["scan_times"]

        # objects: ghosted while the cloud is the subject, gone once the surface stands in
        ghost = envelope(t, 35.2, 67.5, 1.2, 1.4)
        gone = smooth(remap(t, 69.0, 71.0))
        obj_op = (1.0 - 0.86 * ghost) * (1.0 - gone)
        for m in f3.meshes[1:]:
            m.material.opacity = obj_op
            m.material.depth_write = obj_op >= 0.999
            m.visible = obj_op > 0.003
        tray_op = 1.0 - gone     # the real tray leaves too: the map has its own
        f3.meshes[0].material.opacity = tray_op
        f3.meshes[0].material.depth_write = tray_op >= 0.999
        f3.meshes[0].visible = tray_op > 0.003

        # frustum
        fr = max(envelope(t, 7.6, TL.end("back") - 0.4, 0.8, 0.6), envelope(t, SEG0 - 0.4, 67.4, 0.4, 0.6) * 0.7)
        grow = ease_out(remap(t, 7.6, 8.6)) if t < 9 else 1.0
        for tb, (a, b) in zip(f3.frustum, frustum_edges(D_IMG * max(grow, 0.02))):
            tb.set_points([a, b], opacity=fr)

        # image plane: paints row by row after the scan, pixels leave during the flight
        pl = envelope(t, 10.2, T_FLY + FLY_STAGGER + FLY_DUR + 0.4, 0.3, 0.4)
        if pl > 0.003:
            reveal = remap(t, 10.4, 12.4)
            img = s0.image.copy()
            rows = int(round(reveal * SH))
            img[rows:, :, 3] = 0
            if t > T_FLY:
                sv = self.flight(t, s0.vi)
                gone_px = sv > 0.02
                img[s0.vi[gone_px], s0.ui[gone_px], 3] = 0
            f3.tex.update_data(np.flipud(img).copy())
            f3.plane_mat.opacity = pl
            f3.plane.visible = True
        else:
            f3.plane.visible = False

        # the highlighted pixel and its ray
        h = rec.meta["hi"]
        hv = envelope(t, 21.0, 28.6, 0.4, 0.6)
        if hv > 0.003:
            u, v = s0.ui[h] + 0.5, s0.vi[h] + 0.5
            f3.hi_mark.position.set((u - CX) / FX * D_IMG, -(v - CY) / FY * D_IMG, -D_IMG + 0.0005)
            f3.hi_mark.visible = True
            g = ease_out(remap(t, 21.6, 22.8))
            a = s0.origin
            b = a + (s0.pts[h] - a) * max(g, 0.02)
            f3.hi_ray.set_points([a, b], opacity=hv)
        else:
            f3.hi_mark.visible = False
            f3.hi_ray.mesh.visible = False

        # rays: the first scan's fan, and a burst at every later scan
        ray_a, ray_b, ray_c, ray_op = [], [], [], 0.0
        sub0 = (s0.ui % 4 == 2) & (s0.vi % 4 == 2)
        if 8.9 <= t <= 12.0:
            g = remap(t, T_SCAN0, T_SCAN0 + 0.7)
            stag = np.clip((g * 1.6 - s0.vi[sub0] / SH * 0.6), 0, 1)
            ray_a.append(np.repeat(s0.origin[None], sub0.sum(), 0))
            ray_b.append(s0.origin + (s0.pts[sub0] - s0.origin) * ease_out(stag)[:, None])
            ray_c.append(s0.heat[sub0])
            ray_op = envelope(t, T_SCAN0, 12.0, 0.1, 1.2) * 0.9
        for k in range(1, len(rec.scans)):
            ts = stimes[k]
            if ts - 0.05 <= t <= ts + 0.9:
                s = rec.scans[k]
                sub = (s.ui % 5 == 2) & (s.vi % 5 == 2) & s.roi
                g = ease_out(remap(t, ts, ts + 0.3))
                ray_a.append(np.repeat(s.origin[None], sub.sum(), 0))
                ray_b.append(s.origin + (s.pts[sub] - s.origin) * g)
                ray_c.append(s.heat[sub])
                ray_op = max(ray_op, envelope(t, ts, ts + 0.9, 0.05, 0.6) * 0.8)
        if ray_a:
            A, B, Cc = np.concatenate(ray_a), np.concatenate(ray_b), np.concatenate(ray_c)
            f3.rays.set(A, B, Cc * 0.25, Cc, opacity=ray_op)
        else:
            f3.rays.set(np.zeros((0, 3)), None, None, opacity=0.0)

        # the first cloud: pixels fly off the image plane to their 3D points
        c0 = envelope(t, T_FLY, SEG0 + 0.4, 0.01, 0.4)
        if c0 > 0.003:
            sv = self.flight(t, s0.vi)
            moving = sv > 0.0
            P = s0.on_plane + (s0.pts - s0.on_plane) * sv[:, None]
            show = moving & (s0.roi | (t < 34.5))
            # outside the region of interest the floor fades out after the one-view beat
            f3.cloud0.set(P[show], s0.heat[show], opacity=c0)
        else:
            f3.cloud0.set(np.zeros((0, 3)), None, 0.0)

        # accumulated cloud (scans 1..): points shoot out along their rays, turbo -> true colour
        acc_op = envelope(t, stimes[1] - 0.1, 72.6, 0.05, 1.4) if len(stimes) > 1 else 0.0
        vox_fade = 1.0 - smooth(remap(t, 70.4, 72.4))
        if acc_op > 0.003:
            Ps, Cs = [], []
            # scan 0 joins the map once the tour starts, in true colour
            if t >= SEG0:
                k0 = smooth(remap(t, SEG0, SEG0 + 1.0))
                Ps.append(self.acc_pts[0])
                Cs.append(self.acc_heat[0] * (1 - k0) + self.acc_rgb[0] * k0)
            for k in range(1, len(rec.scans)):
                ts = stimes[k]
                if t < ts:
                    break
                g = ease_out(remap(t, ts + 0.05, ts + 0.45))
                o = self.acc_origin[k]
                Ps.append(o + (self.acc_pts[k] - o) * max(g, 0.02))
                kc = smooth(remap(t, ts + 0.5, ts + 1.3))
                Cs.append(self.acc_heat[k] * (1 - kc) + self.acc_rgb[k] * kc)
            if Ps:
                f3.cloud.set(np.concatenate(Ps), np.concatenate(Cs), opacity=acc_op * vox_fade)
            else:
                f3.cloud.set(np.zeros((0, 3)), None, 0.0)
        else:
            f3.cloud.set(np.zeros((0, 3)), None, 0.0)
        if t >= SEG0:
            f3.cloud0.set(np.zeros((0, 3)), None, 0.0)

        # frustum ghosts at every pose already scanned from
        for k, (g, tubes) in enumerate(f3.ghosts):
            op = envelope(t, stimes[k] + 0.2, 69.5, 0.3, 0.8) * 0.55 if t >= SEG0 - 0.5 else 0.0
            for tb in tubes:
                tb.mesh.visible = op > 0.003
                tb.mat.opacity = op
            g.visible = op > 0.003

        # voxels
        vx = envelope(t, T_VOX[0], T_SURF[1], 0.01, 1.2)
        if vx > 0.003:
            sc = ease_out_back((t - f3.vox_start) / 0.55, 1.8).astype(np.float32)
            shrink_k = 1.0 - smooth(remap(t, T_SURF[0] + 0.4, T_SURF[1]))
            sc = sc * 0.88 * shrink_k
            state = (round(t * FPS),)
            if state != f3._vox_state:
                pos = (self.rec.vox_c[:, None, :] + f3.cube_pos[None] * (VOXEL * sc)[:, None, None]).reshape(-1, 3)
                f3.vox_geom.update_attribute("position", pos.astype(np.float32))
                f3._vox_state = state
            f3.voxels.visible = True
        else:
            f3.voxels.visible = False

        # surface: fades in as the cubes melt, then the error sweeps up from the tray
        sf = smooth(remap(t, T_SURF[0], T_SURF[1]))
        if sf > 0.003:
            f3.surface.visible = True
            f3.surf_mat.opacity = sf
            f3.surf_mat.depth_write = sf >= 0.999
            sweep = -0.02 + 0.20 * smoother(remap(t, *T_HEAT))
            k = np.clip((sweep - f3.surf_z) / 0.012, 0, 1)[:, None]
            key = round(float(sweep), 4)
            if key != f3._heat_k:
                col = f3.surf_base * (1 - k) + f3.surf_heat * k
                f3.surf_geom.update_attribute("color", col.astype(np.float32))
                f3._heat_k = key
        else:
            f3.surface.visible = False

    def flight(self, t, vi):
        """0..1 flight progress of each pixel: rows leave the image plane top to bottom."""
        return smoother((t - (T_FLY + FLY_STAGGER * (vi / SH))) / FLY_DUR)

    # ---- 2D ----------------------------------------------------------------
    def draw2d(self, t, i):
        ov, st, rec = self.ov, self.st, self.rec
        m = rec.meta
        s0 = rec.scans[0]

        a = envelope(t, 0.9, 6.4, 0.9, 0.8)
        if a > 0:
            y = 360 + 14 * (1 - ease_out(remap(t, 0.9, 2.2)))
            ov.text(128, y - 58, "A THREEPP LESSON  \u00b7  PART 2", size=22, color=C_ACC, alpha=a, kind="semibold",
                    tracking=4)
            ov.text(122, y, "How a Robot Sees in 3D", size=88, color=C_TEXT, alpha=a, kind="semibold")
            ov.text(128, y + 112, "From depth pixels to a map", size=38, color=C_DIM,
                    alpha=a * smooth(remap(t, 1.6, 2.6)), kind="light")

        for (a0, b0, txt) in self.captions:
            al = envelope(t, a0, b0, 0.45, 0.4)
            if al > 0:
                ov.caption(txt, al)

        self.equations(t)

        # the depth image panel
        dp = envelope(t, 10.6, T_FLY + FLY_STAGGER + FLY_DUR, 0.6, 0.6)
        if dp > 0:
            x, y, w = 1446, 64, 410
            iw, ih = 352, 264
            ov.panel(x, y, w, ih + 150, radius=16, alpha=0.66 * dp, outline=0x8aa0c0, outline_alpha=0.16)
            ov.text(x + 24, y + 22, f"DEPTH IMAGE   {SW} \u00d7 {SH}", size=16, color=C_DIM, alpha=dp,
                    kind="semibold", tracking=2.2)
            ix, iy = x + (w - iw) / 2, y + 56
            ov.panel(ix - 2, iy - 2, iw + 4, ih + 4, radius=4, fill=0x000000, alpha=0.9 * dp)
            ov.image(ix, iy, iw, ih, self.f3.tex, dp)
            ov.colorbar(ix, iy + ih + 22, iw, 10, lambda u: turbo_hex(1.0 - u), dp)
            ov.text(ix, iy + ih + 50, f"{ZMIN:.2f} m", size=16, color=C_DIM, alpha=dp, kind="numeric", anchor="la")
            ov.text(ix + iw, iy + ih + 50, f"{ZMAX:.2f} m", size=16, color=C_DIM, alpha=dp, kind="numeric",
                    anchor="ra")
            ov.text(ix + iw / 2, iy + ih + 50, "depth", size=16, color=C_DIM, alpha=dp, anchor="ma")
            # the highlighted pixel, in the panel and in the scene
            h = m["hi"]
            hv = envelope(t, 21.0, 28.6, 0.4, 0.6)
            if hv > 0:
                px = ix + (s0.ui[h] + 0.5) / SW * iw
                py = iy + (s0.vi[h] + 0.5) / SH * ih
                cell = iw / SW
                ov.panel(px - cell * 1.6, py - cell * 1.6, cell * 3.2, cell * 3.2, radius=2, fill=0xffffff,
                         alpha=0.0, outline=0xffffff, outline_alpha=hv * 0.72 / 0.72, width=2.0)
                ov.text(px, py - 22, f"(u, v) = ({s0.ui[h]}, {s0.vi[h]})", size=18, color=C_TEXT, alpha=hv,
                        kind="numeric", anchor="ms")
                g = ease_out(remap(t, 21.6, 22.8))
                if g > 0.95:
                    p3 = st.project_zup(s0.pts[h])
                    ov.circle(p3[0], p3[1], 6, fill=0xffffff, alpha=hv)
                    ov.text(p3[0] - 16, p3[1] + 30, f"z = {s0.depth[h]:.3f} m", size=24, color=C_TEXT, alpha=hv,
                            kind="numeric", anchor="rm")

        self.stats(t)
        self.error_bar(t)
        self.outro(t)

    def equations(self, t):
        ov = self.ov
        x, y = 70, 64
        eqs = [
            (20.8, 33.8, r"$\mathbf{p} = z\,K^{-1}\,(u,\ v,\ 1)^{T}$", "back-projection"),
            (22.6, 33.8, r"$x = \frac{(u - c_x)\,z}{f_x},\quad y = \frac{(v - c_y)\,z}{f_y}$", None),
            (52.6, 67.4, r"$\mathbf{p}_{\mathrm{world}} = T_{\mathrm{camera}}(\mathbf{q})\ \mathbf{p}$",
             "every scan in one frame"),
            (68.8, 79.8, r"$\mathbf{i} = \lfloor\,\mathbf{p}\,/\,s\,\rfloor$", "voxel index"),
            (80.8, 94.6, r"$F \leftarrow \frac{W\,F + f}{W + 1}$", "signed-distance fusion"),
        ]
        rows = [e for e in eqs if e[0] - 0.1 <= t <= e[1] + 0.6]
        if not rows:
            return
        sizes = [ov.math_size(tex, 40) for _, _, tex, _ in rows]
        w = max(max(s[0] for s in sizes) + 60, 470)
        h = sum(s[1] + (30 if note else 12) for s, (_, _, _, note) in zip(sizes, rows)) + 40
        pa = max(envelope(t, a, b, 0.6, 0.6) for a, b, _, _ in rows)
        ov.panel(x, y, w, h, radius=16, alpha=0.66 * pa, outline=0x8aa0c0, outline_alpha=0.16)
        yy = y + 22
        for (a, b, tex, note), (sw, sh) in zip(rows, sizes):
            al = envelope(t, a, b, 0.6, 0.6)
            if note:
                ov.text(x + 30, yy, note.upper(), size=16, color=C_DIM, alpha=al, kind="semibold", tracking=2.2)
                yy += 24
            ov.math(x + 30, yy + sh / 2, tex, size=40, color=C_TEXT, alpha=al, anchor="lm")
            yy += sh + 16

    def stats(self, t):
        ov, rec = self.ov, self.rec
        m = rec.meta
        a = envelope(t, SEG0 - 0.2, TL.end("surface") - 0.2, 0.6, 0.6)
        if a <= 0:
            return
        stimes = m["scan_times"]
        k = sum(1 for ts in stimes if t >= ts + 0.05)
        pts = sum(int(s.mapped.sum()) for s, ts in zip(rec.scans, stimes) if t >= ts + 0.05)
        rows = [("SCANS", f"{k} / {len(stimes)}", True), ("POINTS ON THE TRAY", f"{pts:,}", True)]
        vox_on = t >= T_VOX[0]
        rows.append(("VOXELS  (1 cm)", f"{m['voxels']:,}" if vox_on else "\u2013", vox_on))
        surf_on = t >= T_SURF[0]
        rows.append(("SURFACE TRIANGLES", f"{m['tris']:,}" if surf_on else "\u2013", surf_on))
        x, y, w = 1488, 64, 368
        rh = 62
        ov.panel(x, y, w, 30 + rh * len(rows), radius=16, alpha=0.66 * a, outline=0x8aa0c0, outline_alpha=0.16)
        for j, (lab, val, on) in enumerate(rows):
            yy = y + 24 + j * rh
            ov.text(x + 24, yy, lab, size=15, color=C_DIM, alpha=a * (1.0 if on else 0.5), kind="semibold",
                    tracking=2.0)
            ov.text(x + 24, yy + 44, val, size=32, color=C_TEXT, alpha=a * (1.0 if on else 0.4), kind="numeric",
                    anchor="ls")

    def error_bar(self, t):
        ov, m = self.ov, self.rec.meta
        a = envelope(t, T_HEAT[0], TL.end("surface") - 0.2, 0.6, 0.6)
        if a <= 0:
            return
        x, y, w = 1488, 356, 368
        ov.panel(x, y, w, 196, radius=16, alpha=0.66 * a, outline=0x8aa0c0, outline_alpha=0.16)
        ov.text(x + 24, y + 22, "DISTANCE TO THE TRUE SHAPES", size=15, color=C_DIM, alpha=a, kind="semibold",
                tracking=2.0)
        ov.colorbar(x + 24, y + 58, w - 48, 12, lambda u: turbo_hex(ERR_T0 + (1 - ERR_T0) * u), a)
        ov.text(x + 24, y + 88, "0", size=16, color=C_DIM, alpha=a, kind="numeric", anchor="la")
        ov.text(x + w - 24, y + 88, f"{ERR_MAX * 1000:.0f} mm", size=16, color=C_DIM, alpha=a, kind="numeric", anchor="ra")
        ov.text(x + 24, y + 164, f"{m['err_mean'] * 1000:.1f} mm", size=38, color=C_TEXT, alpha=a, kind="numeric",
                anchor="ls")
        ov.text(x + w - 24, y + 164, "mean", size=18, color=C_DIM, alpha=a, anchor="rs")

    def outro(self, t):
        ov = self.ov
        a = envelope(t, TL.start("outro") + 0.2, TL.end("outro") + 1, 0.8, 0.1)
        if a <= 0:
            return
        W, H = ov.W, ov.H
        ov.panel(-20, -20, W + 40, H + 40, radius=0, fill=0x070a10, alpha=0.8 * a)
        x = 250
        ov.text(x, 250, "IN SHORT", size=22, color=C_ACC, alpha=a, kind="semibold", tracking=4)
        rows = [
            (r"$(u,\,v,\,z)\ \rightarrow\ \mathbf{p}$", "A depth pixel is a 3D point."),
            (r"$T(\mathbf{q})$", "Knowing the camera pose puts every scan in one map."),
            (r"$\lfloor\mathbf{p}/s\rfloor$", "Voxels compress it, and a surface makes it usable."),
        ]
        for k, (tex, txt) in enumerate(rows):
            al = a * smooth(remap(t, TL.start("outro") + 0.6 + 0.5 * k, TL.start("outro") + 1.3 + 0.5 * k))
            yy = 350 + k * 110
            ov.math(x, yy, tex, size=46, color=C_TEXT, alpha=al, anchor="lm")
            ov.text(x + 340, yy, txt, size=38, color=C_TEXT, alpha=al, anchor="lm")
        al = a * smooth(remap(t, TL.start("outro") + 2.4, TL.start("outro") + 3.2))
        ov.text(x, 780, "Scanned, mapped and rendered with threepp", size=28, color=C_DIM, alpha=al, kind="semibold")
        ov.text(x, 824, "github.com/markaren/threepp", size=26, color=C_CAM, alpha=al)


def fade_amount(t):
    return 1.0 - min(smooth(remap(t, 0.0, 0.8)), 1.0 - smooth(remap(t, TL.duration - 0.9, TL.duration)))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/dev/lesson_out/depth_map.mp4")
    ap.add_argument("--outdir", default="D:/dev/lesson_out")
    ap.add_argument("--stills", default=None, help="comma-separated times (s)")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--every", type=float, default=3.0)
    ap.add_argument("--preview", action="store_true", help="960x540 @ 30 fps")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--fps", type=int, default=None)
    args = ap.parse_args()

    W, H = (960, 540) if args.preview else (1920, 1080)
    fps = args.fps or (30 if args.preview else FPS)
    st, robot, objs, meshes, sensor = build(W, H)
    t0 = time.time()
    rec = choreograph(st, robot, sensor)
    print(f"[choreo] {rec.n} frames in {time.time() - t0:.1f}s")
    f3 = Film3D(st, robot, objs, meshes, sensor, rec)
    ov = Hud(1920, 1080, math_cache=os.path.join(_HERE, "depth_map.math.json"))
    painter = Painter(st, f3, rec, ov)

    def render(t):
        i = min(int(round(t * FPS)), rec.n - 1)
        painter.set3d(t, i)
        ov.begin()
        painter.draw2d(t, i)
        ov.fade(fade_amount(t))
        ov.end()
        return st.frame(t, hud=ov)

    os.makedirs(args.outdir, exist_ok=True)
    if args.stills:
        for tok in args.stills.split(","):
            t = float(tok)
            p = os.path.join(args.outdir, f"depth_still_{t:06.2f}.png")
            write_png(p, render(t))
            print("saved", p)
        return
    if args.sheet:
        f = max(1, W // 480)
        thumbs = [shrink(render(float(t)), f) for t in np.arange(0.5, TL.duration, args.every)]
        cols = 6
        th, tw = thumbs[0].shape[:2]
        rows = (len(thumbs) + cols - 1) // cols
        sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
        for k, im in enumerate(thumbs):
            y, x = (k // cols) * th, (k % cols) * tw
            sheet[y:y + th, x:x + tw] = im
        p = os.path.join(args.outdir, "depth_sheet.png")
        write_png(p, sheet)
        print("saved", p, f"({len(thumbs)} frames, every {args.every} s from 0.5 s, row-major)")
        return

    t_to = args.t_to if args.t_to is not None else TL.duration
    film = Film(args.out, W, H, fps=fps, crf=16 if not args.preview else 22,
                preset="slow" if not args.preview else "veryfast")
    nfr = int(round((t_to - args.t_from) * fps))
    t_start = time.time()
    for f in range(nfr):
        t = args.t_from + f / fps
        film.write(render(t))
        if f % (fps * 5) == 0:
            el = time.time() - t_start
            print(f"[film] {t:6.1f}s  frame {f}/{nfr}  {el / max(f, 1) * 1000:.0f} ms/frame", flush=True)
    path = film.close()
    print(f"[film] wrote {path} ({nfr} frames, {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
