"""The fleet's vehicles beside the boats: the inspection drone, the ROV on its tether, the snake at
the cable entry, the Mariner's landing pad. Kinematic poses at their work stations; update() keeps
the tether and the drone on their moving ends so a film can drive them.

Copied builders, each marked with where it came from (the source files build whole scenes at
import, so they are not imported).
"""
import math

import numpy as np

import threepp as tp
from demo_common import standard_material
from drone_rig import Drone
import turbine_site as ts

UP = np.array([0.0, 1.0, 0.0])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def quat_from_axes(xa, ya):
    """Quaternion (x, y, z, w) of the frame whose local +X is xa and +Y is (close to) ya."""
    from usv_rig import quat_of
    x = _unit(xa)
    z = _unit(np.cross(x, ya))
    y = np.cross(z, x)
    return quat_of(np.stack([x, y, z], axis=1))


# --------------------------------------------------------------------------- #
#  Helpers from netpen/warp_netpen.py (align_y, tube, merged) and its ROV materials
# --------------------------------------------------------------------------- #
def align_y(obj, d):
    """Rotate an object whose axis is +Y onto direction d (warp_netpen.align_y)."""
    d = np.asarray(d, np.float64)
    d /= np.linalg.norm(d)
    axis = np.cross([0.0, 1.0, 0.0], d)
    s = np.linalg.norm(axis)
    if s < 1e-6:
        if d[1] < 0:
            obj.rotation.x = math.pi
        return
    obj.quaternion.set_from_axis_angle(tp.Vector3(*(axis / s)), math.atan2(s, d[1]))


def rod(a, b, r, mat, seg=10):
    """warp_netpen.tube: a cylinder from a to b."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    L = float(np.linalg.norm(b - a))
    m = tp.Mesh(tp.CylinderGeometry(r, r, L, seg, 1), mat)
    m.position.set(*(0.5 * (a + b)))
    align_y(m, b - a)
    return m


def merged(items, mat, shadow=True):
    """warp_netpen.merged: one Mesh from (geometry, (x, y, z), (rx, ry, rz) | None) items."""
    P, N, I, off = [], [], [], 0
    for g, pos, rot in items:
        if rot is not None:
            if rot[0]:
                g.rotate_x(rot[0])
            if rot[1]:
                g.rotate_y(rot[1])
            if rot[2]:
                g.rotate_z(rot[2])
        g.translate(*pos)
        p = np.asarray(g.get_attribute("position"), np.float32).reshape(-1, 3)
        n = np.asarray(g.get_attribute("normal"), np.float32).reshape(-1, 3)
        ix = g.get_index()
        ix = np.arange(len(p), dtype=np.uint32) if ix is None else np.asarray(ix, np.uint32)
        P.append(p)
        N.append(n)
        I.append(ix + off)
        off += len(p)
    geo = tp.BufferGeometry()
    geo.set_attribute("position", np.concatenate(P))
    geo.set_attribute("normal", np.concatenate(N))
    geo.set_index(np.concatenate(I))
    m = tp.Mesh(geo, mat)
    m.cast_shadow = shadow
    return m


hdpe = standard_material(0x0b0d10, roughness=0.55)
alu = standard_material(0xb9bec4, roughness=0.35, metalness=0.95)
dark = standard_material(0x15171a, roughness=0.6)
foam = standard_material(0xf2c21a, roughness=0.75)
acrylic = tp.MeshPhysicalMaterial()
acrylic.color = 0xffffff
acrylic.roughness = 0.05
acrylic.metalness = 0.0
acrylic.transmission = 0.9
acrylic.ior = 1.49
acrylic.thin_walled = True
acrylic.thickness = 0.004


# --------------------------------------------------------------------------- #
#  The ROV: BlueROV2 class (warp_netpen side_plate, thruster, build_rov)
# --------------------------------------------------------------------------- #
def side_plate(th=0.006, cell=0.0025):
    """Rounded HDPE plate with round cut-outs as a cell soup (warp_netpen.side_plate)."""
    hx, hy, rr = 0.20, 0.10, 0.03
    xs = np.arange(-hx, hx - 1e-6, cell)
    ys = np.arange(-hy, hy - 1e-6, cell)
    X, Y = np.meshgrid(xs + 0.5 * cell, ys + 0.5 * cell)
    keep = np.hypot(np.maximum(np.abs(X) - (hx - rr), 0), np.maximum(np.abs(Y) - (hy - rr), 0)) <= rr
    for cx, cy, r in ((-0.02, -0.005, 0.058), (0.135, 0.035, 0.030), (0.135, -0.045, 0.024),
                      (-0.145, 0.035, 0.026), (-0.145, -0.045, 0.026)):
        keep &= np.hypot(X - cx, Y - cy) > r
    P, N = [], []

    def quad(a, b, c, d, n):
        P.append(np.stack([a, b, c, a, c, d], 1).reshape(-1, 3))
        N.append(np.tile(np.float32(n), (len(a) * 6, 1)))

    I, J = np.nonzero(keep)
    xa, ya = xs[J], ys[I]
    xb, yb = xa + cell, ya + cell
    z0, z1 = np.zeros_like(xa), np.full_like(xa, th)
    v = lambda x, y, z: np.stack([x, y, z], 1)
    quad(v(xa, ya, z1), v(xb, ya, z1), v(xb, yb, z1), v(xa, yb, z1), (0, 0, 1))
    quad(v(xa, ya, z0), v(xa, yb, z0), v(xb, yb, z0), v(xb, ya, z0), (0, 0, -1))
    pad = np.pad(keep, 1)
    for di, dj, n in ((0, -1, (-1, 0, 0)), (0, 1, (1, 0, 0)), (-1, 0, (0, -1, 0)), (1, 0, (0, 1, 0))):
        edge = keep & ~pad[1 + di:1 + di + keep.shape[0], 1 + dj:1 + dj + keep.shape[1]]
        I, J = np.nonzero(edge)
        xa, ya = xs[J], ys[I]
        xb, yb = xa + cell, ya + cell
        z0, z1 = np.zeros_like(xa), np.full_like(xa, th)
        if dj:
            x = xa if dj < 0 else xb
            quad(v(x, ya, z0), v(x, yb, z0), v(x, yb, z1), v(x, ya, z1), n)
        else:
            y = ya if di < 0 else yb
            quad(v(xa, y, z0), v(xb, y, z0), v(xb, y, z1), v(xa, y, z1), n)
    g = tp.BufferGeometry()
    g.set_attribute("position", np.concatenate(P).astype(np.float32))
    g.set_attribute("normal", np.concatenate(N).astype(np.float32))
    return g


def thruster(pos, axis):
    """warp_netpen.thruster."""
    g = tp.Group()
    nz = tp.Mesh(tp.CylinderGeometry(0.048, 0.048, 0.052, 24, 1, True), standard_material(0x0d0f12, roughness=0.5, side=tp.Side.Double))
    g.add(nz)
    g.add(tp.Mesh(tp.CylinderGeometry(0.015, 0.015, 0.062, 12, 1), dark))
    for k in range(3):
        st = tp.Mesh(tp.BoxGeometry(0.004, 0.006, 0.034), dark)
        st.position.set(0.0, -0.02, 0.031)
        st.rotation.y = 2 * math.pi * k / 3
        arm = tp.Group()
        arm.add(st)
        arm.rotation.y = 2 * math.pi * k / 3
        g.add(arm)
    prop = tp.Group()
    for k in range(3):
        bl = tp.Mesh(tp.BoxGeometry(0.030, 0.0025, 0.013), standard_material(0x2a2e33, roughness=0.45))
        bl.position.set(0.026, 0.0, 0.0)
        bl.rotation.x = math.radians(32)
        arm = tp.Group()
        arm.add(bl)
        arm.rotation.y = 2 * math.pi * k / 3
        prop.add(arm)
    prop.position.y = 0.008
    g.add(prop)
    g.position.set(*pos)
    align_y(g, axis)
    return g, prop


ROV_SPOT_I, ROV_SPOT_RANGE, ROV_SPOT_DECAY = 60.0, 5.0, 3.0      # netpen: 1000, 3.5, 3; >= 150 burns the 0.13-albedo anode white at 2.7 m


def build_rov():
    """warp_netpen.build_rov: a BlueROV2-class frame, 0.46 x 0.34 x 0.25 m. Local +X forward."""
    rov = tp.Group()
    props = []
    for z in (-0.165, 0.165):
        pl = tp.Mesh(side_plate(), hdpe)
        pl.position.set(0.0, 0.0, z - 0.003)
        pl.cast_shadow = True
        rov.add(pl)
    top = tp.Mesh(tp.BoxGeometry(0.30, 0.005, 0.33), hdpe)
    top.position.set(0.0, 0.105, 0.0)
    rov.add(top)
    for x in (-0.19, 0.19):
        rov.add(rod((x, -0.095, -0.16), (x, -0.095, 0.16), 0.008, hdpe))
    for z in (-0.105, 0.105):
        fb = tp.Mesh(tp.BoxGeometry(0.30, 0.055, 0.085), foam)
        fb.position.set(0.0, 0.135, z)
        rov.add(fb)
    for y, z, r, L, filler in ((0.035, 0.0, 0.051, 0.30, 0x2b2f38), (-0.06, -0.075, 0.038, 0.26, 0x1f3a6a),
                               (-0.06, 0.075, 0.038, 0.26, 0x1f3a6a)):
        t = tp.Mesh(tp.CylinderGeometry(r, r, L, 32, 1, True), acrylic)
        t.rotation.z = math.pi / 2
        t.position.set(0.0, y, z)
        rov.add(t)
        inner = tp.Mesh(tp.BoxGeometry(L * 0.8, r * 0.9, r * 1.3), standard_material(filler, roughness=0.6))
        inner.position.set(0.0, y - 0.15 * r, z)
        rov.add(inner)
        for x in (-0.5 * L - 0.012, 0.5 * L + 0.012):
            cap = tp.Mesh(tp.CylinderGeometry(r + 0.004, r + 0.004, 0.024, 32, 1), alu)
            cap.rotation.z = math.pi / 2
            cap.position.set(x, y, z)
            rov.add(cap)
    dome = tp.Mesh(tp.SphereGeometry(0.052, 24, 16, -math.pi / 2, math.pi), acrylic)
    dome.rotation.y = math.pi
    dome.position.set(0.15 + 0.024 + 0.003, 0.035, 0.0)
    rov.add(dome)
    cam = tp.Mesh(tp.BoxGeometry(0.03, 0.03, 0.03), dark)
    cam.position.set(0.175, 0.035, 0.0)
    rov.add(cam)
    lens = tp.Mesh(tp.CylinderGeometry(0.009, 0.009, 0.012, 16, 1), standard_material(0x0a0c14, roughness=0.15))
    lens.rotation.z = math.pi / 2
    lens.position.set(0.195, 0.035, 0.0)
    rov.add(lens)
    s2 = 0.5 ** 0.5
    for x, z, ax in ((0.17, -0.20, (s2, 0, -s2)), (0.17, 0.20, (s2, 0, s2)),
                     (-0.17, -0.20, (s2, 0, s2)), (-0.17, 0.20, (s2, 0, -s2))):
        g, p = thruster((x, -0.02, z), ax)
        rov.add(g)
        props.append(p)
    for z in (-0.165, 0.165):
        g, p = thruster((-0.02, -0.005, z), (0, 1, 0))
        rov.add(g)
        props.append(p)
    lights = []
    for z in (-0.12, 0.12):
        body = tp.Mesh(tp.CylinderGeometry(0.02, 0.02, 0.045, 16, 1), dark)
        body.rotation.z = math.pi / 2
        body.position.set(0.19, -0.075, z)
        rov.add(body)
        disc = tp.Mesh(tp.CircleGeometry(0.017, 24), standard_material(0xfff4e0, roughness=0.3, emissive=0xfff2d8, emissive_intensity=120.0))
        disc.rotation.y = math.pi / 2
        disc.position.set(0.2135, -0.075, z)
        rov.add(disc)
        spot = tp.SpotLight(tp.Color(1.0, 0.95, 0.85), ROV_SPOT_I, ROV_SPOT_RANGE, math.radians(32.0), 0.95, ROV_SPOT_DECAY)   # netpen 21 deg / 0.55: a hard disc
        spot.position.set(0.215, -0.075, z)
        tgt = tp.Group()
        tgt.position.set(4.0, -0.35, z)
        rov.add(tgt)
        spot.set_target(tgt)
        rov.add(spot)
        lights.append(spot)
    return rov, props, lights


# --------------------------------------------------------------------------- #
#  The snake: the thruster snake of netpen/snake_netpen.py (module_parts, _add, _along_x, cyl_x,
#  the head and its lamps, pose_links); 9 links x 0.18 m (snake_model.SnakeParams)
# --------------------------------------------------------------------------- #
LINK_N, LINK_L, LINK_R = 9, 0.18, 0.0525
yellow = tp.MeshPhysicalMaterial()
yellow.color = 0xffc20e
yellow.roughness = 0.45
yellow.metalness = 0.0
yellow.specular_intensity = 0.4
rubber = standard_material(0x141516, roughness=0.8)
smoked = tp.MeshPhysicalMaterial()
smoked.color = 0x0c0f12
smoked.roughness = 0.08
smoked.metalness = 0.0
smoked.clearcoat = 1.0
smoked.clearcoat_roughness = 0.05
lens_mat = standard_material(0x05070c, roughness=0.12)
ring_mat = standard_material(0x9aa0a6, roughness=0.35, metalness=0.9)
SNAKE_LAMP_I, SNAKE_LAMP_RANGE = 3.0, 3.5          # snake_netpen LAMP_I 60, range 3.5, decay 3; 60 burnt the stiffener white
lamp_disc_mat = standard_material(0xfff2dc, roughness=0.3, emissive=0xffe6c0, emissive_intensity=90.0)
MOD_R = 0.049
SCENE_SIDE_THRUSTERS, SCENE_TUNNEL_BOW, SCENE_TUNNEL_STERN = 2, 1, 7
SIDE_Y, DUCT_R, DUCT_L = 0.079, 0.030, 0.070
MOD_HALF = 0.056
RING_R, RING_W = 0.0515, 0.006
CHAMF_W, BOOT_R = 0.009, 0.036
X_CH = MOD_HALF + RING_W
X_BOOT = X_CH + CHAMF_W
metal = standard_material(0xa3a9ae, roughness=0.38, metalness=0.6)
duct_mat = standard_material(0x2a2e33, roughness=0.5, metalness=0.3, side=tp.Side.Double)
ROT_X = (0.0, 0.0, -math.pi / 2)
ROT_MX = (0.0, 0.0, math.pi / 2)
ROT_TORUS_X = (0.0, math.pi / 2, 0.0)


def _add(parent, mesh, shadow=True):
    mesh.cast_shadow = shadow
    parent.add(mesh)
    return mesh


def _along_x(mesh):
    mesh.rotation.z = -math.pi / 2
    return mesh


def cyl_x(r0, r1, x0, x1, seg=40, open_=False):
    if x1 < x0:
        return (tp.CylinderGeometry(r0, r1, x0 - x1, seg, 1, open_), (0.5 * (x0 + x1), 0.0, 0.0), ROT_MX)
    return (tp.CylinderGeometry(r1, r0, x1 - x0, seg, 1, open_), (0.5 * (x0 + x1), 0.0, 0.0), ROT_X)


def module_parts(i):
    """snake_netpen.module_parts: (yellow, metal, rubber, duct) item lists for link i (0 = head)."""
    Y, M, R, D = [], [], [], []
    ends = []
    if i > 0:
        ends.append(+1.0)
    ends.append(-1.0)
    if i > 0:
        Y.append(cyl_x(MOD_R, MOD_R, -MOD_HALF, MOD_HALF))
        R.append(cyl_x(MOD_R + 0.0006, MOD_R + 0.0006, -0.0013, 0.0013))
        for k in range(10):
            a = 2.0 * math.pi * (k + 0.5) / 10
            M.append((tp.CylinderGeometry(0.0034, 0.0034, 0.004, 8, 1), (0.0075, MOD_R * math.cos(a), MOD_R * math.sin(a)), (a, 0.0, 0.0)))
    for sg in ends:
        rr = LINK_R + 0.002 if i == 0 else RING_R
        M.append(cyl_x(rr, rr, sg * MOD_HALF, sg * X_CH))
        if i == LINK_N - 1 and sg < 0:
            continue
        M.append(cyl_x(rr, BOOT_R + 0.004, sg * X_CH, sg * X_BOOT))
        R.append(cyl_x(BOOT_R, BOOT_R, sg * X_BOOT, sg * 0.5 * LINK_L, seg=24))
        for xr in (0.0765, 0.0855):
            R.append((tp.TorusGeometry(BOOT_R + 0.001, 0.0052, 8, 32), (sg * xr, 0.0, 0.0), ROT_TORUS_X))
    if i < LINK_N - 1:
        R.append((tp.SphereGeometry(BOOT_R + 0.0005, 24, 12), (-0.5 * LINK_L, 0.0, 0.0), None))
    if i in (SCENE_TUNNEL_BOW, SCENE_TUNNEL_STERN):
        for sd in (-1.0, 1.0):
            R.append((tp.CircleGeometry(0.021, 24), (0.0, 0.0, sd * (MOD_R + 0.0008)), (0.0, 0.0 if sd > 0 else math.pi, 0.0)))
            M.append((tp.TorusGeometry(0.0215, 0.0026, 8, 32), (0.0, 0.0, sd * (MOD_R + 0.0004)), None))
            for a in (0.0, math.pi / 2):
                M.append((tp.BoxGeometry(0.040, 0.0028, 0.0025), (0.0, 0.0, sd * (MOD_R + 0.0012)), (0.0, 0.0, a)))
    if i == SCENE_SIDE_THRUSTERS:
        for sd in (-1.0, 1.0):
            zc = sd * SIDE_Y
            D.append(cyl_x(DUCT_R, DUCT_R, -DUCT_L / 2, DUCT_L / 2, seg=32, open_=True))
            D[-1] = (D[-1][0], (0.0, 0.0, zc), D[-1][2])
            Y.append((tp.TorusGeometry(DUCT_R - 0.0005, 0.0042, 10, 36), (DUCT_L / 2, 0.0, zc), ROT_TORUS_X))
            R.append((tp.TorusGeometry(DUCT_R - 0.001, 0.0022, 8, 36), (-DUCT_L / 2, 0.0, zc), ROT_TORUS_X))
            R.append((tp.BoxGeometry(0.048, 0.010, 0.012), (0.0, 0.0, sd * (MOD_R + 0.002)), None))
            M.append((tp.CylinderGeometry(0.0075, 0.0075, 0.030, 16, 1), (-0.004, 0.0, zc), ROT_X))
            M.append((tp.ConeGeometry(0.0075, 0.012, 16, 1), (-0.025, 0.0, zc), ROT_MX))
            M.append((tp.BoxGeometry(0.004, 2 * DUCT_R - 0.004, 0.003), (-0.016, 0.0, zc), None))
    if i == 6:
        R.append((tp.BoxGeometry(0.056, 0.020, 0.034), (0.0, MOD_R + 0.007, 0.0), None))
        M.append((tp.CylinderGeometry(0.009, 0.009, 0.004, 16, 1), (0.029, MOD_R + 0.007, 0.0), ROT_X))
    return Y, M, R, D


def build_snake(scene):
    """snake_netpen's link loop: returns (links, lamps). Link-local +X toward the head."""
    links, lamps = [], []
    for i in range(LINK_N):
        g = tp.Group()
        if i == 0:
            x_dome = 0.5 * LINK_L - LINK_R * 0.98
            m = _add(g, _along_x(tp.Mesh(tp.CylinderGeometry(LINK_R, LINK_R, x_dome + MOD_HALF, 40, 1), yellow)))
            m.position.x = 0.5 * (x_dome - MOD_HALF)
            cr = _add(g, _along_x(tp.Mesh(tp.CylinderGeometry(LINK_R + 0.003, LINK_R + 0.003, 0.012, 40, 1), rubber)))
            cr.position.x = x_dome
            dome = _add(g, tp.Mesh(tp.SphereGeometry(LINK_R * 0.98, 32, 20, 0.0, 2 * math.pi, 0.0, 0.5 * math.pi), smoked))
            dome.rotation.z = -math.pi / 2
            dome.position.x = x_dome
            bezel = _add(g, _along_x(tp.Mesh(tp.CylinderGeometry(0.017, 0.017, 0.006, 20, 1), ring_mat)))
            bezel.position.x = 0.5 * LINK_L - 0.006
            lens = _add(g, _along_x(tp.Mesh(tp.CylinderGeometry(0.012, 0.012, 0.008, 20, 1), lens_mat)))
            lens.position.x = 0.5 * LINK_L - 0.001
            son = _add(g, tp.Mesh(tp.BoxGeometry(0.05, 0.018, 0.04), rubber))
            son.position.set(-0.01, LINK_R + 0.004, 0.0)
            for side in (-1.0, 1.0):
                zc = side * (LINK_R + 0.010)
                hous = _add(g, _along_x(tp.Mesh(tp.CylinderGeometry(0.011, 0.011, 0.034, 16, 1), rubber)))
                hous.position.set(x_dome - 0.03, -0.004, zc)
                disc = tp.Mesh(tp.CircleGeometry(0.0085, 20), lamp_disc_mat)
                disc.rotation.y = math.pi / 2
                disc.position.set(x_dome - 0.03 + 0.0175, -0.004, zc)
                _add(g, disc, shadow=False)
                spot = tp.SpotLight(tp.Color(1.0, 0.90, 0.74), SNAKE_LAMP_I, SNAKE_LAMP_RANGE, math.radians(28.0), 0.95, 3.0)   # netpen 20 deg / 0.6
                spot.position.set(x_dome - 0.03 + 0.02, -0.004, zc)
                tgt = tp.Group()
                tgt.position.set(3.0, 0.0, zc + side * 3.0 * math.tan(math.radians(10.0)))   # a small splay (netpen: 20)
                g.add(tgt)
                spot.set_target(tgt)
                g.add(spot)
                lamps.append(spot)
        Y, M, R, D = module_parts(i)
        for items, mat in ((Y, yellow), (M, metal), (R, rubber), (D, duct_mat)):
            if items:
                _add(g, merged(items, mat))
        if i == LINK_N - 1:
            cap = _add(g, tp.Mesh(tp.ConeGeometry(MOD_R, 0.12, 40, 1), yellow))
            cap.rotation.z = math.pi / 2
            cap.position.x = -X_CH - 0.06
            tip = _add(g, tp.Mesh(tp.SphereGeometry(0.012, 12, 8), rubber))
            tip.position.x = -X_CH - 0.118
        scene.add(g)
        links.append(g)
    return links, lamps


def snake_station(S, sgn=1.0):
    """Joint points (nose first) of a gentle S along the CPS, 0.6 m off it on side sgn, and the neck
    bent onto the sleeve where it meets the rock. Returns (joints (N+1, 3), look point, out side)."""
    cab = np.asarray(S.cable_path)
    az = math.radians(S.spec["empties"]["cable_entry"]["azimuth_deg"])
    rn = np.array([math.cos(az), 0.0, math.sin(az)])
    tn = sgn * np.array([-math.sin(az), 0.0, math.cos(az)])
    r = cab @ rn
    # the touchdown: where the hanging sleeve first rests on the rock (lowest clearance under it)
    gap = cab[:, 1] - ts.CPS_R - ts.seabed_height(cab[:, 0], cab[:, 2])
    r0 = r[0]
    i_td = int(np.argmax((r > r0 + 2.0) & (gap < 0.05)))
    look = cab[i_td] + np.array([0.0, 0.05, 0.0])
    # body: from r_td + 0.9 outward, 0.6 m off the sleeve (to the tangential side, a little up)
    off = _unit(0.75 * tn + 0.65 * UP) * (ts.CPS_R + LINK_R + 0.6)   # 0.6 m skin to skin
    L = LINK_N * LINK_L
    s = np.linspace(0.0, L - 2 * LINK_L, 400)                    # the body behind the 2-joint neck
    r_body = r[i_td] + 0.9 + s
    ci = np.clip(np.searchsorted(r, r_body), 0, len(r) - 1)
    base = cab[ci] + off
    base = base + np.outer(0.10 * np.sin(2 * np.pi * s / 1.3), tn)   # the gentle S (amplitude 0.10 m)
    # arc-length resample into joints, nose end first
    seg = np.linalg.norm(np.diff(base, axis=0), axis=1)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    js = np.arange(0, LINK_N - 1) * LINK_L
    body = np.stack([np.interp(js, cs, base[:, k]) for k in range(3)], 1)      # joints 2..N (7 links)
    # the neck: two links from joint 2 bending toward the look point
    d_body = _unit(body[0] - body[1])
    d_look = _unit(look - body[0])
    d1 = _unit(d_body + 1.2 * d_look)                             # neck link: half way round
    j1 = body[0] + LINK_L * d1
    d0 = _unit(0.3 * d1 + d_look)                                 # head: onto the sleeve
    j0 = j1 + LINK_L * d0
    joints = np.vstack([j0, j1, body])
    return joints, look, tn


def pose_snake(links, joints):
    """Links between joints (nose first); local +X toward the nose, +Y as close to up as it gets."""
    for i, g in enumerate(links):
        a, b = joints[i + 1], joints[i]
        c = 0.5 * (a + b)
        q = quat_from_axes(b - a, UP)
        g.position.set(*map(float, c))
        g.quaternion.set(*map(float, q))


def snake_clearance(joints, S):
    """Minimum clearance (m) of the snake's skin to the rock, the sleeve and the pile."""
    pts = []
    for i in range(LINK_N):
        a, b = joints[i + 1], joints[i]
        for u in np.linspace(0.0, 1.0, 12):
            pts.append(a + (b - a) * u)
    P = np.array(pts)
    rad = LINK_R + 0.03                                           # shell + the side ducts' reach, roughly
    bed = P[:, 1] - ts.seabed_height(P[:, 0], P[:, 2]) - rad
    cab = np.asarray(S.cable_path)
    d = np.min(np.linalg.norm(P[:, None, :] - cab[None, :, :], axis=2), axis=1)
    sleeve = d - rad - (ts.CPS_R + 0.035 + 0.05)                  # collar ribs; stiffener near the hole
    pile = np.hypot(P[:, 0], P[:, 2]) - ts.PILE_R - 0.22 - rad    # marine growth at its thickest
    return float(bed.min()), float(sleeve.min()), float(pile.min())


# --------------------------------------------------------------------------- #
#  The landing pad on the Mariner's aft hatches (a thin disc, offset along the deck normal)
# --------------------------------------------------------------------------- #
# The brief asked for the aft deck, but the sensor tower stands over the aft hatches (shelf 0.94 m
# above them): no 1 m drone lands there. The pad goes on the fore hatch pair (x -0.32 .. 1.16).
PAD_X, PAD_R = 0.42, 0.62


def deck_top(obj, x0, x1, zhalf):
    """Highest vertex y of the boat's meshes over a deck patch (vessel frame, root at identity)."""
    best = [-1e9]

    def visit(o):
        if isinstance(o, tp.Mesh) and o.parent is not None and o.parent.name in ("hull", "deck_fittings"):
            p = np.asarray(o.geometry.get_attribute("position"), float).reshape(-1, 3)   # nodes at identity
            k = (p[:, 0] > x0) & (p[:, 0] < x1) & (np.abs(p[:, 2]) < zhalf)
            if k.any():
                best[0] = max(best[0], float(p[k, 1].max()))
    obj.traverse(visit)
    return best[0]


def build_pad(mariner_obj):
    y = deck_top(mariner_obj, PAD_X - 0.5, PAD_X + 0.5, 0.5)
    pad = tp.Group()
    base = tp.Mesh(tp.CircleGeometry(PAD_R, 48), standard_material(0x1d2226, roughness=0.8))
    base.rotation.x = -math.pi / 2
    pad.add(base)
    ring = tp.Mesh(tp.RingGeometry(PAD_R - 0.07, PAD_R - 0.03, 48), standard_material(0xe8e4d8, roughness=0.7))
    ring.rotation.x = -math.pi / 2
    ring.position.y = 0.002
    pad.add(ring)
    white = standard_material(0xe8e4d8, roughness=0.7)
    for bx, bz, sx, sz in ((0.0, -0.17, 0.46, 0.06), (0.0, 0.17, 0.46, 0.06), (0.0, 0.0, 0.06, 0.34)):
        b = tp.Mesh(tp.BoxGeometry(sx, 0.004, sz), white)                 # the H, bars along the boat's length
        b.position.set(bx, 0.003, bz)
        pad.add(b)
    pad.position.set(PAD_X, y + 0.008, 0.0)
    mariner_obj.add(pad)
    return pad, y


# --------------------------------------------------------------------------- #
#  Stations
# --------------------------------------------------------------------------- #
DRONE_SPAN = 1.1                 # industrial inspection quad, motor to motor (assumed, M300/M350 class)
DRONE_SPAN_AT = 58.0             # m from the blade root: mid-span of 117 m
DRONE_OFF = 11.0                 # m off the leading edge
LE_DIR = _unit([0.9, 0.0, -0.44])    # turbine_site.cam_blade: "off the leading edge"
ROV_AZ, ROV_Y = 78.0, -12.0      # under the Mariner's side of the pile, 12 m down
CAGE = ts.spec["underwater"]["anode_cage"]
ROV_R = CAGE["ring_radius"] + 0.5 * CAGE["anode_size"][0] + 2.5 + 0.25   # 2.5 m clear of the anodes (nose 0.25 ahead of centre)
TETHER_R = 0.0045                # BlueROV2 Fathom tether is 7.6 mm; drawn a hair thicker to read


def industrial_kit(drone):
    """Dress drone_rig's quad as an industrial inspection machine (M300/M350 class, assumed): a bigger
    centre body with twin batteries, landing skids, a GNSS puck and a gimballed camera pod slung
    under the nose (on the rig's gimbal, so it aims at the work)."""
    grey = standard_material(0x2c3036, roughness=0.5, metalness=0.2)
    lite = standard_material(0x8a9098, roughness=0.45, metalness=0.3)
    blk = standard_material(0x111316, roughness=0.6)
    r = drone.root
    body = tp.Mesh(tp.BoxGeometry(0.30, 0.15, 0.46), grey)
    body.position.set(0.0, 0.01, 0.0)
    r.add(body)
    for x in (-0.075, 0.075):                                  # twin batteries on the back
        b = tp.Mesh(tp.BoxGeometry(0.11, 0.07, 0.24), lite)
        b.position.set(x, 0.12, -0.06)
        r.add(b)
    puck = tp.Mesh(tp.CylinderGeometry(0.05, 0.05, 0.03, 16, 1), standard_material(0xd8d8d2, roughness=0.5))
    puck.position.set(0.0, 0.16, 0.12)
    r.add(puck)
    for sx in (-1.0, 1.0):                                     # landing skids
        for sz in (-1.0, 1.0):
            r.add(rod((sx * 0.10, -0.05, sz * 0.12), (sx * 0.20, -0.36, sz * 0.15), 0.011, blk))
        r.add(rod((sx * 0.20, -0.36, -0.26), (sx * 0.20, -0.36, 0.26), 0.013, blk))
    yoke = tp.Mesh(tp.BoxGeometry(0.05, 0.10, 0.05), blk)      # gimbal yoke under the nose
    yoke.position.set(0.0, -0.10, 0.19)
    r.add(yoke)
    g = drone.gimbal
    g.position.set(0.0, -0.20, 0.19)
    pod = tp.Mesh(tp.BoxGeometry(0.15, 0.13, 0.12), standard_material(0x3a3f46, roughness=0.4, metalness=0.3))
    g.add(pod)
    for x in (-0.035, 0.035):                                  # wide + zoom lenses
        lens = tp.Mesh(tp.CylinderGeometry(0.028, 0.028, 0.03, 20, 1), standard_material(0x05070a, roughness=0.08))
        lens.rotate_x(math.pi / 2)
        lens.position.set(x, 0.0, 0.07)
        g.add(lens)


def build_vehicles(scene, S, F):
    V = {}
    # the drone
    bp = S.blade_point(1, DRONE_SPAN_AT)
    V["drone_look"] = bp
    V["drone_pos"] = bp + DRONE_OFF * LE_DIR
    V["drone"] = Drone(scene, span=DRONE_SPAN, start_pos=tuple(V["drone_pos"]), start_look=tuple(bp))
    industrial_kit(V["drone"])
    V["blade_low"] = S.blade_point(1, DRONE_SPAN_AT + 10.0)
    for _ in range(30):
        V["drone"].set_pose(V["drone_pos"], bp, 1.0 / 60.0)
    # the pad
    V["pad"], V["deck_y"] = build_pad(F.mariner_obj)
    # the ROV, facing the cage
    rov, props, lights = build_rov()
    a = math.radians(ROV_AZ)
    rn = np.array([math.cos(a), 0.0, math.sin(a)])
    V["rov_pos"] = np.array([ROV_R * rn[0], ROV_Y, ROV_R * rn[2]])
    V["rov_fwd"] = -rn
    rov.position.set(*map(float, V["rov_pos"]))
    rov.quaternion.set(*map(float, quat_from_axes(-rn, UP)))
    scene.add(rov)
    V["rov"], V["rov_lights"] = rov, lights
    V["tether"] = None
    V["tether_mat"] = standard_material(0xf7d51d, roughness=0.6)    # warp_netpen cable colour
    # the snake
    links, lamps = build_snake(scene)
    for sgn in (1.0, -1.0):                    # the side that puts the lamp pool left of the snake on camera
        joints, look, out = snake_station(S, sgn)
        eye, tgt, _ = snake_cam({"snake_joints": joints, "snake_look": look, "snake_out": out})
        fwd = _unit(tgt - eye)
        right = _unit(np.cross(fwd, UP))
        if np.dot(look - joints[4], right) < 0.0:
            break
    V["snake_out"] = out
    pose_snake(links, joints)
    V["snake_links"], V["snake_lamps"], V["snake_joints"], V["snake_look"] = links, lamps, joints, look
    bed, sleeve, pile = snake_clearance(joints, S)
    ang = [math.degrees(math.acos(np.clip(np.dot(_unit(joints[i] - joints[i + 1]), _unit(joints[i + 1] - joints[i + 2])), -1, 1)))
           for i in range(LINK_N - 1)]
    print(f"[fleet] snake: nose to the look point {np.linalg.norm(look - joints[0]):.2f} m; min clearance rock {bed:.3f} m, sleeve {sleeve:.3f} m, pile {pile:.3f} m; "
          f"joint bends {', '.join('%.0f' % x for x in ang)} deg (limit 60)")
    V["snake_clear"] = (bed, sleeve, pile)
    print(f"[fleet] pad on the fore hatches at vessel y {V['deck_y']:.3f}; drone at {np.round(V['drone_pos'], 1)}, "
          f"blade point {np.round(bp, 1)}; ROV at {np.round(V['rov_pos'], 2)}")
    return V


def tether_path(a, da, b, db, n=120, sag=1.5):
    """A smooth sagging curve from a (leaving along da) to b (arriving along db): a cubic Bezier with a
    belly. Static stand-in for the rope physics of a later phase."""
    L = float(np.linalg.norm(b - a))
    p1 = a + da * 0.35 * L
    p2 = b - db * 0.35 * L
    u = np.linspace(0.0, 1.0, n)[:, None]
    P = ((1 - u) ** 3) * a + 3 * ((1 - u) ** 2) * u * p1 + 3 * (1 - u) * u * u * p2 + u ** 3 * b
    P[:, 1] -= sag * np.sin(np.pi * u[:, 0]) ** 1.5
    return P


def update_vehicles(V, S, F, t, dt):
    d = V["drone"]
    d.set_pose(V["drone_pos"], V["drone_look"], dt if dt > 0 else 1.0 / 60.0)
    d.tick(dt if dt > 0 else 1.0 / 60.0, t, night=0.4)
    # the tether: ROV top aft to the Mariner's moonpool (her vessel frame), rebuilt when the boat moves
    a = V["rov_pos"] + np.array([0.0, 0.11, 0.0]) - 0.2 * V["rov_fwd"]
    b = F.mariner.to_world([0.2, 0.0, 0.0])                     # moonpool (spec: x 0.2, "mid area", assumed)
    key = tuple(np.round(b, 3))
    if V.get("tether_key") != key:
        P = tether_path(a, _unit(-V["rov_fwd"] + 0.6 * UP), b, _unit(UP), sag=2.0)
        geo = ts.tube(P, lambda s: np.full_like(s, TETHER_R), sides=8)
        if V["tether"] is None:
            V["tether"] = tp.Mesh(geo, V["tether_mat"])
            S.scene.add(V["tether"])
        else:                                                   # same vertex count: update in place
            g = V["tether"].geometry
            g.update_attribute("position", np.asarray(geo.get_attribute("position"), np.float32))
            g.update_attribute("normal", np.asarray(geo.get_attribute("normal"), np.float32))
        V["tether_key"] = key


# --------------------------------------------------------------------------- #
#  Cameras
# --------------------------------------------------------------------------- #
def drone_chase(V):
    """Behind and above the drone, looking down the blade past it: the near sea only, no far grid."""
    p, look = V["drone_pos"], V["drone_look"]
    back = _unit(p - look)
    eye = p + 6.5 * back + np.array([0.0, 2.0, 0.0]) + 1.3 * _unit(np.cross(UP, back))
    return eye, p + np.array([0.0, -1.5, 0.0]) + 0.1 * (look - p), 28.0


def rov_cam(V):
    p, f = V["rov_pos"], V["rov_fwd"]
    side = _unit(np.cross(UP, f))
    eye = p - 2.3 * f + 1.4 * side + np.array([0.0, 0.7, 0.0])
    return eye, p + 1.6 * f + np.array([0.0, -0.3, 0.0]), 60.0


def snake_cam(V):
    """Three-quarter front, low, ~2.5 m off the head, on the open side (away from the sleeve)."""
    j, look, out = V["snake_joints"], V["snake_look"], V["snake_out"]
    fwd = _unit((j[2] - j[-1]) * np.array([1.0, 0.0, 1.0]))    # the body's heading, level
    eye = j[0] + 2.5 * _unit(0.6 * fwd + 1.0 * out) + np.array([0.0, 0.15, 0.0])
    tgt = 0.6 * j[4] + 0.4 * look
    return eye, tgt, 32.0
