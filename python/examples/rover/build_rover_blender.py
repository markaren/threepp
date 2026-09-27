"""Build the TP-1 rover blockout in Blender from rover_spec.json and export rover.glb.

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" --background --factory-startup \
        --python build_rover_blender.py -- --spec rover_spec.json --out rover.glb

All geometry is authored in the spec's chassis frame (X forward, Y up, Z right,
metres) and converted to Blender's Z-up only when a mesh or transform is handed
to bpy. The glTF exporter's Y-up conversion maps it straight back, so every
number in the spec is the number in the .glb.

Node hierarchy (all names exact):
    chassis                              root, identity, origin = chassis frame origin
      chassis_body                       keel + upper body (MLI), radiators, brackets, plates,
                                         hazcam bar, patch antenna
      solar_panels                       vertical arrays, both sides + rear
      mast                               mast tube, flange, collars, braces
        mast_pan                         pan drive + yoke, rotates about local +Y
          mast_tilt                      camera head, rotates about local +Z (rest -15 deg)
            navcam_L, navcam_R,          empties: look down local -Z, local +Y up
            headlight_L, headlight_R
      actuator_XX_barrel                 at the chassis mount, extends along local +Y
      leg_XX                             origin at the leg pivot, rotates about local +Z
        actuator_XX_rod                  at the arm mount, extends along local +Y
        wheel_XX                         origin at the hub centre, spins about local +Z
      hazcam_front                       empty: look down local -Z, local +Y up

The crinkled MLI is a procedural tangent-space normal map generated here with
numpy and packed into the .glb; every mesh gets box-projected UVs.
"""
import argparse
import json
import math
import os
import sys

LEGS = ("FL", "FR", "RL", "RR")


# ---------------------------------------------------------------- mesh accumulator (chassis frame)
class Part:
    """Polygons in the chassis (glTF) frame with a material slot per face."""

    def __init__(self):
        self.verts = []
        self.faces = []
        self.mats = []
        self.smooth = []

    def add(self, verts, faces, mat, M=None, smooth=None):
        from mathutils import Vector
        base = len(self.verts)
        for v in verts:
            p = Vector(v)
            if M is not None:
                p = M @ p
            self.verts.append((p.x, p.y, p.z))
        for i, f in enumerate(faces):
            self.faces.append(tuple(base + k for k in f))
            self.mats.append(mat)
            self.smooth.append(bool(smooth[i]) if smooth is not None else False)


def box(sx, sy, sz):
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    v = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
         (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
    return v, f, [False] * 6


def frustum(r0, r1, z0, z1, n, caps=True):
    """Solid of revolution about +Z from (r0, z0) to (r1, z1). Caps have their own vertices."""
    v, f, s = [], [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        c, sn = math.cos(a), math.sin(a)
        v.append((r0 * c, r0 * sn, z0))
        v.append((r1 * c, r1 * sn, z1))
    for i in range(n):
        j = (i + 1) % n
        f.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
        s.append(True)
    if caps:
        b0 = len(v)
        for i in range(n):
            a = 2 * math.pi * i / n
            v.append((r0 * math.cos(a), r0 * math.sin(a), z0))
        b1 = len(v)
        for i in range(n):
            a = 2 * math.pi * i / n
            v.append((r1 * math.cos(a), r1 * math.sin(a), z1))
        if r0 > 1e-9:
            f.append(tuple(b0 + i for i in reversed(range(n))))
            s.append(False)
        if r1 > 1e-9:
            f.append(tuple(b1 + i for i in range(n)))
            s.append(False)
    return v, f, s


def cylinder(r, z0, z1, n):
    return frustum(r, r, z0, z1, n)


def tube(ro, ri, z0, z1, n):
    """Hollow cylinder about +Z: outer and inner walls plus annular end caps."""
    v, f, s = [], [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        c, sn = math.cos(a), math.sin(a)
        v += [(ro * c, ro * sn, z0), (ro * c, ro * sn, z1), (ri * c, ri * sn, z0), (ri * c, ri * sn, z1)]
    for i in range(n):
        j = (i + 1) % n
        o0, o1, i0, i1 = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
        p0, p1, q0, q1 = 4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3
        f.append((o0, p0, p1, o1)); s.append(True)          # outer wall
        f.append((i0, i1, q1, q0)); s.append(True)          # inner wall
    # caps with their own vertices so the walls shade smooth and the rims stay crisp
    for (z, top) in ((z0, False), (z1, True)):
        b = len(v)
        for i in range(n):
            a = 2 * math.pi * i / n
            v += [(ro * math.cos(a), ro * math.sin(a), z), (ri * math.cos(a), ri * math.sin(a), z)]
        for i in range(n):
            j = (i + 1) % n
            q = (b + 2 * i, b + 2 * j, b + 2 * j + 1, b + 2 * i + 1)
            f.append(q if top else tuple(reversed(q)))
            s.append(False)
    return v, f, s


# ---------------------------------------------------------------- transforms (chassis frame)
def T(x, y, z):
    from mathutils import Matrix
    return Matrix.Translation((x, y, z))


def R(axis, deg):
    from mathutils import Matrix
    return Matrix.Rotation(math.radians(deg), 4, axis)


def Z_TO_X():
    return R("Y", 90.0)


def Z_TO_Y():
    return R("X", -90.0)


def frame_from_forward_up(fwd, up):
    """4x4 whose local -Z = fwd and local +Y = up (threepp camera convention)."""
    from mathutils import Matrix, Vector
    zb = -Vector(fwd).normalized()
    y = Vector(up)
    x = y.cross(zb).normalized()
    y = zb.cross(x).normalized()
    M = Matrix.Identity(4)
    for r in range(3):
        M[r][0], M[r][1], M[r][2] = x[r], y[r], zb[r]
    return M


def frame_y_toward(d):
    """4x4 whose local +Y points along d (for actuator halves)."""
    from mathutils import Vector
    d = Vector(d).normalized()
    ref = Vector((0.0, 0.0, 1.0)) if abs(d.z) < 0.9 else Vector((1.0, 0.0, 0.0))
    x = d.cross(ref).normalized()
    zb = x.cross(d).normalized()
    from mathutils import Matrix
    M = Matrix.Identity(4)
    for r in range(3):
        M[r][0], M[r][1], M[r][2] = x[r], d[r], zb[r]
    return M


def pitch_down(deg):
    """Forward and up vectors of a sensor pitched down by deg, looking along +X."""
    a = math.radians(deg)
    return (math.cos(a), -math.sin(a), 0.0), (math.sin(a), math.cos(a), 0.0)


def prism(poly_xz, y0, y1):
    """Vertical prism: convex polygon in (x, z), extruded from y0 to y1. Caps own their vertices."""
    n = len(poly_xz)
    area = sum(poly_xz[i][1] * poly_xz[(i + 1) % n][0] - poly_xz[(i + 1) % n][1] * poly_xz[i][0] for i in range(n))
    if area < 0:                       # want counter-clockwise about +Y (z -> x)
        poly_xz = list(reversed(poly_xz))
    v, f = [], []
    for (x, z) in poly_xz:
        v += [(x, y0, z), (x, y1, z)]
    for i in range(n):
        j = (i + 1) % n
        f.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
    b0 = len(v)
    v += [(x, y0, z) for (x, z) in poly_xz]
    b1 = len(v)
    v += [(x, y1, z) for (x, z) in poly_xz]
    f.append(tuple(b0 + i for i in reversed(range(n))))
    f.append(tuple(b1 + i for i in range(n)))
    return v, f, [False] * len(f)


def strut(part, p0, p1, r, mat, n=12):
    from mathutils import Vector
    d = Vector(p1) - Vector(p0)
    v, f, s = cylinder(r, 0.0, d.length, n)
    part.add(v, f, mat, T(*p0) @ frame_y_toward(d) @ Z_TO_Y(), s)


# ---------------------------------------------------------------- Blender side
def C_MAT():
    """Blender (Z-up) -> chassis/glTF (Y-up): gl = (x, z, -y)."""
    from mathutils import Matrix
    return Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))


def to_bl(M_gl):
    C = C_MAT()
    return C.inverted() @ M_gl @ C


def crinkle_normal_map(size, seed, octaves):
    """Faceted crinkled-foil normals: per octave a periodic jittered-grid Voronoi whose cells are
    tilted planes; slopes add across octaves. Returns float32 RGBA (size*size*4), OpenGL +Y."""
    import numpy as np
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    sx = np.zeros((size, size), np.float32)
    sy = np.zeros((size, size), np.float32)
    for (g, amp) in octaves:
        jit = rng.random((g, g, 2)).astype(np.float32)
        tilt = (rng.standard_normal((g, g, 2)) * amp).astype(np.float32)
        px, py = xx * g, yy * g
        ci, cj = np.floor(px).astype(np.int32), np.floor(py).astype(np.int32)
        best = np.full((size, size), 1e9, np.float32)
        bx = np.zeros((size, size), np.float32)
        by = np.zeros((size, size), np.float32)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = ci + di, cj + dj
                mi, mj = ni % g, nj % g
                cx = ni + jit[mj, mi, 0]
                cy = nj + jit[mj, mi, 1]
                d = (px - cx) ** 2 + (py - cy) ** 2
                m = d < best
                best = np.where(m, d, best)
                bx = np.where(m, tilt[mj, mi, 0], bx)
                by = np.where(m, tilt[mj, mi, 1], by)
        sx += bx
        sy += by
    nz = 1.0 / np.sqrt(1.0 + sx * sx + sy * sy)
    rgba = np.empty((size, size, 4), np.float32)
    rgba[..., 0] = 0.5 - 0.5 * sx * nz
    rgba[..., 1] = 0.5 - 0.5 * sy * nz
    rgba[..., 2] = 0.5 + 0.5 * nz
    rgba[..., 3] = 1.0
    return rgba.reshape(-1)


def make_materials(spec):
    import bpy
    M = spec["materials"]
    images = {}
    for iname, d in sorted(M.get("normal_maps", {}).items()):
        img = bpy.data.images.new(iname, d["size"], d["size"], alpha=False, float_buffer=False)
        img.colorspace_settings.name = "Non-Color"
        img.pixels.foreach_set(crinkle_normal_map(d["size"], d["seed"], d["octaves"]))
        img.pack()
        images[iname] = img
    mats = {}
    for name in sorted(k for k in M if k not in ("note", "normal_maps")):
        d = M[name]
        m = bpy.data.materials.new(name)
        try:
            m.use_nodes = True
        except Exception:
            pass
        m.use_backface_culling = True
        nt = m.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        c = d["color"]
        bsdf.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1.0)
        bsdf.inputs["Metallic"].default_value = d["metallic"]
        bsdf.inputs["Roughness"].default_value = d["roughness"]
        if "emissive" in d:
            e = d["emissive"]
            bsdf.inputs["Emission Color"].default_value = (e[0], e[1], e[2], 1.0)
            bsdf.inputs["Emission Strength"].default_value = d["emissive_strength"]
        if "normal_map" in d:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = images[d["normal_map"]]
            nm = nt.nodes.new("ShaderNodeNormalMap")
            nm.inputs["Strength"].default_value = d.get("normal_scale", 1.0)
            nt.links.new(tex.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
        mats[name] = m
    return mats


def box_uvs(part, tile):
    """Per-corner box-projected UVs in metres / tile, from each face's Newell normal."""
    uvs = []
    V = part.verts
    for f in part.faces:
        nx = ny = nz = 0.0
        for i in range(len(f)):
            a, b = V[f[i]], V[f[(i + 1) % len(f)]]
            nx += (a[1] - b[1]) * (a[2] + b[2])
            ny += (a[2] - b[2]) * (a[0] + b[0])
            nz += (a[0] - b[0]) * (a[1] + b[1])
        ax = max((abs(nx), 0), (abs(ny), 1), (abs(nz), 2))[1]
        for k in f:
            x, y, z = V[k]
            uvs += [(z, y), (x, z), (x, y)][ax]
    return [c / tile for c in uvs]


def new_object(name, part, mats, parent, M_gl, props=None, tile=0.6):
    import bpy
    coll = bpy.context.scene.collection
    if part is None:
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_type = "ARROWS"
        obj.empty_display_size = 0.08
    else:
        used = sorted(set(part.mats))
        slot = {m: i for i, m in enumerate(used)}
        me = bpy.data.meshes.new(name)
        verts = [(x, -z, y) for (x, y, z) in part.verts]     # chassis frame -> Blender Z-up
        me.from_pydata(verts, [], part.faces)
        me.polygons.foreach_set("material_index", [slot[m] for m in part.mats])
        me.polygons.foreach_set("use_smooth", part.smooth)
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", box_uvs(part, tile))
        for m in used:
            me.materials.append(mats[m])
        me.validate(verbose=False)
        me.update()
        obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    obj.matrix_basis = to_bl(M_gl)
    for k, v in (props or {}).items():
        obj[k] = v
    return obj


def text_mesh(body, height, depth):
    """Vertices and faces of a text string: u along the text, v up, w out of the face."""
    import bpy
    cu = bpy.data.curves.new("tmp_text", type="FONT")
    cu.body = body
    cu.size = height
    cu.extrude = 0.5 * depth
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.resolution_u = 4
    ob = bpy.data.objects.new("tmp_text", cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    v = [tuple(p.co) for p in me.vertices]
    f = [tuple(p.vertices) for p in me.polygons]
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    bpy.data.meshes.remove(me)
    ys = [p[1] for p in v]
    k = height / max(1e-9, (max(ys) - min(ys)))
    cx = 0.5 * (max(p[0] for p in v) + min(p[0] for p in v))
    cy = 0.5 * (max(ys) + min(ys))
    return [((p[0] - cx) * k, (p[1] - cy) * k, p[2]) for p in v], f


def sensor_frame(d):
    fwd, up = pitch_down(d["local_pitch_down_deg"])
    return T(*d["local_position"]) @ frame_from_forward_up(fwd, up)


# ---------------------------------------------------------------- the rover
def build_body(spec, mats, root):
    CH, SE = spec["chassis"], spec["sensors"]
    I = T(0, 0, 0)
    body = Part()
    # keel: battery box between the legs, gold MLI
    ks, kc = CH["keel"]["size"], CH["keel"]["center"]
    body.add(*box(*ks)[:2], "mli_gold_foil", T(*kc))
    kr = CH["keel_radiator"]
    for side in (-1, 1):
        zf = side * (0.5 * ks[2] + 0.5 * kr["thickness"])
        body.add(*box(kr["x_range"][1] - kr["x_range"][0], kr["y_range"][1] - kr["y_range"][0], kr["thickness"])[:2],
                 "radiator_white", T(0.5 * sum(kr["x_range"]), 0.5 * sum(kr["y_range"]), zf))
        for k in range(7):     # radiator ribs
            x = kr["x_range"][0] + (k + 0.5) * (kr["x_range"][1] - kr["x_range"][0]) / 7
            body.add(*box(0.012, kr["y_range"][1] - kr["y_range"][0] - 0.02, 0.006)[:2], "radiator_white",
                     T(x, 0.5 * sum(kr["y_range"]), zf + side * 0.008))
    # upper body: chamfered prism spanning the footprint, wheels tucked under its edges
    us, uc, ch = CH["upper_body"]["size"], CH["upper_body"]["center"], CH["upper_body"]["corner_chamfer"]
    hx, hz = 0.5 * us[0], 0.5 * us[2]
    poly = [(hx - ch, hz), (hx, hz - ch), (hx, -hz + ch), (hx - ch, -hz), (-hx + ch, -hz), (-hx, -hz + ch),
            (-hx, hz - ch), (-hx + ch, hz)]
    y0, y1 = uc[1] - 0.5 * us[1], uc[1] + 0.5 * us[1]
    body.add(*prism(poly, y0, y1)[:2], "mli_gold_foil")
    # silver edge trims top and bottom
    for (ya, yb) in ((y0 - 0.004, y0 + 0.018), (y1 - 0.018, y1 + 0.004)):
        pe = [(x * (1 + 0.006 / hx), z * (1 + 0.006 / hz)) for (x, z) in poly]
        body.add(*prism(pe, ya, yb)[:2], "mli_silver_foil")
    # deck: silver foil top with two ribbed white radiators
    body.add(*prism([(x * 0.97, z * 0.97) for (x, z) in poly], y1, y1 + 0.006)[:2], "mli_silver_foil")
    for dr in CH["deck_radiators"]:
        xr, zr = dr["x_range"], dr["z_range"]
        body.add(*box(xr[1] - xr[0], 0.012, zr[1] - zr[0])[:2], "radiator_white",
                 T(0.5 * sum(xr), y1 + 0.012, 0.5 * sum(zr)))
        for k in range(9):
            x = xr[0] + (k + 0.5) * (xr[1] - xr[0]) / 9
            body.add(*box(0.014, 0.008, zr[1] - zr[0] - 0.03)[:2], "radiator_white", T(x, y1 + 0.022, 0.5 * sum(zr)))
    # leg pivot brackets and actuator clevis brackets
    bx, by, bz = CH["pivot_bracket_size"]
    for leg in LEGS:
        p = spec["legs"]["pivots"][leg]
        side = 1 if p[2] > 0 else -1
        span = abs(p[2]) - 0.5 * ks[2] - 0.02
        body.add(*box(bx, by, span)[:2], "leg_aluminium", T(p[0], p[1], side * (0.5 * ks[2] + 0.5 * span)))
        body.add(*box(bx + 0.03, by + 0.03, 0.01)[:2], "leg_aluminium", T(p[0], p[1], side * (0.5 * ks[2] + 0.005)))
        m = spec["actuators"]["chassis_mount"][leg]
        span = abs(m[2]) - 0.5 * ks[2] + 0.03
        body.add(*box(0.07, 0.025, span)[:2], "leg_aluminium",
                 T(m[0], m[1] + 0.0125, (1 if m[2] > 0 else -1) * (0.5 * ks[2] + 0.5 * span - 0.01)))
        for dz in (-0.03, 0.03):
            body.add(*box(0.04, 0.035, 0.008)[:2], "leg_aluminium", T(m[0], m[1] - 0.0175, m[2] + dz))
    # designation plates on the upper body, both sides, reading correctly from outside
    pl = CH["designation_plate"]
    tv, tf = text_mesh(pl["text"], pl["text_height"], 0.003)
    px0, px1 = pl["x_range"]
    py0, py1 = pl["y_range"]
    for side in (-1, 1):
        zf = side * (hz + 0.004)
        body.add(*box(px1 - px0, py1 - py0, 0.004)[:2], "plate_grey", T(0.5 * (px0 + px1), 0.5 * (py0 + py1), zf))
        Mt = T(0.5 * (px0 + px1), 0.5 * (py0 + py1), zf + side * 0.0035) @ (I if side > 0 else R("Y", 180.0))
        body.add(tv, tf, "plate_text_black", Mt)
    # hazcam stereo bar on the upper-body nose
    F = T(0, 0, 0) @ sensor_frame(SE["hazcam_front"])
    body.add(*box(0.22, 0.075, 0.06)[:2], "camera_body_black", F @ T(0.0, 0.0, 0.042))
    for lx in (-0.065, 0.065):
        v, f, s = cylinder(0.018, -0.014, 0.0, 20)
        body.add(v, f, "lens_glass", F @ T(lx, 0.0, 0.0), s)
    # patch antenna: flat, low profile, on a plinth
    an = spec["antenna"]
    ax, ay, az = an["center"]
    body.add(*box(0.10, an["plinth_height"], 0.10)[:2], "leg_aluminium", T(ax, ay + 0.5 * an["plinth_height"] + 0.006, az))
    sx, sy, sz = an["size"]
    body.add(*box(sx, sy, sz)[:2], "antenna_patch", T(ax, ay + an["plinth_height"] + 0.006 + 0.5 * sy, az))
    new_object("chassis_body", body, mats, root, I, tile=CH["mli_tile"])


def build_solar(spec, mats, root):
    SP = spec["solar_panels"]
    part = Part()
    y0, y1 = SP["y_range"]
    th = SP["thickness"]
    nu, nv = SP["cell_grid"]
    orient = {"+Z": T(0, 0, 0), "-Z": R("Y", 180.0), "-X": R("Y", -90.0)}
    for p in SP["panels"]:
        # panel-local: u along, v up, w outward; the plane value is the panel's inner face
        Rm = orient[p["face"]]
        sgn = 1.0 if p["face"] in ("+Z",) else -1.0
        w_in = abs(p["plane"])
        cu = p["center_u"] if p["face"] != "-Z" else -p["center_u"]
        W = p["width"]
        vc = 0.5 * (y0 + y1)

        def P(u, v, w):
            return Rm @ T(u, v, w)
        part.add(*box(W, y1 - y0, th)[:2], "solar_back", P(cu, vc, w_in + 0.5 * th))
        fr = 0.02
        for (du, dv, su, sv) in ((0, 0.5 * (y1 - y0) - 0.5 * fr, W, fr), (0, -0.5 * (y1 - y0) + 0.5 * fr, W, fr),
                                 (0.5 * W - 0.5 * fr, 0, fr, y1 - y0), (-0.5 * W + 0.5 * fr, 0, fr, y1 - y0)):
            part.add(*box(su, sv, 0.008)[:2], "solar_frame", P(cu + du, vc + dv, w_in + th + 0.004))
        gu, gv = (W - 2 * fr) / nu, (y1 - y0 - 2 * fr) / nv
        for i in range(nu):
            for j in range(nv):
                part.add(*box(gu - 0.006, gv - 0.006, 0.003)[:2], "solar_cell",
                         P(cu - 0.5 * W + fr + (i + 0.5) * gu, y0 + fr + (j + 0.5) * gv, w_in + th + 0.0015))
        # standoff brackets back to the body
        for du in (-0.35 * W, 0.35 * W):
            for v in (y0 + 0.06, min(y1 - 0.08, 0.85)):
                part.add(*box(0.04, 0.03, SP["standoff"] + 0.01)[:2], "solar_frame",
                         P(cu + du, v, w_in - 0.5 * SP["standoff"]))
    new_object("solar_panels", part, mats, root, T(0, 0, 0))


def build_mast(spec, mats, root):
    MA, SE = spec["mast"], spec["sensors"]
    mast = Part()
    bx0, by0, bz0 = MA["base"]
    v, f, s = cylinder(MA["base_flange_radius"], 0.0, 0.03, 32)
    mast.add(v, f, "leg_aluminium", T(bx0, by0, bz0) @ Z_TO_Y(), s)
    v, f, s = cylinder(MA["tube_radius"], by0, MA["tube_top_y"], 32)
    mast.add(v, f, "mast_white", T(bx0, 0.0, bz0) @ Z_TO_Y(), s)
    for yc in (by0 + 0.08, 0.5 * (by0 + MA["tube_top_y"]) + 0.08, MA["tube_top_y"] - 0.04):
        v, f, s = cylinder(MA["tube_radius"] + 0.012, yc - 0.02, yc + 0.02, 32)
        mast.add(v, f, "leg_aluminium", T(bx0, 0.0, bz0) @ Z_TO_Y(), s)
    br = MA["braces"]
    for p0 in br["deck"]:
        strut(mast, p0, (bx0 - 0.04, br["mast_y"], bz0 + 0.3 * p0[2]), br["radius"], "leg_aluminium")
    mo = new_object("mast", mast, mats, root, T(0, 0, 0))

    # pan unit: azimuth drive + yoke
    pa, ti = MA["pan"], MA["tilt"]
    pan = Part()
    v, f, s = cylinder(pa["drive_radius"], 0.0, pa["drive_height"], 32)
    pan.add(v, f, "actuator_steel", Z_TO_Y(), s)
    hz = 0.5 * MA["head_size"][2]
    pan.add(*box(0.12, 0.025, 2 * hz + 0.08)[:2], "leg_aluminium", T(0.0, pa["drive_height"] + 0.0125, 0.0))
    ty = ti["origin_in_pan"][1]
    for side in (-1, 1):
        zc = side * (hz + 0.03)
        pan.add(*box(0.11, ty - pa["drive_height"] + 0.05, 0.02)[:2], "leg_aluminium",
                T(0.0, 0.5 * (pa["drive_height"] + ty + 0.05), zc))
        v, f, s = cylinder(0.04, 0.0, 0.05, 24)
        pan.add(v, f, "actuator_steel", T(0.0, ty, zc + side * 0.01) @ (T(0, 0, 0) if side > 0 else R("Y", 180.0)), s)
    po = new_object(pa["node"], pan, mats, mo, T(*pa["origin"]), props={"axis": pa["axis_local"]})

    # tilt head: camera bar with stereo navcams and headlights
    head = Part()
    hs, ho = MA["head_size"], MA["head_offset"]
    head.add(*box(*hs)[:2], "mast_white", T(*ho))
    head.add(*box(hs[0] + 0.07, 0.012, hs[2] + 0.04)[:2], "mli_silver_foil", T(ho[0] + 0.02, 0.5 * hs[1] + 0.006, 0.0))
    v, f, s = cylinder(0.022, -(hz + 0.035), hz + 0.035, 20)
    head.add(v, f, "actuator_steel", T(0, 0, 0), s)
    for name in ("navcam_L", "navcam_R"):
        F = sensor_frame(SE[name])
        head.add(*box(0.075, 0.075, 0.09)[:2], "camera_body_black", F @ T(0.0, 0.0, 0.047))
        v, f, s = cylinder(0.026, -0.004, 0.0, 24)
        head.add(v, f, "camera_body_black", F @ T(0, 0, 0.002), s)
        v, f, s = cylinder(0.02, -0.002, 0.0, 24)
        head.add(v, f, "lens_glass", F, s)
    for name in ("headlight_L", "headlight_R"):
        F = sensor_frame(SE[name])
        v, f, s = cylinder(0.042, 0.004, 0.075, 24)
        head.add(v, f, "camera_body_black", F, s)
        v, f, s = cylinder(0.035, 0.0, 0.004, 24)
        head.add(v, f, "headlight_emitter", F, s)
    Mt = T(*ti["origin_in_pan"]) @ R("Z", ti["rest_deg"])
    to = new_object(ti["node"], head, mats, po, Mt, props={"axis": ti["axis_local"], "rest_deg": ti["rest_deg"]})
    for name in ("navcam_L", "navcam_R", "headlight_L", "headlight_R"):
        d = SE[name]
        props = {"convention": "look -Z, up +Y"}
        props.update({k: v for k, v in d.items() if k in ("hfov_deg", "cone_deg")})
        new_object(name, None, mats, to, sensor_frame(d), props=props)


def build_leg(spec, mats, root, leg):
    from mathutils import Vector
    W, LG, AC = spec["wheel"], spec["legs"], spec["actuators"]
    I = T(0, 0, 0)
    piv = Vector(LG["pivots"][leg])
    hub = Vector(spec["hubs"][leg])
    side = 1 if hub.z > 0 else -1
    arm = hub - piv
    ang = math.degrees(math.atan2(arm.y, arm.x))
    L2 = math.hypot(arm.x, arm.y)
    aw, at = LG["arm_section"]
    legp = Part()
    v, f, s = cylinder(0.048, -0.032, 0.032, 32)
    legp.add(v, f, "leg_aluminium", I, s)
    v, f, s = cylinder(0.02, -0.04, 0.04, 16)
    legp.add(v, f, "hub_titanium", I, s)
    Ra = R("Z", ang)
    # box-section arm with a web and two flanges (reads as a machined beam)
    legp.add(*box(L2, aw, 0.012)[:2], "leg_aluminium", Ra @ T(0.5 * L2, 0.0, 0.0))
    for dz in (-0.5 * at + 0.004, 0.5 * at - 0.004):
        legp.add(*box(L2 - 0.02, aw + 0.012, 0.008)[:2], "leg_aluminium", Ra @ T(0.5 * L2, 0.0, dz))
    for dy in (-0.5 * aw, 0.5 * aw):
        legp.add(*box(L2 - 0.03, 0.006, at)[:2], "leg_aluminium", Ra @ T(0.5 * L2, dy, 0.0))
    kx, ky = arm.x, arm.y
    v, f, s = cylinder(0.055, -0.032, 0.032, 32)
    legp.add(v, f, "leg_aluminium", T(kx, ky, 0.0), s)
    v, f, s = cylinder(0.032, 0.0, arm.z, 20) if side > 0 else cylinder(0.032, arm.z, 0.0, 20)
    legp.add(v, f, "hub_titanium", T(kx, ky, 0.0), s)
    am = Vector(AC["arm_mount_leg_frame"][leg])
    for dz in (-0.03, 0.03):
        legp.add(*box(0.05, 0.05, 0.008)[:2], "leg_aluminium", T(am.x, am.y + 0.022, dz))
    lo = new_object(f"leg_{leg}", legp, mats, root, T(*piv), props={
        "pivot_axis": "+Z", "angle_limits_deg": list(LG["angle_limits_deg"][leg]),
        "hub_up_sign": LG["hub_up_sign"][leg]})

    # wheel: origin at the hub, spin about local +Z. Grousers exactly as the spec says.
    wp = Part()
    r, b, hw = W["rim_radius"], W["width"], 0.5 * W["width"]
    ri = W["rim_inner_radius"]
    v, f, s = tube(r, ri, -hw, hw, 96)
    wp.add(v, f, "wheel_black_anodised", I, s)
    for zc in (-hw + 0.008, hw - 0.008, 0.0):          # stiffening flanges inside the rim
        v, f, s = tube(ri + 0.001, ri - 0.02, zc - 0.006, zc + 0.006, 96)
        wp.add(v, f, "wheel_black_anodised", I, s)
    g = W["grouser"]
    for k in range(g["count"]):
        a = g["phase_deg"] + 360.0 * k / g["count"]
        wp.add(*box(g["height"], g["thickness"], g["length"])[:2], "wheel_black_anodised",
               R("Z", a) @ T(r + 0.5 * g["height"], 0.0, 0.0))
    ns = W.get("spokes", 8)
    for k in range(ns):            # flat blade spokes, staggered on two planes (dished)
        a = 360.0 * (k + 0.5) / ns
        zc = 0.035 if k % 2 == 0 else -0.035
        rs0, rs1 = W["hub_radius"] + 0.01, ri - 0.018
        wp.add(*box(rs1 - rs0, 0.016, 0.028)[:2], "wheel_black_anodised",
               R("Z", a) @ T(0.5 * (rs0 + rs1), 0.0, zc))
    v, f, s = cylinder(W["hub_radius"], -0.5 * W["hub_length"], 0.5 * W["hub_length"], 48)
    wp.add(v, f, "hub_titanium", I, s)
    for zc in (-0.035, 0.035):
        v, f, s = cylinder(W["hub_radius"] + 0.025, zc - 0.012, zc + 0.012, 48)
        wp.add(v, f, "hub_titanium", I, s)
    zo = side * (0.5 * W["hub_length"])
    v, f, s = cylinder(0.05, 0.0, 0.012, 32)
    wp.add(v, f, "hub_titanium", T(0, 0, zo) @ (I if side > 0 else R("Y", 180.0)), s)
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        v, f, s = cylinder(0.009, 0.0, 0.01, 6)
        wp.add(v, f, "actuator_steel", T(0.065 * math.cos(a), 0.065 * math.sin(a), zo) @
               (I if side > 0 else R("Y", 180.0)), s)
    new_object(f"wheel_{leg}", wp, mats, lo, T(*(hub - piv)), props={
        "spin_axis": "+Z", "forward_roll_sign": -1, "rim_radius": r, "width": b,
        "grouser_count": g["count"], "grouser_height": g["height"], "grouser_thickness": g["thickness"]})

    # actuator: barrel (cylinder) on the chassis, rod on the arm, both along local +Y
    cm = Vector(AC["chassis_mount"][leg])
    am_ch = piv + am
    brl = Part()
    v, f, s = cylinder(AC["barrel_radius"], 0.0, AC["barrel_length"], 24)
    brl.add(v, f, "actuator_steel", Z_TO_Y(), s)
    v, f, s = cylinder(AC["barrel_radius"] + 0.006, AC["barrel_length"] - 0.03, AC["barrel_length"], 24)
    brl.add(v, f, "actuator_steel", Z_TO_Y(), s)
    v, f, s = cylinder(0.02, -0.03, 0.03, 16)
    brl.add(v, f, "hub_titanium", I, s)
    v, f, s = cylinder(0.022, 0.03, 0.12, 16)           # motor pod alongside the barrel
    brl.add(v, f, "actuator_steel", T(0.045, 0.0, 0.0) @ Z_TO_Y(), s)
    new_object(f"actuator_{leg}_barrel", brl, mats, root, T(*cm) @ frame_y_toward(am_ch - cm))
    rod = Part()
    v, f, s = cylinder(AC["rod_radius"], 0.0, AC["rod_length"], 20)
    rod.add(v, f, "actuator_rod_chrome", Z_TO_Y(), s)
    v, f, s = cylinder(0.02, -0.03, 0.03, 16)
    rod.add(v, f, "hub_titanium", I, s)
    new_object(f"actuator_{leg}_rod", rod, mats, lo, T(*am) @ frame_y_toward(cm - am_ch))


def build(spec):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    root = new_object("chassis", None, mats, None, T(0, 0, 0), props={
        "frame": "X forward, Y up, Z right; origin = hub-centroid at nominal",
        "spec": "rover_spec.json", "mass_total": spec["mass"]["total"],
        "com": list(spec["mass"]["com"])})
    build_body(spec, mats, root)
    build_solar(spec, mats, root)
    build_mast(spec, mats, root)
    for leg in LEGS:
        build_leg(spec, mats, root, leg)
    d = spec["sensors"]["hazcam_front"]
    new_object("hazcam_front", None, mats, root, sensor_frame(d),
               props={"convention": "look -Z, up +Y", "hfov_deg": d["hfov_deg"]})
    return root


def leg_hub(spec, leg, e):
    """Hub (x, y) in the chassis frame at vertical extension e (pivot -> hub drop)."""
    LG = spec["legs"]
    p, q = LG["pivots"][leg], spec["hubs"][leg]
    sgn = 1 if q[0] > p[0] else -1
    phi = math.asin(e / LG["arm_length"])
    return p[0] + sgn * LG["arm_length"] * math.cos(phi), p[1] - e, phi, sgn


def check_spec(spec):
    """Consistency checks between the spec's authored numbers. Fails loudly."""
    W, LG, AC, CH = spec["wheel"], spec["legs"], spec["actuators"], spec["chassis"]
    rt = W["grouser_tip_radius"]
    assert abs(rt - (W["rim_radius"] + W["grouser"]["height"])) < 1e-9
    h = spec["hubs"]
    assert abs(spec["wheelbase"] - (h["FL"][0] - h["RL"][0])) < 1e-6
    assert abs(spec["track"] - (h["FR"][2] - h["FL"][2])) < 1e-6
    under = CH["upper_body"]["center"][1] - 0.5 * CH["upper_body"]["size"][1]
    panel_bottom = spec["solar_panels"]["y_range"][0]
    worst = 1e9
    for leg in LEGS:
        p, q = LG["pivots"][leg], h[leg]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        assert abs(L - LG["arm_length"]) < 1e-5, (leg, L)
        assert abs((p[1] - q[1]) - LG["nominal_extension"]) < 1e-9
        phi0 = math.atan2(p[1] - q[1], abs(q[0] - p[0]))
        e0, e1 = LG["extension_range"]
        assert abs((e1 - e0) - LG["stroke"]) < 1e-9
        lim = sorted([math.degrees(phi0 - math.asin(e / L)) for e in (e0, e1)])
        if LG["hub_up_sign"][leg] < 0:
            lim = sorted(-a for a in lim)
        want = LG["angle_limits_deg"][leg]
        assert abs(lim[0] - want[0]) < 2e-3 and abs(lim[1] - want[1]) < 2e-3, (leg, lim, want)
        u = ((q[0] - p[0]) / L, (q[1] - p[1]) / L)
        am = AC["arm_mount_leg_frame"][leg]
        assert abs(am[0] - u[0] * AC["arm_mount_distance"]) < 1e-5 and abs(am[1] - u[1] * AC["arm_mount_distance"]) < 1e-5
        cm = AC["chassis_mount"][leg]
        lens = []
        for k in range(41):
            e = e0 + (e1 - e0) * k / 40
            hx, hy, phi, sgn = leg_hub(spec, leg, e)
            ax = p[0] + sgn * AC["arm_mount_distance"] * math.cos(phi)
            ay = p[1] - AC["arm_mount_distance"] * math.sin(phi)
            lens.append(math.hypot(ax - cm[0], ay - cm[1]))
            worst = min(worst, min(under, panel_bottom) - (hy + rt))     # wheel top vs body/panels
        assert AC["barrel_length"] + AC["rod_length"] > max(lens) + 0.02, (leg, lens)
        assert max(AC["barrel_length"], AC["rod_length"]) < min(lens) - 0.02, (leg, lens)
        # lateral: wheel inner face clear of keel, brackets and actuators
        inner = abs(q[2]) - 0.5 * W["width"]
        assert inner - (abs(cm[2]) + AC["barrel_radius"] + 0.006) > 0.005, leg
    assert worst > 0.03, worst
    m = spec["mass"]
    tot = sum(b["mass"] for b in m["budget"])
    assert abs(tot - m["total"]) < 1e-9, tot
    com = [sum(b["mass"] * b["centroid"][i] for b in m["budget"]) / tot for i in range(3)]
    assert all(abs(com[i] - m["com"][i]) < 0.005 for i in range(3)), com
    return com, worst


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="rover_spec.json")
    ap.add_argument("--out", default="rover.glb")
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    spec_path = a.spec if os.path.isabs(a.spec) else os.path.join(here, a.spec)
    out_path = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    com, clear = check_spec(spec)
    print(f"[rover] spec ok; CoM from budget = ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}); "
          f"min wheel-top clearance over stroke = {clear * 1000:.0f} mm")

    import bpy
    build(spec)
    bpy.context.view_layer.update()
    tris, ymax = 0, -1e9
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH":
            tris += sum(len(p.vertices) - 2 for p in ob.data.polygons)
            mw = ob.matrix_world
            ymax = max(ymax, max((mw @ v.co).z for v in ob.data.vertices))    # Blender Z = chassis Y
    bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB", export_yup=True,
                              export_apply=False, export_extras=True, export_cameras=False,
                              export_lights=False, use_selection=False, export_animations=False,
                              export_tangents=True, export_texcoords=True, export_normals=True)
    print(f"[rover] wrote {out_path}  triangles={tris}  top y={ymax:.4f} "
          f"(height above ground {ymax + spec['wheel']['grouser_tip_radius']:.3f} m)")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
