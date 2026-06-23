"""threepp.urdf — import a URDF into a PhysX articulation you can simulate.

    import threepp as tp
    from threepp.urdf import load_articulation

    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81))
    robot = load_articulation(world, "arm.urdf", fixed_base=True,
                              stiffness=200, damping=20, max_force=300)
    for m in robot.meshes:
        scene.add(m)                       # render the colliders (bound to the sim)
    robot.set_targets(target_angles)       # drive the joints (add_link order)
    world.step(dt)

Builds the kinematic tree, places each link by forward kinematics (handling
`<origin rpy=...>`), and creates a reduced-coordinate `Articulation`:

  * collision: `<box>` / `<sphere>` / `<cylinder>` (→ capsule) directly; `<mesh>`
    is approximated by its **bounding box** (articulation links take primitive
    shapes, not trimeshes). Falls back to the `<visual>` mesh if a link has no
    `<collision>`.
  * mass: from `<inertial>` if present, else `default_density` × shape volume.
  * joints: revolute / prismatic become articulation DOFs (with limits + an
    optional PD drive); `fixed` joints are collapsed into their parent frame.

It's an approximation (primitive/bbox collision), not a digital twin — but it
turns "hand-build the robot" into one call. PhysX needs a PhysX-enabled threepp
build (`tp.HAS_PHYSX`).
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

from . import threepp as tp


# --------------------------------------------------------------------------- #
#  small math helpers (4x4 homogeneous transforms)
# --------------------------------------------------------------------------- #
def _floats(s, n, default):
    if not s:
        return list(default)
    v = [float(x) for x in s.split()]
    return (v + list(default))[:n]


def _rpy_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                     # URDF extrinsic XYZ


def _origin_T(elem):
    o = elem.find("origin") if elem is not None else None
    xyz = _floats(o.get("xyz") if o is not None else None, 3, (0, 0, 0))
    rpy = _floats(o.get("rpy") if o is not None else None, 3, (0, 0, 0))
    T = np.eye(4)
    T[:3, :3] = _rpy_R(*rpy)
    T[:3, 3] = xyz
    return T


def _quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return x, y, z, w


# --------------------------------------------------------------------------- #
#  collision shape -> (threepp primitive mesh placed in world, volume)
# --------------------------------------------------------------------------- #
def _shape_from_geometry(geom, base_dir):
    """Return (mesh, local_center, volume) for a <geometry>, or None.
    local_center is the shape centre relative to the link frame's collision origin."""
    box = geom.find("box")
    if box is not None:
        x, y, z = _floats(box.get("size"), 3, (0.1, 0.1, 0.1))
        return tp.Mesh(tp.BoxGeometry(x, y, z), tp.MeshStandardMaterial()), np.zeros(3), x * y * z
    sph = geom.find("sphere")
    if sph is not None:
        r = float(sph.get("radius", "0.05"))
        return tp.Mesh(tp.SphereGeometry(r), tp.MeshStandardMaterial()), np.zeros(3), 4 / 3 * math.pi * r ** 3
    cyl = geom.find("cylinder")
    if cyl is not None:                      # approximate with a capsule along local Y
        r = float(cyl.get("radius", "0.05")); ln = float(cyl.get("length", "0.1"))
        return (tp.Mesh(tp.CapsuleGeometry(r, ln), tp.MeshStandardMaterial()), np.zeros(3),
                math.pi * r * r * ln + 4 / 3 * math.pi * r ** 3)
    msh = geom.find("mesh")
    if msh is not None:                      # approximate the trimesh by its bounding box
        path = os.path.join(base_dir, msh.get("filename"))
        if not os.path.exists(path):
            return None
        if path.lower().endswith(".stl"):
            obj = tp.Mesh(tp.STLLoader().load(path), tp.MeshStandardMaterial())
        else:
            obj = tp.ModelLoader().load(path)
        bb = tp.Box3().set_from_object(obj)
        if bb.is_empty():
            return None
        sz, ctr = bb.get_size(), bb.get_center()
        sc = _floats(msh.get("scale"), 3, (1, 1, 1))
        dims = np.array([max(sz.x * sc[0], 1e-3), max(sz.y * sc[1], 1e-3), max(sz.z * sc[2], 1e-3)])
        center = np.array([ctr.x * sc[0], ctr.y * sc[1], ctr.z * sc[2]])
        return (tp.Mesh(tp.BoxGeometry(*dims), tp.MeshStandardMaterial()), center, float(np.prod(dims)))
    return None


def _link_collider(link, world_T, base_dir):
    """Build the collider mesh for a link, placed at its world pose.
    Returns (mesh, volume, link->collider transform) or None. The last item lets a caller
    place a visual mesh under the collider (the world transform cancels in inv(local) @ visual)."""
    src = link.find("collision")
    if src is None:
        src = link.find("visual")           # fall back to the visual mesh's bbox
    if src is None:
        return None
    geom = src.find("geometry")
    if geom is None:
        return None
    res = _shape_from_geometry(geom, base_dir)
    if res is None:
        return None
    mesh, shape_center, vol = res
    local = _origin_T(src) @ _trans(shape_center)        # link frame -> shape centre
    wp = world_T @ local
    mesh.position.set(*[float(v) for v in wp[:3, 3]])
    qx, qy, qz, qw = _quat_from_R(wp[:3, :3])
    mesh.quaternion.set(float(qx), float(qy), float(qz), float(qw))
    return mesh, vol, local


def _trans(v):
    T = np.eye(4); T[:3, 3] = v
    return T


def _attach_visual(collider, collider_local, link, base_dir):
    """If `link` has a <visual><mesh>, load the real mesh and parent it under `collider` so it
    tracks the sim, then hide the collider's primitive via its material (the object stays visible
    so the child still renders). Returns True if a visual was attached.

    `collider_local` is the link-frame -> collider transform from `_link_collider`; the visual's
    local offset under the collider is inv(collider_local) @ <visual origin>, with the link's
    world transform cancelling out. .obj/.dae/.stl all load through ModelLoader; ignore_up keeps
    the mesh in the link frame (URDF owns orientation) instead of applying a file up-axis."""
    vis = link.find("visual")
    geom = vis.find("geometry") if vis is not None else None
    msh = geom.find("mesh") if geom is not None else None
    if msh is None or not msh.get("filename"):
        return False
    path = os.path.join(base_dir, msh.get("filename"))
    if not os.path.exists(path):
        return False
    loader = tp.ModelLoader(); loader.set_ignore_up_direction(True)
    obj = loader.load(path)                              # Group, or None on failure
    if obj is None:
        return False
    sx, sy, sz = _floats(msh.get("scale"), 3, (1, 1, 1))
    obj.scale.set(float(sx), float(sy), float(sz))
    vloc = np.linalg.inv(collider_local) @ _origin_T(vis)
    obj.position.set(*[float(v) for v in vloc[:3, 3]])
    qx, qy, qz, qw = _quat_from_R(vloc[:3, :3])
    obj.quaternion.set(float(qx), float(qy), float(qz), float(qw))
    obj.traverse(lambda o: setattr(o, "cast_shadow", True))
    collider.add(obj)
    collider.cast_shadow = False                        # the visual casts; the hidden box must not
    collider.material.visible = False
    return True


# --------------------------------------------------------------------------- #
#  the handle
# --------------------------------------------------------------------------- #
class UrdfArticulation:
    def __init__(self, articulation, meshes, joint_names):
        self.articulation = articulation
        self.meshes = meshes                 # collider meshes (bound to the sim → add to a scene)
        self.joint_names = joint_names       # actuated joints, in add_link / drive-target order

    @property
    def num_dof(self):
        return len(self.joint_names)

    def set_targets(self, values):
        self.articulation.set_drive_targets(np.asarray(values, np.float32))

    def positions(self):
        return self.articulation.joint_positions()

    def velocities(self):
        return self.articulation.joint_velocities()


# --------------------------------------------------------------------------- #
#  the importer
# --------------------------------------------------------------------------- #
def load_articulation(world, path, *, fixed_base=False, base_position=(0, 0, 0),
                      default_density=1000.0, stiffness=0.0, damping=0.0, max_force=1e6,
                      self_collision=False, solver_position_iterations=12, render_visuals=True):
    """Import a URDF as a reduced-coordinate `Articulation`. Returns a UrdfArticulation.

    The articulation is built at the zero joint configuration. `stiffness`/`damping`/
    `max_force` configure a PD position drive on every actuated joint (leave stiffness
    0 for passive/force-controlled joints). `base_position` places the root link.

    With `render_visuals` (default), each link's `<visual><mesh>` (.obj/.dae/.stl) is loaded and
    parented under its collider so the robot renders as its real geometry while the primitive
    colliders (hidden) still drive the physics. Links whose visual fails to load keep the
    primitive; `robot.meshes[i].material.visible` is True exactly for those fallbacks.
    """
    root_xml = ET.parse(path).getroot()
    base_dir = os.path.dirname(os.path.abspath(path))
    links = {l.get("name"): l for l in root_xml.findall("link")}
    joints = root_xml.findall("joint")

    children = {}              # parent link -> [(joint, child link)]
    has_parent = set()
    for j in joints:
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        children.setdefault(p, []).append((j, c))
        has_parent.add(c)
    roots = [n for n in links if n not in has_parent]
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, found {roots}")

    art = world.create_articulation(fixed_base=fixed_base,
                                    solver_position_iterations=solver_position_iterations,
                                    disable_self_collision=not self_collision)
    meshes, joint_names, art_links = [], [], {}

    def density_for(link, vol):
        inertial = link.find("inertial")
        if inertial is not None and inertial.find("mass") is not None:
            m = float(inertial.find("mass").get("value", "0"))
            if m > 0 and vol > 1e-9:
                return m / vol
        return default_density

    def add(name, art_parent, world_T, inbound):
        col = _link_collider(links[name], world_T, base_dir)
        if col is None:                      # frame-only link (e.g. a tool tip): give it a tiny proxy
            mesh = tp.Mesh(tp.SphereGeometry(0.02), tp.MeshStandardMaterial())
            mesh.position.set(*[float(v) for v in world_T[:3, 3]])
            mesh.visible = False
            vol = 4 / 3 * math.pi * 0.02 ** 3
            local = None
        else:
            mesh, vol, local = col
        density = density_for(links[name], vol)

        if inbound is None:                  # root
            link_obj = art.add_link(mesh, parent=None, density=density)
        else:
            axis_local = np.array(_floats(inbound.find("axis").get("xyz")
                                          if inbound.find("axis") is not None else None, 3, (1, 0, 0)))
            axis_world = world_T[:3, :3] @ axis_local
            limit = inbound.find("limit")
            lo = hi = None
            if limit is not None and inbound.get("type") != "continuous":
                lo, hi = float(limit.get("lower", "0")), float(limit.get("upper", "0"))
            jtype = "prismatic" if inbound.get("type") == "prismatic" else "revolute"
            link_obj = art.add_link(
                mesh, parent=art_links[art_parent], density=density,
                axis=tuple(float(v) for v in axis_world),
                anchor=tuple(float(v) for v in world_T[:3, 3]),
                lower=lo, upper=hi, stiffness=stiffness, damping=damping,
                max_force=max_force, drive_target=0.0, joint_type=jtype)
            joint_names.append(inbound.get("name"))
        art_links[name] = link_obj
        meshes.append(mesh)
        if render_visuals and local is not None:
            _attach_visual(mesh, local, links[name], base_dir)
        return name

    def walk(name, art_parent, world_T, inbound):
        # fixed (or unmovable) joints don't create a DOF: collapse into the parent frame.
        if inbound is not None and inbound.get("type") == "fixed":
            this_parent = art_parent          # no new articulation link
        else:
            this_parent = add(name, art_parent, world_T, inbound)
        for (j, child) in children.get(name, []):
            walk(child, this_parent, world_T @ _origin_T(j), j)

    base_T = _trans(np.array(base_position, float))
    walk(roots[0], None, base_T, None)
    art.finalize()
    return UrdfArticulation(art, meshes, joint_names)
