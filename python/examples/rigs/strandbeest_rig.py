"""
strandbeest_rig -- a wind-walking machine on Theo Jansen's leg linkage, simulated by PhysX.

Four things, and nothing scene-specific:

  * ``solve`` / ``mirrored`` -- Jansen's linkage in closed form (two-circle intersections): the
                     joints of one leg at a crank angle, from his eleven bar lengths (``HOLY``).
  * ``Machine``   -- the whole walker as ONE PhysX articulation: the frame (free root), the
                     crankshaft in its bearing, and per leg the two rigid triangles (b-d-e, g-h-i)
                     and the four loose bars (j, k, c, f), hinged as a tree; three ball joints per
                     leg close the linkage's loops (``tp.Joint`` between two links of the same
                     articulation). The feet are the g-h-i triangles' lower corners, convex hulls
                     in contact with whatever ground the scene put in the world. Nothing is posed:
                     ``push`` puts the wind on the frame and the ground turns the crank.
  * ``Visual``    -- the bars, crank and frame drawn link by link from the articulation's poses:
                     strandbeest.glb's parts (strandbeest/build_strandbeest_blender.py) where it
                     has been built, plain tubes otherwise.
  * ``Sails``     -- the sails as cloth (position-based, numpy): pinned along a yard ahead of the
                     masts, their lower corners on slack sheets, blown by the same wind. They
                     follow the frame; they do not push it (``sail_thrust`` does that, as drag on
                     the sail area).

Nothing here imports threepp: ``Machine``, ``Visual`` and ``Sails`` are handed the module.

FRAME
-----
The machine frame has its origin on the crank axle: x forward (downwind, the way it walks), y up,
z along the axle. Lengths in metres; ``SCALE`` turns Jansen's numbers into them.

MEASURED (2026-11-10, strandbeest/strandbeest_walk.py --telemetry, 20 legs, 309 kg, 480 Hz, 32
position iterations): the loop joints stay closed to 0.9 mm; with a 7.6 m/s gusting wind it walks
at 0.4-2.0 m/s. At 6 m/s it rocks in place: the rigid feet have to lift the machine over the
gait's humps. The same machine as free rigid bodies tied by joints (no articulation) drifts apart
beyond about six legs.
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "strandbeest")

# ---------------------------------------------------------------- the linkage
HOLY = dict(a=38.0, b=41.5, c=39.3, d=40.1, e=55.8, f=39.4, g=36.7, h=65.7, i=49.0, j=50.0, k=61.9, l=7.8, m=15.0)
SCALE = 0.034                                   # m per linkage unit: about 4 m to the top of the frame
STATIONS = 10                                   # crank throws, two legs each
PITCH = 0.62                                    # m between stations along the axle
PHASES_DEG = [(i % 3) * 120.0 for i in range(STATIONS)]
HALF = (STATIONS - 1) * PITCH / 2 + 0.38        # half length of the rails
MAST = 86.0 * SCALE                             # top spar over the axle
SHOULDER = (17.0 * SCALE, 29.0 * SCALE)         # shoulder rails: |x|, y
YARD_X = 0.30                                   # the sails' yard stands this far ahead of the masts
NSAIL = 8
SAIL_W = (2 * HALF + 0.3) / NSAIL
SAIL_H = 50.0 * SCALE
SAIL_AREA = NSAIL * SAIL_W * 0.9 * SAIL_H
SAIL_CP = np.array([YARD_X, MAST - 0.5 * SAIL_H, 0.0])
LEG_Z, LAYER = 0.08, 0.05                       # a leg's plane beside its throw; the bars' layers in it
BAR_LAYER = {"j": -1, "k": -1, "c": 1, "f": 1}
FRAME_Y = 0.80 * MAST                           # where the frame's own mass sits
BEARING = 90.0                                  # N m s: the crank bearing's drag

BARS = (("j", "C", "P1"), ("k", "C", "P2"), ("b", "B", "P1"), ("c", "B", "P2"), ("d", "B", "P3"),
        ("e", "P1", "P3"), ("f", "P3", "P4"), ("g", "P2", "P4"), ("h", "P4", "F"), ("i", "P2", "F"))


def _cc(p0, r0, p1, r1, pick):
    """Intersection of circles (p0, r0), (p1, r1); pick(q) scores a candidate, the higher one wins."""
    d = p1 - p0
    L = np.linalg.norm(d, axis=-1, keepdims=True)
    a = (r0 * r0 - r1 * r1 + L * L) / (2.0 * L)
    h2 = r0 * r0 - a * a
    if np.any(h2 < 0.0):
        raise ValueError("linkage cannot close")
    u = d / L
    n = np.stack([-u[..., 1], u[..., 0]], -1)
    q0, q1 = p0 + a * u + np.sqrt(h2) * n, p0 + a * u - np.sqrt(h2) * n
    return np.where((pick(q0) >= pick(q1))[..., None], q0, q1)


def solve(theta):
    """One leg's joints (linkage units, crank centre at the origin, the leg towards -x) at crank
    angle(s) theta: dict O, B, C, P1..P4, F -> (..., 2)."""
    H = HOLY
    th = np.asarray(theta, np.float64)
    O = np.zeros(th.shape + (2,))
    B = O + np.array([-H["a"], -H["l"]])
    C = H["m"] * np.stack([np.cos(th), np.sin(th)], -1)
    P1 = _cc(C, H["j"], B, H["b"], lambda q: q[..., 1])
    P2 = _cc(C, H["k"], B, H["c"], lambda q: -q[..., 1])
    P3 = _cc(P1, H["e"], B, H["d"], lambda q: -q[..., 0])
    P4 = _cc(P3, H["f"], P2, H["g"], lambda q: np.linalg.norm(q - B, axis=-1))
    F = _cc(P4, H["h"], P2, H["i"], lambda q: -q[..., 1])
    return {"O": O, "B": B, "C": C, "P1": P1, "P2": P2, "P3": P3, "P4": P4, "F": F}


def mirrored(theta):
    """The opposite leg on the same crank pin, in the same frame."""
    s = solve(np.pi - np.asarray(theta, np.float64))
    return {k: v * np.array([-1.0, 1.0]) for k, v in s.items()}


def lowest_foot(theta=0.0):
    """How far below the axle (m, negative) the lowest foot is at crank angle theta."""
    low = 0.0
    for ph in np.radians(PHASES_DEG):
        low = min(low, float(solve(theta + ph)["F"][1]), float(mirrored(theta + ph)["F"][1]))
    return low * SCALE


# ---------------------------------------------------------------- small maths
def mat_quat(R):
    t = np.trace(R)
    if t > 0:
        k = math.sqrt(t + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / k, (R[0, 2] - R[2, 0]) / k, (R[1, 0] - R[0, 1]) / k, 0.25 * k])
    i = int(np.argmax(np.diag(R)))
    j, m = (i + 1) % 3, (i + 2) % 3
    k = math.sqrt(1.0 + R[i, i] - R[j, j] - R[m, m]) * 2
    q = np.zeros(4)
    q[i], q[j], q[m], q[3] = 0.25 * k, (R[j, i] + R[i, j]) / k, (R[m, i] + R[i, m]) / k, (R[m, j] - R[j, m]) / k
    return q


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------- the machine's parts
TREE = (("j", "crank", "C"), ("k", "crank", "C"), ("t1", "frame", "B"), ("c", "frame", "B"), ("t2", "c", "P2"), ("f", "t1", "P3"))
LOOPS = (("j", "t1", "P1"), ("k", "t2", "P2"), ("f", "t2", "P4"))


def layout(theta0=0.0, legs=None):
    """The machine at crank angle theta0, in its own frame, before any physics.

    Returns (links, loops). links: name -> dict(kind 'box' | 'capsule' | 'hull', centre (3,), rot
    (3x3: the link's axes in the machine frame), the shape's numbers, density, parent (a link's
    name or None), anchor (3,) of its hinge, visuals [(part, local position (3,), local angle
    about z)], and for a g-h-i triangle foot (3,), its foot in its own axes). loops: (link a,
    link b, anchor (3,)) for the ball joints. The dict is in an order a tree can be built in."""
    links = {}
    links["frame"] = dict(kind="box", centre=np.array([0.0, FRAME_Y, 0.0]), rot=np.eye(3), size=(0.5, 0.2, 2 * HALF), density=95.0,
                          parent=None, anchor=None, visuals=[("frame", np.array([0.0, -FRAME_Y, 0.0]), 0.0)])
    # the crankshaft's collider is a box as wide as its throws: a thin rod has next to no inertia about the axle
    crank_size = (0.5, 0.5, 2 * HALF - 0.1)
    links["crank"] = dict(kind="box", centre=np.zeros(3), rot=np.eye(3), size=crank_size, density=25.0 / float(np.prod(crank_size)),
                          parent="frame", anchor=np.zeros(3), visuals=[("crank", np.zeros(3), theta0)])
    loops = []
    n = 0
    for i, ph in enumerate(np.radians(PHASES_DEG)):
        z0 = -(i - (STATIONS - 1) / 2) * PITCH
        for side, solver in ((+1, solve), (-1, mirrored)):
            if legs is not None and n >= legs:
                break
            P = {k: np.array([v[0] * SCALE, v[1] * SCALE, z0 + side * LEG_Z]) for k, v in solver(theta0 + ph).items()}
            leg = f"L{n}"
            part = {}
            for name, tri, bars in (("t1", ("B", "P1", "P3"), (("b", "B", "P1"), ("d", "B", "P3"), ("e", "P1", "P3"))),
                                    ("t2", ("P2", "P4", "F"), (("g", "P2", "P4"), ("h", "P4", "F"), ("i", "P2", "F")))):
                c = np.mean([P[k] for k in tri], 0)
                vis = [(f"bar_{b}", P[a] - c, math.atan2(P[e][1] - P[a][1], P[e][0] - P[a][0])) for b, a, e in bars]
                vis += [("pin", P[k] - c, 0.0) for k in (("P1", "P3") if name == "t1" else ("P2", "P4"))]
                if name == "t2":
                    vis.append(("foot", P["F"] - c, 0.0))
                part[name] = dict(kind="hull", centre=c, rot=np.eye(3), corners=np.array([(P[k] - c) * 0.985 for k in tri]),
                                  density=160.0, visuals=vis, foot=(P["F"] - c) if name == "t2" else None)
            for b, a, e in BARS:
                if b not in BAR_LAYER:
                    continue
                dz = np.array([0.0, 0.0, side * LAYER * BAR_LAYER[b]])
                pa, pe = P[a] + dz, P[e] + dz
                d = pe - pa
                L = float(np.linalg.norm(d))
                shape = dict(kind="capsule", radius=0.014, length=L - 0.16, density=1000.0)
                if b == "c":                      # a flat strip: some inertia about its own length
                    shape = dict(kind="box", size=(0.10, L - 0.16, 0.03), density=260.0)
                part[b] = dict(centre=(pa + pe) / 2, rot=_rz(math.atan2(d[1], d[0]) - math.pi / 2), **shape,
                               visuals=[(f"bar_{b}", np.array([0.0, -L / 2, 0.0]), math.pi / 2)])
            for name, parent, node in TREE:
                part[name]["parent"] = parent if parent in ("frame", "crank") else leg + parent
                part[name]["anchor"] = P[node]
                links[leg + name] = part[name]
            loops += [(leg + a, leg + b, P[node]) for a, b, node in LOOPS]
            n += 1
    return links, loops


def hull_geometry(tp, corners, half=0.012):
    """A triangle of bars as a thin prism (for the convex hull PhysX cooks from it)."""
    pts = np.concatenate([corners + [0, 0, half], corners - [0, 0, half]]).astype(np.float32)
    g = tp.BufferGeometry()
    g.set_attribute("position", pts)
    g.set_index(np.array([0, 1, 2, 3, 5, 4, 0, 3, 4, 0, 4, 1, 1, 4, 5, 1, 5, 2, 2, 5, 3, 2, 3, 0], np.uint32))
    return g


class Machine:
    """The walker in a PhysxWorld: Machine(tp, world, position of the axle, yaw).

    ``links`` name -> ArticulationLink, ``names`` in tree order, ``feet`` [(link name, the foot in
    the link's axes)], ``mass`` kg. Needs a threepp whose tp.Joint accepts an ArticulationLink."""

    def __init__(self, tp, world, position, yaw=0.0, theta0=0.0, legs=None, iterations=32, foot_friction=1.0):
        self.tp, self.world, self.theta0 = tp, world, theta0
        R = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
        pos = np.asarray(position, float)
        self.layout, loops = layout(theta0, legs)
        self.names = list(self.layout)
        self.art = world.create_articulation(False, iterations, True)
        mat = tp.MeshBasicMaterial()
        grip = world.create_material(foot_friction, 0.9 * foot_friction, 0.0, "max")
        axle = [float(v) for v in R[:, 2]]
        self.links, self._meshes = {}, []
        for name, b in self.layout.items():
            if b["kind"] == "box":
                m = tp.Mesh(tp.BoxGeometry(*b["size"]), mat)
            elif b["kind"] == "capsule":
                m = tp.Mesh(tp.CapsuleGeometry(b["radius"], max(0.02, b["length"] - 2 * b["radius"])), mat)
            else:
                m = tp.Mesh(hull_geometry(tp, b["corners"]), mat)
            m.position.set(*[float(v) for v in R @ b["centre"] + pos])
            m.quaternion.set(*[float(v) for v in mat_quat(R @ b["rot"])])
            m.update_matrix_world()
            if b["parent"] is None:
                link = self.art.add_link(m, None, b["density"])
            else:
                anchor = [float(v) for v in R @ b["anchor"] + pos]
                link = self.art.add_link(m, self.links[b["parent"]], b["density"], axle, anchor, None, None, 0.0,
                                         BEARING if name == "crank" else 0.0, 1e6, 0.0, "revolute", 0.0, grip)
            self.links[name] = link
            self._meshes.append(m)
        self.art.finalize()
        self._joints, self._loops = [], []
        q = tp.Quaternion(*[float(v) for v in mat_quat(R)])
        for a, b, anchor in loops:
            p = tp.Joint.Params()
            p.type = tp.Joint.Type.SPHERICAL
            w = R @ anchor + pos
            self._joints.append(tp.Joint(world, self.links[a], self.links[b], tp.Vector3(*[float(v) for v in w]), q, p))
            self._loops.append((a, b, self._local(a, w), self._local(b, w)))
        self.feet = [(n, b["foot"]) for n, b in self.layout.items() if b.get("foot") is not None]
        self.mass = sum(self.links[n].mass for n in self.names)
        # a force on an articulation link lasts one substep: the wind is put on at every one
        self._force, self._point = tp.Vector3(0, 0, 0), tp.Vector3(0, 0, 0)
        self._hook = world.on_pre_substep(lambda dt: self.links["frame"].add_force_at_pos(self._force, self._point))

    def pose(self, name):
        """(position (3,), rotation 3x3) of a link in the world."""
        link = self.links[name]
        p, q = link.position, link.quaternion
        return np.array([p.x, p.y, p.z]), quat_mat((q.x, q.y, q.z, q.w))

    def _local(self, name, w):
        p, R = self.pose(name)
        return R.T @ (w - p)

    def frame(self):
        """(origin on the axle (3,), axes 3x3) of the machine frame in the world."""
        p, R = self.pose("frame")
        return p - R @ np.array([0.0, FRAME_Y, 0.0]), R

    def crank_angle(self):
        """The crank's angle in the frame, rad, in (-pi, pi]."""
        M = self.pose("frame")[1].T @ self.pose("crank")[1]
        a = self.theta0 + math.atan2(M[1, 0], M[0, 0])
        return (a + math.pi) % (2 * math.pi) - math.pi

    def foot_positions(self):
        out = np.zeros((len(self.feet), 3))
        for i, (n, f) in enumerate(self.feet):
            p, R = self.pose(n)
            out[i] = p + R @ f
        return out

    def loop_gap(self):
        """The largest gap (m) at the loop-closing joints: how well the linkage is held together."""
        e = 0.0
        for a, b, la, lb in self._loops:
            pa, Ra = self.pose(a)
            pb, Rb = self.pose(b)
            e = max(e, float(np.linalg.norm(pa + Ra @ la - pb - Rb @ lb)))
        return e

    def push(self, force, point):
        """A force (N, world) on the frame at a world point, held until the next push."""
        tp = self.tp
        self._force, self._point = tp.Vector3(*[float(v) for v in force]), tp.Vector3(*[float(v) for v in point])


def sail_thrust(machine, wind, velocity, area=SAIL_AREA, cd=1.3, rho=1.2):
    """The wind's force on the sails as drag on their area: (force (3,), its point (3,)), world.
    wind and velocity are (3,) m/s in the world; the sails face along the machine's x."""
    o, R = machine.frame()
    n = R[:, 0]
    along = float((np.asarray(wind, float) - np.asarray(velocity, float)) @ n)
    return 0.5 * rho * cd * area * along * abs(along) * n, o + R @ SAIL_CP


# ---------------------------------------------------------------- the picture
def model_path():
    """The generated strandbeest.glb, or None where it has not been built."""
    p = os.path.join(MODEL_DIR, "strandbeest.glb")
    return p if os.path.isfile(p) else None


def _tube(tp, p0, p1, r, material, sides=10):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    m = tp.Mesh(tp.CylinderGeometry(r, r, L, sides), material)
    m.position.set(*[float(v) for v in (p0 + p1) / 2])
    u = d / L
    axis = np.cross([0.0, 1.0, 0.0], u)
    s = float(np.linalg.norm(axis))
    if s > 1e-9:
        m.quaternion.set_from_axis_angle(tp.Vector3(*[float(v) for v in axis / s]), math.atan2(s, float(u[1])))
    elif u[1] < 0:
        m.quaternion.set_from_axis_angle(tp.Vector3(1, 0, 0), math.pi)
    return m


def _plain_parts(tp):
    """The parts as plain tubes, for where strandbeest.glb has not been built: name -> factory."""
    cane = tp.MeshStandardMaterial()
    cane.color = tp.Color(0.62, 0.47, 0.22)
    cane.roughness = 0.6
    dark = tp.MeshStandardMaterial()
    dark.color = tp.Color(0.13, 0.09, 0.06)
    dark.roughness = 0.85
    r = 0.016
    A, Lh, M = HOLY["a"] * SCALE, HOLY["l"] * SCALE, HOLY["m"] * SCALE
    sx, sy = SHOULDER

    def group(tubes):
        def make():
            g = tp.Group()
            for p0, p1, rad, mat in tubes:
                g.add(_tube(tp, p0, p1, rad, mat))
            return g
        return make

    parts = {f"bar_{b}": group([((0, 0, 0), (HOLY[b] * SCALE, 0, 0), r, cane)]) for b, _, _ in BARS}
    parts["pin"] = group([((0, 0, -0.09), (0, 0, 0.09), 0.008, dark)])
    parts["foot"] = group([((0, -0.01, 0), (0, 0.06, 0), 0.03, dark)])
    crank, prev = [], -HALF
    for i, ph in enumerate(np.radians(PHASES_DEG)):
        z0 = -(i - (STATIONS - 1) / 2) * PITCH
        c = (M * math.cos(ph), M * math.sin(ph))
        zs = sorted((z0 - 0.16, z0 + 0.16))
        crank += [((0, 0, z), (c[0], c[1], z), r * 1.15, cane) for z in zs]
        crank.append(((c[0], c[1], zs[0]), (c[0], c[1], zs[1]), r, cane))
    stations = sorted(-(i - (STATIONS - 1) / 2) * PITCH for i in range(STATIONS))
    for z0 in stations:
        crank.append(((0, 0, prev), (0, 0, z0 - 0.16), r * 1.15, cane))
        prev = z0 + 0.16
    crank.append(((0, 0, prev), (0, 0, HALF), r * 1.15, cane))
    parts["crank"] = group(crank)
    frame = []
    for s in (-1, 1):
        frame += [((s * A, -Lh, -HALF - 0.1), (s * A, -Lh, HALF + 0.1), r * 1.1, cane), ((s * sx, sy, -HALF - 0.1), (s * sx, sy, HALF + 0.1), r, cane)]
    ribs = [-HALF] + [(stations[i] + stations[i + 1]) / 2 for i in range(STATIONS - 1)] + [HALF]
    for n, z in enumerate(ribs):
        for s in (-1, 1):
            frame += [((0, 0, z), (s * A, -Lh, z), r, cane), ((s * A, -Lh, z), (s * sx, sy, z), r, cane), ((s * sx, sy, z), (0, 0, z), r, cane)]
        frame.append(((-sx, sy, z), (sx, sy, z), r, cane))
        if n % 2 == 0:
            frame += [((0, sy, z), (0, MAST, z), r, cane), ((0, MAST - 0.04, z), (YARD_X, MAST - 0.04, z), r * 0.8, cane)]
    frame += [((0, MAST, -HALF - 0.25), (0, MAST, HALF + 0.25), r, cane), ((YARD_X, MAST - 0.04, -HALF - 0.2), (YARD_X, MAST - 0.04, HALF + 0.2), r * 0.9, cane)]
    parts["frame"] = group(frame)
    return parts


class Visual:
    """The machine drawn from its articulation: Visual(tp, scene, machine[, model root]).

    One group per link carries that link's bars, pins and foot; update() copies the links' poses
    to them. With ``model`` (strandbeest.glb's root node) the parts are its nodes, cloned; without,
    plain tubes. ``sail_material`` is the model's sail cloth, or None."""

    def __init__(self, tp, scene, machine, model=None):
        self.tp, self.machine = tp, machine
        plain = None if model is not None else _plain_parts(tp)
        self.sail_material = None
        if model is not None:
            needed = {v[0] for b in machine.layout.values() for v in b["visuals"]}
            missing = [n for n in sorted(needed) if model.get_object_by_name(n) is None]
            if missing:
                raise KeyError(f"strandbeest.glb has no node(s) {missing}: rebuild it with strandbeest/build_strandbeest_blender.py")
            sail = model.get_object_by_name("sail_0")
            if sail is not None:
                self.sail_material = sail.children[0].material
        self.groups = {}
        for name, b in machine.layout.items():
            g = tp.Group()
            for part, p, ang in b["visuals"]:
                o = model.get_object_by_name(part).clone() if model is not None else plain[part]()
                holder = tp.Group()
                holder.add(o)
                holder.quaternion.set(*[float(v) for v in mat_quat(_rz(ang))])      # in the link's own axes
                holder.position.set(*[float(v) for v in p])
                g.add(holder)
            scene.add(g)
            self.groups[name] = g

        def shadows(o):
            if isinstance(o, tp.Mesh):
                o.cast_shadow = o.receive_shadow = True
        for g in self.groups.values():
            g.traverse(shadows)
        self.update()

    def update(self):
        for name, g in self.groups.items():
            link = self.machine.links[name]
            p, q = link.position, link.quaternion
            g.position.set(p.x, p.y, p.z)
            g.quaternion.set(q.x, q.y, q.z, q.w)


class Sails:
    """The sails as cloth: Sails(tp, scene, material, (origin, axes) of the machine frame).

    Position-based: every sail a grid pinned along the yard, its two lower corners on slack sheets
    to the front shoulder rail; gravity, wind pressure along the cloth's normal, and the frame
    behind it (the masts and the plane their braces lie in) as a wall. step() moves the pins with
    the frame and writes the mesh."""

    NX, NZ, SUB, ITERS = 9, 14, 8, 5

    def __init__(self, tp, scene, material, frame):
        self.tp = tp
        n, nx, nz = NSAIL, self.NX, self.NZ
        w, h = SAIL_W * 0.9, SAIL_H
        self.dx, self.dy = w / (nx - 1), h / (nz - 1)
        zc = (np.arange(n) - (n - 1) / 2) * SAIL_W
        self.pins = np.zeros((n, nx, 3))
        self.pins[..., 0], self.pins[..., 1] = YARD_X, MAST - 0.04
        self.pins[..., 2] = zc[:, None] + (np.linspace(-0.5, 0.5, nx) * w)[None, :]
        sx, sy = SHOULDER
        self.anchors = np.zeros((n, 2, 3))
        self.anchors[..., 0], self.anchors[..., 1] = sx, sy
        self.anchors[:, 0, 2], self.anchors[:, 1, 2] = zc - 0.5 * w, zc + 0.5 * w
        self.sheet = math.hypot(sx - YARD_X, MAST - 0.04 - h - sy) * 1.6
        bn = np.array([0.67 * MAST - sy, sx])
        self.brace_n, self.brace_a = bn / np.linalg.norm(bn), np.array([sx, sy])
        o, R = frame
        pins = self.pins @ R.T + o
        self.P = np.repeat(pins[:, None], nz, 1)
        self.P[..., 1] -= (np.arange(nz) * self.dy)[None, :, None]
        self.Q = self.P.copy()
        self.gust = np.random.default_rng(3).uniform(0, 6.28, (n, 3))
        self.origin = np.array(o, float)
        idx = []
        for s in range(n):
            for j in range(nz - 1):
                for i in range(nx - 1):
                    a = s * nz * nx + j * nx + i
                    idx += [a, a + nx, a + 1, a + 1, a + nx, a + nx + 1]
        uv = np.zeros((n, nz, nx, 2), np.float32)
        uv[..., 0], uv[..., 1] = np.linspace(0, 1, nx)[None, None, :], np.linspace(0, 1, nz)[None, :, None]
        geo = tp.BufferGeometry()
        geo.set_attribute("position", self._positions())
        geo.set_attribute("normal", self._normals())
        geo.set_attribute("uv", np.ascontiguousarray(uv.reshape(-1, 2)))
        geo.set_index(np.asarray(idx, np.uint32))
        self.geo = geo
        if material is None:
            material = tp.MeshStandardMaterial()
            material.color = tp.Color(0.88, 0.84, 0.74)
            material.roughness = 0.9
        material.side = tp.Side.Double
        self.mesh = tp.Mesh(geo, material)
        self.mesh.position.set(*[float(v) for v in self.origin])
        self.mesh.cast_shadow = self.mesh.receive_shadow = True
        self.mesh.frustum_culled = False
        scene.add(self.mesh)
        self._frame = (np.array(o, float), np.array(R, float))

    def _positions(self):
        return np.ascontiguousarray((self.P - self.origin).reshape(-1, 3), np.float32)

    def _grid_normals(self):
        n = np.cross(np.gradient(self.P, axis=1), np.gradient(self.P, axis=2))
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)

    def _normals(self):
        return np.ascontiguousarray(self._grid_normals().reshape(-1, 3), np.float32)

    @staticmethod
    def _pull(A, B, rest, k):
        d = B - A
        L = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        c = (0.5 * k) * (L - rest) / L * d
        A += c
        B -= c

    def step(self, frame, dt, t, wind):
        """frame: (origin, axes) of the machine now; wind (3,) m/s in the world."""
        P = self.P
        o0, R0 = self._frame
        o1, R1 = np.array(frame[0], float), np.array(frame[1], float)
        self._frame = (o1, R1)
        wind = np.asarray(wind, float)
        speed = float(np.linalg.norm(wind))
        wdir = wind / max(speed, 1e-9)
        h = dt / self.SUB
        dx, dy = self.dx, self.dy
        dd = math.hypot(dx, dy)
        pins0, pins1 = self.pins @ R0.T + o0, self.pins @ R1.T + o1
        anch0, anch1 = self.anchors @ R0.T + o0, self.anchors @ R1.T + o1
        g = self.gust
        for s in range(self.SUB):
            a = (s + 1) / self.SUB
            pins, anch = pins0 + (pins1 - pins0) * a, anch0 + (anch1 - anch0) * a
            ts = t + s * h
            gusty = speed * (1.0 + 0.30 * np.sin(0.9 * ts + g[:, 0]) + 0.18 * np.sin(2.3 * ts + g[:, 1]) + 0.10 * np.sin(5.1 * ts + g[:, 2]))
            nrm = self._grid_normals()
            rel = wdir[None, None, None, :] * gusty[:, None, None, None] - (P - self.Q) / h
            acc = 2.6 * np.sum(rel * nrm, -1, keepdims=True) * nrm
            acc[..., 1] -= 9.81
            new = P + (P - self.Q) * 0.992 + acc * h * h
            self.Q[...] = P
            P[...] = new
            for _ in range(self.ITERS):
                self._pull(P[:, :, 0:-1:2], P[:, :, 1::2], dx, 1.0)
                self._pull(P[:, :, 1:-1:2], P[:, :, 2::2], dx, 1.0)
                self._pull(P[:, 0:-1:2], P[:, 1::2], dy, 1.0)
                self._pull(P[:, 1:-1:2], P[:, 2::2], dy, 1.0)
                for A, B, rest, k in ((P[:, :-1, :-1], P[:, 1:, 1:], dd, 0.20), (P[:, :-1, 1:], P[:, 1:, :-1], dd, 0.20),
                                      (P[:, :, :-2], P[:, :, 2:], 2 * dx, 0.08), (P[:, :-2], P[:, 2:], 2 * dy, 0.08)):
                    d = B - A
                    L = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
                    c = 0.5 * k * (L - rest) / L * d
                    A += c
                    B -= c
                B = (P - o1) @ R1                                    # the frame behind the cloth, in its own axes
                B[..., 0] = np.maximum(B[..., 0], 0.07)
                push = np.maximum(0.045 - (B[..., 0] - self.brace_a[0]) * self.brace_n[0] - (B[..., 1] - self.brace_a[1]) * self.brace_n[1], 0.0)
                B[..., 0] += push * self.brace_n[0]
                B[..., 1] += push * self.brace_n[1]
                P[...] = B @ R1.T + o1
                P[:, 0] = pins
                for k, col in ((0, 0), (1, -1)):                      # the sheets: a longest distance
                    d = P[:, -1, col] - anch[:, k]
                    L = np.linalg.norm(d, axis=-1, keepdims=True)
                    P[:, -1, col] = np.where(L > self.sheet, anch[:, k] + d / np.maximum(L, 1e-9) * self.sheet, P[:, -1, col])
        self.geo.update_attribute("position", self._positions())
        self.geo.update_attribute("normal", self._normals())
