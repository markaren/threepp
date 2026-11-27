"""Draw a line; out comes the solid that rolls along it.

Five lanes on a tilted table. Four hold a solid that was cut for the line painted in its
lane, the fifth an ordinary ball, and PhysX lets go of all of them. It is given the
table, the shapes, gravity and a drag, and nothing about any line. The ink follows the
point of the table under each centre of mass, where the body touches it, and the number
at the start of the lane is how far the ink is from the paint. The ball rolls straight.
Drag the mouse left to right across the front lane to draw your own line: it is cut in
under a second and rolls with the others.

    python trajectoid.py                          # live: draw in the front lane, R rolls again
    python trajectoid.py --check                  # no window: roll them all and print the table
    python trajectoid.py --shot out.png --at 7    # headless still after 7 s of rolling
    python trajectoid.py --video 24               # headless film -> out/trajectoid.mp4

How a solid is made, all of it in this file (the construction is Sobolev, Dong, Tlusty,
Eckmann, Granick and Grzybowski, "Solid-body trajectoids shaped to roll along desired
pathways", Nature 620, 310, 2023):

  1. roll a ball of radius r along one period of the line, on paper, and note how it ends
     up turned. Pick r so that this is exactly half a turn: two periods then bring the
     ball back as it started, whatever the line is;
  2. at every point of those two periods the table is a plane in the ball's own frame.
     Cut a slightly larger ball with all of them. The face that is left is the one that
     rolls on a flat table along the line;
  3. put the weight in the middle, so that rolling never lifts it and leaving the line
     would.

In PhysX each solid is a heavy steel ball with the cut shell fixed around it in sixteen
convex pieces (one articulation, fixed joints). A viscous drag keeps the pace under a
metre per second, and the window and the film play the physics 1.4 times faster than
real. A line that turns too sharply, swings out of its lane, or is bent so hard that the
cuts remove whole faces is flattened until it does none of that, and the panel says by
how much.

Needs a build with PhysX and numpy; draws with Vulkan where there is one, else GL.
"""
import argparse
import gc
import math
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

from demo_common import Encoder, parse_size, resize_handler, standard_material, write_radiance_hdr

R = 0.5                          # m, radius of the rolling face
BALL = 0.75                      # m, radius of the ball it is cut from
SAMPLES = 240                    # points per period of a line: one facet each
FACETS = 30                      # facets per convex piece (PhysX hulls stop at 64 vertices)
HARMONICS = 5                    # a line is smoothed to this many per period
MAX_SWING = 0.85                 # m, side to side, so neighbours stay in their lanes
MAX_STEEP = 3.0                  # sideways per downhill (72 deg): steeper stretches roll too slowly
MIN_FACE = 0.2                   # m, the shortest facet a solid may be left with

SLOPE = math.radians(4.0)
DRAG = 0.8                       # 1/s
DT = 1.0 / 240.0
UNITS = 4.0                      # PhysX units per metre (see Sim)
FPS = 60
INK_POINTS = 3000                # enough for the longest line a lane can hold
PACE = 1.4                       # the window and the film play this much faster than real

LANE = 2.4                       # m between lanes
START = -3.3                     # the table's uphill end (the solids start at x = 0)
FINISH = 8.5                     # ink and the measurement stop here
RAIL = FINISH + 1.35
END = RAIL + 0.35


# ------------------------------------------------------------------------- the lines

U = np.arange(SAMPLES) / SAMPLES


def periodic(z, harmonics=HARMONICS):
    """One period of a line, z at U in units of the period: smoothed, closed, starting at 0."""
    f = np.fft.rfft(z)
    f[0] = 0.0
    f[harmonics + 1:] = 0.0
    z = np.fft.irfft(f, len(z))
    return z - z[0]


def bump(centre, width):
    return np.exp(-(((U - centre + 0.5) % 1.0 - 0.5) / width) ** 2)


WAVE = periodic(0.16 * np.sin(2 * np.pi * U))
PULSE = periodic(0.19 * bump(0.36, 0.085) - 0.19 * bump(0.57, 0.085) + 0.06 * bump(0.83, 0.08))
ZIGZAG = periodic(0.15 * np.arcsin(0.98 * np.sin(2 * np.pi * U)) * 2 / np.pi, 3)
SCRIBBLE = periodic(0.12 * np.sin(2 * np.pi * U) + 0.065 * np.sin(4 * np.pi * U + 1.2))

# (name on the table, the line painted in the lane, ink colour, is it a plain ball)
LANES = [("plain ball", WAVE, (0.85, 0.85, 0.88), True),
         ("wave", WAVE, (0.02, 0.62, 0.80), False),
         ("heartbeat", PULSE, (0.95, 0.03, 0.09), False),
         ("zigzag", ZIGZAG, (1.00, 0.46, 0.01), False),
         ("yours", SCRIBBLE, (0.38, 0.14, 1.00), False)]
DRAWN = len(LANES) - 1


def lane_z(i):
    return (i - 0.5 * (len(LANES) - 1)) * LANE


# ------------------------------------------------------------------- cutting a solid

def qmul(a, b):
    """Hamilton product of quaternions (..., 4) held as (w, x, y, z)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def roll_on_paper(line, radius):
    """The orientations a ball of each `radius` passes through, rolling without slip or
    twist along `line` (n+1 points x, z on a table whose normal is +y): (n+1, m, 4)."""
    step = np.diff(line, axis=0)
    length = np.hypot(step[:, 0], step[:, 1])
    axis = np.stack([step[:, 1], 0 * length, -step[:, 0]], 1) / length[:, None]
    half = 0.5 * length[:, None] / np.atleast_1d(radius)[None, :]
    turn = np.concatenate([np.cos(half)[..., None], np.sin(half)[..., None] * axis[:, None, :]], -1)
    out = np.zeros((len(step) + 1,) + turn.shape[1:])
    out[0, :, 0] = 1.0
    for k in range(len(step)):
        out[k + 1] = qmul(turn[k], out[k])
    return out


def half_turn_radii(line):
    """Radii, largest first, for which one period of `line` leaves the ball half turned.
    A unit quaternion is a half turn where its w crosses zero."""
    length = np.hypot(*np.diff(line, axis=0).T).sum()
    radii = length / np.linspace(2.0, 8.0, 241)
    w = roll_on_paper(line, radii)[-1, :, 0]
    for i in np.nonzero(np.diff(np.sign(w)))[0]:
        lo, hi = radii[i + 1], radii[i]
        for _ in range(3):
            fine = np.linspace(lo, hi, 65)
            j = np.nonzero(np.diff(np.sign(roll_on_paper(line, fine)[-1, :, 0])))[0][0]
            lo, hi = fine[j], fine[j + 1]
        yield 0.5 * (lo + hi)


def table_normals(line):
    """Where "down" points in the ball's own frame, at every sample of two periods."""
    q = roll_on_paper(line, R)[:, 0]
    q = np.concatenate([q[:-1], qmul(q[:-1], q[-1])])    # the second period starts half turned
    w, x, y, z = q.T
    return -np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1)


def facet_edges(n):
    """The edge each facet shares with the next, cut short by the ball and by every other
    facet: its two ends (a, b) and its length, negative where nothing is left of it."""
    m = np.roll(n, -1, 0)
    c = R * (n + m) / (1 + (n * m).sum(1))[:, None]        # the edge's point nearest the centre
    t = np.cross(n, m)
    t /= np.linalg.norm(t, axis=1)[:, None]
    hi = np.sqrt(BALL ** 2 - (c * c).sum(1))
    lo = -hi
    slope, room = t @ n.T, R - c @ n.T                     # edge i against facet j
    i = np.arange(len(n))
    for own in (i, (i + 1) % len(n)):
        slope[i, own] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        at = room / slope
    hi = np.minimum(hi, np.where(slope > 1e-9, at, np.inf).min(1))
    lo = np.maximum(lo, np.where(slope < -1e-9, at, -np.inf).max(1))
    return c + lo[:, None] * t, c + hi[:, None] * t, hi - lo


class Solid:
    def __init__(self, line, normals, a, b, kept):
        self.line, self.normals, self.a, self.b = line, normals, a, b
        self.period = line[-1, 0]
        self.kept = kept                                    # share of the drawn swing that was cut


def cut(shape):
    """The solid for one period `shape` (see periodic), or None if no flattening gives one."""
    steepest = np.abs(np.diff(np.append(shape, shape[0]))).max() * SAMPLES
    gentle = min(1.0, MAX_STEEP / max(steepest, 1e-9))
    for kept in gentle * np.array([1.0, 0.85, 0.72, 0.61, 0.52, 0.44, 0.37]):
        line = np.stack([np.append(U, 1.0), kept * np.append(shape, shape[0])], 1)
        for r in half_turn_radii(line):
            period = R / r                                  # scale the paper so the radius is R
            if np.ptp(line[:, 1]) * period > MAX_SWING:
                continue
            normals = table_normals(line * period)
            a, b, length = facet_edges(normals)
            if length.min() > MIN_FACE:
                return Solid(line * period, normals, a, b, kept)
    return None


def stroke_shape(stroke):
    """A drawn stroke (x, z on the table, x increasing) as one period of a line: scaled to
    unit length, its end brought level with its start so that periods meet, and smoothed.
    None if there is too little of it."""
    pts = np.asarray(stroke, float)
    if len(pts) < 8 or pts[-1, 0] - pts[0, 0] < 0.8:
        return None
    span = pts[-1, 0] - pts[0, 0]
    u = (pts[:, 0] - pts[0, 0]) / span
    z = np.interp(U, u, (pts[:, 1] - pts[0, 1]) / span)
    return periodic(z - U * (pts[-1, 1] - pts[0, 1]) / span)


# ------------------------------------------------------------------------- its skins

def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    f = np.array([0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2,
                  10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5,
                  2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]).reshape(-1, 3)
    for _ in range(level):
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        key, back = np.unique(edges[:, 0] * len(v) + edges[:, 1], return_inverse=True)
        mid = v[key // len(v)] + v[key % len(v)]
        ab, bc, ca = (len(v) + back).reshape(3, -1)
        v = np.concatenate([v, mid / np.linalg.norm(mid, axis=1)[:, None]])
        a, b, c = f.T
        f = np.concatenate([np.stack(tri, 1) for tri in
                            ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return v, f


def skins(solid, sphere):
    """The solid as two triangle soups with a crisp crease between them: the cut face
    (True) and what is left of the ball (False), each (positions, normals)."""
    u, faces = sphere
    reach = np.empty(len(u))
    which = np.empty(len(u), int)
    for k in range(0, len(u), 4096):
        s = u[k:k + 4096] @ solid.normals.T
        which[k:k + 4096] = s.argmax(1)
        reach[k:k + 4096] = s.max(1)
    over = reach - R / BALL                       # > 0: a facet cuts deeper than the ball ends
    face = over > 0
    pos = u * np.where(face, R / np.maximum(reach, 1e-9), BALL)[:, None]
    flat = solid.normals[which]

    def corners(i, side):
        return pos[i], (flat if side else u)[i]

    def crease(i, j, side):
        d = u[i] + (over[i] / (over[i] - over[j]))[:, None] * (u[j] - u[i])
        d /= np.linalg.norm(d, axis=1)[:, None]
        return d * BALL, np.where(face[i][:, None], flat[i], flat[j]) if side else d

    inside = face[faces]
    count = inside.sum(1)
    soup = {side: [[corners(faces[count == (3 if side else 0)][:, k], side) for k in range(3)]]
            for side in (True, False)}
    mixed = faces[(count == 1) | (count == 2)]
    flags = face[mixed]
    odd = np.where(flags.sum(1) == 1, flags.argmax(1), (~flags).argmax(1))
    a, b, c = np.take_along_axis(mixed, (odd[:, None] + np.arange(3)) % 3, 1).T
    for lone in (True, False):                    # which side the odd corner is on
        s = face[a] == lone
        i, j, k = a[s], b[s], c[s]
        rest = not lone
        soup[lone].append([corners(i, lone), crease(i, j, lone), crease(i, k, lone)])
        soup[rest].append([crease(i, j, rest), corners(j, rest), corners(k, rest)])
        soup[rest].append([crease(i, j, rest), corners(k, rest), crease(i, k, rest)])
    return {side: tuple(np.concatenate([np.stack([t[0][n], t[1][n], t[2][n]], 1) for t in tris])
                        .reshape(-1, 3).astype(np.float32) for n in (0, 1))
            for side, tris in soup.items()}


# --------------------------------------------------------------------------- physics

class Sim:
    """PhysX: the tilted table (as tilted gravity) and one body per lane.

    It works in units a quarter of a metre long: every length and gravity are UNITS
    times larger, which leaves the motion unchanged. PhysX's contact tolerances are
    fixed lengths, and a solid four times the size sits more finely within them: the
    typical distance off the line is less than half of what it is at one unit per metre."""

    def __init__(self, solids):
        g = 9.81 * UNITS
        gravity = tp.Vector3(g * math.sin(SLOPE), -g * math.cos(SLOPE), 0.0)
        self.world = tp.PhysxWorld(gravity=gravity, fixed_timestep=DT, max_substeps=12)
        grip = self.world.create_material(1.5, 1.3, 0.0, friction_combine="max")
        width = len(LANES) * LANE
        self.keep = []                       # the meshes PhysX writes poses into
        for size, centre in (((END - START, 1.0, width), (0.5 * (END + START), -0.5, 0.0)),
                             ((0.2, 0.4, width), (RAIL + 0.1, 0.2, 0.0))):
            block = tp.Mesh(tp.BoxGeometry(*(UNITS * v for v in size)), tp.MeshBasicMaterial())
            block.position.set(*(UNITS * v for v in centre))
            self.world.add_static(block, grip)
            self.keep.append(block)
        self.movers = [self._body(i, s, grip) for i, s in enumerate(solids)]
        self.hook = self.world.on_pre_substep(self._drag)

    def _body(self, lane, solid, grip):
        """(the mesh that follows it, velocity(), push(force), mass)"""
        start = (0.0, UNITS * R, UNITS * lane_z(lane))
        radius = UNITS * (R if LANES[lane][3] else 0.6 * R)
        core = tp.Mesh(tp.SphereGeometry(radius), tp.MeshBasicMaterial())
        core.position.set(*start)
        if LANES[lane][3]:
            body = self.world.add(core, density=1000.0, material=grip)
            return core, lambda: body.linear_velocity, body.add_force, body.mass
        art = self.world.create_articulation(False, 16, True)
        root = art.add_link(core, density=8000.0, material=grip)
        n = len(solid.a)
        for k in range(0, n, FACETS):
            ids = np.arange(k, k + FACETS + 1) % n
            points = UNITS * np.concatenate([solid.a[ids], solid.b[ids], np.zeros((1, 3))])
            geometry = tp.BufferGeometry()
            geometry.set_attribute("position", points.astype(np.float32))
            shell = tp.Mesh(geometry, tp.MeshBasicMaterial())
            shell.position.set(*start)
            art.add_link(shell, root, density=2.0, anchor=start, joint_type="fixed", material=grip)
            self.keep.append(shell)
        art.finalize()
        self.keep.append(art)

        def velocity():
            return tp.Vector3(*(float(v) for v in art.root_velocity()[:3]))
        return core, velocity, root.add_force, root.mass

    def _drag(self, _dt):
        for _, velocity, push, mass in self.movers:
            v = velocity()
            k = -DRAG * mass
            push(tp.Vector3(k * v.x, k * v.y, k * v.z))

    def step(self, dt):
        self.world.step(dt)

    def pose(self, lane):
        """Position (metres, in the table's frame) and quaternion of a lane's body."""
        mesh = self.movers[lane][0]
        p, q = mesh.position, mesh.quaternion
        return (p.x / UNITS, p.y / UNITS, p.z / UNITS), (q.x, q.y, q.z, q.w)

    def close(self):
        """PhysX allows one world per process: let go of everything that holds this one."""
        self.world.remove_substep_callback(self.hook)
        self.__dict__.clear()
        gc.collect()


class Trace:
    """Where one body's contact has been, and how far that is from the painted line."""

    def __init__(self, lane, solid):
        periods = int(FINISH / solid.period) + 2
        one = solid.line[:-1] + [0.0, lane_z(lane)]
        self.paint = np.concatenate([one + [k * solid.period, 0.0] for k in range(-1, periods)])
        self.clear()

    def clear(self):
        self.points, self.square, self.worst, self.length = [], 0.0, 0.0, 0.0

    def add(self, x, z):
        """Take the contact point (under the centre of mass); True if the ink grew."""
        if x > FINISH:
            return False
        if self.points:
            step = math.hypot(x - self.points[-1][0], z - self.points[-1][1])
            if step < 0.012:
                return False
        else:
            step = 0.0
        i, j = np.searchsorted(self.paint[:, 0], (x - 0.5, x + 0.5))
        a, ab = self.paint[i:j - 1], np.diff(self.paint[i:j], axis=0)
        t = np.clip((((x, z) - a) * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
        off = np.linalg.norm(a + t[:, None] * ab - (x, z), axis=1).min()
        # the mean is over distance rolled, so a body at rest adds nothing to it
        self.square += off * off * step
        self.length += step
        self.worst = max(self.worst, off)
        self.points.append((x, z))
        return True

    @property
    def rms(self):
        return math.sqrt(self.square / self.length) if self.length else 0.0


def cut_all():
    t0 = time.perf_counter()
    solids = []
    for name, shape, _, _ in LANES:
        solid = cut(shape)
        if solid is None:
            raise RuntimeError(f"no solid for the {name} line")
        solids.append(solid)
    print(f"cut {sum(not lane[3] for lane in LANES)} solids in {time.perf_counter() - t0:.1f} s")
    return solids


def check():
    """Roll everything without a window and print how well each kept to its line."""
    solids = cut_all()
    sim = Sim(solids)
    traces = [Trace(i, s) for i, s in enumerate(solids)]
    for frame in range(50 * FPS):
        sim.step(1.0 / FPS)
        for i, trace in enumerate(traces):
            (x, _, z), _ = sim.pose(i)
            trace.add(x, z)
        if all(sim.pose(i)[0][0] > FINISH for i in range(len(LANES))):
            break
    print(f"after {(frame + 1) / FPS:.1f} s of rolling; distance of the contact from the painted line:")
    print(f"  {'lane':<12}{'period':>8}{'rolled':>10}{'rms':>10}{'worst':>10}")
    for (name, _, _, plain), solid, trace in zip(LANES, solids, traces):
        period = "-" if plain else f"{solid.period:.2f} m"
        print(f"  {name:<12}{period:>8}{trace.length:>8.2f} m{1000 * trace.rms:>7.0f} mm"
              f"{1000 * trace.worst:>7.0f} mm")
    sim.close()


# ----------------------------------------------------------------------- the picture

class Ribbon:
    """A flat stroke on the table, redrawn from a list of (x, z)."""

    def __init__(self, points, width, height, material):
        self.points, self.width = points, width
        self.pos = np.zeros((2 * points, 3), np.float32)
        self.pos[:, 1] = height
        a = 2 * np.arange(points - 1, dtype=np.uint32)[:, None]
        index = (a + np.array([0, 2, 1, 1, 2, 3], np.uint32)).ravel()
        self.geometry = tp.BufferGeometry()
        self.geometry.set_attribute("position", self.pos.copy())
        self.geometry.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (2 * points, 1)))
        self.geometry.set_index(index)
        self.mesh = tp.Mesh(self.geometry, material)
        self.mesh.frustum_culled = False
        self.mesh.receive_shadow = True
        self.draw([])

    def draw(self, line):
        line = np.asarray(line, float)[-self.points:]
        n = len(line)
        self.mesh.visible = n > 1
        if n > 1:
            d = np.gradient(line, axis=0)
            d /= np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
            side = 0.5 * self.width * np.stack([-d[:, 1], d[:, 0]], 1)
            self.pos[0:2 * n:2, 0::2] = line + side
            self.pos[1:2 * n:2, 0::2] = line - side
            self.pos[2 * n:] = self.pos[2 * n - 1]
            self.geometry.update_attribute("position", self.pos)
        self.geometry.set_draw_range(0, max(0, n - 1) * 6)


def colour(rgb, scale=1.0):
    return tp.Color(*(min(1.0, c * scale) for c in rgb))


def make_studio(path, w=1024, h=512):
    """A dark studio with three soft boxes, as an equirectangular HDR."""
    theta = (np.arange(h) + 0.5)[:, None] / h * math.pi
    phi = (np.arange(w) + 0.5)[None, :] / w * 2 * math.pi - math.pi
    y = np.broadcast_to(np.cos(theta), (h, w))
    d = np.stack([np.sin(theta) * np.cos(phi), y, np.sin(theta) * np.sin(phi)], -1)
    sky = 0.025 + 0.10 * np.clip(y, 0, 1)[..., None] * np.array([0.8, 0.9, 1.1])
    for (toward, wide, tall, tint, power) in (
            ((-0.3, 1.0, 0.25), 0.55, 0.30, (1.0, 0.97, 0.92), 2.8),     # key, overhead
            ((0.9, 0.35, 0.35), 0.10, 0.55, (0.75, 0.85, 1.0), 3.0),     # cool strip, downhill
            ((-0.75, 0.3, -0.6), 0.12, 0.45, (1.0, 0.72, 0.45), 2.5)):   # warm strip, behind
        n = np.array(toward) / np.linalg.norm(toward)
        right = np.cross(n, (0.0, 0.0, 1.0) if abs(n[1]) > 0.9 else (0.0, 1.0, 0.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, n)
        facing = d @ n
        a, b = (d @ right) / np.maximum(facing, 0.1), (d @ up) / np.maximum(facing, 0.1)
        box = np.exp(-(a / wide) ** 8 - (b / tall) ** 8) * (facing > 0)
        sky = sky + (power * box)[..., None] * np.array(tint)
    return write_radiance_hdr(path, sky)


def table_texture(width, depth, px=96):
    """Slate with a half-metre grid and a darker gap between lanes."""
    w, h = int(width * px), int(depth * px)
    x = START + (np.arange(w) + 0.5) / px
    z = (np.arange(h) + 0.5) / px - 0.5 * depth
    rng = np.random.default_rng(1)
    img = np.full((h, w, 3), (30.0, 33.0, 40.0)) + rng.normal(0.0, 1.1, (h, w, 1))
    grid = (np.abs((x + 0.25) % 0.5 - 0.25) < 0.008)[None, :] | (np.abs((z + 0.25) % 0.5 - 0.25) < 0.008)[:, None]
    img[grid] += 11.0
    edge = np.abs((z - lane_z(0)) % LANE - 0.5 * LANE) < 0.025
    img[edge, :] = (13.0, 14.0, 18.0)
    return tp.data_texture(np.clip(img, 0, 255).astype(np.uint8))


class Show:
    def __init__(self, args):
        headless = args.shot is not None or args.video > 0
        width, height = parse_size(args.size)
        self.vulkan = not args.gl and tp.HAS_VULKAN and tp.vulkan_available()
        self.canvas = tp.Canvas("threepp - trajectoid", width=width, height=height,
                                antialiasing=0 if self.vulkan else 8, vsync=not headless, headless=headless)
        if self.vulkan:
            self.renderer = tp.VulkanRenderer(self.canvas, flush_frames=1 if headless else 3)
            self.renderer.sun_angular_radius = 1.6
            self.renderer.bloom_intensity = 0.2
            self.renderer.bloom_threshold = 1.3
        else:
            self.renderer = tp.GLRenderer(self.canvas)
        self.renderer.tone_mapping = tp.ToneMapping.ACESFilmic
        self.renderer.tone_mapping_exposure = 0.9
        self.renderer.shadow_map_enabled = True
        self.font = tp.FontLoader().default_font()
        self.sphere = icosphere(6)

        self.scene = tp.Scene()
        self.scene.background = 0x07090d
        self.scene.environment = tp.RGBELoader().load(
            make_studio(os.path.join(tempfile.gettempdir(), "threepp_trajectoid_studio.hdr")))
        sun = tp.DirectionalLight(0xfff1dd, 2.2)
        sun.position.set(-3.0, 11.0, 6.0)
        sun.cast_shadow = True
        sun.set_shadow_frustum(-12, 12, 12, -12)
        self.scene.add(sun)

        self.camera = tp.PerspectiveCamera(34, width / height, 0.1, 200)
        self.target = (3.05, -0.6, 1.0)
        self.aim(math.radians(14), math.radians(38), 19.6)

        self.table = tp.Group()                 # everything in it is in the table's frame
        self.table.rotation.z = -SLOPE
        self.scene.add(self.table)
        self._furniture()

        self.solids = cut_all()
        self.bodies, self.paints, self.inks, self.readouts, self.traces = [], [], [], [], []
        for i, (name, _, rgb, plain) in enumerate(LANES):
            paint = Ribbon(INK_POINTS, 0.085, 0.003, standard_material(colour(rgb, 0.16), roughness=0.75))
            ink_material = standard_material(colour((0.0, 0.0, 0.0)), roughness=0.4,
                                             emissive=colour(rgb), emissive_intensity=1.5)
            ink = Ribbon(INK_POINTS, 0.038, 0.007, ink_material)
            self.table.add(paint.mesh)
            self.table.add(ink.mesh)
            if self.vulkan:
                self.renderer.set_emissive_casts_light(ink.mesh, False)
            self.paints.append(paint)
            self.inks.append(ink)
            self._label(name, 0.40, -0.08, i, colour(rgb, 1.0))
            self.readouts.append(self._label(" ", 0.34, 0.52, i, tp.Color(0.55, 0.60, 0.68)))
            self.bodies.append(None)
            self.traces.append(None)
            self._place(i)
        self.chalk = Ribbon(900, 0.05, 0.012, standard_material(
            colour((0.0, 0.0, 0.0)), emissive=tp.Color(0.8, 0.8, 0.9), emissive_intensity=0.7))
        self.table.add(self.chalk.mesh)
        self.sim = None
        self.restart()

    def aim(self, azimuth, elevation, distance):
        x, y, z = self.target
        self.camera.position.set(x + distance * math.cos(elevation) * math.sin(azimuth),
                                 y + distance * math.sin(elevation),
                                 z + distance * math.cos(elevation) * math.cos(azimuth))
        self.camera.look_at(x, y, z)

    def _furniture(self):
        width = len(LANES) * LANE
        length, middle = END - START, 0.5 * (END + START)

        def block(size, centre, material, parent):
            mesh = tp.Mesh(tp.BoxGeometry(*size), material)
            mesh.position.set(*centre)
            mesh.cast_shadow = mesh.receive_shadow = True
            parent.add(mesh)

        brass = standard_material(tp.Color(0.80, 0.58, 0.28), roughness=0.28, metalness=1.0)
        dark = standard_material(tp.Color(0.012, 0.013, 0.017), roughness=0.35, metalness=0.4)
        block((length, 0.34, width), (middle, -0.171, 0.0), dark, self.table)
        top = tp.Mesh(tp.PlaneGeometry(length, width), standard_material(tp.Color(1, 1, 1), roughness=0.22))
        top.material.map = table_texture(length, width)
        top.rotation.x = -0.5 * math.pi
        top.position.set(middle, 0.0005, 0.0)
        top.receive_shadow = True
        self.table.add(top)
        block((0.2, 0.26, width), (RAIL + 0.1, 0.13, 0.0), brass, self.table)
        for z in (-0.5 * width - 0.06, 0.5 * width + 0.06):
            block((length, 0.12, 0.12), (middle, -0.03, z), brass, self.table)
        for x in (0.0, FINISH):                       # start and finish
            block((0.025, 0.002, width), (x, 0.001, 0.0),
                  standard_material(tp.Color(0.35, 0.37, 0.42), roughness=0.5), self.table)
        floor_y = -2.6
        floor = tp.Mesh(tp.PlaneGeometry(300, 300),
                        standard_material(tp.Color(0.010, 0.011, 0.014), roughness=0.3))
        floor.rotation.x = -0.5 * math.pi
        floor.position.y = floor_y
        floor.receive_shadow = True
        self.scene.add(floor)
        for x in (START + 0.7, END - 0.7):
            under = -x * math.tan(SLOPE) - 0.34
            for z in (-0.5 * width + 0.5, 0.5 * width - 0.5):
                block((0.22, under - floor_y, 0.22),
                      (x * math.cos(SLOPE), 0.5 * (under + floor_y), z), dark, self.scene)

    def _label(self, text, size, down, lane, tint):
        """Lettering painted flat on the table at the uphill end of a lane."""
        material = standard_material(tp.Color(0, 0, 0), emissive=tint, emissive_intensity=0.9)
        label = tp.Text2D(self.font, text, size=size, material=material)
        label.rotation.x = -0.5 * math.pi
        label.position.set(START + 0.25, 0.008, lane_z(lane) + down)
        self.table.add(label)
        if self.vulkan:
            self.renderer.set_emissive_casts_light(label, False)
        return label

    def _place(self, lane):
        """(Re)build what a lane shows: the solid on it and the line painted under it."""
        solid, (_, _, rgb, plain) = self.solids[lane], LANES[lane]
        if self.bodies[lane] is not None:
            self.table.remove(self.bodies[lane])
        body = tp.Group()
        if plain:
            chrome = standard_material(colour(rgb), roughness=0.06, metalness=1.0)
            body.add(tp.Mesh(tp.SphereGeometry(R, 96, 64), chrome))
        else:
            glaze = tp.MeshPhysicalMaterial()
            glaze.color = colour(rgb, 0.42)
            glaze.roughness = 0.32
            glaze.clearcoat, glaze.clearcoat_roughness = 1.0, 0.04
            metal = standard_material(tp.Color(0.90, 0.76, 0.57), roughness=0.31, metalness=1.0)
            for side, (pos, nrm) in skins(solid, self.sphere).items():
                geometry = tp.BufferGeometry()
                geometry.set_attribute("position", pos)
                geometry.set_attribute("normal", nrm)
                body.add(tp.Mesh(geometry, metal if side else glaze))
            # the line itself, wound onto the face it rolls on
            curve = tp.CatmullRomCurve3.from_array((R * 1.001 * solid.normals[::2]).astype(np.float32), closed=True)
            inlay = standard_material(colour(rgb, 0.25), roughness=0.3,
                                      emissive=colour(rgb), emissive_intensity=0.8)
            body.add(tp.Mesh(tp.TubeGeometry(curve, 720, 0.017, 10, True), inlay))
            if self.vulkan:
                self.renderer.set_emissive_casts_light(body.children[-1], False)
        for mesh in body.children:
            mesh.cast_shadow = mesh.receive_shadow = True
        self.table.add(body)
        self.bodies[lane] = body
        self.traces[lane] = Trace(lane, solid)
        paint = self.traces[lane].paint
        self.paints[lane].draw(paint[(paint[:, 0] >= 0.0) & (paint[:, 0] <= FINISH)])

    # ---- one run: roll, rest, wind back, roll again

    def restart(self):
        if self.sim is not None:
            self.sim.close()
        self.sim = Sim(self.solids)
        for trace, ink in zip(self.traces, self.inks):
            trace.clear()
            ink.draw([])
        self.mode, self.clock, self.history, self.frames = "roll", 0.0, [], 0
        self._pose([self.sim.pose(i) for i in range(len(LANES))])

    def _pose(self, poses):
        for body, (p, q) in zip(self.bodies, poses):
            body.position.set(*p)
            body.quaternion.set(*q)

    def _read_out(self):
        for (_, _, _, plain), trace, label in zip(LANES, self.traces, self.readouts):
            label.set_text(f"{1000 * trace.rms:.0f} mm off" if trace.length > 0.3 else " ")

    def advance(self, dt):
        self.clock += dt
        if self.mode == "wind":
            u = min(1.0, self.clock / 2.5)
            k = int((len(self.history) - 1) * (1.0 - u * u * (3.0 - 2.0 * u)))
            poses, counts = self.history[k]
            self._pose(poses)
            for ink, trace, count in zip(self.inks, self.traces, counts):
                ink.draw(trace.points[:count])
            if u >= 1.0:
                self.restart()
            return
        self.sim.step(dt)
        poses = [self.sim.pose(i) for i in range(len(LANES))]
        self._pose(poses)
        for ((x, _, z), _), trace, ink in zip(poses, self.traces, self.inks):
            if trace.add(x, z):
                ink.draw(trace.points)
        self.history.append((poses, [len(t.points) for t in self.traces]))
        self.frames += 1
        if self.frames % 8 == 0:
            self._read_out()
        done = all(p[0] > FINISH for p, _ in poses) or self.clock > 50.0
        if self.mode == "roll" and done:
            self.mode, self.clock = "rest", 0.0
        elif self.mode == "rest" and self.clock > 5.0:
            self.mode, self.clock = "wind", 0.0

    def render(self):
        self.renderer.render(self.scene, self.camera)


# ----------------------------------------------------------------------- the program

def headless(show, args):
    renderer = show.renderer
    if args.video > 0:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
        os.makedirs(out, exist_ok=True)
        w, h = parse_size(args.size)
        path = os.path.join(out, "trajectoid.mp4")
        film = Encoder(path, w, h, FPS)
        show.aim(math.radians(-6), math.radians(31), 19.2)
        for _ in range(32):                  # the temporal passes start from nothing
            show.render()
        for frame in range(int(args.video * FPS)):
            u = frame / (args.video * FPS)
            ease = u * u * (3.0 - 2.0 * u)
            show.aim(math.radians(-6 + 34 * ease), math.radians(31 + 11 * ease), 19.2 + 1.2 * ease)
            show.advance(PACE / FPS)
            if show.vulkan:
                renderer.sim_time = frame / FPS
            show.render()
            film.send(renderer.read_pixels())
        film.close()
        print("wrote", path)
        return
    times = sorted(float(t) for t in args.at.split(","))
    stem, ext = os.path.splitext(args.shot)
    frame = 0
    for n, t in enumerate(times):
        while frame < t * FPS:
            show.advance(1.0 / FPS)
            frame += 1
            if show.vulkan:
                renderer.sim_time = frame / FPS
            if frame > t * FPS - 4:
                show.render()
        show._read_out()
        for _ in range(48):                  # let the temporal passes settle on the still
            show.render()
        path = args.shot if len(times) == 1 else f"{stem}_{n}{ext}"
        if show.vulkan:
            renderer.save_frame(show.scene, show.camera, path)
        else:
            renderer.save_frame(path)
        print(f"wrote {path} after {t:.1f} s")
    for (name, _, _, _), trace in zip(LANES, show.traces):
        print(f"  {name:<12}rolled {trace.length:5.2f} m, {1000 * trace.rms:4.0f} mm rms off the line")


def live(show):
    canvas, camera = show.canvas, show.camera
    state = {"stroke": None, "paused": False, "note": ""}
    ui = tp.ImguiContext(canvas, show.renderer) if tp.HAS_IMGUI else None

    def on_table(x, y):
        """The table point (x, z) under a window pixel, or None."""
        w, h = canvas.size()
        ray = tp.Raycaster()
        ray.set_from_camera(tp.Vector2(2.0 * x / w - 1.0, 1.0 - 2.0 * y / h), camera)
        c, s = math.cos(SLOPE), math.sin(SLOPE)
        o, d = ray.ray.origin, ray.ray.direction
        oy, dy = s * o.x + c * o.y, s * d.x + c * d.y        # into the table's frame
        if dy > -1e-6:
            return None
        t = -oy / dy
        return c * (o.x + t * d.x) - s * (o.y + t * d.y), o.z + t * d.z

    def clear_lane(clear):
        """Hide what stands in the drawing lane while a line is being drawn there."""
        show.bodies[DRAWN].visible = show.paints[DRAWN].mesh.visible = not clear
        show.inks[DRAWN].mesh.visible = not clear and len(show.traces[DRAWN].points) > 1

    # Registered before OrbitControls, so a press in the drawing lane can switch it off first.
    def on_down(button, x, y):
        if button != 0 or (ui is not None and ui.want_capture_mouse):
            return
        p = on_table(x, y)
        if p and abs(p[1] - lane_z(DRAWN)) < 0.5 * LANE and START < p[0] < END:
            state["stroke"] = [p]
            controls.enabled = False
            clear_lane(True)

    def on_move(x, y):
        stroke = state["stroke"]
        if stroke is not None:
            p = on_table(x, y)
            if p and p[0] > stroke[-1][0] + 0.02:            # a line only goes downhill
                stroke.append(p)
                show.chalk.draw(stroke)

    def on_up(button, x, y):
        stroke, state["stroke"] = state["stroke"], None
        if stroke is None:
            return
        controls.enabled = True
        show.chalk.draw([])
        clear_lane(False)
        shape = stroke_shape(stroke)
        if shape is None:
            state["note"] = "draw a longer line, left to right"
            return
        t0 = time.perf_counter()
        solid = cut(shape)
        if solid is None:
            state["note"] = "that line cannot be cut: try gentler turns"
            return
        show.solids[DRAWN] = solid
        show._place(DRAWN)
        show.restart()
        flat = "" if solid.kept == 1.0 else f", swing flattened to {100 * solid.kept:.0f}%"
        state["note"] = f"cut in {time.perf_counter() - t0:.1f} s{flat}"

    canvas.on_mouse_down(on_down)
    canvas.on_mouse_move(on_move)
    canvas.on_mouse_up(on_up)
    controls = tp.OrbitControls(camera, canvas)
    controls.target = tp.Vector3(*show.target)
    controls.enable_damping = True
    controls.max_distance = 60.0

    def on_key(key, _mods):
        if key == "R":
            show.restart()
        elif key == "SPACE":
            state["paused"] = not state["paused"]
    canvas.on_key_pressed(on_key)
    canvas.on_window_resize(resize_handler(camera, show.renderer))
    clock = tp.Clock()

    def panel():
        im = tp.imgui
        # top right, where the table leaves the picture empty; sized to its text
        width = 0.8 * im.calc_text_size("drag across the violet lane to draw")[0] + 24
        im.set_next_window_pos(canvas.size()[0] - width - 10, 10, always=True)
        im.set_next_window_size(width, 0, always=True)
        im.begin("trajectoid")
        im.set_window_font_scale(0.8)
        im.text("drag across the violet lane to draw")
        if state["note"]:
            im.text(state["note"])
        im.separator()
        im.text(f"{'mm off line':<11}{'rms':>5}{'worst':>7}")
        for (name, _, _, _), trace in zip(LANES, show.traces):
            im.text(f"{name:<11}{1000 * trace.rms:5.0f}{1000 * trace.worst:7.0f}")
        im.separator()
        if im.button("roll again (R)"):
            show.restart()
        _, state["paused"] = im.checkbox("pause (space)", state["paused"])
        im.end()

    def frame():
        dt = min(clock.get_delta(), 1.0 / 30.0)
        if not state["paused"] and state["stroke"] is None:     # everything waits for the pen
            show.advance(PACE * dt)
        controls.update()
        show.render()
        if ui is not None:
            ui.render(panel)

    canvas.animate(frame)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="no window: roll and print the tracking table")
    ap.add_argument("--shot", help="headless still(s) to this PNG")
    ap.add_argument("--at", default="7.0", help="seconds of rolling for --shot, comma separated")
    ap.add_argument("--video", type=float, default=0.0, help="headless film of this many seconds")
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--gl", action="store_true", help="draw with GL even where there is Vulkan")
    args = ap.parse_args()
    if not tp.HAS_PHYSX:
        print("This demo needs a threepp build with PhysX.")
        return
    if args.check:
        check()
        return
    show = Show(args)
    if args.shot or args.video > 0:
        headless(show, args)
    else:
        live(show)
    show.sim.close()


if __name__ == "__main__":
    main()
