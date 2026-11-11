"""
flag_rig -- a flag that flies: tp.ClothMesh (threepp's XPBD cloth) on a hoist that moves with
whatever carries it.

Two things, and nothing scene-specific:

  * ``Flag``          -- the flag as cloth. Built from the static flag node it replaces (its place,
                         size and orientation are read from the node at run time, and the node
                         is hidden, not deleted), or from a hoist and a size. Each frame
                         ``flag.update(dt, wind)`` with the TRUE wind, after whatever carries the
                         flag has been moved for the frame.
  * ``nordic_cross``  -- a flag's picture built in numpy: a field and a Nordic cross with an edge
                         (the Norwegian flag by default, 22:16, cross at 6-1-2-1-12 along the fly
                         and 6-1-2-1-6 along the hoist). ``node_colours`` reads the colours of a
                         static flag's materials, so the cloth wears what the model wore.

How a scene uses it (the scene owns the loop; this module never imports threepp, it is handed
the module):

    node = vessel.get_object_by_name("flag")
    c = flag_rig.node_colours(node)
    flag = flag_rig.Flag(tp, scene, node, flag_rig.nordic_cross(c["flag_red"], c["flag_white"], c["flag_blue"]))
    ...each frame, after the vessel is placed:
    flag.update(dt, true_wind_xyz)

FRAME
-----
The cloth is simulated in a frame that TRAVELS WITH THE HEAD OF THE HOIST: its origin is on the
hoist's head and follows it, its axes stay the world's (it never turns), and the mesh sits at the
scene's root, moved to that origin each step, not parented to the carrier. In that frame:

  * the hoist particles are pinned to where the hoist is now relative to its head (the anchor's
    world matrix applied to the hoist line), so the carrier's turning and rolling reach the cloth
    through the pins, carried there over the step's substeps;
  * the air is the true wind minus the head's velocity: the apparent wind, so a flag on a ship
    under way streams aft from her speed with no wind at all, and each triangle then takes off its
    own velocity in the frame, as the solver does;
  * gravity is plainly down, less the head's acceleration (the frame's inertial force), so her
    surging, heaving and the turn's centripetal pull swing the flag as they would.

That is the same motion a world-space simulation gives. What it adds is the solver's gusts on
the air the flag actually meets: the solver varies its wind in waves that travel downwind across
the cloth, scaled by that wind's speed. In world space the wind it is given is the true wind, so
a flag at 28 knots in a 3 m/s breeze would get gusts of 3 m/s travelling at 3 m/s, against 17 m/s
of apparent wind: a frozen sheet. Here they are gusts of the apparent wind, travelling across the
flag at its speed. It also keeps the coordinates small however far the carrier goes.

A jump of the hoist bigger than any carrier can make in one step (a vessel seated somewhere
else, a scene reset) rebuilds the cloth at rest at the new place instead of dragging it there.

THE STAFF
---------
A capsule just off the hoist line on the side away from the fly, the length of the hoist, keeps
the cloth from folding through the staff when the wind blows it back over the hoist. It is the
hoist line, not the model's staff: a staff raked away from the hoist (as on M/S Valderøy, whose
static flag hangs plumb from the staff's top) is not seen below the head of the flag.

The mesh casts and receives shadows. Its positions and normals are rewritten on the CPU every
frame with the topology fixed, which both renderers take as a plain dynamic mesh: GL streams the
buffers, and the Vulkan renderer moves a mesh that stays dirty onto its per-frame path, where
its ray-tracing structure is refit each frame, so ray-traced shadows follow the cloth.

CLOTH
-----
Flag polyester, from the solver's guidance: 0.12 kg/m2, inextensible (stretch compliance 0) and
soft in bending. A 1.1 x 0.8 m flag on a 16 x 12 grid (7 cm cells, 221 particles, about the
solver example's density) at 32 substeps, which hold its longest edge within about 8 % of its
rest length (about 19 % when it flaps hardest); a step costs 0.6-0.9 ms of one CPU core.
"""
import time

import numpy as np

# Flag polyester (Cloth.hpp: 0.1-0.15 kg/m2), soft in bending. Substeps: the solver runs one
# constraint pass per substep, so the stretch it leaves concentrates where the whole flag hangs
# on the pinned hoist, worst at the top corner. Measured on a 1.1 x 0.8 m flag at 30 frames a
# second, 16 x 12 cells, longest edge over rest: 1.31 at the solver's default 12 substeps,
# 1.10 at 24, 1.06-1.08 at 32 in 3 and 12 m/s; with the solver's flapping wave, 1.09-1.19 at 32
# from 3 m/s to 15 m/s. A 22 x 16 grid needs about twice the substeps for the same stretch.
AREAL_DENSITY = 0.12
BEND_COMPLIANCE = 0.5
SUBSTEPS = 32
GUSTINESS = 0.35          # the solver's own default: gusts of +-17 % in speed, +-9 degrees


def linear_to_srgb8(c):
    """Linear RGB (0..1) to 8-bit sRGB, the encoding a colour texture is stored in."""
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    s = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)
    return np.round(s * 255.0).astype(np.uint8)


def node_colours(node):
    """{material name: (r, g, b) linear} for the meshes under a static flag node."""
    out = {}

    def visit(o):
        m = getattr(o, "material", None)
        if getattr(o, "geometry", None) is not None and m is not None:
            out[m.name] = (m.color.r, m.color.g, m.color.b)
        for c in o.children:
            visit(c)
    visit(node)
    return out


def nordic_cross(field, edge, cross, fly=(6, 1, 2, 1, 12), hoist=(6, 1, 2, 1, 6), px=16):
    """A Nordic cross flag as an (H, W, 3) uint8 sRGB image, row 0 at the BOTTOM (texture v = 0)
    and column 0 at the hoist. Colours are linear RGB (a material's colour); ``fly`` and
    ``hoist`` are the bands' widths in units along each side, field-edge-cross-edge-field."""
    def bands(units):
        u = (np.arange(sum(units) * px) + 0.5) / px
        b = np.cumsum(units)
        return np.where((u >= b[1]) & (u < b[2]), 2, np.where((u >= b[0]) & (u < b[3]), 1, 0))
    col, row = bands(fly), bands(hoist)
    which = np.maximum(row[:, None], col[None, :])            # cross over edge over field
    palette = np.stack([linear_to_srgb8(field), linear_to_srgb8(edge), linear_to_srgb8(cross)])
    return np.ascontiguousarray(palette[which])


class Flag:
    """A flag as cloth on a hoist that moves with ``anchor`` (an Object3D), simulated in a frame
    that travels with the head of the hoist (see the module's FRAME).

    Flag(tp, scene, node, picture)
        Replaces a static flag node: the hoist is the node's local x = 0 line over the node's
        height (local y), the fly runs along local x to the side the node's geometry lies on,
        the cloth's face is the node's local z. The node is hidden.
    Flag(tp, scene, None, picture, anchor=obj, hoist=(top, bottom), fly=(dx, dy, dz), length=L)
        The same from a hoist line (two points in ``anchor``'s frame, or the world's if anchor is
        None), the fly's direction in that frame and the fly's length.

    picture: an (H, W, 3|4) uint8 sRGB image (row 0 at the bottom, column 0 at the hoist), as
    nordic_cross makes, or a tp.Texture. segments: grid cells along the fly and the hoist.

    update(dt, wind): after the anchor is placed for the frame; wind is the true wind, m/s, in
    world axes. ``step_ms`` is the last update's cost, ``max_stretch()`` the longest edge over
    its rest length, ``apparent_wind`` the air the head of the hoist met in the last step.
    """

    def __init__(self, tp, scene, node, picture, anchor=None, hoist=None, fly=(-1.0, 0.0, 0.0), length=None,
                 segments=(16, 12), staff_radius=0.022, params=None):
        self.tp = tp
        if node is not None:
            anchor = node
            hoist, fly, length = self._from_node(node)
            node.visible = False
        if hoist is None or length is None:
            raise ValueError("Flag: give the static flag node, or hoist=(top, bottom) and length")
        self.anchor = anchor
        self.top, self.bottom = (np.asarray(p, float) for p in hoist)
        f = np.asarray(fly, float)
        self.fly = f / np.linalg.norm(f)
        self.length = float(length)
        self.nx, self.ny = segments
        self.staff_radius = staff_radius

        if params is None:
            params = tp.Cloth.Params()
            params.areal_density = AREAL_DENSITY
            params.bend_compliance = BEND_COMPLIANCE
            params.substeps = SUBSTEPS
            params.gustiness = GUSTINESS
        self.params = params
        self.gravity = np.array([params.gravity.x, params.gravity.y, params.gravity.z], float)

        # The sheet, in the anchor's frame: column i along the fly, row j up the hoist.
        i = np.arange(self.nx + 1) / self.nx
        j = np.arange(self.ny + 1) / self.ny
        up = self.top - self.bottom
        self.rest_local = (self.bottom + j[:, None, None] * up + i[None, :, None] * self.length * self.fly).reshape(-1, 3)
        uv = np.stack(np.meshgrid(i, j), axis=-1).reshape(-1, 2).astype(np.float32)
        a = (np.arange(self.ny)[:, None] * (self.nx + 1) + np.arange(self.nx)[None, :]).ravel()
        b, c, d = a + 1, a + self.nx + 1, a + self.nx + 2
        # Wound so the face looks along fly x up, whichever way the fly runs.
        idx = np.stack([a, b, d, a, d, c], axis=1).astype(np.uint32).ravel()
        self.hoist_vertices = np.arange(self.ny + 1) * (self.nx + 1)

        self.geometry = tp.BufferGeometry()
        self.geometry.set_attribute("position", self.rest_local.astype(np.float32))
        self.geometry.set_attribute("normal", np.zeros((len(self.rest_local), 3), np.float32))
        self.geometry.set_attribute("uv", uv)
        self.geometry.set_index(idx)

        material = tp.MeshStandardMaterial()
        material.name = "flag_cloth"
        material.map = picture if isinstance(picture, tp.Texture) else tp.data_texture(picture, True)
        material.roughness = 0.85
        material.side = tp.Side.Double
        self.material = material
        # At the scene's root, moved to the frame's origin each step and never rotated: the
        # cloth's positions are relative to the head of the hoist, in world axes.
        self.mesh = tp.Mesh(self.geometry, material)
        self.mesh.name = "flag_cloth"
        self.mesh.cast_shadow = self.mesh.receive_shadow = True
        scene.add(self.mesh)

        self.step_ms = 0.0
        self.apparent_wind = np.zeros(3)
        self._build()

    # -- set-up ----------------------------------------------------------------

    @staticmethod
    def _from_node(node):
        """Hoist, fly direction and length from a static flag node's geometry, in its frame."""
        inv = np.linalg.inv(node.matrix_world.to_numpy().astype(float))
        pts = []

        def visit(o):
            g = getattr(o, "geometry", None)
            if g is not None:
                p = g.get_attribute("position")
                m = inv @ o.matrix_world.to_numpy().astype(float)
                pts.append(p @ m[:3, :3].T + m[:3, 3])
            for c in o.children:
                visit(c)
        visit(node)
        if not pts:
            raise ValueError(f"Flag: node '{node.name}' has no geometry to take the flag's size from")
        p = np.concatenate(pts)
        lo, hi = p.min(axis=0), p.max(axis=0)
        side = -1.0 if (lo[0] + hi[0]) < 0.0 else 1.0           # the fly lies on this side of x = 0
        return ((0.0, hi[1], 0.0), (0.0, lo[1], 0.0)), (side, 0.0, 0.0), float(hi[0] - lo[0])

    def _matrix(self):
        return np.eye(4) if self.anchor is None else self.anchor.matrix_world.to_numpy().astype(float)

    @staticmethod
    def _world(local, M):
        return np.asarray(local, float) @ M[:3, :3].T + M[:3, 3]

    def _build(self):
        """A fresh cloth at rest at the hoist's present place, the frame's origin on its head."""
        tp = self.tp
        M = self._matrix()
        self.origin = self._world(self.top, M)
        self.velocity = None                                    # the frame's, known after a step
        # A new position attribute, then a new ClothMesh over it (the old one pointed at the
        # attribute this replaces).
        self.geometry.set_attribute("position", (self._world(self.rest_local, M) - self.origin).astype(np.float32))
        self.mesh.position.set(*map(float, self.origin))
        self.cloth_mesh = tp.ClothMesh(self.geometry, self.params)
        self.cloth = self.cloth_mesh.cloth
        self.pins = np.array([self.cloth_mesh.particle_of(int(v)) for v in self.hoist_vertices], np.int64)
        self.cloth.pin_many(self.pins)
        self._staff = self.cloth.add_capsule(tp.Vector3(), tp.Vector3(), self.staff_radius)
        tri = self.cloth.indices
        self._edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        x = self.cloth.positions
        self._rest = np.linalg.norm(x[self._edges[:, 0]] - x[self._edges[:, 1]], axis=1)

    def _staff_line(self, M, origin):
        """The staff's axis in the frame: the hoist line, set off by the staff's radius and the
        cloth's padding to the side away from the fly."""
        off = -(self.staff_radius + self.params.thickness) * self.fly
        return self._world([self.top + off, self.bottom + off], M) - origin

    # -- per frame -------------------------------------------------------------

    def update(self, dt, wind):
        """Pin the hoist where it is now, blow the true ``wind`` (m/s, world axes), step dt."""
        if dt <= 0.0:
            return
        tp = self.tp
        M = self._matrix()
        head = self._world(self.top, M)
        if np.linalg.norm(head - self.origin) > max(2.0, 40.0 * dt):
            self._build()                                       # a jump, not a motion

        # The frame moves from the old head to the new over the step. Its velocity takes the
        # place of the cloth's own in the air it meets; its acceleration (the change of that
        # velocity across steps) pushes on every particle the other way, as gravity does. The
        # first step after a build takes the velocity as steady.
        velocity = (head - self.origin) / dt
        accel = np.zeros(3) if self.velocity is None else (velocity - self.velocity) / dt
        self.velocity, self.origin = velocity, head
        self.apparent_wind = np.asarray(wind, float) - velocity
        self.cloth.params.gravity = tp.Vector3(*map(float, self.gravity - accel))
        self.cloth.set_wind(tp.Vector3(*map(float, self.apparent_wind)))

        hoist = self._world(self.rest_local[self.hoist_vertices], M) - head
        self.cloth.pin_many(self.pins, hoist.astype(np.float32))
        s0, s1 = self._staff_line(M, head)
        self.cloth.set_capsule(self._staff, tp.Vector3(*map(float, s0)), tp.Vector3(*map(float, s1)), self.staff_radius)

        t0 = time.perf_counter()
        self.cloth_mesh.update(float(dt))
        self.step_ms = (time.perf_counter() - t0) * 1e3
        self.mesh.position.set(*map(float, head))

    def world_positions(self):
        """(N, 3) the particles in world space."""
        return self.cloth.positions.astype(float) + self.origin

    def max_stretch(self):
        """The longest edge over its rest length (1.0 = no stretch)."""
        x = self.cloth.positions
        return float((np.linalg.norm(x[self._edges[:, 0]] - x[self._edges[:, 1]], axis=1) / self._rest).max())

    def staff_clearance(self):
        """The least distance of a free particle from the staff's axis less the staff's radius
        (below the cloth's padding = into the staff), and how many free particles are more than
        a staff's width on the far side of the hoist line from the fly (blown back round it)."""
        M = self._matrix()
        a, b = self._staff_line(M, self.origin)
        x = self.cloth.positions.astype(float)
        free = ~self.cloth.pinned
        ab = b - a
        u = np.clip((x - a) @ ab / (ab @ ab), 0.0, 1.0)
        d = np.linalg.norm(x - (a + u[:, None] * ab), axis=1) - self.staff_radius
        fly_w = M[:3, :3] @ self.fly
        behind = (x @ fly_w) < -2.0 * (self.staff_radius + self.params.thickness)
        return float(d[free].min()), int((behind & free).sum())
