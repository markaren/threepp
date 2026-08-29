"""Batched terrain queries against the geometry that is actually in the world.

A GpuSim env asks "how high is the ground at these K x N points?" every control step --
for the height scan in the observation, for the drop-settle spawn, for the fell-over
test. Today every env answers that with an analytic formula it evaluates in torch, which
means the terrain is only ever allowed to be something you can write down in closed form:
a tent, a bilinear grid, a plane. Authored geometry, an imported mesh, an overhang, a
scene.json loaded out of the editor -- none of those have a formula.

This module answers the same question by casting a ray down the up-axis into a Warp BVH
built over the world's static triangles, so the answer comes from the geometry the physics
is actually colliding against. It costs about the same:

    K=2048, 45-cell scan, 2.75 M triangles   ->  1.74 ms   (52.8 M rays/s, RTX 4070)
    the same scan as a torch bilinear gather ->  1.04 ms

Warp shares torch's CUDA primary context -- the same one GpuSim hands PhysX -- so the query
reads GpuSim's state tensors and writes its answer with no copy and no context switch.
Nothing here touches the host.

    from threepp.rl.raycast import CollectedWorld, TerrainRays

    rec = CollectedWorld(world)            # forwards to `world`, keeps every added Mesh
    build_my_terrain(rec)                  # unchanged builder code
    rays = TerrainRays.from_objects(rec.meshes, up="z")

    h = rays.heights(px, py)               # [K, N] ground height under each point

Warp is imported lazily, on the first TerrainRays, so `import threepp.rl` still works on a
machine without it.
"""
import numpy as np
import torch

import threepp as tp

# The two horizontal basis vectors and the up vector, per up-axis convention. `heights(a, b)`
# takes its coordinates in this order: (x, y) for up="z", (x, z) for up="y".
_BASIS = {
    "z": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "y": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}

_kernel = None          # (warp module, compiled kernel) -- built once per process


def _compile():
    """Import Warp and build the one kernel. Deferred so importing this module is free."""
    global _kernel
    if _kernel is not None:
        return _kernel
    import warp as wp

    @wp.kernel
    def cast_down(mesh: wp.uint64,
                  a: wp.array(dtype=float),          # first horizontal coordinate
                  b: wp.array(dtype=float),          # second horizontal coordinate
                  ax: wp.vec3, bx: wp.vec3, up: wp.vec3,
                  h0: float, far: float, miss: float,
                  out: wp.array(dtype=float)):
        i = wp.tid()
        org = a[i] * ax + b[i] * bx + h0 * up
        q = wp.mesh_query_ray(mesh, org, -up, far)
        out[i] = wp.where(q.result, h0 - q.t, miss)

    @wp.kernel
    def see_points(mesh: wp.uint64,
                   root: wp.array(dtype=wp.vec3),      # [K] body origin, in (a, b, height) order
                   quat: wp.array(dtype=wp.quat),      # [K] body->world rotation, (x, y, z, w)
                   mount: wp.vec3, mount_rot: wp.quat, # camera pose in the body frame
                   a: wp.array2d(dtype=float),         # [K,N] target, first horizontal coordinate
                   b: wp.array2d(dtype=float),         # [K,N] target, second
                   h: wp.array2d(dtype=float),         # [K,N] target height (the ground there)
                   ax: wp.vec3, bx: wp.vec3, up: wp.vec3,
                   tan_x: float, tan_y: float, near: float, far: float, bias: float,
                   out: wp.array2d(dtype=wp.uint8)):   # [K,N] 1 = the camera can see that point
        k, n = wp.tid()
        eye = root[k] + wp.quat_rotate(quat[k], mount)
        tgt = a[k, n] * ax + b[k, n] * bx + h[k, n] * up
        d = tgt - eye
        dist = wp.length(d)
        out[k, n] = wp.uint8(0)
        if dist < 1.0e-6 or dist > far:
            return
        # into the camera frame, which looks down its own -z (the DepthSensor convention)
        c = wp.quat_rotate_inv(wp.mul(quat[k], mount_rot), d)
        fwd = -c[2]
        if fwd < near or wp.abs(c[0]) > tan_x * fwd or wp.abs(c[1]) > tan_y * fwd:
            return
        # `bias` keeps the ray from being stopped by the surface it is aimed at
        q = wp.mesh_query_ray(mesh, eye, d / dist, dist - bias)
        if not q.result:
            out[k, n] = wp.uint8(1)

    _kernel = (wp, cast_down, see_points)
    return _kernel


# --------------------------------------------------------------------------- #
#  Collecting the static geometry a build_world closure creates
# --------------------------------------------------------------------------- #
class CollectedWorld:
    """A PhysxWorld stand-in that forwards every call and keeps a reference to each Mesh
    passed to an `add*` method.

    A `build_world` closure drops its meshes on the floor the moment it returns -- PhysX has
    cooked them and no longer needs the threepp objects. Wrapping the world is how the scan
    gets at that geometry without every terrain builder growing a return value, and it means
    the BVH is built from whatever was actually added rather than from a second,
    hand-maintained description of it that can drift out of step with the colliders.
    """

    _RECORDED = ("add", "add_static", "add_dynamic_convex", "add_static_trimesh",
                 "add_static_trimesh_tree", "add_instanced")

    def __init__(self, world):
        self._world = world
        self.meshes = []

    def __getattr__(self, name):
        attr = getattr(self._world, name)
        if name not in self._RECORDED:
            return attr

        def record(obj, *args, **kwargs):
            self.meshes.append(obj)
            return attr(obj, *args, **kwargs)
        return record

    @property
    def world(self):
        """The wrapped PhysxWorld, for the calls that must see the real thing."""
        return self._world


def object_soup(objects):
    """Bake an iterable of threepp objects (each traversed) into one world-space triangle
    soup -> (verts [V,3] float32, faces [F,3] int32).

    Non-indexed geometry -- a `set_from_points` soup, as the heightfield env builds -- is
    taken as consecutive triples. Anything without a position attribute is skipped, so
    handing this a whole scene root is safe.
    """
    verts, faces, base = [], [], 0
    for root in objects:
        root.update_matrix_world(True)
        nodes = []
        root.traverse(nodes.append)
        for o in nodes:
            if not isinstance(o, tp.Mesh):
                continue
            p = o.geometry.get_attribute("position")
            if p is None or len(p) == 0:
                continue
            idx = o.geometry.get_index()
            f = (np.arange(len(p), dtype=np.int32).reshape(-1, 3) if idx is None
                 else idx.astype(np.int32).reshape(-1, 3))
            m = o.matrix_world.to_numpy()
            verts.append(p @ m[:3, :3].T + m[:3, 3])
            faces.append(f + base)
            base += len(p)
    if not verts:
        raise ValueError("object_soup: no triangles found (no Mesh with a position attribute)")
    return (np.concatenate(verts).astype(np.float32),
            np.concatenate(faces).astype(np.int32))


# --------------------------------------------------------------------------- #
#  The BVH and the query
# --------------------------------------------------------------------------- #
class TerrainRays:
    """A Warp BVH over static world triangles, queried as batched downward rays.

    `heights(a, b)` returns the height of the first surface below each point, in the shape of
    its inputs. Rays start at `h0` (default: one metre above the tallest triangle) and run
    `far` (default: down past the lowest one), so a point over a hole -- or off the terrain
    entirely -- returns `miss` rather than a plausible wrong number.
    """

    def __init__(self, verts, faces, *, device="cuda", up="z", h0=None, far=None, miss=-10.0):
        if up not in _BASIS:
            raise ValueError(f"up must be 'y' or 'z', got {up!r}")
        wp, self._cast, self._see = _compile()
        self._wp = wp
        wp.init()
        self.device = torch.device(device)
        self._wp_device = f"cuda:{self.device.index or 0}" if self.device.type == "cuda" else "cpu"
        ax, bx, upv = _BASIS[up]
        self._ax, self._bx, self._up = wp.vec3(*ax), wp.vec3(*bx), wp.vec3(*upv)
        # the same basis as torch rows, for callers that need to project a world vector onto the
        # two horizontal axes (PerceivedScan turns a body quaternion into a heading this way)
        self.ax_t = torch.tensor(ax, device=self.device)
        self.bx_t = torch.tensor(bx, device=self.device)
        self.up_t = torch.tensor(upv, device=self.device)
        self.up = up
        self.miss = float(miss)

        verts = np.ascontiguousarray(verts, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        self.n_tris = len(faces)
        axis = "xyz".index(up)
        lo, hi = float(verts[:, axis].min()), float(verts[:, axis].max())
        self.h0 = float(hi + 1.0) if h0 is None else float(h0)
        self.far = float((self.h0 - lo) + 1.0) if far is None else float(far)

        self._points = wp.array(verts, dtype=wp.vec3, device=self._wp_device)
        self.mesh = wp.Mesh(points=self._points,
                            indices=wp.array(faces.reshape(-1), dtype=wp.int32,
                                             device=self._wp_device))

    @classmethod
    def from_objects(cls, objects, **kwargs):
        """Build from threepp objects -- e.g. `CollectedWorld.meshes`."""
        verts, faces = object_soup(objects)
        return cls(verts, faces, **kwargs)

    def heights(self, a, b, *, h0=None, far=None, miss=None, out=None):
        """Ground height under each (a, b) -- (x, y) for up='z', (x, z) for up='y'.

        `a` and `b` are torch tensors of any matching shape; the result has that shape.
        Points with no surface within `far` below the ray origin get `miss`.

        The result is a FRESH tensor unless `out` is given. An earlier version handed back a view
        of one reused buffer, which silently aliased the moment a caller held two results at once
        -- `h_here = heights(x, y)` followed by `heights(px, py)` left `h_here` pointing at the
        second call's output. Pass `out` to reuse a buffer only where that cannot happen.
        """
        if a.shape != b.shape:
            raise ValueError(f"heights: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        wp = self._wp
        af = a.contiguous().reshape(-1).float()
        bf = b.contiguous().reshape(-1).float()
        n = af.numel()
        if out is None:
            out = torch.empty(n, device=self.device)
        elif out.numel() != n:
            raise ValueError(f"heights: out has {out.numel()} elements, need {n}")
        else:
            out = out.reshape(-1)
        wp.launch(self._cast, dim=n,
                  inputs=[self.mesh.id, wp.from_torch(af), wp.from_torch(bf),
                          self._ax, self._bx, self._up,
                          self.h0 if h0 is None else float(h0),
                          self.far if far is None else float(far),
                          self.miss if miss is None else float(miss)],
                  outputs=[wp.from_torch(out)], device=self._wp_device)
        return out.view(a.shape)

    def visible(self, root, quat, a, b, h, *, mount, mount_rot, tan_x, tan_y,
                near=0.05, far=10.0, bias=0.02, out=None):
        """Which of the [K, N] ground points a body-mounted camera can actually see, right now.

        The camera sits at `mount` in the body frame, rotated by `mount_rot`, and looks down its own
        -z the way `tp.DepthSensor` does — so the same numbers that place the sensor in a deploy
        viewer place it here. A point counts as seen when it is inside the frustum (`near`, `far`,
        the two tangent half-angles `tan_x`/`tan_y`, which are the camera's own IMAGE axes and
        have nothing to do with the world basis this class is built on) AND no triangle stands
        between the eye and it. `bias` is how
        far short of the target the occlusion ray stops, so the surface being looked at does not
        occlude itself; make it larger than the terrain's own triangle scale is not needed, a couple
        of centimetres is plenty.

        root [K,3] and quat [K,4] (x,y,z,w) are the body pose — `sim.root_position` and
        `sim.root_quat` straight out of GpuSim. Returns uint8 [K, N].
        """
        wp = self._wp
        K, N = a.shape
        if out is None or out.shape != (K, N):
            out = torch.empty((K, N), dtype=torch.uint8, device=self.device)
        wp.launch(self._see, dim=(K, N),
                  inputs=[self.mesh.id,
                          wp.from_torch(root.contiguous().float(), dtype=wp.vec3),
                          wp.from_torch(quat.contiguous().float(), dtype=wp.quat),
                          wp.vec3(*mount), wp.quat(*mount_rot),
                          wp.from_torch(a.contiguous().float()),
                          wp.from_torch(b.contiguous().float()),
                          wp.from_torch(h.contiguous().float()),
                          self._ax, self._bx, self._up,
                          float(tan_x), float(tan_y), float(near), float(far), float(bias)],
                  outputs=[wp.from_torch(out)], device=self._wp_device)
        return out

    def refit(self, verts):
        """Update the vertex positions in place and refit the BVH -- for terrain that moves
        between episodes. Cheaper than a rebuild, but only valid while the topology and the
        vertex count are unchanged."""
        v = torch.as_tensor(verts, device=self.device, dtype=torch.float32).contiguous()
        if v.numel() != self._points.size * 3:
            raise ValueError(f"refit: expected {self._points.size} vertices, got {v.numel() // 3}")
        self._wp.copy(self._points, self._wp.from_torch(v.view(-1, 3), dtype=self._wp.vec3))
        self.mesh.refit()

    def __repr__(self):
        return (f"TerrainRays({self.n_tris} tris, up={self.up!r}, "
                f"h0={self.h0:.2f}, far={self.far:.2f})")
