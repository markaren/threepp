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

    _kernel = (wp, cast_down)
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
        wp, self._cast = _compile()
        self._wp = wp
        wp.init()
        self.device = torch.device(device)
        self._wp_device = f"cuda:{self.device.index or 0}" if self.device.type == "cuda" else "cpu"
        ax, bx, upv = _BASIS[up]
        self._ax, self._bx, self._up = wp.vec3(*ax), wp.vec3(*bx), wp.vec3(*upv)
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
        self._out = None                                     # result buffer, grown on demand

    @classmethod
    def from_objects(cls, objects, **kwargs):
        """Build from threepp objects -- e.g. `CollectedWorld.meshes`."""
        verts, faces = object_soup(objects)
        return cls(verts, faces, **kwargs)

    def heights(self, a, b, *, h0=None, far=None, miss=None):
        """Ground height under each (a, b) -- (x, y) for up='z', (x, z) for up='y'.

        `a` and `b` are torch tensors of any matching shape; the result has that shape.
        Points with no surface within `far` below the ray origin get `miss`.
        """
        if a.shape != b.shape:
            raise ValueError(f"heights: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        wp = self._wp
        af = a.contiguous().reshape(-1).float()
        bf = b.contiguous().reshape(-1).float()
        n = af.numel()
        if self._out is None or self._out.numel() < n:
            self._out = torch.empty(n, device=self.device)
        out = self._out[:n]
        wp.launch(self._cast, dim=n,
                  inputs=[self.mesh.id, wp.from_torch(af), wp.from_torch(bf),
                          self._ax, self._bx, self._up,
                          self.h0 if h0 is None else float(h0),
                          self.far if far is None else float(far),
                          self.miss if miss is None else float(miss)],
                  outputs=[wp.from_torch(out)], device=self._wp_device)
        return out.view(a.shape)

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
