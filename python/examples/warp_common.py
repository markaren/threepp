"""What the warp_*.py examples share.

Each example is a self-contained demo of one thing: a cloth, a pressurised
shell, a fluid, a sail. The pieces that are the same in every one of them --
command-line parsing, the icosphere and its constraint tables, the Verlet
predict / volume / normal kernels, the particles-to-marching-cubes surfacing,
the standard studio lighting, and the window / headless-shot / benchmark run
loops -- live here so the examples can be read for the physics alone.

Nothing here reads the command line on its own; the examples decide their
flags and pass values in.
"""
import math
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Appended, not inserted: a caller run from a subdirectory still gets its own
# modules first, but `demo_common` below resolves either way.
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import warp as wp

import threepp as tp

# Re-exported so a film has one import; they live in the Warp-free module
# because the hello-world demos share them too. noqa: F401 - re-export.
from demo_common import encode_rgbe, write_radiance_hdr  # noqa: F401

# --- command line --------------------------------------------------------------


def cli_arg(flag, default, cast):
    """`--flag value` from sys.argv, or `default`. A bare flag also yields default."""
    if flag in sys.argv:
        k = sys.argv.index(flag)
        if k + 1 < len(sys.argv) and not sys.argv[k + 1].startswith("--"):
            return cast(sys.argv[k + 1])
    return default


def parse_size(text):
    """'1280x720' -> (1280, 720)."""
    w, h = text.lower().split("x")
    return int(w), int(h)


def find_ffmpeg():
    """The ffmpeg binary on PATH, or imageio-ffmpeg's bundled one, or None."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                      # noqa: BLE001 - optional dependency
        return None


class Encoder:
    """A pipe to x264, fed raw RGB frames.

    The film demos all learned the same lesson: writing a PNG per frame and
    encoding the directory afterwards costs more than the render does (the hull
    film paid 5.4 GB of intermediates and eighteen minutes for eighty seconds of
    picture). So `read_pixels()` off the frame already on the GPU goes straight
    down this pipe and nothing touches the disk but the mp4.

    The keyword flags exist because the films disagree about them and their
    output must not change: `preset`, `faststart`, `an`, `hide_banner`,
    `loglevel`, `vf` (a filter, e.g. the odd-size fixup) and `log` (a path this
    class opens and closes, or an already-open handle it only writes to).
    """

    def __init__(self, path, w, h, fps, crf=18, preset=None, faststart=False,
                 an=True, hide_banner=True, loglevel="warning", vf=None,
                 log=None, ffmpeg=None):
        exe = ffmpeg or find_ffmpeg()
        if exe is None:
            raise RuntimeError("no ffmpeg on PATH and no imageio-ffmpeg")
        cmd = [exe, "-y"]
        if hide_banner:
            cmd += ["-hide_banner"]
        cmd += ["-loglevel", loglevel, "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if an:
            cmd += ["-an"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf)]
        if preset:
            cmd += ["-preset", preset]
        if faststart:
            cmd += ["-movflags", "+faststart"]
        if vf:
            cmd += ["-vf", vf]
        self.cmd = cmd + [path]
        self._own_log = isinstance(log, str)
        self.log = open(log, "w") if self._own_log else log
        redirect = {"stdout": self.log, "stderr": self.log} if self.log is not None else {}
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, **redirect)
        self.n = 0

    def send(self, rgb):
        """One RGB frame, HxWx3 uint8."""
        self.p.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self):
        """Close the pipe and wait for the encoder; returns its exit code."""
        self.p.stdin.close()
        rc = self.p.wait()
        if self._own_log:
            self.log.close()
        return rc


def encode_png_sequence(pattern, path, fps, crf=18, preset=None, faststart=False,
                        an=False, vf=None, loglevel="error", ffmpeg=None, check=True):
    """Encode an already-written PNG sequence (`pattern` is an ffmpeg %0Nd path).

    The other half of the film encoders: the shots that render to disk first,
    either because the frames are wanted as stills too or because the run is
    assembled from segments afterwards.
    """
    exe = ffmpeg or find_ffmpeg()
    if exe is None:
        raise RuntimeError("no ffmpeg on PATH and no imageio-ffmpeg")
    cmd = [exe, "-y", "-loglevel", loglevel, "-framerate", str(fps), "-i", pattern]
    if an:
        cmd += ["-an"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf)]
    if preset:
        cmd += ["-preset", preset]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    return subprocess.run(cmd + [path], check=check)


def load_font(px):
    """A PIL truetype font at `px` pixels, falling back to PIL's default."""
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:                  # noqa: BLE001 - try the next face
            continue
    return ImageFont.load_default()


# --- meshes (numpy, once) ------------------------------------------------------


def icosphere(subdiv):
    """Unit icosphere: (verts (N, 3) float32, faces (F, 3) int32), wound outward."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]
    for _ in range(subdiv):
        cache, new_faces = {}, []

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = verts[i] + verts[j]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    # Outward winding, so accumulated normals and the pressure gradient point out.
    if signed_volume(verts, faces) < 0.0:
        faces = faces[:, ::-1].copy()
    return verts, faces


def signed_volume(pos, faces):
    """Signed volume enclosed by a closed triangle mesh (positive = outward winding)."""
    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def edge_adjacency(faces):
    """Per undirected edge (i, j) with i < j: the faces that own it and the
    opposite vertex in each. Returns (edge_faces, edge_opp), both dicts of lists
    in the same order."""
    edge_faces, edge_opp = {}, {}
    for fi, (fa, fb, fc) in enumerate(faces):
        for i, j, k in ((fa, fb, fc), (fb, fc, fa), (fc, fa, fb)):
            key = (min(i, j), max(i, j))
            edge_faces.setdefault(key, []).append(fi)
            edge_opp.setdefault(key, []).append(k)
    return edge_faces, edge_opp


def unique_edges(faces):
    """Every undirected mesh edge once, as an (E, 2) int32 array with i < j."""
    edges = set()
    for fa, fb, fc in faces:
        for i, j in ((fa, fb), (fb, fc), (fc, fa)):
            edges.add((min(int(i), int(j)), max(int(i), int(j))))
    return np.array(sorted(edges), dtype=np.int32)


def vertex_adjacency(faces, n_verts=None):
    """1-ring neighbours of every vertex as CSR (offsets (N+1,), indices (2E,)).

    What a uniform-Laplacian term -- smoothing a surface, or preconditioning a
    per-vertex gradient -- needs, and the one table that is the same for every
    fixed-topology mesh demo. Built off `unique_edges`, so each neighbour
    appears exactly once per row.
    """
    edges = unique_edges(faces)
    n = int(faces.max()) + 1 if n_verts is None else int(n_verts)
    counts = np.bincount(edges.ravel(), minlength=n)
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    cursor = offsets[:-1].copy()
    indices = np.zeros(int(offsets[-1]), dtype=np.int32)
    for i, j in edges:
        indices[cursor[i]] = j
        cursor[i] += 1
        indices[cursor[j]] = i
        cursor[j] += 1
    return offsets, indices


def shell_pairs(faces, stiff_edge, stiff_bend):
    """Distance constraints for a closed triangle shell.

    Every mesh edge is a stretch constraint; for every interior edge, the two
    opposite vertices of its triangles form a (weaker) bending constraint.
    Returns a list of (i, j, stiffness, kind, face_a, face_b) with kind 0 for
    an edge and 1 for a bend pair, and face_a/face_b the two owning faces.
    """
    edge_faces, edge_opp = edge_adjacency(faces)
    pairs = []
    for key, owners in edge_faces.items():
        i, j = key
        fa, fb = owners[0], owners[-1]
        pairs.append((i, j, stiff_edge, 0, fa, fb))
        opp = edge_opp[key]
        if len(opp) == 2:
            pairs.append((min(opp), max(opp), stiff_bend, 1, fa, fb))
    return pairs


def csr_from_pairs(n_verts, pairs, rest_pos, extra_rows=0):
    """Symmetric CSR adjacency from (i, j, stiffness, ...) pairs.

    Returns (offsets, indices, rests, stiffs, pair_ids) as numpy arrays; row i
    lists every partner of vertex i with the rest length measured on rest_pos,
    the pair's stiffness and its index into `pairs`. `extra_rows` appends that
    many empty rows (for sentinel vertices).
    """
    neighbors = [[] for _ in range(n_verts)]
    for pid, p in enumerate(pairs):
        i, j, s = p[0], p[1], p[2]
        rest = float(np.linalg.norm(rest_pos[i] - rest_pos[j]))
        neighbors[i].append((j, rest, s, pid))
        neighbors[j].append((i, rest, s, pid))
    offsets = np.zeros(n_verts + 1 + extra_rows, dtype=np.int32)
    idx, rest, stiff, pid = [], [], [], []
    for i, ns in enumerate(neighbors):
        offsets[i + 1] = offsets[i] + len(ns)
        for j, r, s, p in ns:
            idx.append(j)
            rest.append(r)
            stiff.append(s)
            pid.append(p)
    offsets[n_verts + 1:] = offsets[n_verts]
    return (offsets, np.array(idx, dtype=np.int32), np.array(rest, dtype=np.float32),
            np.array(stiff, dtype=np.float32), np.array(pid, dtype=np.int32))


# --- warp kernels: particles, shells ------------------------------------------


@wp.kernel
def integrate(x: wp.array(dtype=wp.vec3),
              prev: wp.array(dtype=wp.vec3),
              pred: wp.array(dtype=wp.vec3),
              dt: float, damping: float, gravity: wp.vec3):
    """Verlet predict: pred = x + damped velocity + gravity. pred may alias x."""
    i = wp.tid()
    p = x[i]
    v = (p - prev[i]) * (1.0 - damping)
    prev[i] = p
    pred[i] = p + v + gravity * dt * dt


@wp.kernel
def volume_grad(pos: wp.array(dtype=wp.vec3),
                tris: wp.array(dtype=int),
                faces_per_body: int,
                vol: wp.array(dtype=float),
                grad: wp.array(dtype=wp.vec3)):
    """Signed volume of each closed body and its gradient at every vertex."""
    f = wp.tid()
    ia, ib, ic = tris[f * 3], tris[f * 3 + 1], tris[f * 3 + 2]
    a, b, c = pos[ia], pos[ib], pos[ic]
    wp.atomic_add(vol, f // faces_per_body, wp.dot(a, wp.cross(b, c)) / 6.0)
    wp.atomic_add(grad, ia, wp.cross(b, c) / 6.0)
    wp.atomic_add(grad, ib, wp.cross(c, a) / 6.0)
    wp.atomic_add(grad, ic, wp.cross(a, b) / 6.0)


@wp.kernel
def grad_sumsq(grad: wp.array(dtype=wp.vec3),
               verts_per_body: int,
               out: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(out, i // verts_per_body, wp.dot(grad[i], grad[i]))


@wp.kernel
def volume_apply(pos: wp.array(dtype=wp.vec3),
                 grad: wp.array(dtype=wp.vec3),
                 vol: wp.array(dtype=float),
                 sumsq: wp.array(dtype=float),
                 v_target: wp.array(dtype=float),
                 verts_per_body: int):
    """One global volume constraint per body, acting as internal pressure."""
    i = wp.tid()
    body = i // verts_per_body
    lam = (v_target[body] - vol[body]) / (sumsq[body] + 1.0e-9)
    pos[i] = pos[i] + grad[i] * lam


@wp.kernel
def accum_normals(pos: wp.array(dtype=wp.vec3),
                  tris: wp.array(dtype=int),
                  nrm: wp.array(dtype=wp.vec3)):
    """Area-weighted face normals summed onto the vertices (zero nrm first)."""
    f = wp.tid()
    ia, ib, ic = tris[f * 3], tris[f * 3 + 1], tris[f * 3 + 2]
    n = wp.cross(pos[ib] - pos[ia], pos[ic] - pos[ia])
    wp.atomic_add(nrm, ia, n)
    wp.atomic_add(nrm, ib, n)
    wp.atomic_add(nrm, ic, n)


@wp.kernel
def normalize_vec3(v: wp.array(dtype=wp.vec3)):
    """Unit-length in place; the second half of the accum_normals pattern.

    `accum_normals` leaves AREA-WEIGHTED sums on the vertices, which is what a
    triangle-soup expander normalises on the fly. An INDEXED mesh published
    straight into a vertex buffer has no such pass, and a lit surface with
    unnormalised normals reads as a blown-out or black shell depending on the
    triangle sizes -- so it gets one launch of this instead.
    """
    i = wp.tid()
    n = v[i]
    v[i] = n / wp.max(wp.length(n), 1.0e-9)


@wp.kernel
def smooth_vec3_csr(src: wp.array(dtype=wp.vec3),
                    offsets: wp.array(dtype=wp.int32),
                    indices: wp.array(dtype=wp.int32),
                    alpha: float,
                    dst: wp.array(dtype=wp.vec3)):
    """One Jacobi smoothing sweep over a CSR 1-ring: dst = (1-a) src + a mean(nbrs).

    Used as a GRADIENT preconditioner in the hull optimiser (a cheap stand-in
    for solving (I + lambda L) g' = g), and as a plain surface smoother
    anywhere else. `vertex_adjacency` builds the two index arrays.
    """
    i = wp.tid()
    s = offsets[i]
    e = offsets[i + 1]
    acc = wp.vec3(0.0, 0.0, 0.0)
    for k in range(s, e):
        acc += src[indices[k]]
    n = float(wp.max(e - s, 1))
    dst[i] = src[i] * (1.0 - alpha) + (acc / n) * alpha


@wp.kernel
def scatter_soup(pos: wp.array(dtype=wp.vec3),
                 nrm: wp.array(dtype=wp.vec3),
                 tris: wp.array(dtype=int),
                 out_pos: wp.array(dtype=wp.vec3),
                 out_nrm: wp.array(dtype=wp.vec3)):
    """De-index into a triangle soup: per-corner positions and smooth normals."""
    k = wp.tid()
    i = tris[k]
    out_pos[k] = pos[i]
    n = nrm[i]
    out_nrm[k] = n / wp.max(wp.length(n), 1.0e-9)


# --- warp: particles -> density grid -> marching cubes ------------------------


@wp.func
def _sample(field: wp.array3d(dtype=float), gx: float, gy: float, gz: float,
            nx: int, ny: int, nz: int) -> float:
    i = int(wp.floor(gx))
    j = int(wp.floor(gy))
    k = int(wp.floor(gz))
    if i < 0 or j < 0 or k < 0 or i > nx - 2 or j > ny - 2 or k > nz - 2:
        return 0.0
    fx = gx - float(i)
    fy = gy - float(j)
    fz = gz - float(k)
    c00 = field[i, j, k] * (1.0 - fx) + field[i + 1, j, k] * fx
    c10 = field[i, j + 1, k] * (1.0 - fx) + field[i + 1, j + 1, k] * fx
    c01 = field[i, j, k + 1] * (1.0 - fx) + field[i + 1, j, k + 1] * fx
    c11 = field[i, j + 1, k + 1] * (1.0 - fx) + field[i + 1, j + 1, k + 1] * fx
    a = c00 * (1.0 - fy) + c10 * fy
    b = c01 * (1.0 - fy) + c11 * fy
    return a * (1.0 - fz) + b * fz


@wp.kernel
def _splat(x: wp.array(dtype=wp.vec3), field: wp.array3d(dtype=float),
           origin: wp.vec3, inv_cell: float, nx: int, ny: int, nz: int):
    # Trilinear deposit: one unit of density spread over the 8 cell corners.
    t = wp.tid()
    p = x[t]
    gx = (p[0] - origin[0]) * inv_cell
    gy = (p[1] - origin[1]) * inv_cell
    gz = (p[2] - origin[2]) * inv_cell
    i = int(wp.floor(gx))
    j = int(wp.floor(gy))
    k = int(wp.floor(gz))
    if i < 0 or j < 0 or k < 0 or i > nx - 2 or j > ny - 2 or k > nz - 2:
        return
    fx = gx - float(i)
    fy = gy - float(j)
    fz = gz - float(k)
    for a in range(2):
        wa = wp.where(a == 0, 1.0 - fx, fx)
        for b in range(2):
            wb = wp.where(b == 0, 1.0 - fy, fy)
            for c in range(2):
                wc = wp.where(c == 0, 1.0 - fz, fz)
                wp.atomic_add(field, i + a, j + b, k + c, wa * wb * wc)


@wp.kernel
def _blur_axis(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float),
               axis: int, w: float, nx: int, ny: int, nz: int):
    # 3-tap [w, 1-2w, w] along one axis, edge-clamped. w = 0.25 is the binomial
    # kernel and removes grid-aligned noise exactly; smaller w keeps thin
    # sheets alive.
    i, j, k = wp.tid()
    di = wp.where(axis == 0, 1, 0)
    dj = wp.where(axis == 1, 1, 0)
    dk = wp.where(axis == 2, 1, 0)
    im = wp.max(i - di, 0)
    jm = wp.max(j - dj, 0)
    km = wp.max(k - dk, 0)
    ip = wp.min(i + di, nx - 1)
    jp = wp.min(j + dj, ny - 1)
    kp = wp.min(k + dk, nz - 1)
    dst[i, j, k] = w * src[im, jm, km] + (1.0 - 2.0 * w) * src[i, j, k] + w * src[ip, jp, kp]


@wp.kernel
def _expand(verts: wp.array(dtype=wp.vec3),
            indices: wp.array(dtype=wp.int32),
            field: wp.array3d(dtype=float),
            ntris: int,
            origin: wp.vec3, inv_cell: float, nx: int, ny: int, nz: int,
            sign: float, grain: float, grain_freq: float,
            out_pos: wp.array(dtype=wp.vec3),
            out_nrm: wp.array(dtype=wp.vec3)):
    # De-index into a triangle soup with a smooth normal per corner from the
    # density gradient (a lit mesh with no normals renders black). Triangles
    # past ntris collapse to an off-screen point.
    t = wp.tid()
    if t >= ntris:
        for c in range(3):
            out_pos[t * 3 + c] = wp.vec3(0.0, -50.0, 0.0)
            out_nrm[t * 3 + c] = wp.vec3(0.0, 1.0, 0.0)
        return
    state = wp.uint32(1337)
    for c in range(3):
        p = verts[indices[t * 3 + c]]
        gx = (p[0] - origin[0]) * inv_cell
        gy = (p[1] - origin[1]) * inv_cell
        gz = (p[2] - origin[2]) * inv_cell
        g = wp.vec3(_sample(field, gx + 1.0, gy, gz, nx, ny, nz)
                    - _sample(field, gx - 1.0, gy, gz, nx, ny, nz),
                    _sample(field, gx, gy + 1.0, gz, nx, ny, nz)
                    - _sample(field, gx, gy - 1.0, gz, nx, ny, nz),
                    _sample(field, gx, gy, gz + 1.0, nx, ny, nz)
                    - _sample(field, gx, gy, gz - 1.0, nx, ny, nz))
        l = wp.length(g)
        # Density rises inward, so the outward surface normal is -grad.
        n = wp.where(l > 1.0e-9, g * (-1.0 / l), wp.vec3(0.0, 1.0, 0.0))
        if grain > 0.0:
            # Micro-relief the blurred density field cannot carry: two octaves
            # of world-space Perlin bent into the normal. Position-keyed, so it
            # is stable frame to frame and rides the surface as it flows.
            q = p * grain_freq
            q2 = p * (grain_freq * 2.6)
            nse = wp.vec3(wp.noise(state, q)
                          + 0.5 * wp.noise(state, q2),
                          wp.noise(state, q + wp.vec3(19.1, 47.7, 11.3))
                          + 0.5 * wp.noise(state, q2 + wp.vec3(19.1, 47.7, 11.3)),
                          wp.noise(state, q + wp.vec3(-7.3, 3.9, 29.2))
                          + 0.5 * wp.noise(state, q2 + wp.vec3(-7.3, 3.9, 29.2)))
            n = wp.normalize(n + grain * nse)
        out_pos[t * 3 + c] = p
        out_nrm[t * 3 + c] = n * sign


class DensitySurface:
    """Particles -> blurred density grid -> marching cubes -> triangle soup.

    `origin` is the grid's lower corner, `cell` its spacing, `dims` its node
    counts. build() returns the unclamped triangle count; expand() writes
    `ntris` de-indexed triangles with gradient normals into two (3*ntris,)
    vec3 arrays -- host staging buffers or the renderer's own mapped ones.
    """

    def __init__(self, origin, cell, dims, device, blur=(0.25, 0.125)):
        self.origin = wp.vec3(*origin)
        self.inv_cell = 1.0 / cell
        self.dims = tuple(int(d) for d in dims)
        self.blur = tuple(blur)
        self.device = device
        self.field = wp.zeros(self.dims, dtype=float, device=device)
        self._scratch = wp.zeros(self.dims, dtype=float, device=device)
        nx, ny, nz = self.dims
        self.mc = wp.MarchingCubes(
            nx, ny, nz,
            domain_bounds_lower_corner=self.origin,
            domain_bounds_upper_corner=wp.vec3(origin[0] + (nx - 1) * cell,
                                               origin[1] + (ny - 1) * cell,
                                               origin[2] + (nz - 1) * cell))

    @property
    def verts(self):
        return self.mc.verts

    @property
    def indices(self):
        return self.mc.indices

    def build(self, x, n, iso):
        """Splat `n` particles, blur, surface at `iso`; returns the triangle count."""
        nx, ny, nz = self.dims
        self.field.zero_()
        wp.launch(_splat, dim=n, device=self.device,
                  inputs=[x, self.field, self.origin, self.inv_cell, nx, ny, nz])
        # Ping-pong between the two grids; one pass per axis per blur weight.
        # An even number of passes lands the result back in self.field.
        a, b = self.field, self._scratch
        for w in self.blur:
            for axis in range(3):
                wp.launch(_blur_axis, dim=self.dims, device=self.device,
                          inputs=[a, b, axis, w, nx, ny, nz])
                a, b = b, a
        if a is not self.field:
            wp.copy(self.field, a)
        self.mc.surface(self.field, iso)
        return self.mc.indices.shape[0] // 3

    def expand(self, ntris, out_pos, out_nrm, dim=None, sign=1.0,
               grain=0.0, grain_freq=30.0):
        """De-index `ntris` triangles into out_pos/out_nrm. `dim` overrides the
        launch size to also collapse the rows past ntris.

        `sign` multiplies the gradient normal. The default ships outward
        (-grad) normals -- but wp.MarchingCubes winds its triangles the OTHER
        way, so a Side.Double material's back-face flip (normal *= faceDirection,
        both GL and Vulkan) turns them inward on every fragment seen from
        outside, which lights the surface as pure black. A double-sided lit
        consumer wants sign=-1.0: winding-aligned normals that the rasterizer's
        flip lands outward.

        `grain` > 0 bends two octaves of world-space Perlin noise into the
        normal (amplitude `grain`, base feature size ~1/`grain_freq` metres).
        The blurred density field yields a surface smoother than any granular
        material really is -- a uniform specular over it reads as moulded
        plastic; position-keyed micro-relief breaks the highlight up without
        touching the geometry.
        """
        nx, ny, nz = self.dims
        wp.launch(_expand, dim=ntris if dim is None else dim, device=self.device,
                  inputs=[self.mc.verts, self.mc.indices, self.field, ntris,
                          self.origin, self.inv_cell, nx, ny, nz, float(sign),
                          float(grain), float(grain_freq),
                          out_pos, out_nrm])


def pbf_constants(d, h, mass, s_corr_dq, s_corr_n):
    """Position Based Fluids constants measured from the rest lattice.

    The SPH kernels carry a 1/h^9 scale, so published constants do not
    transfer between resolutions; everything is derived from a cubic lattice
    at spacing `d` instead. Returns a dict with poly6, spiky, rho0, w_dq,
    sum_grad2 (the rest gradient energy), eps_cfm and s_corr_k.
    """
    poly6 = 315.0 / (64.0 * math.pi * h ** 9)
    spiky = -45.0 / (math.pi * h ** 6)
    reach = int(math.ceil(h / d)) + 1
    off = np.arange(-reach, reach + 1) * d
    gx, gy, gz = np.meshgrid(off, off, off, indexing="ij")
    r2 = (gx ** 2 + gy ** 2 + gz ** 2).ravel()
    r2 = r2[r2 < h * h]
    rho0 = float(mass * (poly6 * np.maximum(h * h - r2, 0.0) ** 3).sum())
    w_dq = float(poly6 * (h * h - s_corr_dq ** 2) ** 3)
    r = np.sqrt(r2[r2 > 1e-12])
    g = (mass / rho0) * abs(spiky) * (h - r) ** 2
    sum_grad2 = float((g ** 2).sum())
    # Soften lambda by a fraction of the rest gradient energy, and size the
    # artificial pressure against lambda at a typical compression.
    eps_cfm = 0.05 * sum_grad2
    ratio_typ = (poly6 * (h * h - d * d) ** 3) / w_dq
    s_corr_k = 0.30 * (0.10 / sum_grad2) / ratio_typ ** s_corr_n
    return dict(poly6=poly6, spiky=spiky, rho0=rho0, w_dq=w_dq, sum_grad2=sum_grad2,
                eps_cfm=eps_cfm, s_corr_k=s_corr_k)


# --- threepp scene -------------------------------------------------------------


def standard_material(color, roughness=1.0, metalness=0.0, **props):
    """MeshStandardMaterial with the common knobs set (the defaults are the
    material's own); extra keywords are assigned as attributes (side=,
    emissive=, ...)."""
    m = tp.MeshStandardMaterial()
    m.color = color
    m.roughness = roughness
    m.metalness = metalness
    for k, v in props.items():
        setattr(m, k, v)
    return m


def studio_lights(scene, sun_pos=(4.0, 8.0, 3.0)):
    """A hemisphere fill plus one shadow-casting sun; returns the sun."""
    scene.add(tp.HemisphereLight(0xffffff, 0x33383f, 0.9))
    sun = tp.DirectionalLight(0xffffff, 2.5)
    sun.position.set(*sun_pos)
    sun.cast_shadow = True
    scene.add(sun)
    return sun


def ground_plane(scene, size=30.0, y=0.0, color=0x4a4f55):
    """A shadow-receiving floor at height y."""
    ground = tp.Mesh(tp.PlaneGeometry(size, size), standard_material(color))
    ground.rotate_x(-math.pi / 2)
    ground.position.y = y
    ground.receive_shadow = True
    scene.add(ground)
    return ground


def resize_handler(camera, renderer):
    """The window-resize callback every example installs."""
    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)
    return on_resize


# --- run loops -----------------------------------------------------------------


def orbit_loop(canvas, renderer, scene, camera, step, target=None):
    """Interactive window: step(), orbit controls, render, forever."""
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    if target is not None:
        controls.target.set(*target)
    canvas.on_window_resize(resize_handler(camera, renderer))

    def animate():
        step()
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)


def shot_loop(renderer, scene, camera, step, seconds, out, fps=60):
    """Headless: simulate `seconds` of frames, render once, write `out`."""
    frames = int(round(seconds * fps))
    for _ in range(frames):
        step()
    renderer.render(scene, camera)
    renderer.save_frame(out)
    print(f"simulated {seconds:.1f} s ({frames} frames), wrote {out}")


def bench_loop(step, render, phases, warmup, timed, label):
    """Timed phase breakdown. step() returns one duration in seconds per entry
    of `phases`; render() is timed here as the final phase."""
    for _ in range(warmup):
        step()
        render()
    acc = [0.0] * (len(phases) + 1)
    for _ in range(timed):
        parts = list(step())
        t0 = time.perf_counter()
        render()
        parts.append(time.perf_counter() - t0)
        for i, p in enumerate(parts):
            acc[i] += p
    ms = 1000.0 / timed
    total = sum(acc) * ms
    cols = " | ".join(f"{name} {a * ms:.2f}" for name, a in zip(tuple(phases) + ("render",), acc))
    print(f"bench {label}: {cols} = {total:.2f} ms/frame ({1000.0 / total:.0f} fps)")


def sky_env(sun_dir, w=512, h=256,
            below_horizon=(0.60, 0.71, 0.90), below_nadir=(0.15, 0.19, 0.25)):
    """A procedural float equirect: background AND image-based light, no assets.

    Same trick vulkan_ocean.py uses to avoid shipping an .hdr, minus the RGBE
    round trip -- `float_texture` takes the linear array directly. The lower
    hemisphere fades to a haze rather than to black, or an ocean's grazing
    reflections drop out at the horizon. `sun_dir` is the unit vector TOWARD
    the sun; put the scene's key DirectionalLight on the same line so the disc
    in the reflections, the glint and the shadows agree. Third caller (the
    hull sculpt, now the prop vortex) is what moved it here from the sculpt.

    `below_horizon` / `below_nadir` are the lower hemisphere. The default is
    the haze an above-water ocean wants; a scene whose world under the horizon
    is WATER (the prop vortex: the bronze sits in it and reflects it) passes a
    sea instead, because the renderer's image-based light on a submerged metal
    is this map unattenuated, and a bright haze under the horizon turns bronze
    to white.
    """
    sun_dir = np.asarray(sun_dir, np.float32)
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    elev = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * math.pi
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * math.pi
    d = np.empty((h, w, 3), np.float32)
    d[..., 0] = np.cos(elev)[:, None] * np.cos(az)[None, :]
    d[..., 1] = np.sin(elev)[:, None]
    d[..., 2] = np.cos(elev)[:, None] * np.sin(az)[None, :]
    y = d[..., 1]
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.40
    down = np.clip(-y, 0.0, 1.0)[..., None] ** 0.60
    col = np.where(y[..., None] >= 0.0,
                   np.float32([0.60, 0.71, 0.90]) * (1.0 - up)
                   + np.float32([0.09, 0.23, 0.56]) * up,
                   np.float32(below_horizon) * (1.0 - down)
                   + np.float32(below_nadir) * down).astype(np.float32)
    col += (np.exp(-(y * y) / (2.0 * 0.0040))[..., None]
            * np.float32([0.36, 0.28, 0.20]))
    ang = np.arccos(np.clip(d @ sun_dir, -1.0, 1.0))
    col += ((np.exp(-(ang / math.radians(1.7)) ** 2) * 42.0
             + np.exp(-(ang / math.radians(12.0)) ** 2) * 2.6)[..., None]
            * np.float32([1.0, 0.96, 0.88]))
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = col
    return tp.float_texture(out)
