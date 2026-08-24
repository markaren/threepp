"""How many deformable fish can one GPU convey? A Newton VBD benchmark.

Answers the only question worth asking before building a fish-handling
simulator: at what count does a tetrahedral FEM fish stop running in real time.
Everything else here exists to make that number honest.

The fish is whatever mesh you point --fish at, tetrahedralised into a coarse
cage. The conveyor is a bed of rollers -- kinematic cylinders spun about their
own axis, which is the one conveying mechanism that needs no tricks: a cylinder
rotating about its long axis occupies the same volume forever, so contact reads
the true surface velocity omega*r and nothing ever has to be teleported back.
The solver is Newton's SolverVBD (Vertex Block Descent, stable Neo-Hookean
tets), which is the solver a from-scratch effort would converge on anyway.

--fish IS REQUIRED and has no default. It takes any model threepp's ModelLoader
reads -- .usdz, .usd/.usda/.usdc, .obj, .stl, .gltf/.glb, .dae -- and nothing
about this benchmark is fish-specific beyond the auto-orientation, so any
closed solid works. Point it at a DIRECTORY (or a comma-separated list) and it
loads several scans and deals them out round-robin, which is what a real catch
looks like; --variants caps how many distinct scans it will build cages for.
The scans this was developed against are not redistributable, which is why
there is no bundled asset to fall back on.

    pip install newton
    python newton_fish_conveyor.py --fish scans/ --count 60 --view    # the demo
    python newton_fish_conveyor.py --fish cod.usdz                    # sanity check
    python newton_fish_conveyor.py --fish cod.usdz --sweep            # 1,4,16,64 table
    python newton_fish_conveyor.py --fish cod.usdz --sweep --sweep-to 256 \
        --max-particles 800000
    python newton_fish_conveyor.py --fish scans/ --count 60 --shot catch.png
    python newton_fish_conveyor.py --fish scans/ --count 60 --shot catch.png --shot-time 12
    python newton_fish_conveyor.py --fish scans/ --count 100 --layers 1  # spread, not piled
    python newton_fish_conveyor.py --fish cod.usdz --count 8 --view --cage # what the solver sees
    python newton_fish_conveyor.py --fish cod.usdz --count 32 --tet-res 20 # finer cage
    python newton_fish_conveyor.py --fish cod.usdz --young 3e4 --poisson 0.45
    python newton_fish_conveyor.py --fish cod.usdz --fish-length 0.6 # rescale to 60 cm

Every flag that mattered is now a DEFAULT and has a --no- form: CUDA graph
capture, fish-on-fish contact, and zero-copy skinning are all on.

Reported per count: solve milliseconds per frame, the realtime factor at 60 Hz,
tet/particle counts, VRAM, whether the fish actually conveyed (mean speed along
the belt vs. the belt's own speed), and whether anything went unstable. The
conveying number is a physics check, not a perf number -- if it reads ~0 the
timings are still valid but the contact model is not transmitting drive.

TETRAHEDRALISATION is a voxel lattice carved by the mesh's own signed distance
field, split 5-tets-per-cell with alternating parity so the result is
conforming. No external mesher, no dependency, and the resolution is a single
knob -- which is what you want when the point is to sweep cost against fidelity.
It is carved at --carve cells OUTSIDE the surface, not at the zero level set: a
cell only survives if its CENTRE is inside, so carving at zero leaves the cage
up to half a cell short on every face. At 14 cells that measured a 1.110 m cage
around a 1.412 m cod -- the snout and the whole tail outside the collider, and
outside every tet the skin binds to. 0.25 cells costs 1.7x the tets and gets the
cage's bounding box to match the fish's exactly.

WHAT YOU SEE is not what is simulated. The solver moves a few hundred cage
vertices; the scan's full-resolution surface -- 15k vertices, its own UVs, its
own baked albedo/normal/AO -- rides along, each vertex bound to the tet that
holds it by barycentric weights and skinned on the GPU. That split is the whole
point: a coarse cage is what makes 64 fish affordable, and the scan is what
makes them look like fish. --cage renders the cage itself instead, so you can
see exactly how coarse the thing you are timing really is. The drawn scan is
decimated to --render-ratio of its triangles first (meshoptimizer, through
threepp's simplify_geometry): 25k triangles to 10k moves the bounding box by
less than a micron, because at this distance the detail lives in the baked
normal map rather than in the tessellation.

Binding is by UNCLAMPED barycentric weights, so a vertex reproduces its rest
position exactly even where it sits outside the cage. Extrapolation multiplies,
though -- crush a tet in the middle of a heap and a fin bound at |w| = 3.9
leaves the fish as a metre-long needle -- so anything past |w| = 2.2 is slid
toward the clamped weights, and only as far as that bound requires.

--view and --shot are for looking, not for timing. Fish that reach the end of
the bed ride off it, drop to the floor a metre below, and lie there -- the same
end a real infeed has, and the same behaviour as the PhysX example. --recycle
loops them back to the start instead, for watching a steady stream longer than
one belt-length; run() never recycles either way, since it measures conveying
as displacement over a window and a teleport inside that window reads as the
belt running backwards. And the skinning writes straight into the renderer's own
vertex buffers (GLRenderer.gl_buffer_id + wp.RegisteredGLBuffer), so nothing
crosses host memory: 48 fish measured 30.2 ms/frame that way against 40.5 ms
through geometry.update_attribute. --no-interop takes the host route.

The fish is auto-oriented: longest extent along the belt, thinnest extent
vertical, i.e. lying on its side the way a fish lies on a real conveyor. Yaw,
size and position are jittered per fish (seeded, so runs repeat) -- a rigid
grid of identical clones is neither realistic nor a fair contact load.

CUDA GRAPH CAPTURE IS ON. --no-graph exists to show you why it is: without it
this is launch-overhead bound rather than solve bound, and ONE fish measured
62.3 ms/frame against 4.6 ms captured -- 16 fps for a single cod, and a belt so
slow it looks like the fish is not being conveyed at all. Never turn it off
except to reproduce that. (Graph capture needs an EVEN --substeps; the state
ping-pong would otherwise leave every frame reading a stale buffer.)

WHERE THE TIME GOES, measured at 100 fish on an RTX 4070, end to end -- solve,
skinning and render -- with the defaults above:

    40 fish  15.5 ms   65 fps        100 fish  35.0 ms   29 fps
    60 fish  19.6 ms   51 fps        150 fish  45.7 ms   22 fps

60 Hz lands at roughly FIFTY fish, not a hundred. Everything that buys more than
that buys it by breaking the fish, which is worth knowing precisely:

  --substeps 2   twice the speed, and the single worst thing you can do. Worst
                 tet edge stretch over 8 s with 60 fish piled: 1.55x at 4
                 substeps, 16.0x at 2. Iterations barely matter next to this --
                 8 iterations at 2 substeps still measured 4.85x. Fish stop
                 being fish: they flatten into sheets, fuse into their
                 neighbours, and occasionally come apart altogether.
  --no-self-contact   four times the speed, and a pile stops piling: it
                 interpenetrates into a single flat layer.
  --tet-res 12   another 25%, and a LONE fish stops being conveyed -- 117
                 particles over a 102 mm roller pitch sit in the valleys and
                 stall for two seconds before friction finds them. 14 is the
                 coarsest cage that starts moving immediately at every count.
  --layers 1     spreads the catch instead of piling it, which is cheaper AND
                 gentler, but wants a 27 m belt for 100 fish.

What DID come free, from VBD's stock contact settings being sized for cloth
rather than for a hundred blunt tet cages: detect once per substep instead of
twice; stop preallocating 32 vertex and 64 edge contacts per particle when a
fish in a heap sees a handful; and give fish-on-fish its own contact radius,
smaller than the one that makes a fish sit correctly on a 50 mm roller, since
the detection margin follows the radius and candidate pairs follow the margin
cubed. Those three took 100 fish from 55.5 ms to 17.7 ms with no change in
behaviour. A LARGER self-contact radius is much worse, not better: 0.50 cells
put 179 tets through inversion where 0.25 put none.

STIFFNESS IS NOT THE NUMBER YOU EXPECT. --young defaults to 2e5 Pa, not the 5e4
usually quoted for fish flesh, because a homogeneous Neo-Hookean tet cage has
neither skin nor backbone and the bulk modulus has to stand in for both. At 5e4
the cod at the bottom of a five-deep pile render as empty bags.

Needs a CUDA device. --device cpu runs but is not a meaningful measurement.
The sweep stops at --max-particles rather than running until it OOMs, and the
SDF bake is launched in slabs so no single dispatch is long enough to trip a
display-driver timeout. Sustained GPU load on a thin laptop is its own risk --
keep an eye on the machine rather than walking away from a long sweep.

This file is the Newton half of a two-engine comparison. Everything that is not
the solver -- the scans, the cage, the bind, the belt, the lighting, the skinning
and the quality metrics -- lives in fish_conveyor_common.py, and
physx_fish_conveyor.py runs the same benchmark on PhysX 5 deformable volumes.
fish_conveyor_ab.py drives both and reports the table.

--tets physx swaps the voxel carve for PhysX's own conforming tetrahedralisation
(cooked in a subprocess), so the two solvers can be run on a mesh that is
identical rather than merely equivalent.
"""
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import warp as wp

try:
    import newton
except ImportError:
    sys.exit("newton is not installed -- pip install newton (needs a CUDA GPU, driver 545+)")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fish_conveyor_common as fc
from fish_conveyor_common import (Fish, ab_run, cli_arg, deal_variants, fish_paths,
                                  plan_layout, stage)
# After fish_conveyor_common, which is what puts python/ on sys.path.
import threepp as tp  # noqa: E402


FISH = cli_arg("--fish", "", str)            # required -- see the module docstring
VARIANTS = cli_arg("--variants", 4, int)     # distinct scans to build cages for
COUNT = cli_arg("--count", 1, int)
SWEEP = "--sweep" in sys.argv
SWEEP_TO = cli_arg("--sweep-to", 64, int)    # raise deliberately; see MAX_PARTICLES
# Hard ceiling on model size. This is a laptop with 8 GiB of VRAM whose sustained
# CPU+GPU draw has taken the whole machine down once; a sweep that runs until it
# OOMs is not a measurement, it is a stress test. Raise it on purpose or not at all.
MAX_PARTICLES = cli_arg("--max-particles", 400_000, int)
SDF_SLAB = cli_arg("--sdf-slab", 8, int)     # grid rows per SDF dispatch
TET_RES = cli_arg("--tet-res", 14, int)      # cells along the fish's longest axis
TETS = cli_arg("--tets", "voxel", str)       # voxel | physx -- whose tetrahedralisation
PHYSX_VOXEL_RES = cli_arg("--physx-voxel-res", 10, int)   # only with --tets physx
SOLVER = cli_arg("--solver", "vbd", str)     # vbd | xpbd
ITERATIONS = cli_arg("--iterations", 4, int) # VBD iterations per substep
# The rematch knobs. --detect-interval 0 is Newton's own default (detect twice
# per substep); the shipped default detects once. --contact-buf V,E of 32,64 is
# Newton's own preallocation; the shipped 8,8 was a speed cut whose quality cost
# was never isolated. Both exist so the A/B can run Newton at NEWTON's contact
# settings rather than at this file's.
DETECT_INTERVAL = cli_arg("--detect-interval", -1, int)  # -1 = once per substep (shipped)
RELAXATION = cli_arg("--relaxation", 0.9, float)  # xpbd only: soft-body Jacobi relaxation
EDGE_KE = cli_arg("--edge-ke", 0.0, float)   # xpbd only: surface bending-edge stiffness
SPINE_KE = cli_arg("--spine-ke", 0.0, float) # xpbd only: long-range body springs, N/m
CONTACT_BUF = cli_arg("--contact-buf", "8,8", str)
SUBSTEPS = cli_arg("--substeps", 4, int)    # the quality knob; see the docstring
FRAMES = cli_arg("--frames", 120, int)       # timed frames
WARMUP = cli_arg("--warmup", 90, int)        # settle + kernel compile, untimed
GRAPH = "--no-graph" not in sys.argv   # see the module docstring: never turn this off
VIEW = "--view" in sys.argv
SHOT = cli_arg("--shot", "", str)            # headless: write this PNG and exit
SHOT_TIME = cli_arg("--shot-time", 2.5, float)  # seconds of sim before the shot
CAGE = "--cage" in sys.argv                  # draw the tet cage, not the scan
RENDER_RATIO = cli_arg("--render-ratio", 0.4, float)  # decimate the drawn scan
CARVE = cli_arg("--carve", 0.25, float)      # cells of dilation on the tet cage
PARTICLE_R = cli_arg("--particle-radius", 0.40, float)  # cage cells, vs the machine
CONTACT_MARGIN = cli_arg("--contact-margin", 0.50, float)  # cage cells, vs the machine
# Fish-on-fish is a separate, SMALLER pair. The radius that makes a fish rest
# correctly on a 50 mm roller is not the radius you want between two fish: the
# detection margin scales with it, the candidate pairs scale with the margin
# cubed, and fish-on-fish is where every millisecond over 100 fish goes.
# 0.25/0.40 against 0.40/0.55 measured 17.2 ms against 22.8 at 100 fish, with
# the same pile and one more fish still aboard. Newton wants margin >= 1.5x
# radius and says so at construction time; 1.6x here.
SELF_R = cli_arg("--self-radius", 0.25, float)
SELF_MARGIN = cli_arg("--self-margin", 0.40, float)
INTEROP = "--no-interop" not in sys.argv     # zero-copy skinning into the GL buffers
RECYCLE = "--recycle" in sys.argv            # loop fish back to the start (off: they
                                             # ride off the end, drop, and lie there)
LANES = cli_arg("--lanes", 5, int)           # fish abreast; sets the belt width
BED = cli_arg("--bed", 30.0, float)          # m; hard cap on the belt length
LAYERS = cli_arg("--layers", 2, int)         # how deep the catch may be piled
SELF_CONTACT = "--no-self-contact" not in sys.argv   # fish-on-fish; see the docstring
BELT_SPEED = cli_arg("--speed", 0.5, float)  # m/s along +X
ROLLER_R = cli_arg("--roller-radius", 0.05, float)
MU = cli_arg("--mu", 0.45, float)            # plastic module belt, not steel
YOUNG = cli_arg("--young", 3.0e6, float)     # Pa. Two orders above the 5e4 quoted for fish
                                             # flesh, and it has to be: a homogeneous
                                             # Neo-Hookean cage has neither skin nor
                                             # backbone, so its bulk modulus stands in for
                                             # both, and a real cod is far stiffer in
                                             # bending than its muscle is in compression.
                                             # Measured over 8 s with 60 fish piled, worst
                                             # tet edge stretch: 3.17x at 2e5, 1.53x at
                                             # 3e6, and it costs nothing -- 3e6 measured
                                             # marginally FASTER, because fish that hold
                                             # their shape generate fewer contacts.
POISSON = cli_arg("--poisson", 0.45, float)  # nearly incompressible
DENSITY = cli_arg("--density", 1050.0, float)
FISH_LENGTH = cli_arg("--fish-length", 0.0, float)  # 0 = keep the scan's own size
SCALE_JITTER = cli_arg("--scale-jitter", 0.07, float)  # per-fish size jitter; 0 for the A/B
SEED = cli_arg("--seed", 7, int)             # layout rng seed
AB = "--ab" in sys.argv                      # one measured config, one JSON line
AB_SECONDS = cli_arg("--ab-seconds", 8.0, float)
DEVICE = cli_arg("--device", "cuda:0", str)

# --solver xpbd selects a REGIME, not just a solver, so it brings its own
# defaults for every knob the caller did not set. The VBD defaults are lethal
# to it, not merely suboptimal: E=3e6 diverges XPBD at any substep count tried,
# and a 0.40-cell particle radius explodes a fish from the inside (its
# sphere-sphere contact has no topological filter, so intra-fish neighbours
# start in violation). 16 substeps x 1 iteration is XPBD canon -- small steps
# are its convergence mechanism, iterations are not.
if SOLVER == "xpbd":
    if "--substeps" not in sys.argv:
        SUBSTEPS = 16
    if "--iterations" not in sys.argv:
        ITERATIONS = 2
    if "--particle-radius" not in sys.argv:
        PARTICLE_R = 0.25
    if "--young" not in sys.argv:
        YOUNG = 1.0e6
    if "--relaxation" not in sys.argv:
        RELAXATION = 0.5
    # The COARSE cage is XPBD's stiffness mechanism, not a compromise: its
    # corrections travel one tet-ring per iteration, so body stiffness is set
    # by how many rings a fish is long, not by Young's modulus (res 14 at
    # E=3e6 droops MORE than res 8 at 1e6). The fat contact spheres that come
    # with the coarse cage are also what finally grips the rollers: conveying
    # 0.87x belt at res 8 against 0.71x at res 14.
    if "--tet-res" not in sys.argv:
        TET_RES = 8
FPS = 60.0

fc.configure(FPS=FPS, TET_RES=TET_RES, CARVE=CARVE, SDF_SLAB=SDF_SLAB,
             RENDER_RATIO=RENDER_RATIO, CAGE=CAGE, FISH_LENGTH=FISH_LENGTH,
             LANES=LANES, BED=BED, LAYERS=LAYERS, BELT_SPEED=BELT_SPEED,
             ROLLER_R=ROLLER_R, MU=MU, WARMUP=WARMUP, FRAMES=FRAMES,
             SCALE_JITTER=SCALE_JITTER, SEED=SEED, INTEROP=INTEROP,
             # The belt gets a runway long enough for the window it is measured
             # over, and --ab measures a different window than the sweep does.
             # At 0.5 m/s an 8 s window plus warmup is 4.75 m of travel, and a
             # bed built for 1.75 m conveys a third of the catch off the end
             # mid-measurement -- which reads as "10 of 16 on belt" and a
             # conveying speed averaged over fish that are on the floor.
             MEASURED_FRAMES=int(round(AB_SECONDS * FPS)) if AB else FRAMES)


# -- The conveyor + the sim --------------------------------------------------

@wp.kernel
def drive_rollers(t: wp.array(dtype=wp.float32), omega: float,
                  joint_q: wp.array(dtype=wp.float32), joint_qd: wp.array(dtype=wp.float32)):
    i = wp.tid()
    joint_q[i] = omega * t[0]
    joint_qd[i] = omega


@wp.kernel
def advance_time(dt: float, t: wp.array(dtype=wp.float32)):
    t[0] = t[0] + dt


@wp.kernel
def fish_centroid_x(q: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int),
                    inv_count: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    f = owner[i]
    wp.atomic_add(out, f, q[i][0] * inv_count[f])


@wp.kernel
def shift_fish(q: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int),
               off: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    q[i] = q[i] + off[owner[i]]



_SPINE = {}


def spine_pairs(fish):
    """Skewers for one fish: vertex pairs far apart along the body.

    The spine of each x-slab (the vertex nearest the slab's centroid) chained to
    its neighbours 3 slabs on, plus nose-to-mid and mid-to-tail ties. A few
    dozen springs; O(1) propagation across the whole body.
    """
    key = id(fish)
    if key not in _SPINE:
        v = fish.cverts
        n_slab = 8
        xs = np.linspace(v[:, 0].min(), v[:, 0].max(), n_slab + 1)
        spine = []
        for i in range(n_slab):
            in_slab = np.flatnonzero((v[:, 0] >= xs[i]) & (v[:, 0] <= xs[i + 1]))
            if not len(in_slab):
                continue
            c = v[in_slab].mean(0)
            spine.append(int(in_slab[np.argmin(((v[in_slab] - c) ** 2).sum(1))]))
        pairs = []
        for i in range(len(spine)):
            for j in (i + 2, i + 3):
                if j < len(spine):
                    pairs.append((spine[i], spine[j]))
        if len(spine) >= 5:
            pairs.append((spine[0], spine[len(spine) // 2]))
            pairs.append((spine[len(spine) // 2], spine[-1]))
            pairs.append((spine[0], spine[-1]))
        _SPINE[key] = pairs
    return _SPINE[key]


class Sim:
    def __init__(self, fishes, n_fish, device):
        self.device = device
        self.n_fish = n_fish
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SUBSTEPS
        # Deal the scans out round-robin: fish k is fishes[k % len(fishes)].
        self.variant = deal_variants(n_fish, len(fishes))
        self.fishes = fishes

        # The belt, and every fish's spawn pose. Seeded and shared with the PhysX
        # run, so the contact load -- and therefore the numbers that come out of
        # it -- is the same machine loaded the same way on both sides.
        L = plan_layout(fishes, n_fish, self.variant)
        self.layout = L
        self.part_base = L.part_base
        self.part_count = L.part_count

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))

        # --- roller bed: kinematic cylinders, axis along Y, tops at z = 0 ---
        roller_cfg = newton.ModelBuilder.ShapeConfig()
        roller_cfg.density = 0.0
        roller_cfg.mu = MU
        roller_cfg.ke, roller_cfg.kd = 1.0e5, 1.0e2
        roller_cfg.has_particle_collision = True
        roller_cfg.margin = CONTACT_MARGIN * min(f.cell for f in fishes)

        z_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
        self.roller_joints = []
        for i, x in enumerate(L.roller_x):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01),
                                    is_kinematic=True, label=f"roller_{i}")
            builder.add_shape_cylinder(body, xform=wp.transform(wp.vec3(), z_to_y),
                                       radius=ROLLER_R, half_height=L.half_w, cfg=roller_cfg)
            j = builder.add_joint_revolute(
                parent=-1, child=body, axis=newton.Axis.Y,
                parent_xform=wp.transform(p=wp.vec3(x, 0.0, -ROLLER_R), q=wp.quat_identity()))
            builder.add_articulation([j], label=f"roller_{i}")
            self.roller_joints.append(j)

        # --- side rails, and a floor well below to catch anything that leaves ---
        rail_cfg = newton.ModelBuilder.ShapeConfig()
        rail_cfg.mu = 0.2
        rail_cfg.has_particle_collision = True
        rail_cfg.margin = roller_cfg.margin
        for s in (-1.0, 1.0):
            builder.add_shape_box(body=-1, hx=0.5 * L.bed_len, hy=0.02, hz=L.rail_h,
                                  xform=wp.transform(p=wp.vec3(0.5 * L.bed_len - 0.5,
                                                               s * L.rail_y, 0.10),
                                                     q=wp.quat_identity()),
                                  cfg=rail_cfg)
        builder.add_ground_plane(height=-1.0)

        # --- the fish ---
        lam = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
        mu_lame = YOUNG / (2.0 * (1.0 + POISSON))
        cache = {}
        for k in range(n_fish):
            v = int(self.variant[k])
            fish = fishes[v]
            if v not in cache:
                cache[v] = (fish.cverts.tolist(), fish.ctets.flatten().tolist())
            vert_list, flat_tets = cache[v]
            x, y, z = (float(c) for c in L.spawn[k])
            first_particle = builder.particle_count
            builder.add_soft_mesh(
                pos=wp.vec3(x, y, z),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(L.yaw[k])),
                scale=float(L.scale[k]), vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=vert_list, indices=flat_tets,
                density=DENSITY, k_mu=mu_lame, k_lambda=lam, k_damp=1.0,
                add_surface_mesh_edges=SELF_CONTACT or EDGE_KE > 0.0,
                edge_ke=EDGE_KE,
                particle_radius=PARTICLE_R * fish.cell)
            if SPINE_KE > 0.0:
                # Long-range springs, because tets cannot make the BODY stiff:
                # XPBD moves a correction one tet-ring per iteration, a fish is
                # ~14 rings long, and gravity re-bends it faster than one or two
                # iterations carry stiffness down the chain. A spring that SPANS
                # the body puts nose and tail in the same constraint, making the
                # propagation distance one. Rest lengths come from the spawned
                # positions, so scale and yaw are already in them.
                for a, b_ in spine_pairs(fish):
                    builder.add_spring(first_particle + int(a), first_particle + int(b_),
                                       ke=SPINE_KE, kd=SPINE_KE * 1e-3, control=0.0)

        builder.color()
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = MU

        cell = float(np.mean([f.cell for f in fishes]))
        # Fish-on-fish is where the time goes, and the stock settings are sized for
        # cloth, not for a few hundred blunt tet cages. Three changes, measured at
        # 100 fish: detect once per substep rather than twice (54.2 -> 56.3 ms on
        # its own, but it is what lets the rest be cut), pull the detection margin
        # in from 0.8 to 0.55 cells since a cage particle cannot move half a cell
        # in one substep (-10 ms), and stop preallocating 32 vertex / 64 edge
        # contacts per particle when a fish in a heap sees a handful (-8 ms).
        # Together: 54.2 -> 27.5 ms with the pile behaving the same.
        vbuf, ebuf = (int(x) for x in CONTACT_BUF.split(","))
        interval = ITERATIONS if DETECT_INTERVAL < 0 else DETECT_INTERVAL
        if SOLVER == "xpbd":
            # XPBD's regime is the opposite corner from VBD's: many small steps,
            # one or two iterations. Fish-fish contact is particle-sphere via the
            # model's own radii -- none of the VBD self-contact machinery applies.
            self.solver = newton.solvers.SolverXPBD(
                self.model, iterations=ITERATIONS, soft_body_relaxation=RELAXATION)
        else:
            self.solver = newton.solvers.SolverVBD(
                self.model, iterations=ITERATIONS,
                particle_enable_self_contact=SELF_CONTACT,
                particle_self_contact_radius=SELF_R * cell,
                particle_self_contact_margin=SELF_MARGIN * cell,
                particle_collision_detection_interval=interval,
                particle_vertex_contact_buffer_size=vbuf,
                particle_edge_contact_buffer_size=ebuf,
                rigid_body_particle_contact_buffer_size=256)
        self.pipeline = newton.CollisionPipeline(
            self.model, broad_phase="sap", soft_contact_margin=CONTACT_MARGIN * cell)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.omega = BELT_SPEED / ROLLER_R
        self.n_rollers = L.n_rollers
        self.bed_len = L.bed_len
        self.t = wp.zeros(1, dtype=wp.float32, device=device)
        self.frames = 0
        self.graph = None

        # Per-particle owner, for the two whole-fish operations that are not the
        # solver's business: where is it, and put it back.
        self.n_particles = L.n_particles
        self.owner = wp.array(np.repeat(np.arange(n_fish, dtype=np.int32), self.part_count),
                              dtype=int, device=device)
        self.inv_count = wp.array((1.0 / self.part_count).astype(np.float32),
                                  dtype=float, device=device)
        self.cx = wp.zeros(n_fish, dtype=float, device=device)
        self.shift = wp.zeros(n_fish, dtype=wp.vec3, device=device)
        self.recycle_rng = np.random.default_rng(11)

        # One dof per roller joint and no other joints: joint index == dof index.
        assert self.model.joint_dof_count == L.n_rollers, "roller joint layout changed"

    def positions(self):
        """Cage vertices, all fish concatenated -- on the device, no readback."""
        return self.state_0.particle_q

    def positions_host(self):
        """The same array on the host. Only the metrics ask for this."""
        return self.state_0.particle_q.numpy()[:self.n_particles]

    def simulate(self):
        for _ in range(SUBSTEPS):
            self.state_0.clear_forces()
            wp.launch(drive_rollers, dim=self.n_rollers, device=self.device,
                      inputs=[self.t, self.omega],
                      outputs=[self.state_0.joint_q, self.state_0.joint_qd])
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd,
                           self.state_0, body_flag_filter=newton.BodyFlags.KINEMATIC)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_time, dim=1, device=self.device, inputs=[self.sim_dt], outputs=[self.t])

    def capture(self):
        # simulate() ping-pongs state_0/state_1 per substep. Capture bakes that
        # sequence of device pointers into the graph, so a replay only lands the
        # result back in state_0 -- where the next replay starts reading -- if
        # the swap count is even. An odd count silently simulates from a stale
        # buffer every frame.
        if SUBSTEPS % 2:
            sys.exit(f"--graph needs an even --substeps (got {SUBSTEPS}): the state "
                     f"ping-pong would leave each frame reading a stale buffer")
        with wp.ScopedCapture(device=self.device) as cap:
            self.simulate()
        self.graph = cap.graph

    def step(self):
        self.frames += 1
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

    def recycle(self):
        """Put fish that ran off the end back on the start of the bed.

        View path only. The benchmark measures conveying as mean displacement
        over a fixed window, and a fish teleported back inside that window reads
        as the belt running hard backwards -- so run() never calls this, and the
        window is short enough that nothing reaches the end during it.
        """
        self.cx.zero_()
        wp.launch(fish_centroid_x, dim=self.n_particles, device=self.device,
                  inputs=[self.state_0.particle_q, self.owner, self.inv_count],
                  outputs=[self.cx])
        done = self.cx.numpy() > self.layout.bed_len - 0.9
        if not done.any():
            return
        off = np.zeros((self.n_fish, 3), np.float32)
        off[done, 0] = -(self.layout.bed_len - 1.4)
        off[done, 1] = self.recycle_rng.uniform(-0.12, 0.12, int(done.sum()))
        wp.copy(self.shift, wp.array(off, dtype=wp.vec3, device=self.device))
        wp.launch(shift_fish, dim=self.n_particles, device=self.device,
                  inputs=[self.state_0.particle_q, self.owner], outputs=[self.shift])

    def sim_time(self):
        # Counted host-side on purpose: the device clock is the same number, and
        # reading it back would sync the whole graph replay once per frame.
        return self.frames * self.frame_dt

    def centroids(self):
        n = int(self.part_base[-1] + self.part_count[-1])
        q = self.state_0.particle_q.numpy()[:n]
        return (np.add.reduceat(q, self.part_base, axis=0)
                / self.part_count[:, None].astype(np.float64))


# -- Measurement -------------------------------------------------------------

def vram_mb(device):
    try:
        d = wp.get_device(device)
        return (d.total_memory - d.free_memory) / (1 << 20)
    except Exception:
        return float("nan")


def run(fishes, n_fish, device, verbose=True):
    t_build = time.perf_counter()
    sim = Sim(fishes, n_fish, device)
    build_s = time.perf_counter() - t_build

    for _ in range(WARMUP):          # kernel compile + settle onto the rollers
        sim.simulate()
    wp.synchronize_device(device)
    if GRAPH:
        sim.capture()
        sim.step()
        wp.synchronize_device(device)

    x0 = sim.centroids()[:, 0].copy()
    used0 = vram_mb(device)

    times = []
    for _ in range(FRAMES):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        sim.step()
        wp.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1e3)

    c = sim.centroids()
    dx = c[:, 0] - x0
    elapsed = FRAMES / FPS
    finite = np.isfinite(c).all(1)
    onbelt = finite & (c[:, 2] > -0.5)

    times = np.array(times)
    r = {
        "n": n_fish,
        "tets": sim.model.tet_count,
        "particles": sim.model.particle_count,
        "rollers": sim.n_rollers,
        "ms": float(np.median(times)),
        "ms_p95": float(np.percentile(times, 95)),
        "rt": (1000.0 / FPS) / float(np.median(times)),
        "convey": float(np.mean(dx[onbelt]) / elapsed) if onbelt.any() else float("nan"),
        "belt": BELT_SPEED,
        "onbelt": int(onbelt.sum()),
        "vram": used0,
        "build_s": build_s,
    }
    if verbose:
        print(f"  {n_fish:>4} fish | {r['tets']:>7} tets {r['particles']:>7} parts | "
              f"{r['ms']:7.2f} ms/frame (p95 {r['ms_p95']:6.2f}) | {r['rt']:5.2f}x realtime | "
              f"convey {r['convey']:+.3f} m/s of {BELT_SPEED:.2f} | "
              f"{r['onbelt']}/{n_fish} on belt | {r['vram']:.0f} MB", flush=True)
    return sim, r

def view(sim, fishes):
    """Eyeball it. Not the measured path, but not a slideshow either: the only
    thing that crosses host memory per frame is one float per fish, for the
    recycle check."""
    canvas, renderer, scene, camera, draw = stage(sim, fishes, False, wp.get_device(DEVICE), "newton fish conveyor")
    controls = tp.OrbitControls(camera, canvas)
    controls.target.set(0.5 * sim.bed_len - 0.5 - 0.02 * sim.bed_len, 0.6 * sim.layout.rail_h, 0.0)
    controls.enable_damping = True

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)

    canvas.on_window_resize(on_resize)

    def animate():
        sim.step()
        if RECYCLE and sim.frames % 12 == 0:
            sim.recycle()
        draw()
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)


def shot(sim, fishes, path):
    """Headless PNG, after enough sim that the catch has settled and moved."""
    canvas, renderer, scene, camera, draw = stage(sim, fishes, True, wp.get_device(DEVICE), "newton fish conveyor")
    for _ in range(int(round(SHOT_TIME * FPS))):
        sim.step()
        if RECYCLE and sim.frames % 12 == 0:
            sim.recycle()
        draw()
        renderer.render(scene, camera)      # arms the GL interop on frame one
    renderer.save_frame(path)
    print(f"      simulated {SHOT_TIME:.1f} s, wrote {path}", flush=True)

# -- PhysX's tetrahedralisation, for the like-for-like run --------------------

def physx_tets(paths, voxel_res):
    """Cook each scan into PhysX's conforming tet mesh and hand back the cages.

    Run in a SUBPROCESS: PhysX allows one PxFoundation per process and creates
    its own CUDA context, and the point of this call is to leave neither behind
    in the process that is about to run Warp. What comes back is exactly what
    the PhysX benchmark simulates -- same cook, same voxel resolution -- so
    --tets physx is the "same mesh, different solver" arm of the comparison.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physx_fish_conveyor.py")
    out = os.path.join(tempfile.gettempdir(), f"physx_tets_{os.getpid()}.npz")
    cmd = [sys.executable, script, "--fish", ",".join(paths), "--dump-tets", out,
           "--voxel-res", str(voxel_res), "--render-ratio", str(RENDER_RATIO),
           "--variants", str(len(paths))]
    if FISH_LENGTH > 0:
        cmd += ["--fish-length", str(FISH_LENGTH)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        sys.exit("--tets physx: the cook subprocess failed\n" + (r.stderr or r.stdout)[-3000:])
    with np.load(out) as z:
        cages = [(z[f"v{i}"], z[f"t{i}"]) for i in range(len(paths))]
    os.remove(out)
    return cages


# -- main --------------------------------------------------------------------

def main():
    if not FISH:
        sys.exit("--fish PATH is required: there is no bundled model.\n"
                 "  Any closed solid threepp's ModelLoader reads will do -- .usdz, .usd,\n"
                 "  .obj, .stl, .gltf/.glb, .dae. A directory or a comma-separated list\n"
                 "  loads several and deals them out round-robin. Nothing here is\n"
                 "  fish-specific beyond the auto-orientation (longest extent along the\n"
                 "  belt, thinnest one up).\n"
                 "  e.g. python newton_fish_conveyor.py --fish cod.usdz --graph")
    paths = fish_paths(FISH)
    if not paths:
        sys.exit(f"--fish: nothing to load from {FISH}")
    for p in paths:
        if not os.path.isfile(p):
            sys.exit(f"--fish: no such file: {p}")
    paths = paths[:max(1, VARIANTS)]

    device = wp.get_device(DEVICE)
    if device.is_cpu:
        print("warning: --device cpu is not a meaningful measurement of GPU headroom")

    if TETS == "physx":
        cages = physx_tets(paths, PHYSX_VOXEL_RES)
        fishes = [Fish(p, device, tets=c) for p, c in zip(paths, cages)]
    elif TETS == "voxel":
        fishes = [Fish(p, device) for p in paths]
    else:
        sys.exit(f"--tets: expected 'voxel' or 'physx', got {TETS!r}")
    for f in fishes:
        f.report()
    if len(fishes) > 1:
        print(f"      {len(fishes)} scans, dealt round-robin over the fish")
    print(f"mat   E={YOUNG:.3g} Pa, nu={POISSON}, rho={DENSITY} kg/m3 | "
          f"belt {BELT_SPEED} m/s, mu={MU}, rollers r={ROLLER_R} m")
    print(f"solve {SOLVER.upper()} {ITERATIONS} iters x {SUBSTEPS} substeps @ {FPS:.0f} Hz"
          f"{', CUDA graph' if GRAPH else ''}{', self-contact' if SELF_CONTACT else ''}"
          f" | cage {fishes[0].cage_kind}")
    print()

    if AB:
        # The A/B path: settle, then run a fixed window with skinning and a
        # headless render inside the clock, and report quality alongside speed.
        sim = Sim(fishes, COUNT, device)
        for _ in range(WARMUP):
            sim.simulate()
        wp.synchronize_device(device)
        if GRAPH:
            sim.capture()
            sim.step()
            wp.synchronize_device(device)
        engine = "newton-physx-tets" if TETS == "physx" else "newton-voxel"
        r = ab_run(sim, fishes, device, AB_SECONDS, engine, SEED,
                   "newton fish conveyor", shot=SHOT or None, shot_at=SHOT_TIME, warmup=0)
        fc.ab_report(r)
        return

    if VIEW or SHOT:
        # Warm up (compile + settle onto the rollers) but skip the timed loop --
        # this path is for looking at.
        sim = Sim(fishes, COUNT, device)
        for _ in range(WARMUP):
            sim.simulate()
        wp.synchronize_device(device)
        if GRAPH:
            sim.capture()
        drawn = sum(len(f.render_surface()[1]) for f in fishes) * COUNT // len(fishes)
        print(f"      {COUNT} fish, {sim.model.tet_count} tets simulated, "
              f"{drawn} triangles drawn ({'cage' if CAGE else 'scan'})", flush=True)
        if SHOT:
            shot(sim, fishes, SHOT)
        else:
            print("      drag to orbit, Esc quits", flush=True)
            view(sim, fishes)
        return

    counts = [n for n in (1, 4, 16, 64, 128, 256, 512) if n <= SWEEP_TO] if SWEEP else [COUNT]
    per_fish = float(np.mean([len(f.cverts) for f in fishes]))
    rows = []
    for n in counts:
        if n * per_fish > MAX_PARTICLES:
            print(f"  {n:>4} fish | SKIPPED: {int(n * per_fish)} particles exceeds "
                  f"--max-particles {MAX_PARTICLES}", flush=True)
            break
        print(f"  {n:>4} fish | building...", end="\r", flush=True)
        try:
            _, r = run(fishes, n, device)
            rows.append(r)
        except Exception as e:                       # OOM or buffer overflow at high n
            print(f"  {n:>4} fish | FAILED: {type(e).__name__}: {e}", flush=True)
            break

    if len(rows) > 1:
        print()
        print("  realtime budget is 16.67 ms/frame at 60 Hz")
        last = [r for r in rows if r["rt"] >= 1.0]
        if last:
            print(f"  realtime ceiling on this GPU: ~{last[-1]['n']} fish "
                  f"({last[-1]['tets']} tets) at {last[-1]['ms']:.1f} ms/frame")
        else:
            print("  nothing hit realtime -- lower --tet-res or --iterations")


if __name__ == "__main__":
    main()
