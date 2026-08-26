"""The fish conveyor on a hand-rolled Warp solver: corotational XPBD, mass-split Jacobi.

The third engine of the fish-conveyor comparison, and the only one that is OURS.
newton_fish_conveyor.py runs Newton's solvers, physx_fish_conveyor.py runs PhysX 5
deformable volumes; this file runs the corotational FEM solver developed for the
soft-gripper demo (warp_franka_softgrip.py, phase 11) on the exact same benchmark:
same scans, same voxel cage, same barycentric bind, same belt, same spawn layout,
same GPU skinning, same quality metrics -- everything shared lives in
fish_conveyor_common.py and only the solver differs.

    python warp_fish_conveyor.py --fish scans/ --count 60 --view
    python warp_fish_conveyor.py --fish scans/ --count 60 --ab      # one JSON line
    python warp_fish_conveyor.py --fish cod.usdz --count 8 --view --cage

THE SOLVER is XPBD with two corotational rows per tet, both ZERO AT REST:
deviatoric C = |F - R|_F (R the closest rotation, via SVD with the smallest
singular direction sign-fixed, so an inverted tet is pulled back through flat --
inversion-safe) at stiffness mu, and volumetric C = det(F) - 1 at stiffness
lambda. Zero-at-rest is the property that matters at low iteration counts: the
multipliers carry only the actual load, so a truncated solve is merely a little
soft, never biased (the stable-Neo-Hookean pair, whose rows are nonzero at rest,
sat a fish at half its rest volume with nothing touching it). The solve is ONE
mass-split Jacobi pass per iteration -- Tonge-style n*w denominators, corrections
SUMMED, unconditionally stable with no relaxation knob to tune -- in the
small-steps regime: many substeps, few iterations.

THE CONTACT MODEL is this file's second experiment. Fish-fish is sphere contact
on the cage vertices like Newton-XPBD's, but detection runs ONCE PER FRAME
(hash grid + cached candidate pairs with a margin no particle outruns in a
frame) instead of once per substep -- the pairs are replayed by every substep's
iterations. The machine is ANALYTIC: the rollers are kinematic cylinders solved
in closed form (nearest-roller lookup, since the bed is a uniform lattice), and
belt friction is position-level Coulomb against the roller's own surface
velocity omega x r, with a genuine STICK branch: when the tangential error fits
inside mu times the penetration, it is cancelled exactly, so a resting fish
rides at belt speed rather than creeping behind it. Conveying at 0.71-0.88x of
belt speed is what disqualified Newton-XPBD's sphere contacts; this model is
the attempt to keep XPBD's frame time without that defect.

Everything heavy is captured in ONE CUDA graph per frame (the per-frame contact
detection stays outside it). --no-graph launches kernel by kernel.

The knobs mirror newton_fish_conveyor.py where they mean the same thing
(--tet-res, --substeps, --iterations, --young, --particle-radius, ...). The
defaults are the COARSE-CAGE regime the Newton settings audit converged on:
res-8 cage, 16 substeps x 2 iterations -- body stiffness in a Jacobi-propagated
solver is chain length, not Young's modulus, and the fat contact spheres of a
coarse cage are what grip a 50 mm roller.
"""
import math
import os
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fish_conveyor_common as fc
from fish_conveyor_common import (Fish, ab_run, cli_arg, deal_variants, find_params,
                                  fish_paths, plan_layout, stage)
# After fish_conveyor_common, which is what puts python/ on sys.path.
import threepp as tp  # noqa: E402


FISH = cli_arg("--fish", "", str)            # required -- see the module docstring
VARIANTS = cli_arg("--variants", 4, int)
COUNT = cli_arg("--count", 1, int)
SDF_SLAB = cli_arg("--sdf-slab", 8, int)
TET_RES = cli_arg("--tet-res", 10, int)      # res 8 over-stiffens BENDING (1-2 cells
                                             # through the body cannot curve) and its
                                             # fat spheres keep the pile artificially
                                             # airy; res 10 drapes, nestles (3-4 mm
                                             # real penetrations, like a catch), and
                                             # costs ~0.3 ms at 60 fish
SUBSTEPS = cli_arg("--substeps", 16, int)    # small steps: dt^2 convergence
ITERATIONS = cli_arg("--iterations", 2, int)
FRAMES = cli_arg("--frames", 120, int)
WARMUP = cli_arg("--warmup", 90, int)
GRAPH = "--no-graph" not in sys.argv
VIEW = "--view" in sys.argv
SHOT = cli_arg("--shot", "", str)
SHOT_TIME = cli_arg("--shot-time", 2.5, float)
CAGE = "--cage" in sys.argv
RENDER_RATIO = cli_arg("--render-ratio", 0.4, float)
CARVE = cli_arg("--carve", 0.25, float)
PARTICLE_R = cli_arg("--particle-radius", 0.40, float)  # cage cells, vs the machine
SELF_R = cli_arg("--self-radius", 0.25, float)          # cage cells, fish-on-fish
CONTACT_CAP = cli_arg("--contact-cap", 24, int)         # cached foreign pairs/particle
DETECT_MARGIN = cli_arg("--detect-margin", 0.5, float)  # cells of once-per-frame slack
CONTACT_RELAX = cli_arg("--contact-relax", 0.6, float)  # summed contact push scale
# Fish-on-fish contact has TWO coefficients because slime separates them:
# along the contact normal a pair is dead-inelastic (fish do not bounce off
# each other -- this is what lets a pile come to rest, keep it high), but
# TANGENTIALLY a wet fish is a slippery beast: a fish dropped on a pile slides
# off until the heap is shallow. One shared coefficient made the catch cling
# into steep stacks (user verdict). Both are viscous fractions of the relative
# motion removed per substep -- dead fish smear, they do not stick-slip.
PAIR_FRICTION = cli_arg("--pair-friction", 0.12, float)  # tangential: the slime
PAIR_BOUNCE = cli_arg("--pair-bounce", 0.5, float)       # normal: the dead meat
SLEEP_V = cli_arg("--sleep", 0.012, float)   # m/s; per-particle velocity floor
OMEGA_R = cli_arg("--omega", 1.0, float)     # Jacobi over-relaxation; 1 = plain mass-split
DAMP_RATE = cli_arg("--damp", 5.0, float)    # 1/s. Dead wet flesh, heavily damped --
                                             # but bounded by the belt: the stick
                                             # budget mu*g*dt^2 must exceed the
                                             # damping's per-substep bite v*d*dt or
                                             # a riding fish slips. 8/s measured
                                             # convey 0.84x (bite = the whole
                                             # budget); 5/s keeps a 2x margin.
INTEROP = "--no-interop" not in sys.argv
RECYCLE = "--recycle" in sys.argv
LANES = cli_arg("--lanes", 5, int)
BED = cli_arg("--bed", 30.0, float)
LAYERS = cli_arg("--layers", 2, int)
BELT_SPEED = cli_arg("--speed", 0.5, float)
BELT = cli_arg("--belt", "flat", str)        # flat | rollers. Flat is the default:
                                             # it is the surface MU was measured on
                                             # (plastic modular belt), it is what a
                                             # processing line runs, and the roller
                                             # crowns were a perpetual excitation
                                             # the pile could never settle under.
ROLLER_R = cli_arg("--roller-radius", 0.05, float)
MU = cli_arg("--mu", 0.45, float)
YOUNG = cli_arg("--young", 5.0e4, float)     # Pa, judged AT TRUE SCALE (--params)
                                             # against the PhysX pile, live. The
                                             # literature's 5e4 for fish flesh is
                                             # right ONCE everything else is: at the
                                             # scans' native 2.5x size with 40 kg of
                                             # density-mass it read as thin film, and
                                             # the "compensating" 2e5-1e6 read as
                                             # plastic (user verdicts, 2026-08-24).
                                             # This solve is well-converged (16
                                             # substeps), so E expresses honestly --
                                             # PhysX's softness at its own 1e5 is
                                             # partly its 1-substep under-integration.
POISSON = cli_arg("--poisson", 0.48, float)  # near-incompressible: drape comes from
                                             # soft E, plumpness survives via the
                                             # hard volume row -- soft in bending,
                                             # does not squash. 0.45 at this E reads
                                             # droopier for no gain.
DENSITY = cli_arg("--density", 1050.0, float)
FISH_LENGTH = cli_arg("--fish-length", 0.0, float)
# Physical ground truth. "auto" finds Physical_Parameters/parameters.json next
# to the scans (the FHF database layout); a path uses that file; "none" turns it
# off. With parameters, every scan is rescaled to its MEASURED length (the
# photogrammetry arrives 2.5x life size -- a "1.6 m" cod is a 0.63 m, 3.7 kg
# fish) and each fish weighs its MEASURED weight instead of density x cage
# volume, which made 40 kg fish and was most of why the flesh read as thin film
# and the stacks squirmed: gravity stress scales with size, and E that suits a
# 0.6 m cod is film on a 1.6 m one.
PARAMS = cli_arg("--params", "auto", str)
SCALE_JITTER = cli_arg("--scale-jitter", 0.07, float)
SEED = cli_arg("--seed", 7, int)
AB = "--ab" in sys.argv
AB_SECONDS = cli_arg("--ab-seconds", 8.0, float)
DEVICE = cli_arg("--device", "cuda:0", str)

FPS = 60.0
DT = 1.0 / (FPS * SUBSTEPS)
MU_LAME = YOUNG / (2.0 * (1.0 + POISSON))
LAM_LAME = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))

if PARAMS == "auto":
    PARAMS = find_params(FISH) if FISH else ""
elif PARAMS == "none":
    PARAMS = ""

fc.configure(FPS=FPS, TET_RES=TET_RES, CARVE=CARVE, SDF_SLAB=SDF_SLAB, PARAMS=PARAMS,
             BELT_STYLE=BELT,
             RENDER_RATIO=RENDER_RATIO, CAGE=CAGE, FISH_LENGTH=FISH_LENGTH,
             LANES=LANES, BED=BED, LAYERS=LAYERS, BELT_SPEED=BELT_SPEED,
             ROLLER_R=ROLLER_R, MU=MU, WARMUP=WARMUP, FRAMES=FRAMES,
             SCALE_JITTER=SCALE_JITTER, SEED=SEED, INTEROP=INTEROP,
             # The bed's runway exists so nothing conveys off the end inside a
             # MEASURED window. The interactive view has no measured window and
             # wants a belt sized to the catch, not to a protocol.
             MEASURED_FRAMES=int(round(AB_SECONDS * FPS)) if AB
             else (0 if VIEW else FRAMES))


# -- kernels ------------------------------------------------------------------

@wp.kernel
def integrate_damped(x: wp.array(dtype=wp.vec3),
                     prev: wp.array(dtype=wp.vec3),
                     pred: wp.array(dtype=wp.vec3),
                     dt: float,
                     damp: float,
                     gravity: wp.vec3,
                     npush: wp.array(dtype=float)):
    i = wp.tid()
    p = x[i]
    v = (p - prev[i]) * (1.0 - damp)
    prev[i] = p
    pred[i] = p + v + gravity * dt * dt
    npush[i] = 0.0     # the substep's fresh normal-correction ledger


@wp.kernel
def lambda_reset(lam: wp.array(dtype=float)):
    lam[wp.tid()] = 0.0


@wp.kernel
def apply_delta(pos: wp.array(dtype=wp.vec3),
                dpos: wp.array(dtype=wp.vec3),
                omega: float):
    # Mass-split corrections arrive SUMMED; the split already made that an
    # average, so no division. The buffer self-clears for the next sweep.
    i = wp.tid()
    pos[i] = pos[i] + dpos[i] * omega
    dpos[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def tet_corot(pos: wp.array(dtype=wp.vec3),
              tets: wp.array2d(dtype=int),
              dm_inv: wp.array(dtype=wp.mat33),
              alpha_d: wp.array(dtype=float),
              alpha_h: wp.array(dtype=float),
              invm: wp.array(dtype=float),
              nsplit: wp.array(dtype=float),
              lam_d: wp.array(dtype=float),
              lam_h: wp.array(dtype=float),
              dpos: wp.array(dtype=wp.vec3)):
    # Two XPBD rows per tet, both zero at rest -- the soft-gripper solver
    # verbatim (see the module docstring, and warp_franka_softgrip.py for the
    # full derivation): deviatoric C = |F - R|_F at stiffness mu, volumetric
    # C = det(F) - 1 at stiffness lambda, mass-split Jacobi denominators.
    t = wp.tid()
    ia, ib, ic, id_ = tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]
    x0, x1, x2, x3 = pos[ia], pos[ib], pos[ic], pos[id_]
    w0, w1, w2, w3 = invm[ia], invm[ib], invm[ic], invm[id_]
    s0 = w0 * nsplit[ia]
    s1 = w1 * nsplit[ib]
    s2 = w2 * nsplit[ic]
    s3 = w3 * nsplit[id_]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0
    di = dm_inv[t]
    f0 = e1 * di[0, 0] + e2 * di[1, 0] + e3 * di[2, 0]
    f1 = e1 * di[0, 1] + e2 * di[1, 1] + e3 * di[2, 1]
    f2 = e1 * di[0, 2] + e2 * di[1, 2] + e3 * di[2, 2]
    F = wp.mat33(f0[0], f1[0], f2[0],
                 f0[1], f1[1], f2[1],
                 f0[2], f1[2], f2[2])

    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)
    R = U * wp.transpose(V)
    if wp.determinant(R) < 0.0:
        m = int(0)
        if wp.abs(sig[1]) < wp.abs(sig[m]):
            m = 1
        if wp.abs(sig[2]) < wp.abs(sig[m]):
            m = 2
        um = wp.vec3(U[0, m], U[1, m], U[2, m])
        vm = wp.vec3(V[0, m], V[1, m], V[2, m])
        R = R - wp.outer(um, vm) * 2.0

    da0 = wp.vec3(0.0, 0.0, 0.0)
    da1 = wp.vec3(0.0, 0.0, 0.0)
    da2 = wp.vec3(0.0, 0.0, 0.0)
    da3 = wp.vec3(0.0, 0.0, 0.0)

    D = F - R
    cd = wp.sqrt(D[0, 0] * D[0, 0] + D[0, 1] * D[0, 1] + D[0, 2] * D[0, 2]
                 + D[1, 0] * D[1, 0] + D[1, 1] * D[1, 1] + D[1, 2] * D[1, 2]
                 + D[2, 0] * D[2, 0] + D[2, 1] * D[2, 1] + D[2, 2] * D[2, 2])
    ad = alpha_d[t]
    if cd > 1.0e-9:
        inv_cd = 1.0 / cd
        d0 = wp.vec3(D[0, 0], D[1, 0], D[2, 0])
        d1 = wp.vec3(D[0, 1], D[1, 1], D[2, 1])
        d2 = wp.vec3(D[0, 2], D[1, 2], D[2, 2])
        g1 = (d0 * di[0, 0] + d1 * di[0, 1] + d2 * di[0, 2]) * inv_cd
        g2 = (d0 * di[1, 0] + d1 * di[1, 1] + d2 * di[1, 2]) * inv_cd
        g3 = (d0 * di[2, 0] + d1 * di[2, 1] + d2 * di[2, 2]) * inv_cd
        g0 = -(g1 + g2 + g3)
        den = (s0 * wp.dot(g0, g0) + s1 * wp.dot(g1, g1)
               + s2 * wp.dot(g2, g2) + s3 * wp.dot(g3, g3) + ad)
        dl = (-cd - ad * lam_d[t]) / den
        lam_d[t] = lam_d[t] + dl
        da0 = g0 * (w0 * dl)
        da1 = g1 * (w1 * dl)
        da2 = g2 * (w2 * dl)
        da3 = g3 * (w3 * dl)
    else:
        lam_d[t] = 0.0

    c0 = wp.cross(f1, f2)
    c1 = wp.cross(f2, f0)
    c2 = wp.cross(f0, f1)
    detf = wp.dot(f0, c0)
    h1 = c0 * di[0, 0] + c1 * di[0, 1] + c2 * di[0, 2]
    h2 = c0 * di[1, 0] + c1 * di[1, 1] + c2 * di[1, 2]
    h3 = c0 * di[2, 0] + c1 * di[2, 1] + c2 * di[2, 2]
    h0 = -(h1 + h2 + h3)
    ch = detf - 1.0
    ah = alpha_h[t]
    denh = (s0 * wp.dot(h0, h0) + s1 * wp.dot(h1, h1)
            + s2 * wp.dot(h2, h2) + s3 * wp.dot(h3, h3) + ah)
    dlh = (-ch - ah * lam_h[t]) / denh
    lam_h[t] = lam_h[t] + dlh
    da0 = da0 + h0 * (w0 * dlh)
    da1 = da1 + h1 * (w1 * dlh)
    da2 = da2 + h2 * (w2 * dlh)
    da3 = da3 + h3 * (w3 * dlh)

    wp.atomic_add(dpos, ia, da0)
    wp.atomic_add(dpos, ib, da1)
    wp.atomic_add(dpos, ic, da2)
    wp.atomic_add(dpos, id_, da3)


@wp.kernel
def build_contacts(pos: wp.array(dtype=wp.vec3),
                   grid: wp.uint64,
                   owner: wp.array(dtype=int),
                   rad_pair: wp.array(dtype=float),
                   max_rad: float,
                   margin: float,
                   cap: int,
                   cont_idx: wp.array(dtype=int),
                   cont_n: wp.array(dtype=int)):
    # Once per FRAME: cache every foreign-fish particle within reach-plus-margin.
    # The margin covers a whole frame of motion, so the substeps replay this list
    # instead of re-querying sixteen times. Same-fish pairs are skipped -- sphere
    # contact has no topological filter and rest-state neighbours would start in
    # violation (the defect that limited Newton-XPBD's radius to 0.25 cells).
    i = wp.tid()
    p = pos[i]
    mine = owner[i]
    reach = rad_pair[i]
    n = int(0)
    q = wp.hash_grid_query(grid, p, reach + max_rad + margin)
    for j in q:
        if owner[j] != mine and n < cap:
            if wp.length(p - pos[j]) < reach + rad_pair[j] + margin:
                cont_idx[i * cap + n] = j
                n += 1
    cont_n[i] = n


@wp.func
def machine_push(p: wp.vec3,
                 r: float,
                 flat: int,
                 roller_x0: float,
                 inv_spacing: float,
                 n_rollers: int,
                 rail_y: float,
                 rail_z0: float,
                 rail_z1: float,
                 bed_x0: float,
                 bed_x1: float) -> wp.vec4:
    """Positional push out of the belt (flat or rollers), the rails and the
    floor, packed as (push.xyz, depth) -- depth is the deepest belt/floor
    penetration BEFORE the push, and it feeds the substep's friction budget."""
    push = wp.vec3(0.0, 0.0, 0.0)
    depth = float(0.0)
    if flat != 0:
        # a plastic modular belt: a SLAB, not a half-space over an x-range. The
        # difference is the end face: a particle hanging past the edge that
        # swings back under the lip must be pushed out the NEAREST face -- the
        # end -- not teleported up through the whole belt to the top (that was
        # the slingshot; the softgrip height-field step had the same bug).
        if (p[0] > bed_x0 - r and p[0] < bed_x1 + r
                and p[2] < r and p[2] > -0.09):
            pen_top = r - p[2]
            pen_end = (bed_x1 + r) - p[0]
            pen_back = p[0] - (bed_x0 - r)
            if pen_top <= wp.min(pen_end, pen_back):
                push += wp.vec3(0.0, 0.0, pen_top)
                depth = wp.max(depth, pen_top)
            elif pen_end <= pen_back:
                push += wp.vec3(pen_end, 0.0, 0.0)
            else:
                push += wp.vec3(-pen_back, 0.0, 0.0)
    else:
        # rollers: kinematic cylinders, axis +Y at (x_k, z = -R); a uniform
        # lattice, so the two nearest rollers are an index computation
        k = int(wp.round((p[0] - roller_x0) * inv_spacing))
        for dk in range(-1, 2):
            j = k + dk
            if j >= 0 and j < n_rollers:
                rx = roller_x0 + float(j) / inv_spacing
                d = wp.vec2(p[0] - rx, p[2] + ROLLER_R)
                l = wp.length(d)
                pen = ROLLER_R + r - l
                if pen > 0.0 and l > 1.0e-9:
                    nrm = d / l
                    push += wp.vec3(nrm[0] * pen, 0.0, nrm[1] * pen)
                    depth = wp.max(depth, pen)
    # side rails (their inner faces), only along the bed and over their height
    if p[0] > bed_x0 and p[0] < bed_x1 and p[2] > rail_z0 and p[2] < rail_z1:
        if p[1] > rail_y - r:
            push += wp.vec3(0.0, rail_y - r - p[1], 0.0)
        if p[1] < -(rail_y - r):
            push += wp.vec3(0.0, -(rail_y - r) - p[1], 0.0)
    # the hall floor, a metre below the bed
    if p[2] < -1.0 + r:
        push += wp.vec3(0.0, 0.0, -1.0 + r - p[2])
        depth = wp.max(depth, -1.0 + r - p[2])
    return wp.vec4(push[0], push[1], push[2], depth)


@wp.kernel
def contacts_solve(pos: wp.array(dtype=wp.vec3),
                   prev: wp.array(dtype=wp.vec3),
                   invm: wp.array(dtype=float),
                   rad_pair: wp.array(dtype=float),
                   rad_env: wp.array(dtype=float),
                   cont_idx: wp.array(dtype=int),
                   cont_n: wp.array(dtype=int),
                   cap: int,
                   relax: float,
                   pair_fric: float,
                   pair_bounce: float,
                   roller_x0: float,
                   inv_spacing: float,
                   n_rollers: int,
                   rail_y: float,
                   rail_z0: float,
                   rail_z1: float,
                   bed_x0: float,
                   bed_x1: float,
                   flat: int,
                   belt_v: float,
                   npush: wp.array(dtype=float),
                   dpos: wp.array(dtype=wp.vec3)):
    # Each particle sums the pushes of ITS cached pairs (the pair is cached on
    # both ends, so this is symmetric without atomics) plus the machine's. The
    # roller/floor penetration is LATCHED into npush before the push resolves
    # it: friction budgeted on residual overlap is friction that vanishes the
    # moment the normal constraint converges (the soft-gripper's phase-2 bug).
    i = wp.tid()
    p = pos[i]
    wi = invm[i]
    ri = rad_pair[i]
    disp_i = p - prev[i]
    c = wp.vec3(0.0, 0.0, 0.0)
    for k in range(cont_n[i]):
        j = cont_idx[i * cap + k]
        d = p - pos[j]
        l = wp.length(d)
        thick = ri + rad_pair[j]
        if l < thick and l > 1.0e-9:
            wj = invm[j]
            n = d / l
            c += n * ((thick - l) * (wi / (wi + wj)))
            # wet-on-wet contact, split by direction: the normal component is
            # damped hard (dead fish do not bounce apart -- leaving it elastic
            # made the pile breathe forever under Verlet), the tangential one
            # barely (slime; strong tangential grip made the catch cling into
            # steep stacks instead of sloughing off).
            rel = disp_i - (pos[j] - prev[j])
            reln = n * wp.dot(rel, n)
            c -= reln * (0.5 * pair_bounce) + (rel - reln) * (0.5 * pair_fric)
    m = machine_push(p, rad_env[i], flat, roller_x0, inv_spacing, n_rollers,
                     rail_y, rail_z0, rail_z1, bed_x0, bed_x1)
    npush[i] = npush[i] + m[3]
    dpos[i] = dpos[i] + c * relax + wp.vec3(m[0], m[1], m[2])


@wp.kernel
def machine_friction(pos: wp.array(dtype=wp.vec3),
                     prev: wp.array(dtype=wp.vec3),
                     rad_env: wp.array(dtype=float),
                     npush: wp.array(dtype=float),
                     mu: float,
                     omega: float,
                     dt: float,
                     roller_x0: float,
                     inv_spacing: float,
                     n_rollers: int,
                     rail_y: float,
                     rail_z0: float,
                     rail_z1: float,
                     bed_x0: float,
                     bed_x1: float,
                     flat: int,
                     belt_v: float):
    # Once per substep, after the iterations: a final projection, then Coulomb
    # friction against the surface the particle is touching. On a roller the
    # surface MOVES -- v = omega x r, which at the crown is exactly belt speed --
    # and the friction pulls this substep's tangential motion toward the
    # surface's own, with a real stick branch: an error inside the budget is
    # cancelled EXACTLY, so a resting fish rides at belt speed instead of
    # creeping behind it. (Sphere-contact slip here is what cost Newton-XPBD
    # 12-29% of its conveying speed.) The budget is mu times the substep's
    # ACCUMULATED normal correction -- the positional stand-in for mu*N -- for
    # a resting particle that is g*dt^2 a substep, i.e. friction accelerates a
    # lagging fish at mu*g, which is the Coulomb answer.
    i = wp.tid()
    p = pos[i]
    r = rad_env[i]
    m = machine_push(p, r, flat, roller_x0, inv_spacing, n_rollers,
                     rail_y, rail_z0, rail_z1, bed_x0, bed_x1)
    p = p + wp.vec3(m[0], m[1], m[2])
    total = npush[i] + m[3]
    if total > 0.0:
        # nearest support within half a radius of touching: the belt when over
        # the bed, else the floor. The slack matters -- by this point the normal
        # solve has already put the particle ON the surface, and a support test
        # of "still penetrating" would find nothing to rub against.
        vsurf = wp.vec3(0.0, 0.0, 0.0)
        nrm = wp.vec3(0.0, 0.0, 0.0)
        if flat != 0:
            # top support only: a particle sliding along the end FACE is
            # falling off the machine, not riding it
            if (p[0] > bed_x0 and p[0] < bed_x1
                    and p[2] < 1.5 * r and p[2] > -0.5 * r):
                nrm = wp.vec3(0.0, 0.0, 1.0)
                vsurf = wp.vec3(belt_v, 0.0, 0.0)
        else:
            k = int(wp.round((p[0] - roller_x0) * inv_spacing))
            best = float(1.0e9)
            for dk in range(-1, 2):
                j = k + dk
                if j >= 0 and j < n_rollers:
                    rx = roller_x0 + float(j) / inv_spacing
                    d = wp.vec2(p[0] - rx, p[2] + ROLLER_R)
                    l = wp.length(d)
                    if l < ROLLER_R + 1.5 * r and l < best and l > 1.0e-9:
                        best = l
                        nrm = wp.vec3(d[0] / l, 0.0, d[1] / l)
                        # omega about +Y: v = omega * R * (n_z, 0, -n_x)
                        vsurf = wp.vec3(omega * ROLLER_R * d[1] / l, 0.0,
                                        -omega * ROLLER_R * d[0] / l)
        if wp.length_sq(nrm) < 0.5 and p[2] < -1.0 + 1.5 * r:
            nrm = wp.vec3(0.0, 0.0, 1.0)
        if wp.length_sq(nrm) > 0.5:
            err = (p - prev[i]) - vsurf * dt
            en = wp.dot(err, nrm)
            errt = err - nrm * en
            et = wp.length(errt)
            budget = mu * total
            if et > 1.0e-12:
                if et <= budget:
                    p = p - errt                      # stick
                else:
                    p = p - errt * (budget / et)      # slide
            # a contact does not LAUNCH: damp the separating normal motion so
            # a fish rumbling over the roller crowns loses its hop instead of
            # compounding it
            if en > 0.0:
                p = p - nrm * (en * 0.5)
    # sleep floor: sub-centimetre-per-second residual shimmer is solver noise,
    # not physics -- freeze it, exactly like the big engines' sleep thresholds.
    # A riding fish moves 40x this per substep and never feels it. ONLY where
    # something is holding the particle up, though: gravity's own per-substep
    # step is g*dt^2 = 1.1e-5 m, which is BELOW this floor, so an unsupported
    # particle that starts at rest could never begin to fall.
    if total > 0.0 and wp.length(p - prev[i]) < SLEEP_V * dt:
        prev[i] = p
    pos[i] = p


@wp.kernel
def fish_centroid_x(q: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int),
                    inv_count: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(out, owner[i], q[i][0] * inv_count[owner[i]])


@wp.kernel
def shift_fish(q: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
               owner: wp.array(dtype=int), off: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    o = off[owner[i]]
    q[i] = q[i] + o
    prev[i] = prev[i] + o


# -- the sim ------------------------------------------------------------------

class Sim:
    def __init__(self, fishes, n_fish, device):
        self.device = device
        self.n_fish = n_fish
        self.frame_dt = 1.0 / FPS
        self.variant = deal_variants(n_fish, len(fishes))
        self.fishes = fishes

        L = plan_layout(fishes, n_fish, self.variant)
        self.layout = L
        self.part_base = L.part_base
        self.part_count = L.part_count
        self.n_particles = int(L.n_particles)

        # Spawn every cage: scale about the fish's own origin, yaw about +Z,
        # translate -- the same transform add_soft_mesh applies on the Newton
        # side. Dm_inv and the rest volumes come from the SPAWNED positions, so
        # scale and yaw are already in the rest state.
        p0 = np.empty((self.n_particles, 3), np.float32)
        tets, invm, nsplit = [], np.zeros(self.n_particles, np.float64), np.zeros(self.n_particles, np.float32)
        rad_pair = np.empty(self.n_particles, np.float32)
        rad_env = np.empty(self.n_particles, np.float32)
        dm_inv, rest_vol = [], []
        for k in range(n_fish):
            fish = fishes[int(self.variant[k])]
            b, c = int(L.part_base[k]), int(L.part_count[k])
            yaw = float(L.yaw[k])
            cy, sy = math.cos(yaw), math.sin(yaw)
            v = fish.cverts * float(L.scale[k])
            v = v @ np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], np.float32).T
            p0[b:b + c] = v + L.spawn[k]
            t = fish.ctets.astype(np.int64) + b
            tets.append(t)
            r = p0[t].astype(np.float64)
            Dm = np.stack([r[:, 1] - r[:, 0], r[:, 2] - r[:, 0], r[:, 3] - r[:, 0]], axis=2)
            vol = np.linalg.det(Dm) / 6.0
            dm_inv.append(np.linalg.inv(Dm))
            rest_vol.append(vol)
            # Volume-shaped mass, totalling the MEASURED weight when the
            # parameters know it (density x cage volume is what made 40 kg cod).
            share = np.repeat(np.abs(vol) * 0.25, 4)
            if fish.weight_kg is not None:
                share *= (fish.weight_kg * float(L.scale[k]) ** 3) / max(share.sum(), 1e-12)
            else:
                share *= DENSITY
            np.add.at(invm, t.reshape(-1), share)
            np.add.at(nsplit, t.reshape(-1), 1.0)
            rad_pair[b:b + c] = SELF_R * fish.cell * float(L.scale[k])
            rad_env[b:b + c] = PARTICLE_R * fish.cell * float(L.scale[k])
        tets = np.concatenate(tets).astype(np.int32)
        rest_vol = np.concatenate(rest_vol)
        self.n_tets = len(tets)
        invm = (1.0 / np.maximum(invm, 1e-12)).astype(np.float32)
        nsplit = np.maximum(nsplit, 1.0)

        dt2 = DT * DT
        alpha_d = (1.0 / (MU_LAME * np.abs(rest_vol) * dt2)).astype(np.float32)
        alpha_h = (1.0 / (LAM_LAME * np.abs(rest_vol) * dt2)).astype(np.float32)

        self.x = wp.array(p0, dtype=wp.vec3, device=device)
        self.prev = wp.array(p0, dtype=wp.vec3, device=device)
        self.pred = wp.zeros(self.n_particles, dtype=wp.vec3, device=device)
        self.dpos = wp.zeros(self.n_particles, dtype=wp.vec3, device=device)
        self.tets = wp.array(tets, dtype=int, device=device)
        self.dm_inv = wp.array(np.concatenate(dm_inv).astype(np.float32), dtype=wp.mat33, device=device)
        self.alpha_d = wp.array(alpha_d, dtype=float, device=device)
        self.alpha_h = wp.array(alpha_h, dtype=float, device=device)
        self.lam_d = wp.zeros(self.n_tets, dtype=float, device=device)
        self.lam_h = wp.zeros(self.n_tets, dtype=float, device=device)
        self.invm = wp.array(invm, dtype=float, device=device)
        self.nsplit = wp.array(nsplit, dtype=float, device=device)
        self.rad_pair = wp.array(rad_pair, dtype=float, device=device)
        self.rad_env = wp.array(rad_env, dtype=float, device=device)
        self.owner = wp.array(np.repeat(np.arange(n_fish, dtype=np.int32), self.part_count),
                              dtype=int, device=device)
        self.inv_count = wp.array((1.0 / self.part_count).astype(np.float32),
                                  dtype=float, device=device)
        self.cx = wp.zeros(n_fish, dtype=float, device=device)
        self.shift = wp.zeros(n_fish, dtype=wp.vec3, device=device)
        self.grid = wp.HashGrid(128, 64, 64, device)
        self.cont_idx = wp.zeros(self.n_particles * CONTACT_CAP, dtype=int, device=device)
        self.cont_n = wp.zeros(self.n_particles, dtype=int, device=device)
        self.npush = wp.zeros(self.n_particles, dtype=float, device=device)
        self.max_rad = float(rad_pair.max())

        self.damp_sub = 1.0 - math.exp(-DAMP_RATE * DT)
        self.omega = BELT_SPEED / ROLLER_R
        spacing = L.roller_x[1] - L.roller_x[0]
        self._machine = (float(L.roller_x[0]), float(1.0 / spacing), int(L.n_rollers),
                         float(L.rail_y - 0.02), float(0.10 - L.rail_h), float(0.10 + L.rail_h),
                         float(-0.5), float(L.bed_len - 0.5),
                         int(BELT == "flat"), float(BELT_SPEED))
        cellmax = max(f.cell for f in fishes)
        self.detect_r = 2.0 * SELF_R * cellmax * (1.0 + SCALE_JITTER)
        self.detect_margin = DETECT_MARGIN * cellmax
        self.frames = 0
        self.graph = None
        self.recycle_rng = np.random.default_rng(11)

    def _substep(self):
        dev = self.device
        wp.launch(integrate_damped, dim=self.n_particles, device=dev,
                  inputs=[self.x, self.prev, self.pred, DT, self.damp_sub,
                          wp.vec3(0.0, 0.0, -9.81), self.npush])
        wp.launch(lambda_reset, dim=self.n_tets, device=dev, inputs=[self.lam_d])
        wp.launch(lambda_reset, dim=self.n_tets, device=dev, inputs=[self.lam_h])
        for _ in range(ITERATIONS):
            wp.launch(tet_corot, dim=self.n_tets, device=dev,
                      inputs=[self.pred, self.tets, self.dm_inv, self.alpha_d, self.alpha_h,
                              self.invm, self.nsplit, self.lam_d, self.lam_h, self.dpos])
            wp.launch(apply_delta, dim=self.n_particles, device=dev,
                      inputs=[self.pred, self.dpos, OMEGA_R])
            wp.launch(contacts_solve, dim=self.n_particles, device=dev,
                      inputs=[self.pred, self.prev, self.invm, self.rad_pair, self.rad_env,
                              self.cont_idx, self.cont_n, CONTACT_CAP, CONTACT_RELAX,
                              PAIR_FRICTION / max(ITERATIONS, 1),
                              PAIR_BOUNCE / max(ITERATIONS, 1),
                              *self._machine, self.npush, self.dpos])
            wp.launch(apply_delta, dim=self.n_particles, device=dev,
                      inputs=[self.pred, self.dpos, 1.0])
        wp.launch(machine_friction, dim=self.n_particles, device=dev,
                  inputs=[self.pred, self.prev, self.rad_env, self.npush, MU,
                          self.omega, DT, *self._machine])
        wp.copy(self.x, self.pred)

    def _frame_body(self):
        for _ in range(SUBSTEPS):
            self._substep()

    def _detect(self):
        # Once per frame, outside the graph: rebuild the grid and re-cache the
        # candidate pairs, with a margin no particle outruns in one frame.
        self.grid.build(points=self.x, radius=self.detect_r + self.detect_margin)
        wp.launch(build_contacts, dim=self.n_particles, device=self.device,
                  inputs=[self.x, self.grid.id, self.owner, self.rad_pair, self.max_rad,
                          self.detect_margin, CONTACT_CAP, self.cont_idx, self.cont_n])

    def capture(self):
        try:
            wp.load_module(device=self.device)
            with wp.ScopedCapture(self.device) as cap:
                self._frame_body()
            self.graph = cap.graph
        except Exception as e:  # noqa: BLE001
            self.graph = None
            print(f"  note: CUDA graph capture failed ({e}); launching kernel by kernel")

    def simulate(self):
        self._detect()
        self._frame_body()
        self.frames += 1

    def step(self):
        if self.graph is not None:
            self._detect()
            wp.capture_launch(self.graph)
            self.frames += 1
        else:
            self.simulate()

    def recycle(self):
        """View path only -- ab_run measures displacement and must never see it."""
        self.cx.zero_()
        wp.launch(fish_centroid_x, dim=self.n_particles, device=self.device,
                  inputs=[self.x, self.owner, self.inv_count], outputs=[self.cx])
        done = self.cx.numpy() > self.layout.bed_len - 0.9
        if not done.any():
            return
        off = np.zeros((self.n_fish, 3), np.float32)
        off[done, 0] = -(self.layout.bed_len - 1.4)
        off[done, 1] = self.recycle_rng.uniform(-0.12, 0.12, int(done.sum()))
        wp.copy(self.shift, wp.array(off, dtype=wp.vec3, device=self.device))
        wp.launch(shift_fish, dim=self.n_particles, device=self.device,
                  inputs=[self.x, self.prev, self.owner, self.shift])

    def sim_time(self):
        return self.frames * self.frame_dt

    def positions(self):
        return self.x

    def positions_host(self):
        return self.x.numpy()

    def centroids(self):
        q = self.x.numpy()
        return (np.add.reduceat(q, self.part_base, axis=0)
                / self.part_count[:, None].astype(np.float64))


# -- measurement / viewing ----------------------------------------------------

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

    for _ in range(WARMUP):
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
        "n": n_fish, "tets": sim.n_tets, "particles": sim.n_particles,
        "ms": float(np.median(times)), "ms_p95": float(np.percentile(times, 95)),
        "rt": (1000.0 / FPS) / float(np.median(times)),
        "convey": float(np.mean(dx[onbelt]) / elapsed) if onbelt.any() else float("nan"),
        "belt": BELT_SPEED, "onbelt": int(onbelt.sum()), "vram": used0, "build_s": build_s,
    }
    if verbose:
        print(f"  {n_fish:>4} fish | {r['tets']:>7} tets {r['particles']:>7} parts | "
              f"{r['ms']:7.2f} ms/frame (p95 {r['ms_p95']:6.2f}) | {r['rt']:5.2f}x realtime | "
              f"convey {r['convey']:+.3f} m/s of {BELT_SPEED:.2f} | "
              f"{r['onbelt']}/{n_fish} on belt | {r['vram']:.0f} MB", flush=True)
    return sim, r


def view(sim, fishes, device):
    canvas, renderer, scene, camera, draw = stage(sim, fishes, False, device, "warp fish conveyor")
    controls = tp.OrbitControls(camera, canvas)
    controls.target.set(0.5 * sim.layout.bed_len - 0.5, 0.6 * sim.layout.rail_h, 0.0)
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


def shot(sim, fishes, device, path):
    canvas, renderer, scene, camera, draw = stage(sim, fishes, True, device, "warp fish conveyor")
    for _ in range(int(round(SHOT_TIME * FPS))):
        sim.step()
        if RECYCLE and sim.frames % 12 == 0:
            sim.recycle()
        draw()
        renderer.render(scene, camera)
    renderer.save_frame(path)
    print(f"      simulated {SHOT_TIME:.1f} s, wrote {path}", flush=True)


# -- main ---------------------------------------------------------------------

def main():
    if not FISH:
        sys.exit("--fish PATH is required: there is no bundled model.\n"
                 "  Any closed solid threepp's ModelLoader reads will do; a directory or a\n"
                 "  comma-separated list loads several and deals them out round-robin.\n"
                 "  e.g. python warp_fish_conveyor.py --fish cod.usdz --count 8 --view")
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

    fishes = [Fish(p, device) for p in paths]
    for f in fishes:
        f.report()
    if len(fishes) > 1:
        print(f"      {len(fishes)} scans, dealt round-robin over the fish")
    print(f"mat   E={YOUNG:.3g} Pa, nu={POISSON}, rho={DENSITY} kg/m3 | "
          f"belt {BELT_SPEED} m/s, mu={MU}, rollers r={ROLLER_R} m")
    print(f"solve corotational XPBD (mass-split Jacobi) {ITERATIONS} iters x {SUBSTEPS} "
          f"substeps @ {FPS:.0f} Hz{', CUDA graph' if GRAPH else ''} | cage voxel {TET_RES}")
    print()

    if AB:
        sim = Sim(fishes, COUNT, device)
        for _ in range(WARMUP):
            sim.simulate()
        wp.synchronize_device(device)
        if GRAPH:
            sim.capture()
            sim.step()
            wp.synchronize_device(device)
        r = ab_run(sim, fishes, device, AB_SECONDS, "warp-corot", SEED,
                   "warp fish conveyor", shot=SHOT or None, shot_at=SHOT_TIME, warmup=0)
        fc.ab_report(r)
        return

    if VIEW or SHOT:
        sim = Sim(fishes, COUNT, device)
        for _ in range(WARMUP):
            sim.simulate()
        wp.synchronize_device(device)
        if GRAPH:
            sim.capture()
        drawn = sum(len(f.render_surface()[1]) for f in fishes) * COUNT // len(fishes)
        print(f"      {COUNT} fish, {sim.n_tets} tets simulated, "
              f"{drawn} triangles drawn ({'cage' if CAGE else 'scan'})", flush=True)
        if SHOT:
            shot(sim, fishes, device, SHOT)
        else:
            print("      drag to orbit, Esc quits", flush=True)
            view(sim, fishes, device)
        return

    run(fishes, COUNT, device)


if __name__ == "__main__":
    main()
