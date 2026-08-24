"""The same fish conveyor, on PhysX 5 deformable volumes instead of Newton.

The A/B twin of newton_fish_conveyor.py. Same scans, same auto-orientation, same
belt, same rollers, same lighting, same camera, same GPU skinning of the same
decimated render surface, same quality metrics -- all of it out of
fish_conveyor_common.py. The solver is the only difference, which is the whole
point: run both, and whatever the numbers disagree about is the solver.

    python physx_fish_conveyor.py --fish scans/ --count 60 --view
    python physx_fish_conveyor.py --fish scans/ --count 16 --shot catch.png
    python physx_fish_conveyor.py --fish scans/ --count 60 --ab      # one JSON line
    python physx_fish_conveyor.py --fish scans/ --dump-tets tets.npz # for --tets physx

WHAT PHYSX SIMULATES is two meshes per fish, not one. PxTetMaker cooks the
surface into a CONFORMING tet mesh (it follows the skin, and it is what contact
is resolved against) plus a VOXELISED simulation mesh the solver integrates,
with the conforming vertices skinned to the voxel ones. --voxel-res sizes the
second. That is a different bargain from Newton's, where one voxel cage is both,
and it is the main reason the two engines do not cost the same per fish: PhysX
is carrying a finer collision surface for the same solver work.

THE CAGE THIS BENCHMARK REPORTS IS PHYSX'S CONFORMING MESH. The render surface is
bound to it barycentrically by the shared binder, exactly as the Newton run binds
to its voxel cage, and the quality metrics are computed on it. So "stretch 1.6x"
here and "stretch 1.6x" there are the same measurement on two different meshes --
which is precisely why newton_fish_conveyor.py --tets physx exists: that arm runs
Newton on THIS cage, and then the meshes are identical too.

ROLLERS ARE CAPSULES, not the 24-gon convex prisms a naive port would use. PhysX
has no cylinder primitive, but a capsule's cylindrical section IS a cylinder, and
its hemispherical caps sit 50 mm outside rails the fish cannot pass anyway. That
matters more than it sounds: a prism rotating about its axis sweeps a different
volume every step, so the fish would ride a 0.4 mm ripple that Newton's true
cylinder does not have -- and the whole comparison is about which solver behaves,
not which one was handed a bumpier belt.

They are driven the way a kinematic body has to be driven: a fresh
setKinematicTarget every SUBSTEP (--roller-drive substep), because PhysX derives
a kinematic actor's velocity from (target - pose) / dt and a target set once per
frame leaves three of four substeps with a stationary belt. That is ~300 bound
calls per substep and it is not free; --roller-drive frame shows what the
shortcut buys and costs.

MASS IS PROBED, NOT ASSUMED. PhysX takes a total mass in kg, and the tet volume
it would use for a density is only knowable after the cook -- so the first body of
each species is spawned, its cooked tet volume measured, removed, and the real
ones spawned at density x that volume. The cook cache makes the second spawn
free. Getting this wrong is not cosmetic: at PhysX's default unit density a 1.4 m
cod weighs five grams, and a five-gram fish against a 3 MPa flesh modulus is a
fish made of steel.

Needs a CUDA GPU: PhysX cooks and solves deformable volumes on the device, so
the world is built with gpu_dynamics=True and there is no CPU fallback.
"""
import math
import os
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fish_conveyor_common as fc
from fish_conveyor_common import (Fish, ab_run, cli_arg, deal_variants, fish_paths,
                                  plan_layout, stage)
# After fish_conveyor_common, which is what puts python/ on sys.path.
import threepp as tp  # noqa: E402

if not tp.HAS_PHYSX:
    sys.exit("this build of threepp has no PhysX backend (threepp.HAS_PHYSX is False)")


FISH = cli_arg("--fish", "", str)            # required -- see the module docstring
VARIANTS = cli_arg("--variants", 4, int)     # distinct scans to cook cages for
COUNT = cli_arg("--count", 1, int)
SWEEP = "--sweep" in sys.argv
SWEEP_TO = cli_arg("--sweep-to", 64, int)
VOXEL_RES = cli_arg("--voxel-res", 10, int)  # PhysX simulation-mesh resolution
SOLVER_ITERS = cli_arg("--solver-iters", 20, int)   # PhysX's own shipped default
SUBSTEPS = cli_arg("--substeps", 1, int)
# 1, which is PhysX's own shipped rate: a deformable volume is meant to be
# simulated once per 60 Hz frame and to buy its stability from solver iterations
# (20 of them) rather than from substepping. Newton's VBD defaults to 4 because
# it has to. Measured at 16 fish: 8.8 ms at 1 substep, 14.7 at 2, 26.3 at 4 --
# almost exactly linear -- and what the extra substeps buy is bending stiffness
# (fish bbox length 0.62 of rest at 1 substep, 0.84 at 4), not integrity: volume
# ratio and edge stretch are already 1.00 and 1.01 at one substep.
FRAMES = cli_arg("--frames", 120, int)
WARMUP = cli_arg("--warmup", 90, int)
VIEW = "--view" in sys.argv
SHOT = cli_arg("--shot", "", str)
SHOT_TIME = cli_arg("--shot-time", 2.5, float)
CAGE = "--cage" in sys.argv                  # draw the cooked tet cage, not the scan
RENDER_RATIO = cli_arg("--render-ratio", 0.4, float)
LANES = cli_arg("--lanes", 5, int)
BED = cli_arg("--bed", 30.0, float)
LAYERS = cli_arg("--layers", 2, int)
SELF_CONTACT = "--no-self-contact" not in sys.argv
BELT_SPEED = cli_arg("--speed", 0.5, float)
ROLLER_R = cli_arg("--roller-radius", 0.05, float)
MU = cli_arg("--mu", 0.45, float)
YOUNG = cli_arg("--young", 1.0e5, float)     # same as the Newton run's default
POISSON = cli_arg("--poisson", 0.45, float)
DENSITY = cli_arg("--density", 1050.0, float)
FISH_LENGTH = cli_arg("--fish-length", 0.0, float)
# Physical ground truth (see warp_fish_conveyor.py): "auto" finds the FHF
# database's Physical_Parameters/parameters.json next to the scans, rescales
# every scan to its MEASURED length and gives it its MEASURED weight. The scans
# arrive 2.5x life size, and density x cooked volume made 40 kg cod.
PARAMS = cli_arg("--params", "auto", str)
SCALE_JITTER = cli_arg("--scale-jitter", 0.0, float)
# 0, and not by accident. Newton scales each soft mesh at spawn for free; PhysX's
# cook cache keys on the source geometry and applies the spawn's rotation and
# translation only, so a per-fish scale would mean one cook per fish. Both engines
# are therefore run at 0 for the A/B and keep the yaw and position jitter, which
# is where the contact-load variety actually comes from.
SEED = cli_arg("--seed", 7, int)
INTEROP = "--no-interop" not in sys.argv
AB = "--ab" in sys.argv
AB_SECONDS = cli_arg("--ab-seconds", 8.0, float)
DUMP_TETS = cli_arg("--dump-tets", "", str)  # write the cooked cages and exit
ROLLER_DRIVE = cli_arg("--roller-drive", "substep", str)   # substep | frame
DEVICE = cli_arg("--device", "cuda:0", str)  # the Warp device for skinning/metrics
FPS = 60.0

if PARAMS == "auto":
    PARAMS = fc.find_params(FISH) if FISH else ""
elif PARAMS == "none":
    PARAMS = ""

fc.configure(FPS=FPS, RENDER_RATIO=RENDER_RATIO, CAGE=CAGE, FISH_LENGTH=FISH_LENGTH,
             PARAMS=PARAMS,
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


def tet_volume(verts, tets):
    t = np.asarray(verts, np.float64)[tets]
    return float(abs(np.einsum("ij,ij->i", np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                               t[:, 3] - t[:, 0]).sum()) / 6.0)


class Cooker:
    """Cooks a scan into PhysX's tet mesh, in a world that outlives the call.

    PhysX allows one PxFoundation per process, so the world that answers "what
    tets would you make of this?" has to be the same world that then simulates
    them -- which is why this is a small stateful object rather than a function.
    Each probe is spawned, measured, and removed; the cook cache means the real
    spawn afterwards pays nothing.
    """

    def __init__(self, world, material, voxel_res):
        self.world = world
        self.material = material
        self.voxel_res = voxel_res
        self.geometry = {}      # variant key -> the shared render-surface geometry
        self.mass = {}          # variant key -> kg, from the cooked tet volume

    def template_geometry(self, fish):
        """The cook source: the decimated render surface, at the origin.

        PxTetMaker remeshes the input to --voxel-res before it tetrahedralises,
        so feeding it the scan's full 25k triangles buys nothing and costs
        minutes. The decimated surface is also what gets DRAWN, which keeps the
        cage and the skin describing the same fish.
        """
        key = fish.path
        if key not in self.geometry:
            g = tp.BufferGeometry()
            g.set_attribute("position", np.ascontiguousarray(fish.rverts, np.float32))
            g.set_index(np.ascontiguousarray(fish.rfaces, np.uint32).reshape(-1))
            self.geometry[key] = g
        return self.geometry[key]

    def cook(self, fish):
        """(rest vertices, tets) of the conforming collision mesh for this scan."""
        g = self.template_geometry(fish)
        mesh = tp.Mesh(g, tp.MeshStandardMaterial())
        t0 = time.perf_counter()
        # The cache key carries the fish's length: the parameters file rescales
        # the scan, and a cache keyed on the basename alone would happily serve
        # the cook of the OLD size.
        probe = self.world.add_soft_body(mesh, material=self.material,
                                         voxel_resolution=self.voxel_res,
                                         solver_iterations=SOLVER_ITERS,
                                         self_collision=SELF_CONTACT,
                                         cache_key=f"{os.path.basename(fish.path)}"
                                                   f"_{fish.extent[0]:.4f}")
        verts, tets = probe.tet_mesh()
        if fish.weight_kg is not None:
            self.mass[fish.path] = fish.weight_kg
        else:
            self.mass[fish.path] = DENSITY * tet_volume(verts, tets)
        self.world.remove_soft_body(probe)
        print(f"cook  {os.path.basename(fish.path)} -> {len(verts)} verts, {len(tets)} tets, "
              f"{self.mass[fish.path]:.2f} kg "
              f"{'measured' if fish.weight_kg is not None else f'at {DENSITY:.0f} kg/m3'} "
              f"({time.perf_counter() - t0:.2f} s)", flush=True)
        return verts, tets


class Sim:
    """The PhysX side: one world, a bed of kinematic capsules, and the catch."""

    def __init__(self, fishes, n_fish, device, cooker):
        self.device = device
        self.n_fish = n_fish
        self.fishes = fishes
        self.frame_dt = 1.0 / FPS
        self.variant = deal_variants(n_fish, len(fishes))
        self.world = cooker.world
        L = plan_layout(fishes, n_fish, self.variant)
        self.layout = L
        self.part_base = L.part_base
        self.part_count = L.part_count
        self.omega = BELT_SPEED / ROLLER_R
        self.frames = 0
        self.readback_s = 0.0

        # --- roller bed: kinematic capsules, axis along Y, tops at z = 0 ---
        # A capsule, not a convex prism: its cylindrical section is an exact
        # cylinder, so a fish rides the same surface Newton's cylinder gives it.
        # The caps stick out 50 mm past rails the fish cannot reach.
        belt_mat = self.world.create_material(static_friction=MU, dynamic_friction=MU,
                                              restitution=0.0, friction_combine="min")
        roller_geo = tp.CapsuleGeometry(ROLLER_R, 2.0 * L.half_w)
        self.roller_bodies = []
        self.roller_pos = []
        for x in L.roller_x:
            m = tp.Mesh(roller_geo, tp.MeshStandardMaterial())
            m.position.set(x, 0.0, -ROLLER_R)
            b = self.world.add(m, density=1000.0, material=belt_mat)
            b.set_kinematic(True)
            self.roller_bodies.append(b)
            self.roller_pos.append(tp.Vector3(x, 0.0, -ROLLER_R))

        # --- side rails, and a floor well below to catch anything that leaves ---
        rail_mat = self.world.create_material(static_friction=0.2, dynamic_friction=0.2,
                                              restitution=0.0, friction_combine="min")
        mid_x = 0.5 * L.bed_len - 0.5
        for s in (-1.0, 1.0):
            rail = tp.Mesh(tp.BoxGeometry(L.bed_len, 0.04, 2.0 * L.rail_h),
                           tp.MeshStandardMaterial())
            rail.position.set(mid_x, s * L.rail_y, 0.10)
            self.world.add_static(rail, material=rail_mat)
        floor = tp.Mesh(tp.BoxGeometry(80.0, 80.0, 0.2), tp.MeshStandardMaterial())
        floor.position.set(mid_x, 0.0, -1.1)
        self.world.add_static(floor, material=rail_mat)

        # --- the fish ---
        self.bodies = []
        for k in range(n_fish):
            fish = fishes[int(self.variant[k])]
            mesh = tp.Mesh(cooker.template_geometry(fish), tp.MeshStandardMaterial())
            mesh.position.set(*(float(c) for c in L.spawn[k]))
            mesh.rotate_z(float(L.yaw[k]))
            sb = self.world.add_soft_body(mesh, material=cooker.material,
                                          voxel_resolution=cooker.voxel_res,
                                          solver_iterations=SOLVER_ITERS,
                                          self_collision=SELF_CONTACT,
                                          cache_key=os.path.basename(fish.path),
                                          mass=cooker.mass[fish.path])
            # The visual meshes here are never added to a scene -- the render
            # surface is skinned by the shared Warp path, exactly as Newton's is.
            # GPU skinning is still the right switch: it drops the per-step CPU
            # skin, normal recompute and bounds pass over ~7k vertices per fish
            # down to one few-hundred-texel write, which is the honest cost of a
            # solver whose visual someone else owns.
            sb.enable_gpu_skinning()
            self.bodies.append(sb)
            if len(self.part_count) > k and self.part_count[k] != sb.num_vertices:
                raise RuntimeError(
                    f"fish {k}: cage has {self.part_count[k]} vertices but PhysX cooked "
                    f"{sb.num_vertices} -- the Fish cage and the spawned body disagree")

        self.host = np.zeros((int(L.n_particles), 3), np.float32)
        self.angle = 0.0
        if ROLLER_DRIVE == "substep":
            self.world.on_pre_substep(self._drive)
        elif ROLLER_DRIVE != "frame":
            sys.exit(f"--roller-drive: expected 'substep' or 'frame', got {ROLLER_DRIVE!r}")

    def _drive(self, dt):
        """Advance every roller's kinematic target by omega*dt about its own axis.

        PhysX reads a kinematic actor's velocity out of (target - pose) / dt, so
        the target has to be reset every step it is expected to keep turning.
        The pose is unchanged apart from the spin: a capsule about its own axis
        is the one motion that never sweeps new volume.
        """
        self.angle = (self.angle + self.omega * dt) % (2.0 * math.pi)
        q = tp.Quaternion()
        q.set_from_axis_angle(tp.Vector3(0.0, 1.0, 0.0), self.angle)
        for b, p in zip(self.roller_bodies, self.roller_pos):
            b.set_kinematic_target(p, q)

    def step(self):
        self.frames += 1
        if ROLLER_DRIVE == "frame":
            self._drive(self.frame_dt)
        self.world.step(self.frame_dt)
        t0 = time.perf_counter()
        for k, sb in enumerate(self.bodies):
            b = int(self.part_base[k])
            self.host[b:b + int(self.part_count[k])] = sb.sim_positions()
        self.readback_s += time.perf_counter() - t0

    def sim_time(self):
        return self.frames * self.frame_dt

    def positions(self):
        """Host-side, because that is the only side PhysX offers here.

        The tet positions live in GPU memory PhysX owns; there is a zero-copy
        bridge into the renderer's own buffers (PhysxWorld's Vulkan/GL interop)
        but not one into a foreign CUDA context, which is what Warp is. So this
        is a device->host copy per fish per frame, and its cost is reported
        separately -- it is a real difference between the two engines, not a
        measurement artefact, but it is not the solver either.
        """
        return self.host

    def positions_host(self):
        return self.host

    def centroids(self):
        return (np.add.reduceat(self.host.astype(np.float64), self.part_base, axis=0)
                / self.part_count[:, None].astype(np.float64))


def build(fishes_paths, n_fish, device):
    """One world, cooked cages, and a Sim on top. Returns (sim, fishes, cooker)."""
    world = tp.PhysxWorld(gravity=tp.Vector3(0.0, 0.0, -9.81),
                          fixed_timestep=1.0 / (FPS * SUBSTEPS),
                          max_substeps=SUBSTEPS, gpu_dynamics=True)
    material = world.create_soft_body_material(young=YOUNG, poisson=POISSON, friction=MU)
    cooker = Cooker(world, material, VOXEL_RES)
    fishes = [Fish(p, device, tets=cooker.cook) for p in fishes_paths]
    for f in fishes:
        f.report()
    if len(fishes) > 1:
        print(f"      {len(fishes)} scans, dealt round-robin over the fish")
    print(f"mat   E={YOUNG:.3g} Pa, nu={POISSON}, rho={DENSITY} kg/m3 | "
          f"belt {BELT_SPEED} m/s, mu={MU}, rollers r={ROLLER_R} m (capsules)")
    print(f"solve PhysX deformable volumes, {SOLVER_ITERS} iters x {SUBSTEPS} substeps "
          f"@ {FPS:.0f} Hz, voxel-res {VOXEL_RES}"
          f"{', self-collision' if SELF_CONTACT else ''} | cage {fishes[0].cage_kind}")
    print()
    if n_fish is None:
        return None, fishes, cooker
    return Sim(fishes, n_fish, device, cooker), fishes, cooker


def view(sim, fishes, device):
    canvas, renderer, scene, camera, draw = stage(sim, fishes, False, device,
                                                  "physx fish conveyor")
    controls = tp.OrbitControls(camera, canvas)
    controls.target.set(0.5 * sim.layout.bed_len - 0.5 - 0.02 * sim.layout.bed_len,
                        0.6 * sim.layout.rail_h, 0.0)
    controls.enable_damping = True

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)

    canvas.on_window_resize(on_resize)

    def animate():
        sim.step()
        draw()
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)


def shot(sim, fishes, device, path):
    canvas, renderer, scene, camera, draw = stage(sim, fishes, True, device,
                                                  "physx fish conveyor")
    for _ in range(int(round(SHOT_TIME * FPS))):
        sim.step()
        draw()
        renderer.render(scene, camera)      # arms the GL interop on frame one
    renderer.save_frame(path)
    print(f"      simulated {SHOT_TIME:.1f} s, wrote {path}", flush=True)


def main():
    if not FISH:
        sys.exit("--fish PATH is required: there is no bundled model. A file, a\n"
                 "  directory of scans, or a comma-separated list -- anything\n"
                 "  threepp's ModelLoader reads.")
    paths = fish_paths(FISH)
    if not paths:
        sys.exit(f"--fish: nothing to load from {FISH}")
    for p in paths:
        if not os.path.isfile(p):
            sys.exit(f"--fish: no such file: {p}")
    paths = paths[:max(1, VARIANTS)]

    device = wp.get_device(DEVICE)

    if DUMP_TETS:
        # Cook only, then hand the cages to whoever asked (newton_fish_conveyor.py
        # --tets physx runs this in a subprocess so PhysX's foundation and CUDA
        # context die before Warp builds its own).
        _, fishes, _ = build(paths, None, device)
        np.savez(DUMP_TETS, **{f"v{i}": f.cverts for i, f in enumerate(fishes)},
                 **{f"t{i}": f.ctets for i, f in enumerate(fishes)})
        print(f"      wrote {len(fishes)} cooked cages to {DUMP_TETS}", flush=True)
        return

    counts = [n for n in (1, 4, 16, 64, 128) if n <= SWEEP_TO] if SWEEP else [COUNT]
    for n in counts:
        # One world per count, and the old one has to be GONE before the next is
        # built: PhysX allows a single PxFoundation per process, so a sweep that
        # leaks a world dies on the second entry rather than the last.
        sim, fishes, cooker = build(paths, n, device)
        drawn = sum(len(f.render_surface()[1]) for f in fishes) * n // len(fishes)
        print(f"      {n} fish, {sum(len(f.ctets) for f in fishes) * n // len(fishes)} "
              f"tets simulated, {drawn} triangles drawn "
              f"({'cage' if CAGE else 'scan'})", flush=True)

        if AB:
            for _ in range(WARMUP):
                sim.step()
            r = ab_run(sim, fishes, device, AB_SECONDS, "physx", SEED,
                       "physx fish conveyor", shot=SHOT or None, shot_at=SHOT_TIME,
                       warmup=0)
            r["readback_ms"] = sim.readback_s * 1000.0 / max(sim.frames, 1)
            fc.ab_report(r)
        elif SHOT:
            shot(sim, fishes, device, SHOT)
        elif VIEW:
            print("      drag to orbit, Esc quits", flush=True)
            view(sim, fishes, device)
        else:
            for _ in range(WARMUP):
                sim.step()
            r = ab_run(sim, fishes, device, FRAMES / FPS, "physx", SEED,
                       "physx fish conveyor", warmup=0)
            r["readback_ms"] = sim.readback_s * 1000.0 / max(sim.frames, 1)
            fc.ab_report(r)
        if len(counts) > 1:
            del sim, fishes, cooker
            import gc
            gc.collect()


if __name__ == "__main__":
    main()
