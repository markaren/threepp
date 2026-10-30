"""threepp.usd — import a USD robot into a PhysX articulation you can simulate.

    import threepp as tp
    from threepp.usd import load_articulation

    world = tp.PhysxWorld()
    robot = load_articulation(world, "crx10ia_l.usd")
    for m in robot.meshes:
        scene.add(m)                       # render the link bodies (bound to the sim)
    robot.set_targets(target_angles)       # drive the joints (parents first)
    world.step(dt)

A thin wrapper over the C++ `loadUsdArticulation`. The file decides what it authors, and an argument
left `None` takes the file's value: whether the base is fixed, drive gains and effort limits, joint
limits, mass, centre of mass and principal inertia, one convex hull per collider mesh, and the solver
iterations, self-collision and joint speed limits of a `PhysX` variant, which is selected. A Z-up file
comes in Y-up, matching `tp.PhysxWorld`'s default gravity.

Revolute and prismatic joints become DOFs; a fixed joint between bodies stays a link with its own mass.
A PhysxMimicJointAPI coupling becomes a PhysX mimic joint, so a gripper's follower fingers close with the
finger its drive moves; `robot.mimics` lists them. `robot.diagnostics` lists what the file holds that
could not be imported: spherical and D6 joints, kinematic loops, a decomposition collider cooked as one
hull.

Needs a build with PhysX and `THREEPP_WITH_USD` (`tp.HAS_USD`).
"""
import os

from . import threepp as tp
from .urdf import UrdfArticulation


class UsdArticulation(UrdfArticulation):
    """Handle to a USD robot imported as a PhysX `Articulation`: `.articulation`, `.meshes` (bound to the
    sim, add them to a scene), actuated `.joint_names`, `.diagnostics`, `.scene`, whatever the file
    holds outside the robot's bodies, and `.mimics`, the couplings built: one dict per coupling with
    `joint`, `reference_joint`, `link` and `reference_link` (indices into `.meshes`), `gearing`,
    `offset`, `natural_frequency` and `damping_ratio`, holding q + gearing * q_ref + offset = 0 in
    radians and metres."""

    def __init__(self, articulation, meshes, joint_names, diagnostics, scene, mimics, world):
        super().__init__(articulation, meshes, joint_names, world)
        self.diagnostics = diagnostics
        self.scene = scene
        self.mimics = mimics


def load_articulation(world, path, *, fixed_base=None, self_collision=None, solver_position_iterations=None,
                      stiffness=None, damping=None, max_force=None, base_position=(0, 0, 0),
                      base_rotation=(0, 0, 0, 1), default_density=1000.0, render_visuals=True,
                      drive_limits_are_forces=True):
    """Import a USD robot as a reduced-coordinate `Articulation`. Returns a UsdArticulation.

    The articulation is in the pose the file authors, each drive on the file's target position.
    `stiffness`/`damping`/`max_force` override every drive, in N·m/rad or N/m: USD authors angular
    gains per degree, and these are not. `base_position` and `base_rotation` (x, y, z, w) move the
    whole robot from where the file puts it. `default_density` applies to links with neither mass
    nor density authored. `drive_limits_are_forces` enforces a drive's max force as a force rather
    than as PhysX's per-substep impulse.
    """
    if not tp.HAS_USD:
        raise RuntimeError("threepp.usd.load_articulation needs a threepp build with PhysX and THREEPP_WITH_USD")
    art, meshes, joint_names, diagnostics, scene, mimics = world.load_usd_articulation(
        os.fspath(path),
        fixed_base=fixed_base,
        self_collision=self_collision,
        solver_position_iterations=solver_position_iterations,
        stiffness=stiffness,
        damping=damping,
        max_force=max_force,
        base_position=tuple(float(x) for x in base_position),
        base_rotation=tuple(float(x) for x in base_rotation),
        default_density=float(default_density),
        render_visuals=render_visuals,
        drive_limits_are_forces=drive_limits_are_forces)
    return UsdArticulation(art, meshes, joint_names, diagnostics, scene, mimics, world)
