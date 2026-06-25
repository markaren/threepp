"""threepp.urdf — import a URDF into a PhysX articulation you can simulate.

    import threepp as tp
    from threepp.urdf import load_articulation

    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81))
    robot = load_articulation(world, "arm.urdf", fixed_base=True,
                              stiffness=200, damping=20, max_force=300)
    for m in robot.meshes:
        scene.add(m)                       # render the link bodies (bound to the sim)
    robot.set_targets(target_angles)       # drive the joints (add order)
    world.step(dt)

A thin wrapper over the C++ URDF articulation loader (threepp's `URDFLoader` → a reduced-coordinate
`Articulation`) — ONE parser shared with C++, xacro supported. It builds the kinematic tree, places each
link by forward kinematics, and creates the articulation:

  * collision: `<box>` / `<sphere>` / `<cylinder>` (→ capsule) directly; `<mesh>` by its **bounding box**
    (articulation links take primitive shapes, not trimeshes).
  * mass: from `<inertial>` if present, else `default_density` × shape volume.
  * joints: revolute / prismatic become DOFs (with limits + an optional PD drive); `fixed` joints are
    collapsed into their parent frame.
  * with `render_visuals` (default), each link's `<visual><mesh>` is parented under its (hidden) collider
    so the robot renders as its real geometry while the primitive colliders drive the physics.

It's an approximation (primitive/bbox collision), not a digital twin. Needs a PhysX-enabled threepp
build (`tp.HAS_PHYSX`).
"""
import os

import numpy as np

from . import threepp as tp


class UrdfArticulation:
    """Handle to a URDF imported as a PhysX `Articulation` (the `.articulation`), plus its collider
    meshes (`.meshes`, bound to the sim — add them to a scene) and actuated `.joint_names`."""

    def __init__(self, articulation, meshes, joint_names, world):
        self.articulation = articulation
        self.meshes = meshes
        self.joint_names = joint_names       # actuated joints, in drive-target order
        self._world = world                  # the articulation holds a C++ ref to the world: keep it alive

    @property
    def num_dof(self):
        return len(self.joint_names)

    def set_targets(self, values):
        self.articulation.set_drive_targets(np.asarray(values, np.float32))

    def positions(self):
        return self.articulation.joint_positions()

    def velocities(self):
        return self.articulation.joint_velocities()


def load_articulation(world, path, *, fixed_base=False, base_position=(0, 0, 0),
                      default_density=1000.0, stiffness=0.0, damping=0.0, max_force=1e6,
                      self_collision=False, solver_position_iterations=12, render_visuals=True):
    """Import a URDF/xacro as a reduced-coordinate `Articulation`. Returns a UrdfArticulation.

    The articulation is built at the zero joint configuration. `stiffness`/`damping`/`max_force`
    configure a PD position drive on every actuated joint (leave stiffness 0 for passive/force-controlled
    joints). `base_position` places the root link.
    """
    if not tp.HAS_PHYSX:
        raise RuntimeError("threepp.urdf.load_articulation needs a PhysX-enabled threepp build")
    art, meshes, joint_names = world.load_articulation(
        os.fspath(path),
        fixed_base=fixed_base,
        base_position=tuple(float(x) for x in base_position),
        default_density=float(default_density),
        stiffness=float(stiffness),
        damping=float(damping),
        max_force=float(max_force),
        self_collision=self_collision,
        solver_position_iterations=int(solver_position_iterations),
        render_visuals=render_visuals)
    return UrdfArticulation(art, meshes, joint_names, world)
