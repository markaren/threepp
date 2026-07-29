"""
The runtime face of editor-authored data.
"""
from __future__ import annotations
import typing
import threepp
__all__: list[str] = ['Articulation', 'RigidBody', 'SoftBody', 'SplinePath', 'articulation_from_object', 'rigid_body_from_object', 'soft_body_from_object', 'spline_from_object']
class SplinePath:
    def get_point_at(self, u: typing.SupportsFloat | typing.SupportsIndex) -> threepp.Vector3:
        """
        WORLD-SPACE point at fraction u in [0, 1] of the arc length.
        """
    def get_tangent_at(self, u: typing.SupportsFloat | typing.SupportsIndex) -> threepp.Vector3:
        """
        WORLD-SPACE unit tangent at fraction u of the arc length.
        """
    def get_length(self) -> float:
        """
        Arc length in the spline's LOCAL space.
        """
    def refresh(self) -> None:
        """
        Re-capture the control points and config. The world transform is live regardless.
        """
    @property
    def closed(self) -> bool:
        ...
    @property
    def tension(self) -> float:
        ...
    @property
    def curve_type(self) -> threepp.CatmullRomCurve3.CurveType:
        ...
    @property
    def curve(self) -> threepp.CatmullRomCurve3:
        """
        The captured LOCAL-SPACE CatmullRomCurve3.
        """
class RigidBody:
    """
    A body PhysX is simulating for a scene object. Only exists during Play, and
    only in a build with the PhysX SDK.
    """
    @property
    def object(self) -> threepp.Object3D:
        """
        The scene object this body governs.
        """
    @property
    def valid(self) -> bool:
        """
        False once the play session that created it has stopped.
        """
    @property
    def is_static(self) -> bool:
        ...
    @property
    def is_kinematic(self) -> bool:
        ...
    @property
    def position(self) -> threepp.Vector3:
        """
        WORLD-SPACE position of the body itself.
        """
    @property
    def rotation(self) -> threepp.Quaternion:
        """
        WORLD-SPACE orientation of the body itself.
        """
    @property
    def velocity(self) -> threepp.Vector3:
        """
        Linear velocity in m/s. Dynamic bodies only.
        """
    @velocity.setter
    def velocity(self, value: threepp.Vector3) -> None:
        ...
    @property
    def angular_velocity(self) -> threepp.Vector3:
        """
        Angular velocity in rad/s. Dynamic bodies only.
        """
    @angular_velocity.setter
    def angular_velocity(self, value: threepp.Vector3) -> None:
        ...
    @property
    def mass(self) -> float:
        """
        Mass in kg; setting it recomputes the inertia tensor.
        """
    @mass.setter
    def mass(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sleeping(self) -> bool:
        """
        True when the solver has parked this body.
        """
    def apply_force(self, force: threepp.Vector3) -> None:
        """
        Add a force in newtons for the coming step. Call it every update while the force should act - it is not a setting.
        """
    def apply_impulse(self, impulse: threepp.Vector3) -> None:
        """
        Add an instantaneous impulse in newton-seconds.
        """
    def apply_torque(self, torque: threepp.Vector3) -> None:
        """
        Add a torque in newton-metres for the coming step.
        """
    def apply_torque_impulse(self, torque: threepp.Vector3) -> None:
        """
        Add an instantaneous angular impulse.
        """
    def set_kinematic_target(self, position: threepp.Vector3, rotation: threepp.Quaternion | None = None) -> None:
        """
        Where a Kinematic body should be by the end of the next step. PhysX sweeps it there, so it pushes dynamics on the way instead of teleporting through them. Keeps the current orientation when rotation is None.
        """
    def wake_up(self) -> None:
        """
        Take the body out of sleep.
        """
class SoftBody:
    """
    A deformable volume PhysX is simulating for a scene object. Only exists
    during Play, and only in a build with the PhysX SDK.
    """
    @property
    def object(self) -> threepp.Object3D:
        """
        The scene object whose mesh this body deforms.
        """
    @property
    def valid(self) -> bool:
        """
        False once the play session that created it has stopped.
        """
    @property
    def center(self) -> threepp.Vector3:
        """
        WORLD-SPACE centre of the deformed body. The object's own position is zero throughout Play - the mesh carries world-space vertices - so this is how a script follows it.
        """
    @property
    def bounds_min(self) -> threepp.Vector3:
        ...
    @property
    def bounds_max(self) -> threepp.Vector3:
        ...
    @property
    def vertex_count(self) -> int:
        ...
    @property
    def recompute_normals(self) -> bool:
        """
        Recompute vertex normals every step (on by default). The bulk of a soft body's CPU cost, and pointless for a flat-shaded body.
        """
    @recompute_normals.setter
    def recompute_normals(self, value: bool) -> None:
        ...
class Articulation:
    """
    A robot PhysX is simulating as a reduced-coordinate articulation. Only
    exists during Play, and only in a build with the PhysX SDK.
    """
    @property
    def object(self) -> threepp.Object3D:
        """
        The Robot this articulation simulates - the robot itself, even when the handle was asked for from one of its links.
        """
    @property
    def valid(self) -> bool:
        """
        False once the play session that created it has stopped.
        """
    @property
    def joint_names(self) -> list[str]:
        """
        The simulated DOFs, in the articulation's own order. Fixed URDF joints are collapsed, so this is NOT the visual Robot's joint order - match by name.
        """
    @property
    def num_dof(self) -> int:
        ...
    @property
    def joint_positions(self) -> list[float]:
        """
        Joint positions in joint_names order: radians for a revolute joint, metres for a prismatic one.
        """
    @property
    def joint_velocities(self) -> list[float]:
        """
        Joint velocities in joint_names order: rad/s or m/s.
        """
    def set_drive_targets(self, targets: list[float]) -> None:
        """
        One PD setpoint per DOF, in joint_names order. The drive authored in the Articulation section pulls each joint toward its target over the coming steps - a setpoint, not a teleport, and inert with zero authored stiffness.
        """
    @typing.overload
    def set_drive_target(self, joint: str, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        PD setpoint for one DOF, by its URDF joint name.
        """
    @typing.overload
    def set_drive_target(self, index: typing.SupportsInt | typing.SupportsIndex, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        PD setpoint for one DOF, by its index in joint_names.
        """
    @property
    def root_position(self) -> threepp.Vector3:
        """
        WORLD-SPACE position of the root link.
        """
    @property
    def root_rotation(self) -> threepp.Quaternion:
        """
        WORLD-SPACE orientation of the root link.
        """
    @property
    def root_velocity(self) -> threepp.Vector3:
        """
        Root link linear velocity in m/s, world frame.
        """
    @property
    def root_angular_velocity(self) -> threepp.Vector3:
        """
        Root link angular velocity in rad/s, world frame.
        """
def articulation_from_object(object: threepp.Object3D | None) -> Articulation | None:
    """
    The Articulation PhysX is simulating for `object`, or None when Play is not running or no articulated robot governs it. The lookup walks up the scene graph, so a script on any link of a robot finds the robot's articulation. Robots simulate only when their Articulation section says Simulate.
    """
def rigid_body_from_object(object: threepp.Object3D | None) -> RigidBody | None:
    """
    The RigidBody PhysX is simulating for `object`, or None when Play is not running or the object has no physics. The lookup walks up the scene graph, so a script on a child finds the body governing it.
    """
def soft_body_from_object(object: threepp.Object3D | None) -> SoftBody | None:
    """
    The SoftBody PhysX is simulating for `object`, or None when Play is not running or the object is not a soft body.
    """
def spline_from_object(object: threepp.Object3D | None) -> SplinePath | None:
    """
    The SplinePath an authored spline describes, or None when `object` is not a spline or has fewer than two control points.
    """
