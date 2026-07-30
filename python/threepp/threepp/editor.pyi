"""
The runtime face of editor-authored data.
"""
from __future__ import annotations
import typing
import threepp
__all__: list[str] = ['Articulation', 'Collision', 'Contact', 'ContactSample', 'Encoder', 'EncoderSample', 'ForceTorque', 'Imu', 'ImuSample', 'RaycastHit', 'RigidBody', 'SoftBody', 'SplinePath', 'WrenchSample', 'add', 'articulation_from_object', 'contact_from_object', 'encoder_from_object', 'encoders_from_object', 'force_torque_from_object', 'imu_from_object', 'is_key_down', 'raycast', 'rigid_body_from_object', 'scene', 'soft_body_from_object', 'spline_from_object']
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
class RaycastHit:
    """
    What threepp.editor.raycast answers with when the ray hit something.

    Values, all of it - nothing here points into PhysX, so keeping one is safe.
    """
    @property
    def object(self) -> threepp.Object3D | None:
        """
        The object the physics was authored on, as its concrete type (Mesh, Group, Robot, ...) - or None when the actor answers to nothing the script can name.
        """
    @property
    def point(self) -> threepp.Vector3:
        """
        WORLD-SPACE point of the hit.
        """
    @property
    def normal(self) -> threepp.Vector3:
        """
        Unit surface normal there, pointing OUT of the surface hit.
        """
    @property
    def distance(self) -> float:
        """
        Metres from `origin` to `point`, along the ray.
        """
class ImuSample:
    """
    One IMU measurement, as the sensor reported it (noise, bias and all).
    """
    @property
    def time(self) -> float:
        """
        Sim time of the measurement (s).
        """
    @property
    def angular_velocity(self) -> threepp.Vector3:
        """
        Gyro reading in rad/s, in the sensor's own frame.
        """
    @property
    def acceleration(self) -> threepp.Vector3:
        """
        Accelerometer reading in m/s^2, in the sensor's own frame. SPECIFIC FORCE, so a level sensor at rest reads +9.81 on its up axis and one in free fall reads ~0.
        """
class EncoderSample:
    """
    One joint-encoder reading: quantized to whole ticks and noise-corrupted,
    which is why a controller tuned on it survives contact with hardware.
    """
    @property
    def time(self) -> float:
        """
        Sim time of the reading (s).
        """
    @property
    def position(self) -> float:
        """
        Joint position: radians for a revolute joint, metres for a prismatic one.
        """
    @property
    def velocity(self) -> float:
        """
        Joint velocity, differentiated from the QUANTIZED position - so it chatters at standstill, exactly as a real one does.
        """
class WrenchSample:
    """
    One six-component load-cell reading, in the measured joint's child frame.
    """
    @property
    def time(self) -> float:
        """
        Sim time of the reading (s).
        """
    @property
    def force(self) -> threepp.Vector3:
        """
        Force in newtons.
        """
    @property
    def torque(self) -> threepp.Vector3:
        """
        Torque in newton-metres.
        """
class ContactSample:
    """
    One contact reading: the touch latch and the force behind it.
    """
    @property
    def time(self) -> float:
        """
        Sim time of the reading (s).
        """
    @property
    def touching(self) -> bool:
        """
        Latched touch state. Stays True while resting on something, including after the contact pair falls asleep and `force` goes quiet - this is the channel a foot-down check wants.
        """
    @property
    def force(self) -> threepp.Vector3:
        """
        Mean contact force over the interval (N). Zero while the pair sleeps, even though the touch is real.
        """
class Imu:
    """
    A live IMU the play session is running for a scene object. Only exists
    during Play, and only in a build with the PhysX SDK.
    """
    @property
    def object(self) -> threepp.Object3D:
        """
        The scene object this sensor was authored on.
        """
    @property
    def valid(self) -> bool:
        """
        False once the play session that created it has stopped.
        """
    def latest(self) -> ImuSample | None:
        """
        The newest measurement, or None before the first one. Does not move this handle's read cursor.
        """
    def read_new(self) -> list[ImuSample]:
        """
        Every measurement since this handle last read, oldest first, and advances its cursor. A fresh handle starts empty - it reports what arrives from now on. Each handle has its own cursor, so two of them never steal each other's samples; falling more than 256 behind loses the oldest.
        """
class Encoder:
    """
    A live joint encoder the play session is running. Only exists during Play,
    and only in a build with the PhysX SDK.
    """
    @property
    def object(self) -> threepp.Object3D:
        """
        The scene object this sensor was authored on.
        """
    @property
    def valid(self) -> bool:
        """
        False once the play session that created it has stopped.
        """
    @property
    def joint(self) -> str:
        """
        The URDF joint name this encoder measures.
        """
    def latest(self) -> EncoderSample | None:
        """
        The newest reading, or None before the first one. Does not move this handle's read cursor.
        """
    def read_new(self) -> list[EncoderSample]:
        """
        Every reading since this handle last read, oldest first, and advances its cursor. A fresh handle starts empty.
        """
class ForceTorque:
    """
    A live six-axis load cell the play session is running. Only exists during
    Play, and only in a build with the PhysX SDK.
    """
    @property
    def object(self) -> threepp.Object3D:
        """
        The scene object this sensor was authored on.
        """
    @property
    def valid(self) -> bool:
        """
        False once the play session that created it has stopped.
        """
    @property
    def joint(self) -> str:
        """
        The URDF joint name this load cell sits in.
        """
    def latest(self) -> WrenchSample | None:
        """
        The newest wrench, or None before the first one. Does not move this handle's read cursor.
        """
    def read_new(self) -> list[WrenchSample]:
        """
        Every wrench since this handle last read, oldest first, and advances its cursor. A fresh handle starts empty.
        """
class Contact:
    """
    A live contact sensor the play session is running. Only exists during Play,
    and only in a build with the PhysX SDK.
    """
    @property
    def object(self) -> threepp.Object3D:
        """
        The scene object this sensor was authored on.
        """
    @property
    def valid(self) -> bool:
        """
        False once the play session that created it has stopped.
        """
    def latest(self) -> ContactSample | None:
        """
        The newest reading, or None before the first one. Does not move this handle's read cursor.
        """
    def read_new(self) -> list[ContactSample]:
        """
        Every reading since this handle last read, oldest first, and advances its cursor. A fresh handle starts empty.
        """
class Collision:
    """
    One touch, as on_collision_enter / on_collision_exit are handed it.

    A value copied out of the physics report that produced it, so keeping it is safe. `other` is the object on the far side of the touch, as its concrete type - or None when that body belongs to nothing the script can see. The contact geometry describes the ENTER only; an exit has no manifold left to read and carries zeros.
    """
    @property
    def other(self) -> threepp.Object3D | None:
        """
        The other object, as its concrete type (Mesh, Robot, ...), or None.
        """
    @property
    def point(self) -> threepp.Vector3:
        """
        WORLD-SPACE point of the hardest-hit manifold point, at the substep the touch began.
        """
    @property
    def normal(self) -> threepp.Vector3:
        """
        Unit contact normal at that point, pointing INTO this script's body - the direction the other body is pushing it.
        """
    @property
    def impulse(self) -> threepp.Vector3:
        """
        Total impulse over the manifold (N*s), same orientation. Divide by the substep to read it as a force.
        """
def articulation_from_object(object: threepp.Object3D | None) -> Articulation | None:
    """
    The Articulation PhysX is simulating for `object`, or None when Play is not running or no articulated robot governs it. The lookup walks up the scene graph, so a script on any link of a robot finds the robot's articulation. Robots simulate only when their Articulation section says Simulate.
    """
def contact_from_object(object: threepp.Object3D | None) -> Contact | None:
    """
    The live contact sensor authored on `object`, or None when Play is not running or none measures it.
    """
def encoder_from_object(object: threepp.Object3D | None, joint: str | None = None) -> Encoder | None:
    """
    The live joint encoder authored on `object`, or None when Play is not running or none measures it. An encoder authored for All joints becomes one live encoder per DOF, so pass joint="name" to pick one; with no joint this raises rather than guessing when more than one answers.
    """
def encoders_from_object(object: threepp.Object3D | None) -> list[Encoder]:
    """
    Every live joint encoder authored on `object`, in the articulation's DOF order - the whole-robot joint state an All-joints encoder fans out into. Empty outside Play. Read `joint` on each to know which DOF it is.
    """
def force_torque_from_object(object: threepp.Object3D | None) -> ForceTorque | None:
    """
    The live force/torque sensor authored on `object`, or None when Play is not running or none measures it. A load cell sits in ONE joint, so there is never more than one to choose between.
    """
def imu_from_object(object: threepp.Object3D | None) -> Imu | None:
    """
    The live IMU authored on `object`, or None when Play is not running or no IMU measures it. The lookup walks up the scene graph, so a script on a child finds the sensor on its link.
    """
def raycast(origin: threepp.Vector3, direction: threepp.Vector3, max_distance: float = 3.4028234663852886e+38, ignore: threepp.Object3D | None = None) -> RaycastHit | None:
    """
    Cast a ray through the playing physics world and return the NEAREST RaycastHit, or None when it hits nothing.

    `origin` and `direction` are Vector3, world space; direction is normalised here, and a zero-length one raises ValueError. `max_distance` is in metres and defaults to unbounded. `ignore` excludes every actor governing that object - pass your own object for a ground check, or the ray starts inside your own collider and hits it.

    Raises RuntimeError when no physics world is playing: a miss is None, so 'not playing' cannot also be None without making the two the same answer.
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
def add(object: threepp.Object3D, parent: threepp.Object3D | None = None) -> threepp.Object3D:
    """
    Add `object` to what this generator is building, and return it. With no `parent` it goes at the generator's root; pass one of your own earlier adds to nest. Raises outside a generator run.
    """
def scene() -> threepp.Object3D:
    """
    The scene this generator is authoring into. READ it to place content relative to what already exists. Objects reached this way are NOT this generator's output and are not replaced when it re-runs.
    """
def is_key_down(key: str) -> bool:
    """
    Poll whether a key is currently held — 'W', 'SPACE', 'UP', 'LEFT', 'KP8', the same names Canvas.is_key_down takes. Answers False while the user is typing into a field, and False in a build or a pass with no window. Query it every update() for continuous control; it never sticks.
    """
