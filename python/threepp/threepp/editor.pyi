"""
The runtime face of editor-authored data.
"""
from __future__ import annotations
import typing
import threepp
__all__: list[str] = ['Articulation', 'Collision', 'Contact', 'ContactSample', 'Encoder', 'EncoderSample', 'ForceTorque', 'Imu', 'ImuSample', 'Joint', 'RaycastHit', 'RigidBody', 'SoftBody', 'SplinePath', 'Task', 'Time', 'Vehicle', 'WrenchSample', 'add', 'articulation_from_object', 'contact_from_object', 'encoder_from_object', 'encoders_from_object', 'force_torque_from_object', 'imu_from_object', 'is_key_down', 'joint_from_object', 'raycast', 'rigid_body_from_object', 'scene', 'script_from_object', 'soft_body_from_object', 'spline_from_object', 'start_coroutine', 'time', 'until', 'vehicle_from_object', 'wait']
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
class Joint:
    """
    One authored joint, played: the constraint PhysX built from a joint node's
    userData. Only exists during Play, and only in a build with the PhysX SDK.
    """
    @property
    def object(self) -> threepp.Object3D:
        """
        The joint NODE this handle was built from - the node whose transform is the joint frame.
        """
    @property
    def valid(self) -> bool:
        """
        False once the play session that created it has stopped.
        """
    @property
    def type(self) -> str:
        """
        "fixed" | "revolute" | "prismatic" | "spherical" | "distance".
        """
    @property
    def position(self) -> float:
        """
        The joint coordinate: radians about the axis for a revolute (and a spherical's twist), metres along it for a prismatic, anchor distance for a distance joint. Zero for fixed.
        """
    @property
    def velocity(self) -> float:
        """
        Its rate: rad/s or m/s, same convention as position.
        """
    @property
    def broken(self) -> bool:
        """
        True once the solver exceeded the break threshold; the constraint never comes back. A script ON the joint node hears on_break() at that moment.
        """
    def set_drive_target(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        PD setpoint along the motion axis (radians / metres). Acts through the authored stiffness - inert while stiffness is zero.
        """
    def set_drive_velocity(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Velocity setpoint (rad/s or m/s). Acts through the authored damping - inert while damping is zero.
        """
    @property
    def reaction_force(self) -> threepp.Vector3:
        """
        Force (N, world axes) the solver applied to hold the constraint on the last step. Zero once broken - a broken joint transmits nothing; the failure load is break_force.
        """
    @property
    def reaction_torque(self) -> threepp.Vector3:
        """
        Torque (N*m, world axes) alongside reaction_force.
        """
    @property
    def break_force(self) -> threepp.Vector3:
        """
        The force (N, world axes) the solver applied on the step that BROKE the joint - the true failure load, necessarily past the authored break threshold. Zero until broken; an on_break() reads it directly, the break already happened.
        """
    @property
    def break_torque(self) -> threepp.Vector3:
        """
        Torque (N*m, world axes) alongside break_force.
        """
class Vehicle:
    """
    One authored vehicle, played: the PhysX vehicle built from a model root's
    userData. Only exists during Play, and only in a build with the PhysX SDK.
    """
    @property
    def object(self) -> threepp.Object3D:
        """
        The model root the vehicle was authored on - the root itself, even when the handle was asked for from a wheel.
        """
    @property
    def valid(self) -> bool:
        """
        False once the play session that created it has stopped.
        """
    def set_throttle(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Throttle in [0, 1]. A held pedal, not an impulse - it stays where it was set until set again.
        """
    def set_brake(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Brake in [0, 1].
        """
    def set_steer(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Steer in [-1, 1]; positive steers left, matching the editor's A key.
        """
    @property
    def reverse(self) -> bool:
        """
        Direction selector. The transmission is automatic all the way - True selects reverse, False drive; there is nothing to shift.
        """
    @reverse.setter
    def reverse(self, value: bool) -> None:
        ...
    @property
    def speed(self) -> float:
        """
        Forward speed in m/s (negative while rolling backwards).
        """
    @property
    def wheel_spin_rates(self) -> list[float]:
        """
        Wheel spin in rad/s, [FR, FL, RR, RL].
        """
    @property
    def wheels_grounded(self) -> list[bool]:
        """
        Whether each wheel's suspension found ground, [FR, FL, RR, RL].
        """
    @property
    def position(self) -> threepp.Vector3:
        """
        WORLD-SPACE position of the chassis centre.
        """
    @property
    def rotation(self) -> threepp.Quaternion:
        """
        WORLD-SPACE orientation of the chassis (+Z is forward).
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
class Task:
    """
    A running coroutine, as start_coroutine hands it back.

    Keep it to cancel(), or to ask whether it is done. Dropping it does NOT stop the coroutine - the scheduler holds it until it finishes, its owning script is disabled, or Play stops.
    """
    def cancel(self) -> None:
        """
        Stop the coroutine now. Not an error, and harmless if already done.

        The generator is close()d, so a `finally:` inside it runs - which is where anything a half-finished mission has to put back belongs.
        """
    @property
    def done(self) -> bool:
        """
        True once the coroutine returned, raised, or was cancelled.
        """
class Time:
    """
    The play session's clocks, live - read `threepp.editor.time`.

    There are two, and they do not agree. WALL time is what update(dt) rides: real seconds, however many the last frame took. SIM time is what the physics world advances in fixed substeps, and what fixed_update(dt) and every sensor timestamp are stamped with. A frame that hitches advances wall time in full but simulates at most a few substeps and drops the remainder, so the two drift apart for good - which is why anything integrating toward a physics quantity should be reading sim_time (or living in fixed_update) rather than summing update's dt.

    Every field is zero outside Play.
    """
    @property
    def frame_dt(self) -> float:
        """
        Wall-clock seconds the last frame took - the same number update(dt) is handed, readable from the methods that are not handed it.
        """
    @property
    def wall_time(self) -> float:
        """
        Real seconds since Play started.
        """
    @property
    def sim_time(self) -> float:
        """
        Simulated seconds since Play started: the physics world's own clock, which advances only when substeps run. This is the SAME clock that stamps sensor samples, so comparing a sample's timestamp against it is meaningful. Inside fixed_update it reads the time at the START of the substep about to be solved.
        """
    @property
    def sim_dt(self) -> float:
        """
        The fixed substep in seconds - constant for the run, and exactly what fixed_update(dt) is handed.
        """
    @property
    def steps(self) -> int:
        """
        Fixed substeps completed since Play started. Advances by 0, 1 or more per frame depending on how long the frame took.
        """
    @property
    def fixed_clock(self) -> bool:
        """
        True when sim_time and sim_dt come from a playing physics world. False in a build or a pass without one, where sim_time falls back to wall_time and sim_dt to frame_dt - still an elapsed-time answer, but not a simulated one, and this is how to tell.
        """
    @property
    def playing(self) -> bool:
        """
        True between the start of a play session and its stop.
        """

time: Time

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
def script_from_object(object: threepp.Object3D | None) -> typing.Any | None:
    """
    The live script instance running on `object`, or None.

    The instance IS the API: call its methods and read or write its attributes to signal it. Returns None when nothing is playing, when `object` carries no script, or when that script's instance failed - a disabled script is dead to a lookup. The object must be the exact one the script is authored on; unlike rigid_body_from_object this does not walk up the scene graph.

    Every instance exists before any start() runs, so resolving a neighbour in start() works whatever order the scene is in. Do not keep the reference across Play sessions - the next Play builds new instances.

    Typed Any because it honestly is: what comes back is your own script class, which this stub has never seen. `X | None` rather than a bare Any so the None case still reads as a case a checker will make you handle.
    """
def world() -> threepp.PhysxWorld | None:
    """
    The PhysxWorld this Play session is stepping, or None outside Play.

    The ordinary threepp.PhysxWorld - world.add(mesh), add_static, add_dynamic_convex, remove, create_material - so a script that spawns a mesh into editor.scene() gives it a body exactly as a standalone threepp program would. Bodies added this way die with the world at Stop, and meshes spawned into the scene die with the stop-restore, so neither needs cleaning up.

    This is the ONLY way to a world inside the editor: threepp.PhysxWorld's own constructor raises there, because the session owns the one world and a second would bring up a second PhysX foundation beside it.

    NOTE the handle difference: what world.add returns is a raw threepp.RigidBody, valid while the world is alive - which includes stop(), since sessions stop in reverse order and physics goes down last - but NOT invalidated when the world dies, so one stashed beyond its session dereferences a released actor. rigid_body_from_object returns the lifetime-checked threepp.editor.RigidBody, which raises instead. Prefer that one for anything held longer than the session.
    """
def joint_from_object(object: threepp.Object3D | None) -> Joint | None:
    """
    The live Joint built from `object`'s authored joint entry, or None when Play is not running or the object is not a joint node. NO ancestor walk, unlike the other from_object verbs: a joint is its own node, so the script asking is normally sitting on it.
    """
def vehicle_from_object(object: threepp.Object3D | None) -> Vehicle | None:
    """
    The Vehicle PhysX is simulating for `object`, or None when Play is not running or no authored vehicle governs it. The lookup walks up the scene graph, so a script on a wheel (or anywhere in the model) finds the car.
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
    The live scene: what a generator is authoring into, or what a behaviour script is playing in. READ it to reach what you did not author — scene.get_object_by_name("Ground"), scene.children. Objects a generator reaches this way are NOT its output and are not replaced when it re-runs. During Play this answers from start() onwards, including update(), fixed_update() and the collision and trigger callbacks. Raises when nothing is generating and nothing is playing.
    """
def is_key_down(key: str) -> bool:
    """
    Poll whether a key is currently held — 'W', 'SPACE', 'UP', 'LEFT', 'KP8', the same names Canvas.is_key_down takes. Answers False while the user is typing into a field, and False in a build or a pass with no window. Query it every update() for continuous control; it never sticks.
    """
def draw_line(a: threepp.Vector3, b: threepp.Vector3, color: int | threepp.Color | None = None) -> None:
    """
    Draw a world-space line from a to b, over the scene, for the CURRENT frame only — call it every update() to keep it visible. color is a hex int or a threepp.Color; default white. No-op outside Play, so a script that draws still runs headless. The lines are editor furniture: never saved, and invisible to the sensors.
    """
def draw_ray(origin: threepp.Vector3, direction: threepp.Vector3, length: float = 1.0, color: int | threepp.Color | None = None) -> None:
    """
    Draw origin plus direction times length — the shape of a raycast, so draw_ray(origin, direction, hit.distance) shows exactly the ray that hit. direction is used as given, not normalised. Current frame only; no-op outside Play.
    """
def draw_point(point: threepp.Vector3, size: float = 0.25, color: int | threepp.Color | None = None) -> None:
    """
    Draw a small axis-aligned cross at point — a position made visible. Current frame only; no-op outside Play.
    """
def draw_box(center: threepp.Vector3, size: threepp.Vector3, color: int | threepp.Color | None = None) -> None:
    """
    Draw the 12 edges of an axis-aligned box: center and size as full extents — an AABB, a spawn region, a sensor's reach. Current frame only; no-op outside Play.
    """
def draw_sphere(center: threepp.Vector3, radius: float = 1.0, color: int | threepp.Color | None = None) -> None:
    """
    Draw a wireframe sphere as three great circles — a trigger radius, a sensor range, a clearance. Current frame only; no-op outside Play.
    """
def draw_axes(object: threepp.Object3D, size: float = 1.0) -> None:
    """
    Draw object's world-space frame: X red, Y green, Z blue — the one question every attitude bug comes down to. Current frame only; no-op outside Play.
    """
def start_coroutine(generator: typing.Generator[typing.Any, typing.Any, typing.Any]) -> Task:
    """
    Run generator as a coroutine, and return the Task driving it.

    The generator is resumed once per frame, AFTER physics has stepped and after every script's update() has run - so what it sees is the settled state of the frame. Nothing of it executes inside this call; the body runs up to its first yield at the next pump.

    What it may yield: nothing (resume next frame); threepp.editor.wait(seconds) to resume after that much SIMULATED time; threepp.editor.until(predicate) to resume once the predicate is truthy, receiving that value (`hit = yield editor.until(...)`); or another generator, which is run to completion with its return value handed back.

    The task belongs to the script instance whose method is running, and raises when there is none. A raise inside it is reported once and disables that instance whole, exactly as a raise in update() does; the instance's other tasks go with it. Every task is dropped at Stop. Start as many as you like.
    """
def wait(seconds: typing.SupportsFloat) -> typing.Any:
    """
    Yield this from a coroutine to suspend for `seconds` of SIMULATED time.

    Simulated, not wall: a mission written this way freezes when physics is paused or starved, and takes the same number of substeps on a machine that renders half as fast. Without a physics world sim time degrades to wall time, so this still measures something honest there. See threepp.editor.time.
    """
def until(predicate: typing.Callable[[], typing.Any]) -> typing.Any:
    """
    Yield this from a coroutine to suspend until predicate() is truthy.

    The truthy value is sent back in, so `hit = yield threepp.editor.until(lambda: threepp.editor.raycast(o, d, 2.0))` both waits for the hit and hands you the hit. Polled once per frame, after physics has stepped and after every script's update() has run.
    """
