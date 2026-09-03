"""
threepp — a cross-platform C++ 3D library with the high-level API of three.js, exposed to Python (scene graph, geometries, materials, cameras, lights, OpenGL renderer + headless render-to-numpy).
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
from . import editor
from . import imgui
__all__: list[str] = ['AmbientLight', 'AnimationAction', 'AnimationBlendMode', 'AnimationClip', 'AnimationMixer', 'Approach', 'ArrowHelper', 'Articulation', 'ArticulationLink', 'Audio', 'AudioDistanceModel', 'AudioListener', 'AxesHelper', 'BVH', 'Background', 'BakePose', 'BarkStyle', 'BirdPlumage', 'BirdRole', 'BirdShape', 'BirdState', 'Blending', 'Blob', 'Box3', 'Box3Helper', 'BoxGeometry', 'BoxHelper', 'BranchingMode', 'BufferGeometry', 'Camera', 'CameraHelper', 'Canvas', 'CapsuleGeometry', 'CatmullRomCurve3', 'CircleGeometry', 'Clock', 'Cluster', 'Colonise', 'Color', 'ColorKeyframeTrack', 'ColorSpace', 'CombineOperation', 'Cone', 'ConeGeometry', 'ContactPoint', 'ContactSample', 'ContactSensor', 'ConvexGeometry', 'CrossQuad', 'CrownShape', 'Cruise', 'Curve2', 'Curve3', 'Cylinder', 'CylinderGeometry', 'DepthPacking', 'DepthSensor', 'DirectionalLight', 'DirectionalLightHelper', 'DisplacedMesh', 'EXPONENTIAL', 'EXRLoader', 'Ellipsoid', 'ErosionType', 'Euler', 'Evade', 'ExtrudeGeometry', 'FBXLoader', 'Filter', 'Flare', 'Flock', 'FlockParams', 'Fog', 'FogExp2', 'Follower', 'Font', 'FontLoader', 'ForceTorqueSensor', 'FrameChannel', 'Frond', 'Furrowed', 'GLRenderer', 'GLTFLoader', 'GLTFResult', 'Gait', 'GrassMesh', 'GridHelper', 'Group', 'HAS_AUDIO', 'HAS_IMGUI', 'HAS_PHYSX', 'HAS_VULKAN', 'Hemisphere', 'HemisphereLight', 'HemisphereLightHelper', 'Hop', 'HorizontalAlignment', 'INVERSE', 'IcosahedronGeometry', 'IcpOptions', 'IcpResult', 'IkOptions', 'IkResult', 'IkSolver', 'IkTask', 'ImguiContext', 'Imu', 'ImuSample', 'InstancedMesh', 'Interpolation', 'IsoMesh', 'Joint', 'JointEncoder', 'JointInfo', 'JointRange', 'JointSample', 'JointType', 'KeyframeTrack', 'LINEAR', 'Lanceolate', 'Launch', 'Layers', 'Leader', 'LeafShape', 'LeafStyle', 'LidarBeam', 'LidarModel', 'LidarParams', 'LidarReturn', 'Light', 'Line', 'LineBasicMaterial', 'LineCurve', 'LineCurve3', 'LineSegments', 'Lobed', 'Loner', 'Loop', 'Mapping', 'Material', 'Matrix3', 'Matrix4', 'Mesh', 'MeshBasicMaterial', 'MeshDepthMaterial', 'MeshLambertMaterial', 'MeshNormalMaterial', 'MeshPhongMaterial', 'MeshPhysicalMaterial', 'MeshStandardMaterial', 'ModelLoader', 'NONE', 'NoiseModel', 'NoiseType', 'NumberKeyframeTrack', 'OBJLoader', 'Object3D', 'Ocean', 'OctahedronGeometry', 'OrbitControls', 'OrthographicCamera', 'Ovate', 'Papery', 'ParticleField', 'Path', 'PathTracedLidarSensor', 'Perched', 'PerspectiveCamera', 'PhysxGpuBatch', 'PhysxMaterial', 'PhysxSoftBodyMaterial', 'PhysxVehicle', 'PhysxWorld', 'PlaneGeometry', 'Plated', 'PointLight', 'PointLightHelper', 'Points', 'PointsMaterial', 'PolarGridHelper', 'PositionalAudio', 'Quad', 'Quaternion', 'QuaternionKeyframeTrack', 'RGBELoader', 'RangeNoiseModel', 'Ray', 'RayHit', 'RectAreaLight', 'RigidBody', 'RingGeometry', 'Robot', 'RotationOrder', 'STLLoader', 'SVGLoader', 'ScalarField', 'Scene', 'Sensor', 'Serrate', 'ShadowMaterial', 'Shape', 'ShapeGeometry', 'Side', 'SkeletonHelper', 'SoftBody', 'SogLoader', 'Sphere', 'SphereGeometry', 'SplatCloud', 'SplatData', 'SplatLoader', 'SplatPoseSet', 'SplineCurve', 'SpotLight', 'SpotLightHelper', 'Sprite', 'SpriteMaterial', 'SurfaceMesh', 'TerrainFalloff', 'TerrainGenerator', 'TerrainParams', 'Text2D', 'Text3D', 'TextSprite', 'Texture', 'TextureLoader', 'TextureWrapping', 'ToneMapping', 'TorusGeometry', 'TorusKnotGeometry', 'TransformControls', 'TreeGenerator', 'TreeParams', 'TubeGeometry', 'URDFLoader', 'Vector2', 'Vector3', 'Vector4', 'VectorKeyframeTrack', 'VerticalAlignment', 'VoxelGrid', 'VulkanRenderer', 'Walk', 'Whorl', 'WrenchSample', 'apply_terrain_preset', 'apply_tree_preset', 'bake_surface', 'clamp', 'damp', 'data_texture', 'deg_to_rad', 'editor', 'euclidean_modulo', 'float_texture', 'generate_uuid', 'icp_point_to_point', 'imgui', 'inverse_lerp', 'is_power_of_two', 'iso_mesh_to_geometry', 'lerp', 'make_bark_textures', 'make_flower_texture', 'make_leaf_texture', 'make_needle_frond_texture', 'make_sensor_mesh', 'map_linear', 'marching_cubes', 'merge_buffer_geometries', 'merge_vertices', 'rad_to_deg', 'rand_float', 'rand_float_range', 'rand_float_spread', 'rand_int', 'rotate_equirect', 'select_lod', 'simplify_geometry', 'splat_points_to_field', 'terrain_from_json', 'terrain_load_config', 'terrain_save_config', 'terrain_to_json', 'voxel_downsample', 'vulkan_available', 'vulkan_validation_active', 'vulkan_validation_error_count', 'write_wav']
class AmbientLight(Light):
    def __init__(self, color: Color = ..., intensity: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
class AnimationAction:
    blend_mode: AnimationBlendMode
    def cross_fade_to(self, other: AnimationAction, duration: typing.SupportsFloat | typing.SupportsIndex, warp: bool = False) -> AnimationAction:
        ...
    def fade_in(self, duration: typing.SupportsFloat | typing.SupportsIndex) -> AnimationAction:
        ...
    def fade_out(self, duration: typing.SupportsFloat | typing.SupportsIndex) -> AnimationAction:
        ...
    def get_effective_time_scale(self) -> float:
        ...
    def get_effective_weight(self) -> float:
        ...
    def halt(self, duration: typing.SupportsFloat | typing.SupportsIndex) -> AnimationAction:
        ...
    def is_running(self) -> bool:
        ...
    def is_scheduled(self) -> bool:
        ...
    def play(self) -> AnimationAction:
        ...
    def reset(self) -> AnimationAction:
        ...
    def set_duration(self, duration: typing.SupportsFloat | typing.SupportsIndex) -> AnimationAction:
        ...
    def set_effective_time_scale(self, time_scale: typing.SupportsFloat | typing.SupportsIndex) -> AnimationAction:
        ...
    def set_effective_weight(self, weight: typing.SupportsFloat | typing.SupportsIndex) -> AnimationAction:
        ...
    def set_loop(self, mode: Loop, repetitions: typing.SupportsInt | typing.SupportsIndex = -1) -> AnimationAction:
        ...
    def start_at(self, time: typing.SupportsFloat | typing.SupportsIndex) -> AnimationAction:
        ...
    def stop(self) -> AnimationAction:
        ...
    def stop_fading(self) -> AnimationAction:
        ...
    def stop_warping(self) -> AnimationAction:
        ...
    def sync_with(self, action: AnimationAction) -> AnimationAction:
        ...
    def warp(self, start_time_scale: typing.SupportsFloat | typing.SupportsIndex, end_time_scale: typing.SupportsFloat | typing.SupportsIndex, duration: typing.SupportsFloat | typing.SupportsIndex) -> AnimationAction:
        ...
class AnimationBlendMode:
    """
    Members:
    
      NORMAL
    
      ADDITIVE
    """
    ADDITIVE: typing.ClassVar[AnimationBlendMode]  # value = <AnimationBlendMode.ADDITIVE: 1>
    NORMAL: typing.ClassVar[AnimationBlendMode]  # value = <AnimationBlendMode.NORMAL: 0>
    __members__: typing.ClassVar[dict[str, AnimationBlendMode]]  # value = {'NORMAL': <AnimationBlendMode.NORMAL: 0>, 'ADDITIVE': <AnimationBlendMode.ADDITIVE: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class AnimationClip:
    blend_mode: AnimationBlendMode
    @staticmethod
    def find_by_name(clips: collections.abc.Sequence[AnimationClip], name: str) -> AnimationClip:
        ...
    def __init__(self, name: str, duration: typing.SupportsFloat | typing.SupportsIndex = 1.0, tracks: collections.abc.Sequence[KeyframeTrack] = [], blend_mode: AnimationBlendMode = AnimationBlendMode.NORMAL) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def make_additive(self) -> None:
        """
        Convert every track to additive form and mark the clip Additive.
        """
    def reset_duration(self) -> None:
        """
        Recompute duration as the maximum track end time.
        """
    @property
    def duration(self) -> float:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def uuid(self) -> str:
        ...
class AnimationMixer:
    def __init__(self, root: typing.Any) -> None:
        ...
    def clip_action(self, clip: AnimationClip, blend_mode: AnimationBlendMode | None = None) -> AnimationAction:
        """
        Return (creating if needed) the AnimationAction for a clip on this mixer's root.
        """
    def stop_all_action(self) -> None:
        ...
    def update(self, dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Advance all active actions by dt seconds and write the result into the scene graph.
        """
    @property
    def time(self) -> float:
        ...
    @time.setter
    def time(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def time_scale(self) -> float:
        ...
    @time_scale.setter
    def time_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class ArrowHelper(Object3D):
    def __init__(self, dir: Vector3 = ..., origin: Vector3 = ..., length: typing.SupportsFloat | typing.SupportsIndex = 1.0, color: Color = ..., head_length: typing.SupportsFloat | typing.SupportsIndex | None = None, head_width: typing.SupportsFloat | typing.SupportsIndex | None = None) -> None:
        ...
    def set_color(self, color: Color) -> None:
        ...
    def set_direction(self, dir: Vector3) -> None:
        ...
    def set_length(self, length: typing.SupportsFloat | typing.SupportsIndex, head_length: typing.SupportsFloat | typing.SupportsIndex | None = None, head_width: typing.SupportsFloat | typing.SupportsIndex | None = None) -> None:
        ...
class Articulation:
    """
    A reduced-coordinate articulation (robot): a tree of links joined by motorized revolute joints. Build with add_link (root first), then finalize(); stepping the world drives the bound meshes.
    """
    def add_link(self, mesh: Mesh, parent: typing.Any = None, density: typing.SupportsFloat | typing.SupportsIndex = 1000.0, axis: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"] = [0.0, 0.0, 1.0], anchor: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"] = [0.0, 0.0, 0.0], lower: typing.Any = None, upper: typing.Any = None, stiffness: typing.SupportsFloat | typing.SupportsIndex = 0.0, damping: typing.SupportsFloat | typing.SupportsIndex = 0.0, max_force: typing.SupportsFloat | typing.SupportsIndex = 1000000.0, drive_target: typing.SupportsFloat | typing.SupportsIndex = 0.0, joint_type: str = 'revolute', joint_friction: typing.SupportsFloat | typing.SupportsIndex = 0.0, material: typing.Any = None) -> ArticulationLink:
        """
        Add a link. parent=None → the fixed/free root; otherwise attach an inbound joint at world-space `anchor` along world-space `axis`. joint_type='revolute' (hinge about axis) or 'prismatic' (slider along axis). lower/upper set the joint limits (radians for revolute, metres for prismatic; omit both for a free axis); stiffness/damping/max_force configure the PD drive (stiffness>0 motorizes it; leave 0 for a passive/force-controlled joint). Shape is inferred from the mesh (Box/Sphere/Capsule). `material` (from world.create_material) overrides the contact friction/restitution for this link's shape — e.g. a grippy, restitution-0 foot, or a per-env material for friction domain randomization; default uses the world's shared material. Returns an ArticulationLink.
        """
    def dof_order(self) -> numpy.typing.NDArray[numpy.int32]:
        """
        Per add-order joint, its low-level DOF slot in the direct-GPU joint buffers (PhysX cache order != add-order). Use to map a GPU-trained policy back to the CPU getters: obs_gpu[dof_order[i]] = cpu[i]; cpu_target[i] = gpu_target[dof_order[i]].
        """
    def finalize(self) -> None:
        """
        Add the finished articulation to the scene. No links may be added afterwards.
        """
    def joint_positions(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        All revolute joint angles (radians) as one numpy array, in add_link order.
        """
    def joint_velocities(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        All revolute joint angular velocities (rad/s) as one numpy array.
        """
    def link(self, key: typing.Any) -> ArticulationLink:
        """
        The ArticulationLink at add-order index `key` (int, 0 = root, negatives count from the end) or registered under name `key` (str). For a URDF-loaded articulation every URDF link name resolves — a link attached by a FIXED joint maps to the link it was welded into, so a tool frame like a hand TCP still finds its body. Use it to add_force/add_impulse on one specific link (e.g. load the tool link with a catch impulse for two-way coupling) or to read its pose. The handle keeps the articulation alive; the world must outlive both.
        """
    def reset(self, position: Vector3, quaternion: Quaternion = ...) -> None:
        """
        Episode reset: teleport the root to `position` with optional `quaternion` orientation (default upright/identity), zero velocity, and zero all joint positions/velocities.
        """
    def root_state(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Root link world pose as numpy [px,py,pz, qx,qy,qz,qw] in one call.
        """
    def root_velocity(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Root link world-frame velocity as numpy [vx,vy,vz, wx,wy,wz] — the base linear + angular velocity a locomotion observation needs.
        """
    def set_drive_targets(self, targets: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """
        Set every joint's PD drive target from one numpy array — the batched hot path for vectorized stepping (one call instead of one per joint).
        """
    def set_joint_positions(self, positions: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """
        Set all joint positions (DOF order) and zero velocities — e.g. place a cart-pole hanging straight down for a swing-up demo.
        """
    @property
    def link_names(self) -> list[str]:
        """
        All names link(name) resolves, in registration order. Populated by load_articulation with every URDF link name (root and fixed-collapsed children included); empty for a hand-built articulation.
        """
    @property
    def links(self) -> list[ArticulationLink]:
        """
        Every link as a list of ArticulationLink, in add_link order (root first). Handles are valid only while the articulation and its world live — hold both; prefer link(key) for a single handle, which keeps the articulation alive itself.
        """
class ArticulationLink:
    """
    A link of an Articulation plus its inbound revolute joint (the root has none). Valid while its Articulation/world live.
    """
    def add_force(self, force: Vector3) -> None:
        """
        Apply an external force (N) to this link.
        """
    def add_impulse(self, impulse: Vector3) -> None:
        """
        Apply an external impulse (kg·m/s) — e.g. a random shove. PhysX takes no impulse on an articulation link, so this goes in as the force that carries the same momentum through one substep (force = impulse / fixed_timestep), consumed by the next step().
        """
    def set_drive_target(self, target: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the PD drive's target angle (radians).
        """
    def set_drive_velocity(self, velocity: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def is_root(self) -> bool:
        ...
    @property
    def joint_position(self) -> float:
        """
        Joint angle (radians).
        """
    @property
    def joint_velocity(self) -> float:
        """
        Joint angular velocity (rad/s).
        """
    @property
    def position(self) -> Vector3:
        ...
    @property
    def quaternion(self) -> Quaternion:
        ...
class Audio:
    def __init__(self, listener: AudioListener, file: str) -> None:
        ...
    def play(self) -> None:
        ...
    def seek_to_start(self) -> None:
        ...
    def set_looping(self, loop: bool) -> None:
        ...
    def set_playback_rate(self, rate: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_volume(self, volume: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def stop(self) -> None:
        ...
    def toggle_play(self) -> None:
        ...
    @property
    def is_playing(self) -> bool:
        ...
class AudioDistanceModel:
    """
    Members:
    
      NONE
    
      INVERSE
    
      LINEAR
    
      EXPONENTIAL
    """
    EXPONENTIAL: typing.ClassVar[AudioDistanceModel]  # value = <AudioDistanceModel.EXPONENTIAL: 3>
    INVERSE: typing.ClassVar[AudioDistanceModel]  # value = <AudioDistanceModel.INVERSE: 1>
    LINEAR: typing.ClassVar[AudioDistanceModel]  # value = <AudioDistanceModel.LINEAR: 2>
    NONE: typing.ClassVar[AudioDistanceModel]  # value = <AudioDistanceModel.NONE: 0>
    __members__: typing.ClassVar[dict[str, AudioDistanceModel]]  # value = {'NONE': <AudioDistanceModel.NONE: 0>, 'INVERSE': <AudioDistanceModel.INVERSE: 1>, 'LINEAR': <AudioDistanceModel.LINEAR: 2>, 'EXPONENTIAL': <AudioDistanceModel.EXPONENTIAL: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class AudioListener(Object3D):
    def __init__(self) -> None:
        ...
    @property
    def master_volume(self) -> float:
        ...
    @master_volume.setter
    def master_volume(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class AxesHelper(LineSegments):
    def __init__(self, size: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
class BVH:
    @staticmethod
    def distance(a: BVH, b: BVH, m1: Matrix4 = ..., m2: Matrix4 = ..., max_distance: typing.SupportsFloat | typing.SupportsIndex = ...) -> float:
        """
        Smallest surface-to-surface distance: 0 when the meshes intersect, inf when either is empty or nothing is closer than max_distance. A finite max_distance seeds the pruning bound and is much cheaper than an exact search.
        """
    @staticmethod
    def distance_many(candidates: collections.abc.Sequence[BVH], targets: collections.abc.Sequence[BVH], max_distance: typing.SupportsFloat | typing.SupportsIndex = ...) -> numpy.typing.NDArray[numpy.float32]:
        """
        (C, T) float32 array of surface-to-surface distances; 0 where they intersect, inf beyond max_distance.
        """
    @staticmethod
    def intersect(a: BVH, b: BVH, m1: Matrix4 = ..., m2: Matrix4 = ..., accurate: bool = True) -> numpy.typing.NDArray[numpy.float32]:
        """
        (N, 3) float32 points of intersection, one per intersecting triangle pair. With accurate=True each point lies on both surfaces; accurate=False is the cheap conservative form that reports the centres of overlapping leaf boxes.
        """
    @staticmethod
    def intersect_many(candidates: collections.abc.Sequence[BVH], targets: collections.abc.Sequence[BVH], accurate: bool = True) -> list:
        """
        C x T nested list of (N, 3) float32 point arrays — the intersection points of every candidate/target pair. An empty (0, 3) array means no contact.
        """
    @staticmethod
    def intersect_pairs(a: BVH, b: BVH, m1: Matrix4 = ..., m2: Matrix4 = ..., accurate: bool = True) -> tuple[numpy.typing.NDArray[numpy.int32], numpy.typing.NDArray[numpy.int32], numpy.typing.NDArray[numpy.float32]]:
        """
        As intersect(), plus which triangles met: (idx_a, idx_b, points). The indices are -1 when accurate=False, which reports nodes rather than triangles.
        """
    @staticmethod
    def intersects(a: BVH, b: BVH, m1: Matrix4 = ..., m2: Matrix4 = ...) -> bool:
        """
        Do the two surfaces touch? Exact at the triangle level, early-exiting on the first hit.
        """
    @staticmethod
    def intersects_many(candidates: collections.abc.Sequence[BVH], targets: collections.abc.Sequence[BVH]) -> numpy.typing.NDArray[numpy.bool]:
        """
        (C, T) bool array: does candidate i touch target j?
        """
    def __init__(self, max_triangles_per_node: typing.SupportsInt | typing.SupportsIndex = 8, max_subdivisions: typing.SupportsInt | typing.SupportsIndex = 10) -> None:
        """
        max_triangles_per_node is the leaf size; smaller gives a deeper tree that prunes harder.
        """
    def __repr__(self) -> str:
        ...
    def bounding_box(self) -> Box3:
        """
        Root bounds, in the BVH's own space.
        """
    def build(self, geometry: BufferGeometry) -> None:
        """
        Build from a BufferGeometry (indexed or a raw soup). The triangles are COPIED, so the geometry may be dropped afterwards.
        """
    def build_arrays(self, positions: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], indices: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32] | None = None) -> None:
        """
        Build from raw arrays: `positions` flat or (N, 3) float, `indices` a flat or (M, 3) uint32 triangle list, or None for a soup (three consecutive vertices per triangle) — the layout of BufferGeometry.get_attribute('position') / get_index().
        """
    def collect_boxes(self, leaves_only: bool = False) -> list[Box3]:
        """
        Every node's box, for debug visualisation.
        """
    def intersect_box(self, box: Box3, matrix: Matrix4 = ...) -> list[int]:
        """
        Indices of the triangles whose boxes overlap `box`.
        """
    def raycast(self, ray: Ray, max_distance: typing.SupportsFloat | typing.SupportsIndex = ...) -> RayHit | None:
        """
        Closest hit, or None. The ray is in the BVH's own space.
        """
    def raycast_any(self, ray: Ray, max_distance: typing.SupportsFloat | typing.SupportsIndex = ...) -> bool:
        """
        Early-out occlusion query: is anything hit within max_distance?
        """
    @property
    def triangle_count(self) -> int:
        ...
class Background:
    @typing.overload
    def __init__(self, color: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, color: Color) -> None:
        ...
    @typing.overload
    def __init__(self, texture: Texture) -> None:
        ...
    def is_color(self) -> bool:
        ...
    def is_texture(self) -> bool:
        ...
class BakePose:
    """
    One capture viewpoint for bake_surface. `fov` is vertical degrees; the bake renders at the renderer's own framebuffer extent, so the horizontal field follows from its aspect.
    """
    position: Vector3
    target: Vector3
    up: Vector3
    def __init__(self, position: Vector3, target: Vector3, up: Vector3 = ..., fov: typing.SupportsFloat | typing.SupportsIndex = 55.0) -> None:
        ...
    @property
    def fov(self) -> float:
        ...
    @fov.setter
    def fov(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class BarkStyle:
    """
    Members:
    
      Furrowed
    
      Plated
    
      Papery
    """
    Furrowed: typing.ClassVar[BarkStyle]  # value = <BarkStyle.Furrowed: 0>
    Papery: typing.ClassVar[BarkStyle]  # value = <BarkStyle.Papery: 2>
    Plated: typing.ClassVar[BarkStyle]  # value = <BarkStyle.Plated: 1>
    __members__: typing.ClassVar[dict[str, BarkStyle]]  # value = {'Furrowed': <BarkStyle.Furrowed: 0>, 'Plated': <BarkStyle.Plated: 1>, 'Papery': <BarkStyle.Papery: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class BirdPlumage:
    leg: Color
    def __init__(self) -> None:
        ...
    @property
    def back(self) -> Color:
        """
        dorsal colour (LINEAR rgb; multiplies albedo)
        """
    @back.setter
    def back(self, arg0: Color) -> None:
        ...
    @property
    def belly(self) -> Color:
        """
        ventral colour
        """
    @belly.setter
    def belly(self, arg0: Color) -> None:
        ...
    @property
    def cap(self) -> Color:
        """
        crown + bill
        """
    @cap.setter
    def cap(self, arg0: Color) -> None:
        ...
    @property
    def cap_strength(self) -> float:
        ...
    @cap_strength.setter
    def cap_strength(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lightness_jitter(self) -> float:
        ...
    @lightness_jitter.setter
    def lightness_jitter(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def tail_band_dark(self) -> float:
        ...
    @tail_band_dark.setter
    def tail_band_dark(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wingtip_dark(self) -> float:
        """
        0..1 multiplier at the primaries
        """
    @wingtip_dark.setter
    def wingtip_dark(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class BirdRole:
    """
    Members:
    
      Follower
    
      Leader
    
      Loner
    """
    Follower: typing.ClassVar[BirdRole]  # value = <BirdRole.Follower: 0>
    Leader: typing.ClassVar[BirdRole]  # value = <BirdRole.Leader: 1>
    Loner: typing.ClassVar[BirdRole]  # value = <BirdRole.Loner: 2>
    __members__: typing.ClassVar[dict[str, BirdRole]]  # value = {'Follower': <BirdRole.Follower: 0>, 'Leader': <BirdRole.Leader: 1>, 'Loner': <BirdRole.Loner: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class BirdShape:
    def __init__(self) -> None:
        ...
    @property
    def body_length(self) -> float:
        """
        m, bill tip to tail tip
        """
    @body_length.setter
    def body_length(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def body_radius(self) -> float:
        """
        m, max half-width of the body spindle
        """
    @body_radius.setter
    def body_radius(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def tail_fork(self) -> float:
        """
        -1 forked .. +1 wedge; 0 = square
        """
    @tail_fork.setter
    def tail_fork(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wing_span(self) -> float:
        """
        m, tip to tip, fully extended
        """
    @wing_span.setter
    def wing_span(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class BirdState:
    """
    Members:
    
      Cruise
    
      Approach
    
      Flare
    
      Perched
    
      Launch
    
      Evade
    """
    Approach: typing.ClassVar[BirdState]  # value = <BirdState.Approach: 1>
    Cruise: typing.ClassVar[BirdState]  # value = <BirdState.Cruise: 0>
    Evade: typing.ClassVar[BirdState]  # value = <BirdState.Evade: 5>
    Flare: typing.ClassVar[BirdState]  # value = <BirdState.Flare: 2>
    Launch: typing.ClassVar[BirdState]  # value = <BirdState.Launch: 4>
    Perched: typing.ClassVar[BirdState]  # value = <BirdState.Perched: 3>
    __members__: typing.ClassVar[dict[str, BirdState]]  # value = {'Cruise': <BirdState.Cruise: 0>, 'Approach': <BirdState.Approach: 1>, 'Flare': <BirdState.Flare: 2>, 'Perched': <BirdState.Perched: 3>, 'Launch': <BirdState.Launch: 4>, 'Evade': <BirdState.Evade: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Blending:
    """
    Members:
    
      NoBlending
    
      Normal
    
      Additive
    
      Subtractive
    
      Multiply
    
      Custom
    """
    Additive: typing.ClassVar[Blending]  # value = <Blending.Additive: 2>
    Custom: typing.ClassVar[Blending]  # value = <Blending.Custom: 5>
    Multiply: typing.ClassVar[Blending]  # value = <Blending.Multiply: 4>
    NoBlending: typing.ClassVar[Blending]  # value = <Blending.NoBlending: 0>
    Normal: typing.ClassVar[Blending]  # value = <Blending.Normal: 1>
    Subtractive: typing.ClassVar[Blending]  # value = <Blending.Subtractive: 3>
    __members__: typing.ClassVar[dict[str, Blending]]  # value = {'NoBlending': <Blending.NoBlending: 0>, 'Normal': <Blending.Normal: 1>, 'Additive': <Blending.Additive: 2>, 'Subtractive': <Blending.Subtractive: 3>, 'Multiply': <Blending.Multiply: 4>, 'Custom': <Blending.Custom: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Box3:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, min: Vector3, max: Vector3) -> None:
        ...
    def contains_point(self, point: Vector3) -> bool:
        ...
    def get_center(self) -> Vector3:
        ...
    def get_size(self) -> Vector3:
        ...
    def is_empty(self) -> bool:
        ...
    def max(self) -> Vector3:
        ...
    def min(self) -> Vector3:
        ...
    def set_from_object(self, object: typing.Any, precise: bool = False) -> Box3:
        ...
class Box3Helper(LineSegments):
    def __init__(self, box: Box3, color: Color = ...) -> None:
        ...
class BoxGeometry(BufferGeometry):
    def __init__(self, width: typing.SupportsFloat | typing.SupportsIndex = 1.0, height: typing.SupportsFloat | typing.SupportsIndex = 1.0, depth: typing.SupportsFloat | typing.SupportsIndex = 1.0, width_segments: typing.SupportsInt | typing.SupportsIndex = 1, height_segments: typing.SupportsInt | typing.SupportsIndex = 1, depth_segments: typing.SupportsInt | typing.SupportsIndex = 1) -> None:
        ...
    @property
    def depth(self) -> float:
        ...
    @property
    def height(self) -> float:
        ...
    @property
    def width(self) -> float:
        ...
class BoxHelper(LineSegments):
    def __init__(self, object: typing.Any, color: Color = ...) -> None:
        ...
    def set_from_object(self, object: typing.Any) -> BoxHelper:
        ...
    def update(self) -> None:
        ...
class BranchingMode:
    """
    Members:
    
      Colonise
    
      Whorl
    """
    Colonise: typing.ClassVar[BranchingMode]  # value = <BranchingMode.Colonise: 0>
    Whorl: typing.ClassVar[BranchingMode]  # value = <BranchingMode.Whorl: 1>
    __members__: typing.ClassVar[dict[str, BranchingMode]]  # value = {'Colonise': <BranchingMode.Colonise: 0>, 'Whorl': <BranchingMode.Whorl: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class BufferGeometry:
    name: str
    def __init__(self) -> None:
        ...
    def apply_matrix4(self, matrix: Matrix4) -> BufferGeometry:
        ...
    def attribute_names(self) -> list[str]:
        """
        Sorted names of the attributes this geometry carries.
        """
    def center(self) -> BufferGeometry:
        ...
    def compute_bounding_box(self) -> None:
        ...
    def compute_bounding_sphere(self) -> None:
        ...
    def compute_vertex_normals(self) -> None:
        ...
    def dispose(self) -> None:
        ...
    def get_attribute(self, name: str) -> numpy.typing.NDArray[numpy.float32] | None:
        """
        Read a float attribute back as an (N, item_size) float32 array, or None if the geometry has no attribute of that name. Returns a copy.
        """
    def get_index(self) -> numpy.typing.NDArray[numpy.uint32] | None:
        """
        Read the index buffer back as a flat uint32 array, or None if the geometry is non-indexed (a triangle soup, which is what an STL always is).
        """
    def rotate_x(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> BufferGeometry:
        ...
    def rotate_y(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> BufferGeometry:
        ...
    def rotate_z(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> BufferGeometry:
        ...
    def scale(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> BufferGeometry:
        ...
    def set_attribute(self, name: str, data: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> BufferGeometry:
        ...
    def set_draw_range(self, start: typing.SupportsInt | typing.SupportsIndex, count: typing.SupportsInt | typing.SupportsIndex) -> BufferGeometry:
        """
        Render only vertices [start, start+count). Use with a fixed-capacity attribute + update_attribute.
        """
    def set_from_points(self, points: collections.abc.Sequence[Vector3]) -> BufferGeometry:
        ...
    def set_index(self, data: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32]) -> BufferGeometry:
        """
        Give this geometry an index buffer, so its vertices can be shared between faces. Accepts a flat or (M, 3) integer array. Validated against the vertex count of the attributes already set, so set_attribute first. Sets the draw range to the whole index buffer; once indexed, a later set_attribute leaves the draw range alone.
        """
    def to_non_indexed(self) -> BufferGeometry:
        """
        A new geometry with the index resolved into duplicated vertices (a triangle soup).
        """
    def translate(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> BufferGeometry:
        ...
    def update_attribute(self, name: str, data: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> BufferGeometry:
        ...
class Camera(Object3D):
    def update_projection_matrix(self) -> None:
        ...
    @property
    def far(self) -> float:
        ...
    @far.setter
    def far(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def far_plane(self) -> float:
        ...
    @far_plane.setter
    def far_plane(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def matrix_world_inverse(self) -> Matrix4:
        ...
    @property
    def near(self) -> float:
        ...
    @near.setter
    def near(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def near_plane(self) -> float:
        ...
    @near_plane.setter
    def near_plane(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def projection_matrix(self) -> Matrix4:
        ...
    @property
    def projection_matrix_inverse(self) -> Matrix4:
        ...
    @property
    def zoom(self) -> float:
        ...
    @zoom.setter
    def zoom(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CameraHelper(LineSegments):
    def __init__(self, camera: Camera) -> None:
        ...
    def update(self) -> None:
        ...
class Canvas:
    def __init__(self, title: str = 'threepp', width: typing.SupportsInt | typing.SupportsIndex = -1, height: typing.SupportsInt | typing.SupportsIndex = -1, antialiasing: typing.SupportsInt | typing.SupportsIndex = 4, vsync: bool = True, resizable: bool = True, headless: bool = False, fullscreen: bool = False, position: tuple[typing.SupportsInt | typing.SupportsIndex, typing.SupportsInt | typing.SupportsIndex] | None = None) -> None:
        """
        A window (or a hidden surface when headless=True). width/height default to half the primary monitor. fullscreen=True gives BORDERLESS windowed fullscreen: an undecorated, non-resizable window covering the primary monitor, which ignores width/height and resizable. It never changes the display mode, so alt-tab behaves like any other window. headless=True wins over it (there is no window to show). position=(x, y) places the window in screen coordinates instead of letting the OS choose (ignored with fullscreen=True; no-op on Wayland).
        """
    def animate(self, callback: collections.abc.Callable[[], None]) -> None:
        """
        Run the render loop, calling callback() every frame until the window closes.
        """
    def animate_once(self, callback: collections.abc.Callable[[], None]) -> bool:
        """
        Render a single frame; returns False when the app should quit.
        """
    def aspect(self) -> float:
        ...
    def close(self) -> None:
        ...
    def is_key_down(self, key: str) -> bool:
        """
        Poll whether a key is currently held — e.g. 'W','A','S','D','SPACE','UP','LEFT'. Query per-frame for continuous controls (WASD driving); never sticks.
        """
    def is_open(self) -> bool:
        ...
    def on_window_resize(self, callback: collections.abc.Callable[[typing.SupportsInt | typing.SupportsIndex, typing.SupportsInt | typing.SupportsIndex], None]) -> None:
        """
        Register callback(width, height), called when the window is resized. Use it to update the camera aspect (+ update_projection_matrix) and the renderer size.
        """
    def position(self) -> tuple[int, int]:
        ...
    def set_position(self, x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Move the window's outer top-left corner (title bar included) to (x, y) in screen coordinates — (0, 0) keeps the whole window on screen. No-op on Wayland.
        """
    def set_size(self, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def size(self) -> tuple[int, int]:
        ...
    @property
    def graphics_api(self) -> str:
        ...
class CapsuleGeometry(BufferGeometry):
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex = 0.5, length: typing.SupportsFloat | typing.SupportsIndex = 1.0, cap_segments: typing.SupportsInt | typing.SupportsIndex = 8, radial_segments: typing.SupportsInt | typing.SupportsIndex = 16) -> None:
        ...
    @property
    def length(self) -> float:
        ...
    @property
    def radius(self) -> float:
        ...
class CatmullRomCurve3(Curve3):
    class CurveType:
        """
        Members:
        
          centripetal
        
          chordal
        
          catmullrom
        """
        __members__: typing.ClassVar[dict[str, CatmullRomCurve3.CurveType]]  # value = {'centripetal': <CurveType.centripetal: 0>, 'chordal': <CurveType.chordal: 1>, 'catmullrom': <CurveType.catmullrom: 2>}
        catmullrom: typing.ClassVar[CatmullRomCurve3.CurveType]  # value = <CurveType.catmullrom: 2>
        centripetal: typing.ClassVar[CatmullRomCurve3.CurveType]  # value = <CurveType.centripetal: 0>
        chordal: typing.ClassVar[CatmullRomCurve3.CurveType]  # value = <CurveType.chordal: 1>
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    catmullrom: typing.ClassVar[CatmullRomCurve3.CurveType]  # value = <CurveType.catmullrom: 2>
    centripetal: typing.ClassVar[CatmullRomCurve3.CurveType]  # value = <CurveType.centripetal: 0>
    chordal: typing.ClassVar[CatmullRomCurve3.CurveType]  # value = <CurveType.chordal: 1>
    closed: bool
    curve_type: CatmullRomCurve3.CurveType
    def __init__(self, points: collections.abc.Sequence[Vector3] = [], closed: bool = False, curve_type: CatmullRomCurve3.CurveType = CatmullRomCurve3.CurveType.centripetal, tension: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def points(self) -> list[Vector3]:
        ...
    @points.setter
    def points(self, arg0: collections.abc.Sequence[Vector3]) -> None:
        ...
    @property
    def tension(self) -> float:
        ...
    @tension.setter
    def tension(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CircleGeometry(BufferGeometry):
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex = 1.0, segments: typing.SupportsInt | typing.SupportsIndex = 16, theta_start: typing.SupportsFloat | typing.SupportsIndex = 0.0, theta_length: typing.SupportsFloat | typing.SupportsIndex = 6.2831854820251465) -> None:
        ...
class Clock:
    running: bool
    def __init__(self, auto_start: bool = True) -> None:
        ...
    def get_delta(self) -> float:
        ...
    def get_elapsed_time(self) -> float:
        ...
    def start(self) -> None:
        ...
    def stop(self) -> None:
        ...
    @property
    def elapsed_time(self) -> float:
        ...
    @elapsed_time.setter
    def elapsed_time(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Color:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: Color) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, hex: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, r: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, style: str) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def clone(self) -> Color:
        ...
    def copy(self, color: Color) -> Color:
        ...
    def get_hex(self) -> int:
        ...
    def get_hex_string(self) -> str:
        ...
    def lerp(self, color: Color, alpha: typing.SupportsFloat | typing.SupportsIndex) -> Color:
        ...
    def set_hex(self, hex: typing.SupportsInt | typing.SupportsIndex) -> Color:
        ...
    def set_rgb(self, r: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> Color:
        ...
    def set_style(self, style: str) -> Color:
        ...
    @property
    def b(self) -> float:
        ...
    @b.setter
    def b(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def g(self) -> float:
        ...
    @g.setter
    def g(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def r(self) -> float:
        ...
    @r.setter
    def r(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class ColorKeyframeTrack(KeyframeTrack):
    def __init__(self, name: str, times: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], values: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], interpolation: Interpolation | None = None) -> None:
        ...
class ColorSpace:
    """
    Members:
    
      NoColorSpace
    
      Linear
    
      SRGB
    """
    Linear: typing.ClassVar[ColorSpace]  # value = <ColorSpace.Linear: 3000>
    NoColorSpace: typing.ClassVar[ColorSpace]  # value = <ColorSpace.NoColorSpace: -1>
    SRGB: typing.ClassVar[ColorSpace]  # value = <ColorSpace.SRGB: 3001>
    __members__: typing.ClassVar[dict[str, ColorSpace]]  # value = {'NoColorSpace': <ColorSpace.NoColorSpace: -1>, 'Linear': <ColorSpace.Linear: 3000>, 'SRGB': <ColorSpace.SRGB: 3001>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CombineOperation:
    """
    Members:
    
      Multiply
    
      Mix
    
      Add
    """
    Add: typing.ClassVar[CombineOperation]  # value = <CombineOperation.Add: 2>
    Mix: typing.ClassVar[CombineOperation]  # value = <CombineOperation.Mix: 1>
    Multiply: typing.ClassVar[CombineOperation]  # value = <CombineOperation.Multiply: 0>
    __members__: typing.ClassVar[dict[str, CombineOperation]]  # value = {'Multiply': <CombineOperation.Multiply: 0>, 'Mix': <CombineOperation.Mix: 1>, 'Add': <CombineOperation.Add: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ConeGeometry(BufferGeometry):
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex = 1.0, height: typing.SupportsFloat | typing.SupportsIndex = 1.0, radial_segments: typing.SupportsInt | typing.SupportsIndex = 16, height_segments: typing.SupportsInt | typing.SupportsIndex = 1, open_ended: bool = False, theta_start: typing.SupportsFloat | typing.SupportsIndex = 0.0, theta_length: typing.SupportsFloat | typing.SupportsIndex = 6.2831854820251465) -> None:
        ...
class ContactPoint:
    """
    One manifold point, world space.
    """
    @property
    def impulse(self) -> float:
        """
        Normal impulse magnitude at this point over the substep (N*s).
        """
    @property
    def normal(self) -> Vector3:
        """
        Unit normal pointing INTO the sensor's body.
        """
    @property
    def position(self) -> Vector3:
        ...
class ContactSample:
    """
    One contact reading.
    """
    def __repr__(self) -> str:
        ...
    @property
    def force(self) -> Vector3:
        """
        Mean contact force over the interval (N). Zero while asleep.
        """
    @property
    def in_contact(self) -> bool:
        """
        Latched touch state — stays true while resting, including after the contact pair falls asleep and stops producing points.
        """
    @property
    def observed_points(self) -> int:
        """
        Manifold points seen this interval, before the report cap.
        """
    @property
    def points(self) -> list[ContactPoint]:
        """
        Reported manifold points (capped; see observed_points).
        """
    @property
    def t(self) -> float:
        ...
    @property
    def touch_began(self) -> bool:
        ...
    @property
    def touch_ended(self) -> bool:
        ...
class ContactSensor(Sensor):
    """
    Reports whether the attached body is touching anything, where, and how hard — a bumper, a foot-contact switch, a grasp detector.
    """
    def __init__(self, node: typing.Any, rate_hz: typing.SupportsFloat | typing.SupportsIndex = 0.0, buffer_capacity: typing.SupportsInt | typing.SupportsIndex = 256) -> None:
        """
        Attach to `node`; the rigid body in its ancestry is the one whose contacts are reported.
        """
    def drain(self) -> list[ContactSample]:
        """
        Move all buffered ContactSamples (oldest-first) out as a list; empties the buffer.
        """
    def latest(self) -> ContactSample | None:
        """
        The most recent ContactSample, or None. Survives drain().
        """
    def reset(self) -> None:
        """
        Re-arm: clear the buffer, latch and pending observations.
        """
    @property
    def attached(self) -> bool:
        """
        True while bound to a live rigid body.
        """
    @property
    def available(self) -> int:
        """
        Number of buffered samples.
        """
    @property
    def in_contact(self) -> bool:
        """
        Current latched touch state — the cheap read for a control loop that only wants a foot-down boolean.
        """
    @property
    def rate_hz(self) -> float:
        ...
    @rate_hz.setter
    def rate_hz(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def touch_count(self) -> int:
        """
        How many distinct bodies are currently being touched.
        """
class ConvexGeometry(BufferGeometry):
    def __init__(self, points: collections.abc.Sequence[Vector3]) -> None:
        ...
    def contains_point(self, point: Vector3, tolerance: typing.SupportsFloat | typing.SupportsIndex = -1.0) -> bool:
        ...
class CrownShape:
    """
    Members:
    
      Sphere
    
      Ellipsoid
    
      Cone
    
      Hemisphere
    
      Cylinder
    """
    Cone: typing.ClassVar[CrownShape]  # value = <CrownShape.Cone: 2>
    Cylinder: typing.ClassVar[CrownShape]  # value = <CrownShape.Cylinder: 4>
    Ellipsoid: typing.ClassVar[CrownShape]  # value = <CrownShape.Ellipsoid: 1>
    Hemisphere: typing.ClassVar[CrownShape]  # value = <CrownShape.Hemisphere: 3>
    Sphere: typing.ClassVar[CrownShape]  # value = <CrownShape.Sphere: 0>
    __members__: typing.ClassVar[dict[str, CrownShape]]  # value = {'Sphere': <CrownShape.Sphere: 0>, 'Ellipsoid': <CrownShape.Ellipsoid: 1>, 'Cone': <CrownShape.Cone: 2>, 'Hemisphere': <CrownShape.Hemisphere: 3>, 'Cylinder': <CrownShape.Cylinder: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Curve2:
    def get_length(self) -> float:
        """
        Total arc length, from the cached table.
        """
    def get_lengths(self) -> list[float]:
        """
        Cumulative segment lengths, arc_length_divisions + 1 of them.
        """
    def get_point(self, t: typing.SupportsFloat | typing.SupportsIndex) -> Vector2:
        """
        Point at curve parameter t in [0, 1] (NOT arc length).
        """
    def get_point_at(self, u: typing.SupportsFloat | typing.SupportsIndex) -> Vector2:
        """
        Point at fraction u in [0, 1] of the ARC LENGTH — equidistant.
        """
    def get_points(self, divisions: typing.SupportsInt | typing.SupportsIndex = 5) -> list[Vector2]:
        """
        divisions + 1 points, evenly spaced in t.
        """
    def get_spaced_points(self, divisions: typing.SupportsInt | typing.SupportsIndex = 5) -> list[Vector2]:
        """
        divisions + 1 points, evenly spaced along the curve.
        """
    def get_tangent(self, t: typing.SupportsFloat | typing.SupportsIndex) -> Vector2:
        """
        Unit tangent at curve parameter t.
        """
    def get_tangent_at(self, u: typing.SupportsFloat | typing.SupportsIndex) -> Vector2:
        """
        Unit tangent at fraction u of the arc length.
        """
    def update_arc_lengths(self) -> None:
        """
        Rebuild the arc-length table after editing the curve in place.
        """
    @property
    def arc_length_divisions(self) -> int:
        """
        Segments used to build the arc-length table get_point_at and get_length work from.
        """
    @arc_length_divisions.setter
    def arc_length_divisions(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class Curve3:
    def get_length(self) -> float:
        """
        Total arc length, from the cached table.
        """
    def get_lengths(self) -> list[float]:
        """
        Cumulative segment lengths, arc_length_divisions + 1 of them.
        """
    def get_point(self, t: typing.SupportsFloat | typing.SupportsIndex) -> Vector3:
        """
        Point at curve parameter t in [0, 1] (NOT arc length).
        """
    def get_point_at(self, u: typing.SupportsFloat | typing.SupportsIndex) -> Vector3:
        """
        Point at fraction u in [0, 1] of the ARC LENGTH — equidistant.
        """
    def get_points(self, divisions: typing.SupportsInt | typing.SupportsIndex = 5) -> list[Vector3]:
        """
        divisions + 1 points, evenly spaced in t.
        """
    def get_spaced_points(self, divisions: typing.SupportsInt | typing.SupportsIndex = 5) -> list[Vector3]:
        """
        divisions + 1 points, evenly spaced along the curve.
        """
    def get_tangent(self, t: typing.SupportsFloat | typing.SupportsIndex) -> Vector3:
        """
        Unit tangent at curve parameter t.
        """
    def get_tangent_at(self, u: typing.SupportsFloat | typing.SupportsIndex) -> Vector3:
        """
        Unit tangent at fraction u of the arc length.
        """
    def update_arc_lengths(self) -> None:
        """
        Rebuild the arc-length table after editing the curve in place.
        """
    @property
    def arc_length_divisions(self) -> int:
        """
        Segments used to build the arc-length table get_point_at and get_length work from.
        """
    @arc_length_divisions.setter
    def arc_length_divisions(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class CylinderGeometry(BufferGeometry):
    def __init__(self, radius_top: typing.SupportsFloat | typing.SupportsIndex = 1.0, radius_bottom: typing.SupportsFloat | typing.SupportsIndex = 1.0, height: typing.SupportsFloat | typing.SupportsIndex = 1.0, radial_segments: typing.SupportsInt | typing.SupportsIndex = 16, height_segments: typing.SupportsInt | typing.SupportsIndex = 1, open_ended: bool = False, theta_start: typing.SupportsFloat | typing.SupportsIndex = 0.0, theta_length: typing.SupportsFloat | typing.SupportsIndex = 6.2831854820251465) -> None:
        ...
class DepthPacking:
    """
    Members:
    
      Basic
    
      RGBA
    """
    Basic: typing.ClassVar[DepthPacking]  # value = <DepthPacking.Basic: 3200>
    RGBA: typing.ClassVar[DepthPacking]  # value = <DepthPacking.RGBA: 3201>
    __members__: typing.ClassVar[dict[str, DepthPacking]]  # value = {'Basic': <DepthPacking.Basic: 3200>, 'RGBA': <DepthPacking.RGBA: 3201>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DepthSensor(Object3D, Sensor):
    """
    GPU depth sensor: scans the scene from its own pose and returns a world-space point cloud.
    
    Beam convention: beams are cast down the sensor's LOCAL -Z axis (camera convention), and look_at() honours it -- look_at(target) turns the beams toward the target, exactly as it would for a camera. (Older releases gave the sensor the plain-Object3D convention instead, turning local +Z toward the target -- beams aimed exactly backwards, every scan an empty (0, 3) cloud -- and callers compensated by aiming at the mirror point 2*position - target. Such call sites now aim backwards: pass the target itself.)
    """
    def __init__(self, fov_y: typing.SupportsFloat | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex, near: typing.SupportsFloat | typing.SupportsIndex = 0.10000000149011612, far: typing.SupportsFloat | typing.SupportsIndex = 100.0) -> None:
        """
        Depth sensor with a vertical FOV (deg), output resolution, and the near/far RANGE bounds (m) it reports in -- a blind sphere of radius near out to a max range of far, inclusive at both ends and identical on GL and Vulkan. These are ranges, not view-space clip planes: an off-axis surface is judged by its distance, not by its depth along the view axis.
        """
    def reset_noise(self) -> None:
        """
        Re-seed the noise stream from noise.seed and clear last_scan_time — call between episodes so two runs with the same seed produce the same clouds.
        """
    def scan(self, renderer: typing.Any, scene: Scene) -> numpy.typing.NDArray[numpy.float32]:
        """
        Depth scan -> (N,3) float32 world-space hit points (N = points that hit within far). Works with a GLRenderer (raster depth) or a VulkanRenderer (path-traced through the renderer's acceleration structure -- render() the scene at least once first). Beams go down the sensor's local -Z; aim them with look_at(target) (see the class docstring).
        """
    def scan_begin(self, renderer: typing.Any, scene: Scene) -> bool:
        """
        Fire a scan without waiting for it. Call it AFTER render() on the frame you want sampled: the beams snapshot the sensor's pose (and stamp last_scan_time) here, not at scan_collect. Take delivery with scan_collect on a later frame — on Vulkan a collect with at least one intervening render() is essentially free, whereas scan() blocks on the readback and so pays for every frame already queued behind the fence.
        
            if sensor.scan_due and not sensor.scan_pending:
                sensor.scan_begin(renderer, scene)
            if sensor.scan_ready(renderer):
                pts = sensor.scan_collect(renderer)
        
        Returns True when the cloud is ALREADY complete — the raster (GLRenderer) path has nothing to pipeline, so it does the whole scan here. On Vulkan it returns False and the cloud arrives at a later scan_collect. Either way scan_collect is what hands the points over, so the loop above is correct on both backends; only the frame the cloud lands on differs.
        """
    def scan_collect(self, renderer: typing.Any) -> numpy.typing.NDArray[numpy.float32]:
        """
        Take delivery of a scan_begin -> (N,3) float32 world-space hit points, exactly like scan(). Returns an EMPTY (0,3) array when there was nothing to deliver: no scan outstanding, or a scan_begin the backend refused because too many traces were already in flight. Check scan_ready first (or accept the empty array as 'not yet').
        """
    def scan_ready(self, renderer: typing.Any) -> bool:
        """
        True when a fired scan can be collected without waiting. A poll, never a wait. False when no scan is outstanding. Raster: True as soon as scan_begin has run.
        """
    def scan_rgbd(self, renderer: typing.Any, scene: Scene) -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32]]:
        """
        RGB-D scan -> (points (N,3) float32 world-space, colors (N,3) float32 in [0,1]). On GL the colors are sampled sRGB; on Vulkan they are LIDAR intensity as greyscale.
        """
    @property
    def far(self) -> float:
        ...
    @property
    def fov(self) -> float:
        ...
    @property
    def height(self) -> int:
        ...
    @property
    def last_scan_time(self) -> float:
        """
        Sim time (s) stamped on the most recent scan — the timestamp to record alongside the cloud. Register the sensor with a PhysxWorld or drive `sim_time` yourself; see threepp.Sensor.
        """
    @property
    def near(self) -> float:
        ...
    @property
    def noise(self) -> RangeNoiseModel:
        """
        The full RangeNoiseModel (stddev, stddev_per_metre, bias, seed). Set `seed` to make a captured dataset replayable; the stream re-seeds on the next scan when the seed changes.
        """
    @noise.setter
    def noise(self, arg0: RangeNoiseModel) -> None:
        ...
    @property
    def range_noise(self) -> float:
        """
        Gaussian range-noise std-dev in metres (0 = perfect sensor). Shorthand for noise.stddev.
        """
    @range_noise.setter
    def range_noise(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def scan_due(self) -> bool:
        """
        True when the rate gate says a scan is due (always true unless rate_hz is set and the sensor is registered with a PhysxWorld).
        """
    @property
    def scan_pending(self) -> bool:
        """
        True between scan_begin and its scan_collect. Firing again while one is outstanding throws the earlier scan away, so a driver on a rate gate should skip a due scan while this is True.
        """
    @property
    def width(self) -> int:
        ...
class DirectionalLight(Light):
    def __init__(self, color: Color = ..., intensity: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    def get_target(self) -> Object3D:
        ...
    def set_shadow_bias(self, bias: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    def set_shadow_frustum(self, left: typing.SupportsFloat | typing.SupportsIndex = -10.0, right: typing.SupportsFloat | typing.SupportsIndex = 10.0, top: typing.SupportsFloat | typing.SupportsIndex = 10.0, bottom: typing.SupportsFloat | typing.SupportsIndex = -10.0) -> None:
        """
        Resize the directional-light shadow ortho frustum. Call after creation; default ±1 clips almost everything.
        """
    def set_target(self, target: Object3D) -> None:
        ...
class DirectionalLightHelper(Object3D):
    def __init__(self, light: DirectionalLight, size: typing.SupportsFloat | typing.SupportsIndex = 1.0, color: Color | None = None) -> None:
        ...
    def update(self) -> None:
        ...
class DisplacedMesh(Mesh):
    class HullExclusion:
        def set_pose(self, x: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex = 0.0, yaw: typing.SupportsFloat | typing.SupportsIndex = 0.0, pitch: typing.SupportsFloat | typing.SupportsIndex = 0.0, roll: typing.SupportsFloat | typing.SupportsIndex = 0.0, half_length: typing.SupportsFloat | typing.SupportsIndex = -1.0, half_beam: typing.SupportsFloat | typing.SupportsIndex = -1.0) -> None:
            """
            One-call per-frame update: world XZ, waterline height, heading (rad, 0 = +Z) and the two plane angles. half_length/half_beam are left alone when negative, so the usual pattern is to set them once and then call set_pose(x, z, y, yaw, pitch, roll) every frame.
            """
        @property
        def center_x(self) -> float:
            """
            World X of the hull's centre (the exclusion origin).
            """
        @center_x.setter
        def center_x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def center_y(self) -> float:
            """
            World Y of the hull's DESIGN WATERLINE at (center_x, center_z) — the height the water should meet the hull at, not the deck. 0 (the default) = the ocean rest plane, i.e. the pre-2026-08 behaviour.
            """
        @center_y.setter
        def center_y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def center_z(self) -> float:
            ...
        @center_z.setter
        def center_z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def cos_yaw(self) -> float:
            ...
        @cos_yaw.setter
        def cos_yaw(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def half_beam(self) -> float:
            """
            Half the beam (m).
            """
        @half_beam.setter
        def half_beam(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def half_length(self) -> float:
            """
            Half the vessel's length (m). 0 DISABLES hull exclusion and the wake.
            """
        @half_length.setter
        def half_length(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def pitch(self) -> float:
            """
            Waterline-plane pitch (rad), POSITIVE = bow up. Clamped to +/-1 rad.
            """
        @pitch.setter
        def pitch(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def roll(self) -> float:
            """
            Waterline-plane roll (rad), POSITIVE = starboard up. Clamped to +/-1 rad.
            """
        @roll.setter
        def roll(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def sin_yaw(self) -> float:
            """
            sin/cos of the heading: forward = (sin_yaw, cos_yaw), starboard = (cos_yaw, -sin_yaw).
            """
        @sin_yaw.setter
        def sin_yaw(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
    class MeshWarp:
        @property
        def center_x(self) -> float:
            ...
        @center_x.setter
        def center_x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def center_z(self) -> float:
            ...
        @center_z.setter
        def center_z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def coef_a(self) -> float:
            """
            1 = uniform; lower = denser centre (~0.1 ≈ 10 cm centre / 2.7 m edge).
            """
        @coef_a.setter
        def coef_a(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def half_range(self) -> float:
            """
            Half-extent the warp covers; 0 = uniform grid (disabled).
            """
        @half_range.setter
        def half_range(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
    class Params:
        @property
        def choppiness(self) -> float:
            """
            Horizontal pull / crest sharpness; ~0.45 realistic, >=0.8 folds crests.
            """
        @choppiness.setter
        def choppiness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def fetch(self) -> float:
            """
            Fetch (m of open water upwind). 0 = fully developed Phillips/PM sea (long swell, peak ~8 V^2/g). Finite = JONSWAP young sea: shorter, steeper waves with more energy at the 10-40 m scale and less swell; 20e3-40e3 reads as a coastal sea. Live-tunable; saturates at ~1600 V^2.
            """
        @fetch.setter
        def fetch(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def foam_amount(self) -> float:
            """
            Natural whitecap foam scale, live-tunable (1 = ocean whitewater, 0 = none; wake/disturbance foam unaffected). Ocean auto-derives ~size/300.
            """
        @foam_amount.setter
        def foam_amount(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def texture_size_0(self) -> int:
            """
            Cascade-0 FFT resolution (power of two).
            """
        @texture_size_0.setter
        def texture_size_0(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        @property
        def texture_size_1(self) -> int:
            """
            Cascade-1 FFT resolution (power of two).
            """
        @texture_size_1.setter
        def texture_size_1(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        @property
        def texture_size_2(self) -> int:
            """
            Cascade-2 FFT resolution (power of two).
            """
        @texture_size_2.setter
        def texture_size_2(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        @property
        def tile_size_0(self) -> float:
            """
            Cascade-0 world tile (m); the largest swell wavelength. Must be > 0.
            """
        @tile_size_0.setter
        def tile_size_0(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def tile_size_1(self) -> float:
            """
            Cascade-1 tile (m); 0 disables.
            """
        @tile_size_1.setter
        def tile_size_1(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def tile_size_2(self) -> float:
            """
            Cascade-2 tile (m); 0 disables.
            """
        @tile_size_2.setter
        def tile_size_2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def wave_scale(self) -> float:
            """
            Global wave-height multiplier; 1.0 = physical.
            """
        @wave_scale.setter
        def wave_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def wind_speed(self) -> float:
            """
            Wind speed (m/s); the dominant wave-height lever (Phillips amplitude ~ V^4).
            """
        @wind_speed.setter
        def wind_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def wind_theta(self) -> float:
            """
            Wind direction (radians, 0 = +X).
            """
        @wind_theta.setter
        def wind_theta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
    class VesselWake:
        enabled: bool
        @property
        def forward_speed(self) -> float:
            """
            m/s along +heading. The whole wake fades out below ~0.5 m/s; 0 disables it.
            """
        @forward_speed.setter
        def forward_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def trail(self) -> list[DisplacedMesh.WakeSample]:
            """
            Historical pose snapshots (list of WakeSample). Reading COPIES and writing REPLACES — mutating the returned list does not write through. Use mesh.add_wake_sample()/age_wake() for the per-frame path.
            """
        @trail.setter
        def trail(self, arg0: collections.abc.Sequence[DisplacedMesh.WakeSample]) -> None:
            ...
    class WakeSample:
        def __init__(self) -> None:
            ...
        def __repr__(self) -> str:
            ...
        @property
        def age(self) -> float:
            """
            Seconds since emission.
            """
        @age.setter
        def age(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def cos_yaw(self) -> float:
            ...
        @cos_yaw.setter
        def cos_yaw(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def sin_yaw(self) -> float:
            ...
        @sin_yaw.setter
        def sin_yaw(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def speed(self) -> float:
            """
            m/s along +heading at emission.
            """
        @speed.setter
        def speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def world_x(self) -> float:
            ...
        @world_x.setter
        def world_x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def world_z(self) -> float:
            ...
        @world_z.setter
        def world_z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
    def __init__(self, geometry: BufferGeometry, material: typing.Any) -> None:
        """
        Low-level constructor. Most callers want Ocean instead, which builds the plane + water material + cascade defaults for you.
        """
    def add_foam_disturbance(self, world_x: typing.SupportsFloat | typing.SupportsIndex, world_z: typing.SupportsFloat | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex, intensity: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Splat a gaussian foam blob at a world XZ (radius m, intensity in [0,1]).
        """
    def add_wake_sample(self, x: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex, sin_yaw: typing.SupportsFloat | typing.SupportsIndex, cos_yaw: typing.SupportsFloat | typing.SupportsIndex, speed: typing.SupportsFloat | typing.SupportsIndex, max_samples: typing.SupportsInt | typing.SupportsIndex = 64) -> None:
        """
        Emit one wake snapshot at the vessel's current pose (age 0), dropping the oldest once the trail is full. The renderer's hard cap is 64 samples; overflow beyond it is dropped silently on upload. The C++ showcase's cadence is 10 Hz OR every 1 m travelled, whichever fires first.
        """
    def age_wake(self, dt: typing.SupportsFloat | typing.SupportsIndex, max_age: typing.SupportsFloat | typing.SupportsIndex = 6.0, max_samples: typing.SupportsInt | typing.SupportsIndex = 64) -> int:
        """
        Age every trail sample by dt, drop anything older than max_age, and keep at most max_samples (newest). Returns the surviving count. Call once per frame.
        """
    def clear_foam_disturbances(self) -> None:
        ...
    def clear_wake(self) -> None:
        """
        Drop the whole trail (e.g. after teleporting the vessel, so the wake does not stretch across the map).
        """
    def sample_height(self, world_x: typing.SupportsFloat | typing.SupportsIndex, world_z: typing.SupportsFloat | typing.SupportsIndex, cascade_mask: typing.SupportsInt | typing.SupportsIndex = 7) -> float:
        """
        Combined wave height (m) at a world XZ. cascade_mask selects cascades (bit i = cascade i). Returns 0 until a Vulkan render() has run.
        """
    def sample_wake_height(self, world_x: typing.SupportsFloat | typing.SupportsIndex, world_z: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        CPU mirror of the shader's wake height (bow bump + bow V-wedge + the trail-summed Kelvin V) at a world XZ. 0 with no active vessel or below the speed gate. Add to sample_height() to make a buoy bob through a passing wake.
        """
    @property
    def hull_exclusion(self) -> DisplacedMesh.HullExclusion:
        """
        The vessel's footprint + waterline plane; set each frame before render(). half_length = 0 (the default) disables it AND the wake.
        """
    @property
    def params(self) -> DisplacedMesh.Params:
        ...
    @property
    def wake(self) -> DisplacedMesh.VesselWake:
        """
        Kelvin V-wake / bow bump / foam trail. Shares the hull_exclusion pose, so set that first.
        """
    @property
    def warp(self) -> DisplacedMesh.MeshWarp:
        ...
class EXRLoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str, flip_y: bool = True) -> Texture:
        """
        Load an OpenEXR .exr equirectangular environment as a float Texture.
        """
class ErosionType:
    """
    Members:
    
      Off
    
      Hydraulic
    
      Thermal
    
      Both
    """
    Both: typing.ClassVar[ErosionType]  # value = <ErosionType.Both: 3>
    Hydraulic: typing.ClassVar[ErosionType]  # value = <ErosionType.Hydraulic: 1>
    Off: typing.ClassVar[ErosionType]  # value = <ErosionType.Off: 0>
    Thermal: typing.ClassVar[ErosionType]  # value = <ErosionType.Thermal: 2>
    __members__: typing.ClassVar[dict[str, ErosionType]]  # value = {'Off': <ErosionType.Off: 0>, 'Hydraulic': <ErosionType.Hydraulic: 1>, 'Thermal': <ErosionType.Thermal: 2>, 'Both': <ErosionType.Both: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Euler:
    order: RotationOrder
    def __init__(self, x: typing.SupportsFloat | typing.SupportsIndex = 0.0, y: typing.SupportsFloat | typing.SupportsIndex = 0.0, z: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def set(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> Euler:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def z(self) -> float:
        ...
    @z.setter
    def z(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class ExtrudeGeometry(BufferGeometry):
    def __init__(self, shapes: typing.Any, depth: typing.SupportsFloat | typing.SupportsIndex = 1.0, steps: typing.SupportsInt | typing.SupportsIndex = 1, bevel_enabled: bool = True, curve_segments: typing.SupportsInt | typing.SupportsIndex = 12, bevel_thickness: typing.SupportsFloat | typing.SupportsIndex = 0.20000000298023224, bevel_size: typing.SupportsFloat | typing.SupportsIndex = 0.10000000149011612, bevel_offset: typing.SupportsFloat | typing.SupportsIndex = 0.0, bevel_segments: typing.SupportsInt | typing.SupportsIndex = 3) -> None:
        ...
class FBXLoader:
    class MaterialMode:
        """
        Members:
        
          Auto
        
          Phong
        
          PBR
        """
        Auto: typing.ClassVar[FBXLoader.MaterialMode]  # value = <MaterialMode.Auto: 0>
        PBR: typing.ClassVar[FBXLoader.MaterialMode]  # value = <MaterialMode.PBR: 2>
        Phong: typing.ClassVar[FBXLoader.MaterialMode]  # value = <MaterialMode.Phong: 1>
        __members__: typing.ClassVar[dict[str, FBXLoader.MaterialMode]]  # value = {'Auto': <MaterialMode.Auto: 0>, 'Phong': <MaterialMode.Phong: 1>, 'PBR': <MaterialMode.PBR: 2>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def __init__(self) -> None:
        ...
    def load(self, path: str) -> Group:
        """
        Load an .fbx file as a Group.
        """
    @property
    def emissive_scale(self) -> float:
        """
        Multiplier on every emissive material's intensity (1.0 = file values).
        """
    @emissive_scale.setter
    def emissive_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def material_mode(self) -> FBXLoader.MaterialMode:
        """
        How the FBX SPECULAR slot is interpreted (Auto/Phong/PBR).
        """
    @material_mode.setter
    def material_mode(self, arg0: FBXLoader.MaterialMode) -> None:
        ...
class Filter:
    """
    Members:
    
      Nearest
    
      NearestMipmapNearest
    
      NearestMipmapLinear
    
      Linear
    
      LinearMipmapNearest
    
      LinearMipmapLinear
    """
    Linear: typing.ClassVar[Filter]  # value = <Filter.Linear: 1006>
    LinearMipmapLinear: typing.ClassVar[Filter]  # value = <Filter.LinearMipmapLinear: 1008>
    LinearMipmapNearest: typing.ClassVar[Filter]  # value = <Filter.LinearMipmapNearest: 1007>
    Nearest: typing.ClassVar[Filter]  # value = <Filter.Nearest: 1003>
    NearestMipmapLinear: typing.ClassVar[Filter]  # value = <Filter.NearestMipmapLinear: 1005>
    NearestMipmapNearest: typing.ClassVar[Filter]  # value = <Filter.NearestMipmapNearest: 1004>
    __members__: typing.ClassVar[dict[str, Filter]]  # value = {'Nearest': <Filter.Nearest: 1003>, 'NearestMipmapNearest': <Filter.NearestMipmapNearest: 1004>, 'NearestMipmapLinear': <Filter.NearestMipmapLinear: 1005>, 'Linear': <Filter.Linear: 1006>, 'LinearMipmapNearest': <Filter.LinearMipmapNearest: 1007>, 'LinearMipmapLinear': <Filter.LinearMipmapLinear: 1008>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Flock(Mesh):
    @typing.overload
    def __init__(self, params: FlockParams) -> None:
        """
        Build a flock from a FlockParams. Add it to a Scene and call update(dt) once per frame.
        """
    @typing.overload
    def __init__(self, bird_count: typing.SupportsInt | typing.SupportsIndex = 18, home: Vector3 = ..., roam_radius: typing.SupportsFloat | typing.SupportsIndex = 42.0, cruise_altitude: typing.SupportsFloat | typing.SupportsIndex = 14.0, cruise_speed: typing.SupportsFloat | typing.SupportsIndex = 9.0, perching: bool = True, seed: typing.SupportsInt | typing.SupportsIndex = 1337) -> None:
        """
        Convenience constructor over the handful of fields most hosts set. For the rest, build a FlockParams and pass that instead.
        """
    def add_perch(self, world_pos: Vector3, world_normal: Vector3, walkable: bool) -> None:
        """
        Add one perch by hand, instead of (or as well as) baking.
        """
    def bake_complete(self) -> bool:
        ...
    def bake_perches(self, scene_root: Object3D) -> None:
        """
        Scan a scene for landable surfaces (async; poll bake_complete()).
        """
    def bake_perches_blocking(self, scene_root: Object3D) -> None:
        """
        Same, but finish before returning.
        """
    def bake_progress(self) -> float:
        ...
    def bird_count(self) -> int:
        ...
    def bird_position(self, i: typing.SupportsInt | typing.SupportsIndex) -> Vector3:
        ...
    def bird_velocity(self, i: typing.SupportsInt | typing.SupportsIndex) -> Vector3:
        ...
    def flying_count(self) -> int:
        ...
    def perch_count(self) -> int:
        ...
    def perched_count(self) -> int:
        ...
    def role_of(self, i: typing.SupportsInt | typing.SupportsIndex) -> BirdRole:
        ...
    def set_disturbance_source(self, source: Object3D) -> None:
        """
        Birds take flight when this object comes within flight_initiation_distance. Pass None to clear.
        """
    def set_observer(self, camera: Camera) -> None:
        """
        Soft repulsion from the camera, so birds do not fly through the lens.
        """
    def set_perch_filter(self, predicate: collections.abc.Callable[[Mesh], bool]) -> None:
        """
        predicate(mesh) -> bool, deciding which meshes may be landed on.
        """
    def set_wind(self, x: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        World XZ wind direction; perched birds turn to face into it.
        """
    def stalled_updates(self) -> int:
        ...
    def startle(self, epicentre: Vector3, radius: typing.SupportsFloat | typing.SupportsIndex = 1000000000.0, strength: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        """
        Flush the flock away from a world point.
        """
    def state_of(self, i: typing.SupportsInt | typing.SupportsIndex) -> BirdState:
        ...
    def update(self, dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Advance the simulation and rebake the geometry. CALL ONCE PER FRAME. dt is clamped internally to [0, 0.05] s; dt <= 0 is a no-op.
        """
    def update_count(self) -> int:
        ...
    @property
    def params(self) -> FlockParams:
        """
        A COPY of the construction params (read-only).
        """
class FlockParams:
    birds_cast_shadow: bool
    gait: Gait
    perching: bool
    plumage: BirdPlumage
    shape: BirdShape
    def __init__(self) -> None:
        ...
    @property
    def abort_chance(self) -> float:
        ...
    @abort_chance.setter
    def abort_chance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def altitude_spread(self) -> float:
        ...
    @altitude_spread.setter
    def altitude_spread(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def bird_count(self) -> int:
        """
        Hard-clamped to [0, 256]: neighbour search is O(N^2).
        """
    @bird_count.setter
    def bird_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cruise_altitude(self) -> float:
        """
        m above the baked ground under home
        """
    @cruise_altitude.setter
    def cruise_altitude(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cruise_speed(self) -> float:
        ...
    @cruise_speed.setter
    def cruise_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def flight_initiation_distance(self) -> float:
        ...
    @flight_initiation_distance.setter
    def flight_initiation_distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ground_bias(self) -> float:
        ...
    @ground_bias.setter
    def ground_bias(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def home(self) -> Vector3:
        """
        Centre of the loiter volume (world m); drifts at runtime.
        """
    @home.setter
    def home(self, arg0: Vector3) -> None:
        ...
    @property
    def home_drift_rate(self) -> float:
        ...
    @home_drift_rate.setter
    def home_drift_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def launch_contagion(self) -> float:
        ...
    @launch_contagion.setter
    def launch_contagion(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def leader_fraction(self) -> float:
        ...
    @leader_fraction.setter
    def leader_fraction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lod_far_distance(self) -> float:
        ...
    @lod_far_distance.setter
    def lod_far_distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def loner_fraction(self) -> float:
        ...
    @loner_fraction.setter
    def loner_fraction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lookahead_time(self) -> float:
        ...
    @lookahead_time.setter
    def lookahead_time(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mass_kg(self) -> float:
        """
        drives wingbeat_hz allometrically
        """
    @mass_kg.setter
    def mass_kg(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_accel_along(self) -> float:
        ...
    @max_accel_along.setter
    def max_accel_along(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_accel_lateral(self) -> float:
        ...
    @max_accel_lateral.setter
    def max_accel_lateral(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_bank(self) -> float:
        ...
    @max_bank.setter
    def max_bank(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_perched_fraction(self) -> float:
        ...
    @max_perched_fraction.setter
    def max_perched_fraction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_speed(self) -> float:
        ...
    @max_speed.setter
    def max_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_turn_rate(self) -> float:
        ...
    @max_turn_rate.setter
    def max_turn_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def min_ground_clearance(self) -> float:
        ...
    @min_ground_clearance.setter
    def min_ground_clearance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def min_speed(self) -> float:
        ...
    @min_speed.setter
    def min_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def neighbour_count(self) -> int:
        """
        Topological, not metric: a fixed NUMBER of nearest neighbours.
        """
    @neighbour_count.setter
    def neighbour_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def neighbour_radius(self) -> float:
        ...
    @neighbour_radius.setter
    def neighbour_radius(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def obstacle_margin(self) -> float:
        ...
    @obstacle_margin.setter
    def obstacle_margin(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def perch_contagion(self) -> float:
        ...
    @perch_contagion.setter
    def perch_contagion(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def perch_interval_max(self) -> float:
        ...
    @perch_interval_max.setter
    def perch_interval_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def perch_interval_min(self) -> float:
        ...
    @perch_interval_min.setter
    def perch_interval_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def perch_search_radius(self) -> float:
        ...
    @perch_search_radius.setter
    def perch_search_radius(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rest_interval_max(self) -> float:
        ...
    @rest_interval_max.setter
    def rest_interval_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rest_interval_min(self) -> float:
        ...
    @rest_interval_min.setter
    def rest_interval_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def roam_radius(self) -> float:
        ...
    @roam_radius.setter
    def roam_radius(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def seed(self) -> int:
        ...
    @seed.setter
    def seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def separation_distance(self) -> float:
        ...
    @separation_distance.setter
    def separation_distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def size_variation(self) -> float:
        ...
    @size_variation.setter
    def size_variation(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def startle_wave_speed(self) -> float:
        ...
    @startle_wave_speed.setter
    def startle_wave_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def w_alignment(self) -> float:
        ...
    @w_alignment.setter
    def w_alignment(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def w_altitude(self) -> float:
        ...
    @w_altitude.setter
    def w_altitude(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def w_bounds(self) -> float:
        ...
    @w_bounds.setter
    def w_bounds(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def w_cohesion(self) -> float:
        ...
    @w_cohesion.setter
    def w_cohesion(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def w_goal(self) -> float:
        ...
    @w_goal.setter
    def w_goal(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def w_ground(self) -> float:
        ...
    @w_ground.setter
    def w_ground(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def w_observer(self) -> float:
        ...
    @w_observer.setter
    def w_observer(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def w_obstacle(self) -> float:
        ...
    @w_obstacle.setter
    def w_obstacle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def w_separation(self) -> float:
        ...
    @w_separation.setter
    def w_separation(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def w_wander(self) -> float:
        ...
    @w_wander.setter
    def w_wander(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wind(self) -> Vector2:
        """
        world XZ; perched birds face into it
        """
    @wind.setter
    def wind(self, arg0: Vector2) -> None:
        ...
    @property
    def wingbeat_hz(self) -> float:
        """
        0 => derived from mass_kg
        """
    @wingbeat_hz.setter
    def wingbeat_hz(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Fog:
    color: Color
    def __init__(self, color: Color, near: typing.SupportsFloat | typing.SupportsIndex = 1.0, far: typing.SupportsFloat | typing.SupportsIndex = 1000.0) -> None:
        ...
    @property
    def far(self) -> float:
        ...
    @far.setter
    def far(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def near(self) -> float:
        ...
    @near.setter
    def near(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class FogExp2:
    color: Color
    def __init__(self, color: Color, density: typing.SupportsFloat | typing.SupportsIndex = 0.0002500000118743628) -> None:
        """
        Exponential (Beer-Lambert) participating-media fog. density = σ_t (extinction per metre). On Vulkan this drives full volumetric single-scattering; on GL it falls back to GL_EXP2.
        """
    @property
    def density(self) -> float:
        ...
    @density.setter
    def density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Font:
    def __repr__(self) -> str:
        ...
    @property
    def family_name(self) -> str:
        ...
class FontLoader:
    def __init__(self) -> None:
        ...
    def default_font(self) -> Font:
        """
        The built-in embedded font (no file needed).
        """
    def load(self, path: str) -> typing.Any:
        """
        Load a typeface (.json) or TrueType (.ttf) font; None on failure.
        """
class ForceTorqueSensor(Sensor):
    """
    Load cell on an articulation joint: the wrench the parent link transmits to the child through their joint, as the solver computed it. The input to force control, admittance control and payload estimation.
    """
    @typing.overload
    def __init__(self, node: typing.Any, articulation: ..., link: ..., rate_hz: typing.SupportsFloat | typing.SupportsIndex = 0.0, buffer_capacity: typing.SupportsInt | typing.SupportsIndex = 2048) -> None:
        """
        Measure `link`'s inbound joint. Raises if `link` is the root, or if the articulation has not been finalized.
        """
    @typing.overload
    def __init__(self, node: typing.Any, joint: ..., rate_hz: typing.SupportsFloat | typing.SupportsIndex = 0.0, buffer_capacity: typing.SupportsInt | typing.SupportsIndex = 2048) -> None:
        """
        The same load cell across a plain Joint: the wrench is the solver's constraint force on it, in world axes.
        """
    def drain(self) -> list[WrenchSample]:
        """
        Move all buffered WrenchSamples (oldest-first) out as a list; empties the buffer.
        """
    def drain_array(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Drain all buffered samples as a (N, 7) float64 numpy array with columns [t, fx, fy, fz, tx, ty, tz]. Empties the buffer.
        """
    def latest(self) -> WrenchSample | None:
        """
        The most recent WrenchSample, or None. Survives drain().
        """
    def reset(self) -> None:
        """
        Re-arm: clear the buffer and re-seed the noise.
        """
    @property
    def available(self) -> int:
        """
        Number of buffered samples.
        """
    @property
    def force_noise(self) -> NoiseModel:
        """
        NoiseModel for the force channel (N). Change then call reset().
        """
    @force_noise.setter
    def force_noise(self, arg0: NoiseModel) -> None:
        ...
    @property
    def torque_noise(self) -> NoiseModel:
        """
        NoiseModel for the torque channel (N*m). Change then call reset().
        """
    @torque_noise.setter
    def torque_noise(self, arg0: NoiseModel) -> None:
        ...
class FrameChannel:
    """
    A per-frame image the Vulkan renderer can export for zero-copy CUDA/torch consumption.
    
    Members:
    
      Color
    
      Depth
    
      Normal
    
      Motion
    
      Ids
    
      Albedo
    
      SplatDepth
    """
    Albedo: typing.ClassVar[FrameChannel]  # value = <FrameChannel.Albedo: 5>
    Color: typing.ClassVar[FrameChannel]  # value = <FrameChannel.Color: 0>
    Depth: typing.ClassVar[FrameChannel]  # value = <FrameChannel.Depth: 1>
    Ids: typing.ClassVar[FrameChannel]  # value = <FrameChannel.Ids: 4>
    Motion: typing.ClassVar[FrameChannel]  # value = <FrameChannel.Motion: 3>
    Normal: typing.ClassVar[FrameChannel]  # value = <FrameChannel.Normal: 2>
    SplatDepth: typing.ClassVar[FrameChannel]  # value = <FrameChannel.SplatDepth: 6>
    __members__: typing.ClassVar[dict[str, FrameChannel]]  # value = {'Color': <FrameChannel.Color: 0>, 'Depth': <FrameChannel.Depth: 1>, 'Normal': <FrameChannel.Normal: 2>, 'Motion': <FrameChannel.Motion: 3>, 'Ids': <FrameChannel.Ids: 4>, 'Albedo': <FrameChannel.Albedo: 5>, 'SplatDepth': <FrameChannel.SplatDepth: 6>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GLRenderer:
    auto_clear: bool
    check_shader_errors: bool
    shadow_map_enabled: bool
    sort_objects: bool
    tone_mapping: ToneMapping
    def __init__(self, canvas: Canvas) -> None:
        ...
    def clear(self, color: bool = True, depth: bool = True, stencil: bool = True) -> None:
        ...
    def gl_buffer_id(self, geometry: BufferGeometry, attribute: str) -> typing.Any:
        ...
    def read_pixels(self, flip: bool = True) -> numpy.typing.NDArray[numpy.uint8]:
        ...
    def render(self, scene: Object3D, camera: Camera) -> None:
        ...
    def save_frame(self, path: str) -> None:
        ...
    def set_clear_color(self, color: Color, alpha: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    def set_pixel_ratio(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_scissor(self, x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_scissor_test(self, enabled: bool) -> None:
        ...
    def set_size(self, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_viewport(self, x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def size(self) -> tuple[int, int]:
        ...
    @property
    def tone_mapping_exposure(self) -> float:
        ...
    @tone_mapping_exposure.setter
    def tone_mapping_exposure(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class GLTFLoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str) -> GLTFResult:
        ...
class GLTFResult:
    def __repr__(self) -> str:
        ...
    @property
    def animations(self) -> list[AnimationClip]:
        """
        All AnimationClips in the file.
        """
    @property
    def scene(self) -> Group:
        """
        Root Group of the loaded model.
        """
    @property
    def scenes(self) -> list[Group]:
        """
        All scenes in the file.
        """
class Gait:
    """
    Members:
    
      Hop
    
      Walk
    """
    Hop: typing.ClassVar[Gait]  # value = <Gait.Hop: 0>
    Walk: typing.ClassVar[Gait]  # value = <Gait.Walk: 1>
    __members__: typing.ClassVar[dict[str, Gait]]  # value = {'Hop': <Gait.Hop: 0>, 'Walk': <Gait.Walk: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GrassMesh(Mesh):
    wind_dir: Vector2
    def __init__(self, geometry: BufferGeometry, material: typing.Any) -> None:
        ...
    @property
    def time(self) -> float:
        """
        Animation clock (seconds); set per frame to advance the wind.
        """
    @time.setter
    def time(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wind_strength(self) -> float:
        ...
    @wind_strength.setter
    def wind_strength(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class GridHelper(LineSegments):
    def __init__(self, size: typing.SupportsInt | typing.SupportsIndex = 10, divisions: typing.SupportsInt | typing.SupportsIndex = 10, color1: Color = ..., color2: Color = ...) -> None:
        ...
class Group(Object3D):
    def __init__(self) -> None:
        ...
class HemisphereLight(Light):
    ground_color: Color
    def __init__(self, sky_color: Color = ..., ground_color: Color = ..., intensity: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
class HemisphereLightHelper(Object3D):
    color: Color | None
    def __init__(self, light: HemisphereLight, size: typing.SupportsFloat | typing.SupportsIndex = 1.0, color: Color | None = None) -> None:
        ...
    def update(self) -> None:
        ...
class HorizontalAlignment:
    """
    Members:
    
      Left
    
      Center
    
      Right
    """
    Center: typing.ClassVar[HorizontalAlignment]  # value = <HorizontalAlignment.Center: 1>
    Left: typing.ClassVar[HorizontalAlignment]  # value = <HorizontalAlignment.Left: 0>
    Right: typing.ClassVar[HorizontalAlignment]  # value = <HorizontalAlignment.Right: 2>
    __members__: typing.ClassVar[dict[str, HorizontalAlignment]]  # value = {'Left': <HorizontalAlignment.Left: 0>, 'Center': <HorizontalAlignment.Center: 1>, 'Right': <HorizontalAlignment.Right: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class IcosahedronGeometry(BufferGeometry):
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex = 1.0, detail: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
class IcpOptions:
    def __init__(self) -> None:
        ...
    @property
    def max_correspondence_distance(self) -> float:
        ...
    @max_correspondence_distance.setter
    def max_correspondence_distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_iterations(self) -> int:
        ...
    @max_iterations.setter
    def max_iterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def min_correspondence_distance(self) -> float:
        ...
    @min_correspondence_distance.setter
    def min_correspondence_distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def robust_sigma(self) -> float:
        ...
    @robust_sigma.setter
    def robust_sigma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rotation_tolerance(self) -> float:
        ...
    @rotation_tolerance.setter
    def rotation_tolerance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def translation_tolerance(self) -> float:
        ...
    @translation_tolerance.setter
    def translation_tolerance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class IcpResult:
    def __repr__(self) -> str:
        ...
    @property
    def converged(self) -> bool:
        ...
    @property
    def correspondences(self) -> int:
        ...
    @property
    def iterations(self) -> int:
        ...
class IkOptions:
    task: IkTask
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def damping(self) -> float:
        """
        DLS damping; must be > 0 — it is what lets the solve succeed at a singularity instead of flinging the arm.
        """
    @damping.setter
    def damping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_iterations(self) -> int:
        ...
    @max_iterations.setter
    def max_iterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_joint_speed(self) -> float:
        """
        Per-call joint speed cap in rad/s or m/s, applied against the joint values as they arrived. Zero disables it; needs a non-zero dt.
        """
    @max_joint_speed.setter
    def max_joint_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_orientation_step(self) -> float:
        """
        radians
        """
    @max_orientation_step.setter
    def max_orientation_step(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_position_step(self) -> float:
        """
        Largest correction one iteration will attempt, in metres. A Gauss-Newton step is a LOCAL statement; clamping keeps it inside the trust region. Travel per solve is bounded by max_iterations * step. Zero disables the clamp.
        """
    @max_position_step.setter
    def max_position_step(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def null_space_damping(self) -> float:
        """
        Damping for the null-space PROJECTION — much smaller than `damping`, or the posture bias leaks into the tool pose and the arm never reports convergence.
        """
    @null_space_damping.setter
    def null_space_damping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def orientation_tolerance(self) -> float:
        """
        radians
        """
    @orientation_tolerance.setter
    def orientation_tolerance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def orientation_weight(self) -> float:
        """
        Weight on the orientation rows relative to position. Below 1 the solver reaches the point first and straightens up after, which reads as natural motion.
        """
    @orientation_weight.setter
    def orientation_weight(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def position_tolerance(self) -> float:
        """
        metres
        """
    @position_tolerance.setter
    def position_tolerance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def prismatic_step(self) -> float:
        """
        Finite-difference probe for prismatic joints (metres).
        """
    @prismatic_step.setter
    def prismatic_step(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rest_pose(self) -> list[float]:
        """
        Rest posture, indexed by GLOBAL dof like every other joint vector.
        """
    @rest_pose.setter
    def rest_pose(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def rest_pose_gain(self) -> float:
        """
        Null-space rest-posture pull per iteration; zero disables it. Only does anything on a redundant arm.
        """
    @rest_pose_gain.setter
    def rest_pose_gain(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def revolute_step(self) -> float:
        """
        Finite-difference probe for revolute joints (radians).
        """
    @revolute_step.setter
    def revolute_step(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def target_axis(self) -> Vector3:
        """
        AxisAlign only, in WORLD space.
        """
    @target_axis.setter
    def target_axis(self, arg0: Vector3) -> None:
        ...
    @property
    def tool_axis(self) -> Vector3:
        """
        AxisAlign only, in the TOOL frame (+Z is the URDF convention for an approach direction).
        """
    @tool_axis.setter
    def tool_axis(self, arg0: Vector3) -> None:
        ...
    @property
    def tool_offset(self) -> Matrix4:
        """
        Flange -> tool centre point (Matrix4). The solve drives the TCP, so a tool of any length or mounting is described here rather than in the URDF.
        """
    @tool_offset.setter
    def tool_offset(self, arg0: Matrix4) -> None:
        ...
class IkResult:
    def __repr__(self) -> str:
        ...
    @property
    def converged(self) -> bool:
        """
        Both errors are inside tolerance — judged on the TRUE error, so the step clamp never fakes it.
        """
    @property
    def iterations(self) -> int:
        ...
    @property
    def orientation_error(self) -> float:
        """
        radians
        """
    @property
    def position_error(self) -> float:
        """
        metres
        """
class IkSolver:
    def __init__(self, robot: Robot, options: IkOptions = ...) -> None:
        """
        Damped-least-squares IK over the robot's root-to-end-effector chain.
        
        Joint ranges and the solvable DOF set are cached at construction, so a Robot that is re-parsed or given a new end effector needs a fresh solver.
        """
    def solve(self, q: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], target: typing.Any, dt: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> tuple[list[float], IkResult]:
        """
        Step q toward placing the tool at `target` (Vector3 = point, Matrix4 = full pose). q is a FULL joint vector indexed by global dof; only `solved_dofs` are modified. Returns (new_q, IkResult) — the input list is left alone. `dt` is used solely by the max_joint_speed cap.
        """
    def tool_transform(self, q: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> Matrix4:
        """
        The tool centre point for a joint vector, in the robot's PARENT frame (FK composed with tool_offset). Call robot.update_matrix() first if the robot itself has moved.
        """
    @property
    def options(self) -> IkOptions:
        """
        Solver options. Reading gives a COPY — assign back to change them.
        """
    @options.setter
    def options(self, arg1: IkOptions) -> None:
        ...
    @property
    def solved_dofs(self) -> list[int]:
        """
        The DOF indices this solver is allowed to move (robot.chain_dofs).
        """
class IkTask:
    """
    How much of the tool pose the solve must reproduce.
    
    Members:
    
      Position : 3-DOF: reach the point, any orientation.
    
      AxisAlign : 5-DOF: reach the point AND aim the tool axis; spin about that axis is left free — what a drill, a suction cup or a symmetric two-finger grasp wants.
    
      Pose : 6-DOF: reproduce the full target transform.
    """
    AxisAlign: typing.ClassVar[IkTask]  # value = <IkTask.AxisAlign: 1>
    Pose: typing.ClassVar[IkTask]  # value = <IkTask.Pose: 2>
    Position: typing.ClassVar[IkTask]  # value = <IkTask.Position: 0>
    __members__: typing.ClassVar[dict[str, IkTask]]  # value = {'Position': <IkTask.Position: 0>, 'AxisAlign': <IkTask.AxisAlign: 1>, 'Pose': <IkTask.Pose: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ImguiContext:
    def __init__(self, canvas: Canvas, renderer: typing.Any) -> None:
        """
        Dear ImGui UI. Pass the renderer (GLRenderer or VulkanRenderer); create the context AFTER the renderer. The same call works on both backends.
        """
    def render(self, draw: collections.abc.Callable[[], None]) -> None:
        """
        Build + draw one UI frame; call inside animate() after renderer.render().
        """
    @property
    def want_capture_keyboard(self) -> bool:
        ...
    @property
    def want_capture_mouse(self) -> bool:
        """
        True when the pointer is over UI — gate OrbitControls on `not ui.want_capture_mouse`.
        """
class Imu(Sensor):
    """
    Gyroscope + accelerometer attached to an Object3D. Its measurement frame is that node's world frame; register it with a PhysxWorld (world.register_sensor) AFTER adding the body it rides. Each substep it measures the body's angular velocity (rad/s) and the specific force at the sensor point (m/s^2, gravity reaction + lever-arm terms), in the sensor frame. A level body at rest reads accel (0, +9.81, 0); free fall reads ~0.
    """
    def __init__(self, node: typing.Any, rate_hz: typing.SupportsFloat | typing.SupportsIndex = 0.0, buffer_capacity: typing.SupportsInt | typing.SupportsIndex = 2048) -> None:
        """
        Attach an IMU to `node` (its world frame is the sensor frame). rate_hz=0 samples every physics substep; buffer_capacity is the ring depth (oldest dropped on overflow).
        """
    def drain(self) -> list[ImuSample]:
        """
        Move all buffered ImuSamples (oldest-first) out as a list; empties the buffer.
        """
    def drain_array(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Drain all buffered samples as a (N, 7) float64 numpy array with columns [t, gx, gy, gz, ax, ay, az]. Empties the buffer.
        """
    def latest(self) -> ImuSample | None:
        """
        The most recent ImuSample, or None. Survives drain().
        """
    def reset(self) -> None:
        """
        Re-arm after an episode reset or a noise change: clears the finite-difference history + buffer and re-seeds the noise from the current configs.
        """
    @property
    def accel_noise(self) -> NoiseModel:
        """
        NoiseModel for the accelerometer (m/s^2 units). Change then call reset().
        """
    @accel_noise.setter
    def accel_noise(self, arg0: NoiseModel) -> None:
        ...
    @property
    def attached(self) -> bool:
        """
        True while bound to a live rigid body. False before registering, after unregistering, and after the body was removed from the world (remove_actor) — sampling is a silent no-op in all three cases.
        """
    @property
    def available(self) -> int:
        """
        Number of buffered samples.
        """
    @property
    def gyro_noise(self) -> NoiseModel:
        """
        NoiseModel for the gyroscope (rad/s units). Change then call reset().
        """
    @gyro_noise.setter
    def gyro_noise(self, arg0: NoiseModel) -> None:
        ...
    @property
    def rate_hz(self) -> float:
        ...
    @rate_hz.setter
    def rate_hz(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class ImuSample:
    """
    One IMU measurement. t: sim time (s). angular_velocity: rad/s. linear_acceleration: specific force (m/s^2). Both in the sensor frame.
    """
    def __repr__(self) -> str:
        ...
    @property
    def angular_velocity(self) -> Vector3:
        ...
    @property
    def linear_acceleration(self) -> Vector3:
        ...
    @property
    def t(self) -> float:
        ...
class InstancedMesh(Mesh):
    def __init__(self, geometry: BufferGeometry, material: typing.Any, count: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def get_matrix_at(self, index: typing.SupportsInt | typing.SupportsIndex) -> Matrix4:
        ...
    def instance_color_needs_update(self) -> None:
        ...
    def instance_matrix_needs_update(self) -> None:
        ...
    def set_color_at(self, index: typing.SupportsInt | typing.SupportsIndex, color: Color) -> None:
        ...
    def set_count(self, count: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_matrix_at(self, index: typing.SupportsInt | typing.SupportsIndex, matrix: Matrix4) -> None:
        ...
    @property
    def count(self) -> int:
        ...
class Interpolation:
    """
    Keyframe interpolation mode.
    
    Members:
    
      DISCRETE
    
      LINEAR
    
      SMOOTH
    """
    DISCRETE: typing.ClassVar[Interpolation]  # value = <Interpolation.DISCRETE: 0>
    LINEAR: typing.ClassVar[Interpolation]  # value = <Interpolation.LINEAR: 1>
    SMOOTH: typing.ClassVar[Interpolation]  # value = <Interpolation.SMOOTH: 2>
    __members__: typing.ClassVar[dict[str, Interpolation]]  # value = {'DISCRETE': <Interpolation.DISCRETE: 0>, 'LINEAR': <Interpolation.LINEAR: 1>, 'SMOOTH': <Interpolation.SMOOTH: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class IsoMesh:
    def __init__(self) -> None:
        ...
    @property
    def empty(self) -> bool:
        ...
    @property
    def normals(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        (N,3) float32 per-vertex normals.
        """
    @property
    def positions(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        (N,3) float32 vertex positions (3 per triangle, unwelded).
        """
class Joint:
    """
    A joint between two rigid bodies, or one body and the world (pass None for that side). fixed/revolute/prismatic/spherical ride one configured PxD6Joint; distance is a tether. The frame is world-space: anchor at `position`, hinge/slide axis along the frame's local X (`rotation` aims it). Valid only while its world lives.
    """
    class Params:
        """
        Everything a joint is configured with. Angles in radians, lengths in metres. The drive is force-mode PD: `target` acts through stiffness, `velocity` through damping. break_force / break_torque of 0 = unbreakable.
        """
        collide: bool
        limited: bool
        type: Joint.Type
        def __init__(self) -> None:
            ...
        @property
        def break_force(self) -> float:
            ...
        @break_force.setter
        def break_force(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def break_torque(self) -> float:
            ...
        @break_torque.setter
        def break_torque(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def cone_y(self) -> float:
            ...
        @cone_y.setter
        def cone_y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def cone_z(self) -> float:
            ...
        @cone_z.setter
        def cone_z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def damping(self) -> float:
            ...
        @damping.setter
        def damping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def lower(self) -> float:
            ...
        @lower.setter
        def lower(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def max_force(self) -> float:
            ...
        @max_force.setter
        def max_force(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def stiffness(self) -> float:
            ...
        @stiffness.setter
        def stiffness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def target(self) -> float:
            ...
        @target.setter
        def target(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def upper(self) -> float:
            ...
        @upper.setter
        def upper(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def velocity(self) -> float:
            ...
        @velocity.setter
        def velocity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
    class Type:
        """
        Members:
        
          FIXED
        
          REVOLUTE
        
          PRISMATIC
        
          SPHERICAL
        
          DISTANCE
        """
        DISTANCE: typing.ClassVar[Joint.Type]  # value = <Type.DISTANCE: 4>
        FIXED: typing.ClassVar[Joint.Type]  # value = <Type.FIXED: 0>
        PRISMATIC: typing.ClassVar[Joint.Type]  # value = <Type.PRISMATIC: 2>
        REVOLUTE: typing.ClassVar[Joint.Type]  # value = <Type.REVOLUTE: 1>
        SPHERICAL: typing.ClassVar[Joint.Type]  # value = <Type.SPHERICAL: 3>
        __members__: typing.ClassVar[dict[str, Joint.Type]]  # value = {'FIXED': <Type.FIXED: 0>, 'REVOLUTE': <Type.REVOLUTE: 1>, 'PRISMATIC': <Type.PRISMATIC: 2>, 'SPHERICAL': <Type.SPHERICAL: 3>, 'DISTANCE': <Type.DISTANCE: 4>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def __init__(self, world: ..., body_a: typing.Any, body_b: typing.Any, position: Vector3, rotation: Quaternion = ..., params: Joint.Params = ...) -> None:
        """
        Create a joint in `world` between body_a and body_b (either may be None, meaning the world itself — not both). Default params are a fixed weld.
        """
    def break_wrench(self) -> tuple[Vector3, Vector3]:
        """
        (force N, torque N*m) the solver applied on the step that BROKE the joint - the true failure load, past the break threshold. Zero until broken.
        """
    def reaction(self) -> tuple[Vector3, Vector3]:
        """
        (force N, torque N*m) the solver applied to hold the constraint on the last step, world axes. Zero once broken - the failure load is break_wrench().
        """
    def set_drive_target(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        PD setpoint along the motion axis (radians / metres). Acts through stiffness — inert at zero stiffness.
        """
    def set_drive_velocity(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Velocity setpoint (rad/s or m/s). Acts through damping.
        """
    @property
    def broken(self) -> bool:
        """
        True once the solver exceeded the break threshold; the constraint never comes back.
        """
    @property
    def position(self) -> float:
        """
        The joint coordinate: radians (revolute / a spherical's twist), metres (prismatic), anchor distance (distance).
        """
    @property
    def type(self) -> Joint.Type:
        ...
    @property
    def velocity(self) -> float:
        """
        Its rate: rad/s or m/s, same convention as position.
        """
class JointEncoder(Sensor):
    """
    Joint position/velocity encoder on an articulation link's inbound joint. Adds what a real encoder has and Articulation.joint_positions does not: tick quantization, noise, rate gating and buffering.
    """
    @typing.overload
    def __init__(self, node: typing.Any, link: ..., rate_hz: typing.SupportsFloat | typing.SupportsIndex = 0.0, buffer_capacity: typing.SupportsInt | typing.SupportsIndex = 2048) -> None:
        """
        Attach to `node` (normally the mesh bound to the joint's child link) and measure `link`'s inbound joint. Raises if `link` is the root.
        """
    @typing.overload
    def __init__(self, node: typing.Any, joint: ..., rate_hz: typing.SupportsFloat | typing.SupportsIndex = 0.0, buffer_capacity: typing.SupportsInt | typing.SupportsIndex = 2048) -> None:
        """
        The same encoder on a plain Joint: the coordinate is the joint's scalar axis (twist for a hinge, displacement for a slider, anchor distance for a tether).
        """
    def drain(self) -> list[JointSample]:
        """
        Move all buffered JointSamples (oldest-first) out as a list; empties the buffer.
        """
    def drain_array(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Drain all buffered samples as a (N, 3) float64 numpy array with columns [t, position, velocity]. Empties the buffer.
        """
    def latest(self) -> JointSample | None:
        """
        The most recent JointSample, or None. Survives drain().
        """
    def reset(self) -> None:
        """
        Re-arm after an episode reset: clear the buffer and differentiation history and re-seed the noise.
        """
    def set_counts_per_rev(self, counts: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set `resolution` from a rotary encoder's counts per revolution.
        """
    @property
    def available(self) -> int:
        """
        Number of buffered samples.
        """
    @property
    def differentiate_velocity(self) -> bool:
        """
        True (default): differentiate the quantized, noisy position, as a real encoder-fed controller does. False: report the simulator's true velocity.
        """
    @differentiate_velocity.setter
    def differentiate_velocity(self, arg0: bool) -> None:
        ...
    @property
    def position_noise(self) -> NoiseModel:
        """
        Position noise; only the X component of each Vector3 is used.
        """
    @position_noise.setter
    def position_noise(self, arg0: NoiseModel) -> None:
        ...
    @property
    def rate_hz(self) -> float:
        ...
    @rate_hz.setter
    def rate_hz(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def resolution(self) -> float:
        """
        Quantization step: rad (revolute) or m (prismatic) per tick. 0 = ideal continuous encoder.
        """
    @resolution.setter
    def resolution(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def velocity_noise(self) -> NoiseModel:
        """
        Velocity noise; applied only when differentiate_velocity is False.
        """
    @velocity_noise.setter
    def velocity_noise(self, arg0: NoiseModel) -> None:
        ...
class JointInfo:
    def __repr__(self) -> str:
        ...
    @property
    def axis(self) -> Vector3:
        ...
    @property
    def child(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def parent(self) -> str:
        ...
    @property
    def range(self) -> typing.Any:
        ...
    @property
    def type(self) -> JointType:
        ...
class JointRange:
    def __repr__(self) -> str:
        ...
    def clamp(self, value: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def mid(self) -> float:
        ...
    @property
    def max(self) -> float:
        ...
    @property
    def min(self) -> float:
        ...
class JointSample:
    """
    One encoder reading. Units follow the joint: rad and rad/s for a revolute joint, m and m/s for a prismatic one.
    """
    def __repr__(self) -> str:
        ...
    @property
    def position(self) -> float:
        ...
    @property
    def t(self) -> float:
        ...
    @property
    def velocity(self) -> float:
        ...
class JointType:
    """
    Members:
    
      Revolute
    
      Prismatic
    
      Fixed
    """
    Fixed: typing.ClassVar[JointType]  # value = <JointType.Fixed: 2>
    Prismatic: typing.ClassVar[JointType]  # value = <JointType.Prismatic: 1>
    Revolute: typing.ClassVar[JointType]  # value = <JointType.Revolute: 0>
    __members__: typing.ClassVar[dict[str, JointType]]  # value = {'Revolute': <JointType.Revolute: 0>, 'Prismatic': <JointType.Prismatic: 1>, 'Fixed': <JointType.Fixed: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class KeyframeTrack:
    def make_additive(self) -> None:
        """
        Convert this track's values to deltas from its first frame (for additive layering).
        """
    def set_interpolation(self, interpolation: Interpolation) -> None:
        ...
    @property
    def interpolation(self) -> Interpolation:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def times(self) -> list[float]:
        ...
    @property
    def value_type_name(self) -> str:
        ...
    @property
    def values(self) -> list[float]:
        ...
class Layers:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def disable(self, channel: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def disable_all(self) -> None:
        ...
    def enable(self, channel: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def enable_all(self) -> None:
        ...
    def is_enabled(self, channel: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def mask(self) -> int:
        ...
    def set(self, channel: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Membership of exactly this one channel.
        """
    def test(self, layers: Layers) -> bool:
        """
        True if the two masks share a channel.
        """
    def toggle(self, channel: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class LeafShape:
    """
    Members:
    
      Ovate
    
      Lobed
    
      Serrate
    
      Lanceolate
    """
    Lanceolate: typing.ClassVar[LeafShape]  # value = <LeafShape.Lanceolate: 3>
    Lobed: typing.ClassVar[LeafShape]  # value = <LeafShape.Lobed: 1>
    Ovate: typing.ClassVar[LeafShape]  # value = <LeafShape.Ovate: 0>
    Serrate: typing.ClassVar[LeafShape]  # value = <LeafShape.Serrate: 2>
    __members__: typing.ClassVar[dict[str, LeafShape]]  # value = {'Ovate': <LeafShape.Ovate: 0>, 'Lobed': <LeafShape.Lobed: 1>, 'Serrate': <LeafShape.Serrate: 2>, 'Lanceolate': <LeafShape.Lanceolate: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LeafStyle:
    """
    Members:
    
      Quad
    
      Cluster
    
      CrossQuad
    
      Blob
    
      Frond
    """
    Blob: typing.ClassVar[LeafStyle]  # value = <LeafStyle.Blob: 3>
    Cluster: typing.ClassVar[LeafStyle]  # value = <LeafStyle.Cluster: 1>
    CrossQuad: typing.ClassVar[LeafStyle]  # value = <LeafStyle.CrossQuad: 2>
    Frond: typing.ClassVar[LeafStyle]  # value = <LeafStyle.Frond: 4>
    Quad: typing.ClassVar[LeafStyle]  # value = <LeafStyle.Quad: 0>
    __members__: typing.ClassVar[dict[str, LeafStyle]]  # value = {'Quad': <LeafStyle.Quad: 0>, 'Cluster': <LeafStyle.Cluster: 1>, 'CrossQuad': <LeafStyle.CrossQuad: 2>, 'Blob': <LeafStyle.Blob: 3>, 'Frond': <LeafStyle.Frond: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LidarBeam:
    direction: Vector3
    origin: Vector3
    def __init__(self, origin: Vector3 = ..., direction: Vector3 = ...) -> None:
        ...
class LidarModel:
    @staticmethod
    def hdl32e() -> LidarModel:
        """
        Velodyne HDL-32E: 32 beams, -30.67..+10.67deg.
        """
    @staticmethod
    def os0_128() -> LidarModel:
        """
        Ouster OS0-128: 128 beams, +/-45deg elevation.
        """
    @staticmethod
    def os1_64() -> LidarModel:
        """
        Ouster OS1-64: 64 beams, +/-22.5deg elevation.
        """
    @staticmethod
    def vlp16() -> LidarModel:
        """
        Velodyne VLP-16: 16 beams, +/-15deg elevation.
        """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def azimuth_max(self) -> float:
        ...
    @azimuth_max.setter
    def azimuth_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def azimuth_min(self) -> float:
        ...
    @azimuth_min.setter
    def azimuth_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def azimuth_resolution(self) -> float:
        ...
    @azimuth_resolution.setter
    def azimuth_resolution(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def elevation_angles(self) -> list[float]:
        ...
    @elevation_angles.setter
    def elevation_angles(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
class LidarParams:
    def __init__(self) -> None:
        ...
    @property
    def atmospheric_extinction(self) -> float:
        ...
    @atmospheric_extinction.setter
    def atmospheric_extinction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def beam_divergence_mrad(self) -> float:
        ...
    @beam_divergence_mrad.setter
    def beam_divergence_mrad(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def detector_threshold(self) -> float:
        ...
    @detector_threshold.setter
    def detector_threshold(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def laser_power(self) -> float:
        ...
    @laser_power.setter
    def laser_power(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_range(self) -> float:
        ...
    @max_range.setter
    def max_range(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_returns(self) -> int:
        ...
    @max_returns.setter
    def max_returns(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def medium_albedo(self) -> float:
        ...
    @medium_albedo.setter
    def medium_albedo(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def medium_anisotropy(self) -> float:
        ...
    @medium_anisotropy.setter
    def medium_anisotropy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def medium_extinction(self) -> float:
        ...
    @medium_extinction.setter
    def medium_extinction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def medium_surface_y(self) -> float:
        ...
    @medium_surface_y.setter
    def medium_surface_y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def min_range(self) -> float:
        ...
    @min_range.setter
    def min_range(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def paired_clean_trace(self) -> bool:
        """
        Trace every beam twice in one dispatch -- as-is, and with the ParticleField density medium off, same beams and RNG keys. The difference IS the degradation. PathTracedLidarSensor.scan() returns the clean leg under the 'clean' key.
        """
    @paired_clean_trace.setter
    def paired_clean_trace(self, arg0: bool) -> None:
        ...
    @property
    def reference_range(self) -> float:
        ...
    @reference_range.setter
    def reference_range(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def samples_per_beam(self) -> int:
        ...
    @samples_per_beam.setter
    def samples_per_beam(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class LidarReturn:
    normal: Vector3
    position: Vector3
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def distance(self) -> float:
        ...
    @distance.setter
    def distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hit_instance_id(self) -> int:
        ...
    @hit_instance_id.setter
    def hit_instance_id(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def intensity(self) -> float:
        ...
    @intensity.setter
    def intensity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def return_kind(self) -> int:
        ...
    @return_kind.setter
    def return_kind(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def return_no(self) -> int:
        ...
    @return_no.setter
    def return_no(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class Light(Object3D):
    color: Color
    @property
    def intensity(self) -> float:
        ...
    @intensity.setter
    def intensity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Line(Object3D):
    cast_shadow: bool
    frustum_culled: bool
    layers: Layers
    matrix_auto_update: bool
    name: str
    position: Vector3
    quaternion: Quaternion
    receive_shadow: bool
    rotation: Euler
    scale: Vector3
    up: Vector3
    visible: bool
    def __init__(self, geometry: BufferGeometry = None, material: typing.Any = None) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def add(self, *args) -> None:
        ...
    def clear(self) -> None:
        ...
    def compute_line_distances(self) -> None:
        ...
    def get_object_by_name(self, name: str) -> Object3D:
        ...
    def get_user_data(self, key: str) -> typing.Any:
        ...
    def get_world_direction(self) -> Vector3:
        ...
    def get_world_position(self) -> Vector3:
        ...
    def get_world_quaternion(self) -> Quaternion:
        ...
    def get_world_scale(self) -> Vector3:
        ...
    def local_to_world(self, vector: Vector3) -> Vector3:
        ...
    @typing.overload
    def look_at(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def look_at(self, vector: Vector3) -> None:
        ...
    def remove(self, object: typing.Any) -> None:
        ...
    def remove_from_parent(self) -> None:
        ...
    def rotate_on_axis(self, axis: Vector3, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_x(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_y(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_z(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_user_data(self, key: str, value: str) -> None:
        ...
    def translate_x(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def translate_y(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def translate_z(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def traverse(self, callback: collections.abc.Callable[[typing.Any], None]) -> None:
        ...
    def update_matrix(self) -> None:
        ...
    def update_matrix_world(self, force: bool = False) -> None:
        ...
    def world_to_local(self, vector: Vector3) -> Vector3:
        ...
    @property
    def children(self) -> list[Object3D]:
        ...
    @property
    def geometry(self) -> BufferGeometry:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def material(self) -> Material | None:
        ...
    @property
    def matrix_world(self) -> Matrix4:
        ...
    @property
    def parent(self) -> Object3D:
        ...
    @property
    def render_order(self) -> int:
        ...
    @render_order.setter
    def render_order(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def uuid(self) -> str:
        ...
class LineBasicMaterial(Material):
    blending: Blending
    color: Color
    depth_test: bool
    depth_write: bool
    fog: bool
    name: str
    premultiplied_alpha: bool
    side: Side
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def linewidth(self) -> float:
        ...
    @linewidth.setter
    def linewidth(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class LineCurve(Curve2):
    v1: Vector2
    v2: Vector2
    def __init__(self, v1: Vector2, v2: Vector2) -> None:
        ...
    def __repr__(self) -> str:
        ...
class LineCurve3(Curve3):
    v1: Vector3
    v2: Vector3
    def __init__(self, v1: Vector3, v2: Vector3) -> None:
        ...
    def __repr__(self) -> str:
        ...
class LineSegments(Line):
    def __init__(self, geometry: BufferGeometry = None, material: typing.Any = None) -> None:
        ...
class Loop:
    """
    Looping mode for an AnimationAction.
    
    Members:
    
      ONCE
    
      REPEAT
    
      PING_PONG
    """
    ONCE: typing.ClassVar[Loop]  # value = <Loop.ONCE: 2200>
    PING_PONG: typing.ClassVar[Loop]  # value = <Loop.PING_PONG: 2202>
    REPEAT: typing.ClassVar[Loop]  # value = <Loop.REPEAT: 2201>
    __members__: typing.ClassVar[dict[str, Loop]]  # value = {'ONCE': <Loop.ONCE: 2200>, 'REPEAT': <Loop.REPEAT: 2201>, 'PING_PONG': <Loop.PING_PONG: 2202>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Mapping:
    """
    Members:
    
      UV
    
      CubeReflection
    
      CubeRefraction
    
      EquirectangularReflection
    
      EquirectangularRefraction
    """
    CubeReflection: typing.ClassVar[Mapping]  # value = <Mapping.CubeReflection: 301>
    CubeRefraction: typing.ClassVar[Mapping]  # value = <Mapping.CubeRefraction: 302>
    EquirectangularReflection: typing.ClassVar[Mapping]  # value = <Mapping.EquirectangularReflection: 303>
    EquirectangularRefraction: typing.ClassVar[Mapping]  # value = <Mapping.EquirectangularRefraction: 304>
    UV: typing.ClassVar[Mapping]  # value = <Mapping.UV: 300>
    __members__: typing.ClassVar[dict[str, Mapping]]  # value = {'UV': <Mapping.UV: 300>, 'CubeReflection': <Mapping.CubeReflection: 301>, 'CubeRefraction': <Mapping.CubeRefraction: 302>, 'EquirectangularReflection': <Mapping.EquirectangularReflection: 303>, 'EquirectangularRefraction': <Mapping.EquirectangularRefraction: 304>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Material:
    blending: Blending
    depth_test: bool
    depth_write: bool
    fog: bool
    name: str
    premultiplied_alpha: bool
    side: Side
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Matrix3:
    def __init__(self) -> None:
        ...
    def determinant(self) -> float:
        ...
    def elements(self) -> list[float]:
        ...
    def identity(self) -> Matrix3:
        ...
    def invert(self) -> Matrix3:
        ...
    def to_numpy(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
    def transpose(self) -> Matrix3:
        ...
class Matrix4:
    def __init__(self) -> None:
        ...
    def compose(self, position: Vector3, quaternion: Quaternion, scale: Vector3) -> Matrix4:
        ...
    def copy(self, m: Matrix4) -> Matrix4:
        ...
    def determinant(self) -> float:
        ...
    def elements(self) -> list[float]:
        ...
    def identity(self) -> Matrix4:
        ...
    def invert(self) -> Matrix4:
        ...
    def make_rotation_x(self, theta: typing.SupportsFloat | typing.SupportsIndex) -> Matrix4:
        ...
    def make_rotation_y(self, theta: typing.SupportsFloat | typing.SupportsIndex) -> Matrix4:
        ...
    def make_rotation_z(self, theta: typing.SupportsFloat | typing.SupportsIndex) -> Matrix4:
        ...
    def make_scale(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> Matrix4:
        ...
    def make_translation(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> Matrix4:
        ...
    def multiply(self, m: Matrix4) -> Matrix4:
        ...
    def premultiply(self, m: Matrix4) -> Matrix4:
        ...
    def set(self, n11: typing.SupportsFloat | typing.SupportsIndex, n12: typing.SupportsFloat | typing.SupportsIndex, n13: typing.SupportsFloat | typing.SupportsIndex, n14: typing.SupportsFloat | typing.SupportsIndex, n21: typing.SupportsFloat | typing.SupportsIndex, n22: typing.SupportsFloat | typing.SupportsIndex, n23: typing.SupportsFloat | typing.SupportsIndex, n24: typing.SupportsFloat | typing.SupportsIndex, n31: typing.SupportsFloat | typing.SupportsIndex, n32: typing.SupportsFloat | typing.SupportsIndex, n33: typing.SupportsFloat | typing.SupportsIndex, n34: typing.SupportsFloat | typing.SupportsIndex, n41: typing.SupportsFloat | typing.SupportsIndex, n42: typing.SupportsFloat | typing.SupportsIndex, n43: typing.SupportsFloat | typing.SupportsIndex, n44: typing.SupportsFloat | typing.SupportsIndex) -> Matrix4:
        ...
    def set_position(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> Matrix4:
        ...
    def to_numpy(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
    def transpose(self) -> Matrix4:
        ...
class Mesh(Object3D):
    cast_shadow: bool
    frustum_culled: bool
    layers: Layers
    matrix_auto_update: bool
    name: str
    position: Vector3
    quaternion: Quaternion
    receive_shadow: bool
    rotation: Euler
    scale: Vector3
    up: Vector3
    visible: bool
    def __init__(self, geometry: BufferGeometry = None, material: typing.Any = None) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def add(self, *args) -> None:
        ...
    def clear(self) -> None:
        ...
    def get_object_by_name(self, name: str) -> Object3D:
        ...
    def get_user_data(self, key: str) -> typing.Any:
        ...
    def get_world_direction(self) -> Vector3:
        ...
    def get_world_position(self) -> Vector3:
        ...
    def get_world_quaternion(self) -> Quaternion:
        ...
    def get_world_scale(self) -> Vector3:
        ...
    def local_to_world(self, vector: Vector3) -> Vector3:
        ...
    @typing.overload
    def look_at(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def look_at(self, vector: Vector3) -> None:
        ...
    def remove(self, object: typing.Any) -> None:
        ...
    def remove_from_parent(self) -> None:
        ...
    def rotate_on_axis(self, axis: Vector3, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_x(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_y(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_z(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_geometry(self, geometry: BufferGeometry) -> None:
        ...
    def set_material(self, material: typing.Any) -> None:
        ...
    def set_user_data(self, key: str, value: str) -> None:
        ...
    def translate_x(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def translate_y(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def translate_z(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def traverse(self, callback: collections.abc.Callable[[typing.Any], None]) -> None:
        ...
    def update_matrix(self) -> None:
        ...
    def update_matrix_world(self, force: bool = False) -> None:
        ...
    def world_to_local(self, vector: Vector3) -> Vector3:
        ...
    @property
    def children(self) -> list[Object3D]:
        ...
    @property
    def geometry(self) -> BufferGeometry:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def material(self) -> Material | None:
        ...
    @property
    def matrix_world(self) -> Matrix4:
        ...
    @property
    def parent(self) -> Object3D:
        ...
    @property
    def render_order(self) -> int:
        ...
    @render_order.setter
    def render_order(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def uuid(self) -> str:
        ...
class MeshBasicMaterial(Material):
    alpha_map: Texture
    ao_map: Texture
    blending: Blending
    color: Color
    combine: CombineOperation
    depth_test: bool
    depth_write: bool
    env_map: Texture
    fog: bool
    map: Texture
    name: str
    premultiplied_alpha: bool
    side: Side
    specular_map: Texture
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    wireframe: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def reflectivity(self) -> float:
        ...
    @reflectivity.setter
    def reflectivity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def refraction_ratio(self) -> float:
        ...
    @refraction_ratio.setter
    def refraction_ratio(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wireframe_linewidth(self) -> float:
        ...
    @wireframe_linewidth.setter
    def wireframe_linewidth(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MeshDepthMaterial(Material):
    alpha_map: Texture
    blending: Blending
    depth_packing: DepthPacking
    depth_test: bool
    depth_write: bool
    displacement_map: Texture
    fog: bool
    map: Texture
    name: str
    premultiplied_alpha: bool
    side: Side
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    wireframe: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def displacement_bias(self) -> float:
        ...
    @displacement_bias.setter
    def displacement_bias(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def displacement_scale(self) -> float:
        ...
    @displacement_scale.setter
    def displacement_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wireframe_linewidth(self) -> float:
        ...
    @wireframe_linewidth.setter
    def wireframe_linewidth(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MeshLambertMaterial(Material):
    alpha_map: Texture
    ao_map: Texture
    blending: Blending
    color: Color
    depth_test: bool
    depth_write: bool
    emissive: Color
    emissive_map: Texture
    env_map: Texture
    fog: bool
    map: Texture
    name: str
    premultiplied_alpha: bool
    side: Side
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    wireframe: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def emissive_intensity(self) -> float:
        ...
    @emissive_intensity.setter
    def emissive_intensity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def reflectivity(self) -> float:
        ...
    @reflectivity.setter
    def reflectivity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MeshNormalMaterial(Material):
    blending: Blending
    bump_map: Texture
    depth_test: bool
    depth_write: bool
    displacement_map: Texture
    flat_shading: bool
    fog: bool
    name: str
    normal_map: Texture
    normal_scale: Vector2
    premultiplied_alpha: bool
    side: Side
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    wireframe: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MeshPhongMaterial(Material):
    alpha_map: Texture
    ao_map: Texture
    blending: Blending
    bump_map: Texture
    color: Color
    combine: CombineOperation
    depth_test: bool
    depth_write: bool
    emissive: Color
    emissive_map: Texture
    env_map: Texture
    flat_shading: bool
    fog: bool
    map: Texture
    name: str
    normal_map: Texture
    premultiplied_alpha: bool
    side: Side
    specular: Color
    specular_map: Texture
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    wireframe: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def emissive_intensity(self) -> float:
        ...
    @emissive_intensity.setter
    def emissive_intensity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def reflectivity(self) -> float:
        ...
    @reflectivity.setter
    def reflectivity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def shininess(self) -> float:
        ...
    @shininess.setter
    def shininess(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MeshPhysicalMaterial(MeshStandardMaterial):
    def __init__(self) -> None:
        ...
    @property
    def attenuation_color(self) -> Color:
        """
        Beer-Lambert tint per attenuation_distance of travel — the water colour lever.
        """
    @attenuation_color.setter
    def attenuation_color(self, arg0: Color) -> None:
        ...
    @property
    def attenuation_distance(self) -> float:
        """
        Distance (m) over which attenuation_color is applied once; smaller = murkier.
        """
    @attenuation_distance.setter
    def attenuation_distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def clearcoat(self) -> float:
        ...
    @clearcoat.setter
    def clearcoat(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def clearcoat_roughness(self) -> float:
        ...
    @clearcoat_roughness.setter
    def clearcoat_roughness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dispersion(self) -> float:
        ...
    @dispersion.setter
    def dispersion(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ior(self) -> float:
        ...
    @ior.setter
    def ior(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def iridescence(self) -> float:
        ...
    @iridescence.setter
    def iridescence(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def iridescence_ior(self) -> float:
        ...
    @iridescence_ior.setter
    def iridescence_ior(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def iridescence_thickness_nm(self) -> float:
        ...
    @iridescence_thickness_nm.setter
    def iridescence_thickness_nm(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def specular_color(self) -> Color:
        """
        Tints dielectric F0; applied together with specular_intensity.
        """
    @specular_color.setter
    def specular_color(self, arg0: Color) -> None:
        ...
    @property
    def specular_intensity(self) -> float:
        """
        Scales dielectric F0 linearly; 0 kills the specular lobe (direct and environment) entirely. Default 1.
        """
    @specular_intensity.setter
    def specular_intensity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def thickness(self) -> float:
        """
        Thin-shell in-medium proxy distance (m) for Beer-Lambert; also scales the water body veil (attenuation_color^(2*thickness/attenuation_distance)).
        """
    @thickness.setter
    def thickness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def thin_walled(self) -> bool:
        """
        Surface is a thin shell (ocean plane, lens), not a closed volume.
        """
    @thin_walled.setter
    def thin_walled(self, arg0: bool) -> None:
        ...
    @property
    def transmission(self) -> float:
        """
        0 = opaque, 1 = fully transmissive (water/glass).
        """
    @transmission.setter
    def transmission(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MeshStandardMaterial(Material):
    alpha_map: Texture
    ao_map: Texture
    blending: Blending
    bump_map: Texture
    color: Color
    depth_test: bool
    depth_write: bool
    displacement_map: Texture
    emissive: Color
    emissive_map: Texture
    env_map: Texture
    flat_shading: bool
    fog: bool
    map: Texture
    metalness_map: Texture
    name: str
    normal_map: Texture
    normal_scale: Vector2
    premultiplied_alpha: bool
    roughness_map: Texture
    side: Side
    tone_mapped: bool
    translucency_color: Color
    transparent: bool
    vertex_colors: bool
    visible: bool
    wireframe: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def emissive_intensity(self) -> float:
        ...
    @emissive_intensity.setter
    def emissive_intensity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def env_map_intensity(self) -> float:
        ...
    @env_map_intensity.setter
    def env_map_intensity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def metalness(self) -> float:
        ...
    @metalness.setter
    def metalness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def roughness(self) -> float:
        ...
    @roughness.setter
    def roughness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def translucency(self) -> float:
        ...
    @translucency.setter
    def translucency(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wireframe_linewidth(self) -> float:
        ...
    @wireframe_linewidth.setter
    def wireframe_linewidth(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class ModelLoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str) -> Group:
        """
        Load a model (.obj/.gltf/.glb/.stl/.dae/.json) as a Group. .json is a three.js scene document - what the threepp editor saves - with the document root adopted as the group's child, so a Scene root keeps its environment and userData and is reachable with group.children[0].
        """
    def set_ignore_up_direction(self, ignore: bool) -> None:
        ...
class NoiseModel:
    """
    Per-axis Gaussian noise config shared by every sensor. Densities are continuous-time so the noise is invariant to sample rate: white_noise_density [X/sqrt(Hz)] -> per-sample stddev density/sqrt(dt); random_walk [X/(s*sqrt(Hz))] -> bias increment stddev random_walk*sqrt(dt); constant_bias [X] is a fixed turn-on offset. Same seed + call sequence is deterministic. All-zero = a perfect sensor.
    """
    constant_bias: Vector3
    random_walk: Vector3
    white_noise_density: Vector3
    def __init__(self, white_noise_density: Vector3 = ..., random_walk: Vector3 = ..., constant_bias: Vector3 = ..., seed: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def seed(self) -> int:
        ...
    @seed.setter
    def seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class NoiseType:
    """
    Members:
    
      fBm
    
      Ridged
    
      Hybrid
    """
    Hybrid: typing.ClassVar[NoiseType]  # value = <NoiseType.Hybrid: 2>
    Ridged: typing.ClassVar[NoiseType]  # value = <NoiseType.Ridged: 1>
    __members__: typing.ClassVar[dict[str, NoiseType]]  # value = {'fBm': <NoiseType.fBm: 0>, 'Ridged': <NoiseType.Ridged: 1>, 'Hybrid': <NoiseType.Hybrid: 2>}
    fBm: typing.ClassVar[NoiseType]  # value = <NoiseType.fBm: 0>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class NumberKeyframeTrack(KeyframeTrack):
    def __init__(self, name: str, times: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], values: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], interpolation: Interpolation | None = None) -> None:
        ...
class OBJLoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str, try_load_mtl: bool = True) -> Group:
        ...
class Object3D:
    cast_shadow: bool
    frustum_culled: bool
    layers: Layers
    matrix_auto_update: bool
    name: str
    position: Vector3
    quaternion: Quaternion
    receive_shadow: bool
    rotation: Euler
    scale: Vector3
    up: Vector3
    visible: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def add(self, *args) -> None:
        ...
    def clear(self) -> None:
        ...
    def clone(self, recursive: bool = True) -> Object3D:
        ...
    def copy(self, source: typing.Any, recursive: bool = True) -> None:
        ...
    def get_object_by_name(self, name: str) -> Object3D:
        ...
    def get_user_data(self, key: str) -> typing.Any:
        """
        String userData entry for `key`, or None when absent or not a string. The editor's spline/physics/script configs live here as flat 'key=value;...' strings.
        """
    def get_world_direction(self) -> Vector3:
        ...
    def get_world_position(self) -> Vector3:
        ...
    def get_world_quaternion(self) -> Quaternion:
        ...
    def get_world_scale(self) -> Vector3:
        ...
    def local_to_world(self, vector: Vector3) -> Vector3:
        ...
    @typing.overload
    def look_at(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def look_at(self, vector: Vector3) -> None:
        ...
    def remove(self, object: typing.Any) -> None:
        ...
    def remove_from_parent(self) -> None:
        ...
    def rotate_on_axis(self, axis: Vector3, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_x(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_y(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_z(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_user_data(self, key: str, value: str) -> None:
        """
        Set the string userData entry for `key`; an empty value removes it. This is how a GENERATOR script authors physics, a sensor or a script onto the content it builds - the same flat 'key=value;...' strings the inspector writes.
        """
    def translate_x(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def translate_y(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def translate_z(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def traverse(self, callback: collections.abc.Callable[[typing.Any], None]) -> None:
        ...
    def update_matrix(self) -> None:
        ...
    def update_matrix_world(self, force: bool = False) -> None:
        ...
    def world_to_local(self, vector: Vector3) -> Vector3:
        ...
    @property
    def children(self) -> list[Object3D]:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def matrix_world(self) -> Matrix4:
        ...
    @property
    def parent(self) -> Object3D:
        ...
    @property
    def render_order(self) -> int:
        ...
    @render_order.setter
    def render_order(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def uuid(self) -> str:
        ...
class Ocean(DisplacedMesh):
    def __init__(self, size: typing.SupportsFloat | typing.SupportsIndex = 1000.0, resolution: typing.SupportsInt | typing.SupportsIndex = 512, wind_speed: typing.SupportsFloat | typing.SupportsIndex = 10.0, wind_theta: typing.SupportsFloat | typing.SupportsIndex = 0.6000000238418579, choppiness: typing.SupportsFloat | typing.SupportsIndex = 0.550000011920929, wave_scale: typing.SupportsFloat | typing.SupportsIndex = 1.0, tile_size_1: typing.SupportsFloat | typing.SupportsIndex = -1.0, tile_size_2: typing.SupportsFloat | typing.SupportsIndex = -1.0, fft_size: typing.SupportsInt | typing.SupportsIndex = 1024, size_z: typing.SupportsFloat | typing.SupportsIndex = 0.0, resolution_z: typing.SupportsInt | typing.SupportsIndex = 0, look: str = 'auto', fetch: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        """
        A ready-to-use FFT ocean. Add it to a Scene and render with the Vulkan renderer. size is the local-X extent (m); size_z=0 makes a square, >0 a rectangle (vertices only where the water is — the wave field is unaffected). resolution is the vertex grid along X; resolution_z=0 keeps cells square-ish. fft_size caps the per-cascade FFT resolutions (band-passed cascades auto-size below it). tile_size_1/2 default to -1 = auto: scaled from the larger extent (a 1000 m ocean gets the classic 127/9.3 bands, a 16 m pond gets dm-scale ripples); 0 disables a cascade, >0 pins it. Ponds also want wind_speed 2-5. look picks the water material: 'auto' = pond recipe under 100 m, ocean above; 'ocean'/'pond' pin it regardless of scale. fetch (m) = 0 is a fully developed sea (long swell); 20e3-40e3 gives the shorter, steeper JONSWAP chop of a coastal sea (see Params.fetch).
        """
    def set_wind(self, speed: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set wind speed (m/s) and direction (radians). Live: the renderer regenerates the spectra next frame and the sea morphs smoothly into the new state.
        """
    def warp_toward(self, world_x: typing.SupportsFloat | typing.SupportsIndex, world_z: typing.SupportsFloat | typing.SupportsIndex, coef_a: typing.SupportsFloat | typing.SupportsIndex = 0.10000000149011612) -> None:
        """
        Pack vertex density toward a world-space focus point (e.g. the camera). Call each frame before render().
        """
class OctahedronGeometry(BufferGeometry):
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex = 1.0, detail: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
class OrbitControls:
    auto_rotate: bool
    enable_damping: bool
    enable_pan: bool
    enable_rotate: bool
    enable_zoom: bool
    enabled: bool
    target: Vector3
    def __init__(self, camera: Camera, canvas: Canvas) -> None:
        ...
    def update(self) -> bool:
        ...
    @property
    def auto_rotate_speed(self) -> float:
        ...
    @auto_rotate_speed.setter
    def auto_rotate_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def damping_factor(self) -> float:
        ...
    @damping_factor.setter
    def damping_factor(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_distance(self) -> float:
        ...
    @max_distance.setter
    def max_distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def min_distance(self) -> float:
        ...
    @min_distance.setter
    def min_distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rotate_speed(self) -> float:
        ...
    @rotate_speed.setter
    def rotate_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def zoom_speed(self) -> float:
        ...
    @zoom_speed.setter
    def zoom_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class OrthographicCamera(Camera):
    def __init__(self, left: typing.SupportsFloat | typing.SupportsIndex = -1.0, right: typing.SupportsFloat | typing.SupportsIndex = 1.0, top: typing.SupportsFloat | typing.SupportsIndex = 1.0, bottom: typing.SupportsFloat | typing.SupportsIndex = -1.0, near: typing.SupportsFloat | typing.SupportsIndex = 0.10000000149011612, far: typing.SupportsFloat | typing.SupportsIndex = 2000.0) -> None:
        ...
    def update_projection_matrix(self) -> None:
        ...
    @property
    def bottom(self) -> float:
        ...
    @bottom.setter
    def bottom(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def left(self) -> float:
        ...
    @left.setter
    def left(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def right(self) -> float:
        ...
    @right.setter
    def right(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def top(self) -> float:
        ...
    @top.setter
    def top(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class ParticleField(Mesh):
    class BillboardRepr:
        enabled: bool
        @property
        def alpha_over(self) -> bool:
            """
            Composite premultiplied SRC_ALPHA-over instead of additive, so a sprite OCCLUDES what is behind it. Nothing is sorted: draws go in field order and, within a field, in SLOT order, so submit back-to-front for correct blending.
            """
        @alpha_over.setter
        def alpha_over(self, arg0: bool) -> None:
            ...
        @property
        def bright_jitter(self) -> float:
            """
            Per-particle brightness spread, hashed.
            """
        @bright_jitter.setter
        def bright_jitter(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def color_cool(self) -> Color:
            """
            Linear HDR radiance at end of life.
            """
        @color_cool.setter
        def color_cool(self, arg0: Color) -> None:
            ...
        @property
        def color_hot(self) -> Color:
            """
            Linear HDR radiance at age 0.
            """
        @color_hot.setter
        def color_hot(self, arg0: Color) -> None:
            ...
        @property
        def core_weight(self) -> float:
            """
            Weight of the fixed 1-2 px core dot under the softness-shaped skirt. 0.85 (the default) is the pre-knob look; big soft smoke parcels set ~0 so a dense field stops speckling.
            """
        @core_weight.setter
        def core_weight(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def fade_power(self) -> float:
            """
            Brightness over life: (1 - age_frac)^fade_power. 0 = no fade (rain).
            """
        @fade_power.setter
        def fade_power(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def glow(self) -> float:
            """
            > 0 gives this field its own bloom pyramid. 0 skips the whole chain (what weather wants — 300k rain streaks have nothing to bloom).
            """
        @glow.setter
        def glow(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def glow_threshold(self) -> float:
            """
            Bright-pass knee for this field's own pyramid. 0 = no bright pass.
            """
        @glow_threshold.setter
        def glow_threshold(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def intensity(self) -> float:
            """
            HDR scale on both colours. Additive over a field the camera stands INSIDE — rain wants ~0.07, not ~0.5.
            """
        @intensity.setter
        def intensity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def lit(self) -> bool:
            """
            Per-particle radiance = colour x (ambient + sun x HG phase), from the scene's own sun. One lobe per particle — no shadow ray, no cluster walk.
            """
        @lit.setter
        def lit(self, arg0: bool) -> None:
            ...
        @property
        def lit_ambient(self) -> float:
            """
            Ambient radiance FLOOR added to the scene's summed AmbientLights, in the same linear units — how dark the shaded side of a sprite is allowed to get. An IBL-lit scene carries no AmbientLight, so a scale on it would be a scale on zero.
            """
        @lit_ambient.setter
        def lit_ambient(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def lit_phase_g(self) -> float:
            """
            Henyey-Greenstein asymmetry for that lobe. ~0.35 = the forward-ish scattering of a water parcel; 0 = isotropic.
            """
        @lit_phase_g.setter
        def lit_phase_g(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def lod_fade(self) -> float:
            """
            Metres of ramp above lod_near over which the quad fades IN.
            """
        @lod_fade.setter
        def lod_fade(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def lod_near(self) -> float:
            """
            Quads CLOSER than this collapse — MeshRepr.lod_far is the complementary gate on the same field. 0 = off.
            """
        @lod_near.setter
        def lod_near(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def near_fade(self) -> float:
            """
            Fade the sprite out below this camera distance (m). 0 = off.
            """
        @near_fade.setter
        def near_fade(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def opacity(self) -> float:
            """
            Coverage scale in alpha_over mode, before the texture/procedural falloff. Ignored when additive.
            """
        @opacity.setter
        def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def size_scale(self) -> float:
            """
            Multiplies the particle radius.
            """
        @size_scale.setter
        def size_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def size_taper(self) -> float:
            """
            Radius over life: r *= (1 - size_taper * age_frac).
            """
        @size_taper.setter
        def size_taper(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def softness(self) -> float:
            """
            0 = a tight spark, 1 = a broad glow.
            """
        @softness.setter
        def softness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def splash_ring_width(self) -> float:
            """
            Annulus width as a fraction of the splash ring's radius (1 = a filled disc). Only means anything with emitter.surface.splash_seconds > 0.
            """
        @splash_ring_width.setter
        def splash_ring_width(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def stretch_max(self) -> float:
            """
            Streak cap in multiples of the radius (WORLD units).
            """
        @stretch_max.setter
        def stretch_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def stretch_max_screen(self) -> float:
            """
            Streak cap as a fraction of the frame HEIGHT (NDC). 0 = off; ~0.045 stops the nearest drop painting a bar across the frame.
            """
        @stretch_max_screen.setter
        def stretch_max_screen(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def stretch_seconds(self) -> float:
            """
            Seconds of travel to smear the quad over, along the particle's own analytic velocity. 0 = a round sprite; rain without this reads as HAIL.
            """
        @stretch_seconds.setter
        def stretch_seconds(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def sun_geometry_shadow(self) -> bool:
            """
            Let the SCENE shadow the sprites, not only the field's own dust: one ray query per particle toward the sun against the scene's acceleration structure, folded into the same T_sun the volume march produces. A wake under a ship's counter goes dark; a wake in open water is unchanged. False by default and read only when volume_shadow > 0, so nothing that predates the flag moves. WATER IS NOT AN OCCLUDER (the ray steps past it) because the renderer carries water on the opaque visibility mask, and glass and blended geometry are culled outright. Needs VK_KHR_ray_query; without it the sun term is silently the volume-only one.
            """
        @sun_geometry_shadow.setter
        def sun_geometry_shadow(self, arg0: bool) -> None:
            ...
        @property
        def texture(self) -> Texture:
            """
            Optional sprite that MODULATES the procedural quad (rgb tint x alpha). None binds a 1x1 white default.
            """
        @texture.setter
        def texture(self, arg0: Texture) -> None:
            ...
        @property
        def volume_ambient(self) -> float:
            """
            Floor under the sun term, so the shadowed side of the volume does not go black. Only read when volume_shadow > 0.
            """
        @volume_ambient.setter
        def volume_ambient(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def volume_extinction(self) -> float:
            """
            Dim each sprite by the transmittance of the field's OWN DensityRepr volume between it and the camera, as pow(T_cam, this). 1 is the physically honest answer; >1 is a 'more dust' grade that does not disturb sigma_per_particle (which the deferred fog march also reads). 0 is an EXACT no-op — the shader takes a uniform branch around the march. This is what puts DUST LANES across a nebula: additive blending is orderless and therefore carries zero occlusion information, and this is the occlusion, from the medium the same particles collectively are. Needs DensityRepr on.
            """
        @volume_extinction.setter
        def volume_extinction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def volume_shadow(self) -> float:
            """
            Mix each sprite toward T_sun * (volume_ambient + volume_sun_gain * HG(V.L)) — the transmittance from the sprite to the sun through the same volume, times one phase lobe. 0 = unshadowed (the pre-change look and an exact no-op), 1 = fully replace. This is the LIT RIM and the self-shadowed interior. Needs DensityRepr on.
            """
        @volume_shadow.setter
        def volume_shadow(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def volume_sun_gain(self) -> float:
            """
            Scale on the sun's own contribution through T_sun. Unitless — it multiplies the HG lobe (whose asymmetry is lit_phase_g), not a radiance, because the sprite's colour already IS its radiance.
            """
        @volume_sun_gain.setter
        def volume_sun_gain(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
    class Config:
        def __init__(self) -> None:
            ...
        @property
        def attributes(self) -> bool:
            """
            Allocate the per-particle vec4 appearance buffer (rgb = linear HDR radiance, a reserved). It rides the POSITIONS' path exactly: under Ownership.Interop it is a second exported allocation handed back by the same enable_particle_field_interop call and snapshotted by the same per-frame copy, so positions and colours can never diverge. With it on, BillboardRepr uses attribute.rgb INSTEAD of the color_hot/color_cool ramp — one scheme or the other, never a blend.
            """
        @attributes.setter
        def attributes(self, arg0: bool) -> None:
            ...
        @property
        def capacity(self) -> int:
            """
            Fixed for life — a field is created ONCE at its final capacity and never resized (creating/destroying one is a structural scene change: entry re-expansion, device idle, TAA history clear). Park a field with set_live_count(0) instead.
            """
        @capacity.setter
        def capacity(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        @property
        def host_stable_slots(self) -> bool:
            """
            HostRing only: the host promises index i is the SAME particle in every submit (fixed pool, dead slots left at w < 0, no compaction). That makes the previous ring slot a real prevPositions buffer, which is what BillboardRepr.stretch_seconds — the velocity streak — needs; without the promise the stretch stays off on a host field. A frame the host skips falls back to round sprites rather than smearing over two steps, and a freshly spawned slot streaks from its predecessor for one frame (bounded by stretch_max; spend it under a fade-in).
            """
        @host_stable_slots.setter
        def host_stable_slots(self, arg0: bool) -> None:
            ...
        @property
        def orientations(self) -> bool:
            """
            Allocate the snorm16x4 per-particle orientation buffer.
            """
        @orientations.setter
        def orientations(self, arg0: bool) -> None:
            ...
        @property
        def ownership(self) -> ParticleField.Ownership:
            """
            Who writes the positions.
            """
        @ownership.setter
        def ownership(self, arg0: ParticleField.Ownership) -> None:
            ...
        @property
        def uniform_radius(self) -> float:
            """
            World radius the MeshRepr proxy geometry is authored at; also the billboard/density size reference.
            """
        @uniform_radius.setter
        def uniform_radius(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def w_semantic(self) -> ParticleField.WSemantic:
            """
            What the w channel means.
            """
        @w_semantic.setter
        def w_semantic(self, arg0: ParticleField.WSemantic) -> None:
            ...
    class DensityRepr:
        enabled: bool
        @property
        def albedo(self) -> Color:
            """
            Scattering albedo of THIS medium. Snow is bright; a rain curtain is dark.
            """
        @albedo.setter
        def albedo(self, arg0: Color) -> None:
            ...
        @property
        def anisotropy(self) -> float:
            """
            HG g for this medium.
            """
        @anisotropy.setter
        def anisotropy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def center(self) -> Vector3:
            """
            World centre of the volume box.
            """
        @center.setter
        def center(self, arg0: Vector3) -> None:
            ...
        @property
        def emissive_intensity(self) -> float:
            """
            HDR radiance scale of the analytic blackbody flame ramp. 0 = the exact no-op (pure dust).
            """
        @emissive_intensity.setter
        def emissive_intensity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def half_extent(self) -> Vector3:
            """
            World half-size per axis.
            """
        @half_extent.setter
        def half_extent(self, arg0: Vector3) -> None:
            ...
        @property
        def resolution(self) -> int:
            """
            Voxels per axis — LATCHED at the first enable (set it via set_density_repr).
            """
        @property
        def sigma_per_particle(self) -> float:
            """
            Extinction (1/m) one particle contributes. Total optical mass is N * sigma — a 300k weather field wants hundredths, not units.
            """
        @sigma_per_particle.setter
        def sigma_per_particle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def temp_bottom_k(self) -> float:
            ...
        @temp_bottom_k.setter
        def temp_bottom_k(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def temp_falloff(self) -> float:
            ...
        @temp_falloff.setter
        def temp_falloff(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def temp_top_k(self) -> float:
            ...
        @temp_top_k.setter
        def temp_top_k(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
    class EmitterParams:
        class Surface:
            def __init__(self) -> None:
                ...
            @property
            def bias(self) -> float:
                """
                Metres above the baked surface the particle CENTRE rests; ~its own size.
                """
            @bias.setter
            def bias(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
                ...
            @property
            def enabled(self) -> bool:
                """
                Solve each slot's landing against a top-down height bake of the scene, inside the same closed form. Snow rests; rain splashes.
                """
            @enabled.setter
            def enabled(self, arg0: bool) -> None:
                ...
            @property
            def extent(self) -> float:
                """
                Half-size of the square bake footprint (field-local m). 0 = use spawn_half_extent.xz, which is also the toroidal wrap period.
                """
            @extent.setter
            def extent(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
                ...
            @property
            def fade_seconds(self) -> float:
                ...
            @fade_seconds.setter
            def fade_seconds(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
                ...
            @property
            def resolution(self) -> int:
                """
                Texels per axis, clamped to [16, 1024]. Sampled NEAREST by design.
                """
            @resolution.setter
            def resolution(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
                ...
            @property
            def rest_jitter(self) -> float:
                ...
            @rest_jitter.setter
            def rest_jitter(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
                ...
            @property
            def rest_seconds(self) -> float:
                ...
            @rest_seconds.setter
            def rest_seconds(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
                ...
            @property
            def search_bottom(self) -> float:
                """
                Vertical search band, field-local. top == bottom = derive it.
                """
            @search_bottom.setter
            def search_bottom(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
                ...
            @property
            def search_top(self) -> float:
                ...
            @search_top.setter
            def search_top(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
                ...
            @property
            def splash_grow(self) -> float:
                """
                Ring radius at the end of the splash, in multiples of the drop radius.
                """
            @splash_grow.setter
            def splash_grow(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
                ...
            @property
            def splash_seconds(self) -> float:
                """
                > 0 makes a landed drop an expanding flat RING for this long instead of resting. Drawn by the billboard representation.
                """
            @splash_seconds.setter
            def splash_seconds(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
                ...
        def __init__(self) -> None:
            ...
        @property
        def accel(self) -> Vector3:
            """
            Constant acceleration (m/s^2). Falling snow and rain use ZERO — they are at terminal velocity.
            """
        @accel.setter
        def accel(self, arg0: Vector3) -> None:
            ...
        @property
        def drift_amplitude(self) -> float:
            """
            metres
            """
        @drift_amplitude.setter
        def drift_amplitude(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def drift_frequency(self) -> float:
            """
            Hz of the slowest term.
            """
        @drift_frequency.setter
        def drift_frequency(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def drift_growth(self) -> float:
            """
            0 = constant, 1 = ramps in over the life.
            """
        @drift_growth.setter
        def drift_growth(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def drift_scale(self) -> float:
            """
            Metres of spatial wavelength (turns wobble into travelling gusts); 0 = per-slot phase only.
            """
        @drift_scale.setter
        def drift_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def duty_cycle(self) -> float:
            """
            Alive fraction of the period, (0,1].
            """
        @duty_cycle.setter
        def duty_cycle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def follow(self) -> bool:
            """
            Wrap the field toroidally into a lateral box centred on set_follow_center() — weather instead of a patch. The wrap PERIOD is 2 * spawn_half_extent.xz, so author the slab accordingly. Y is never wrapped.
            """
        @follow.setter
        def follow(self, arg0: bool) -> None:
            ...
        @property
        def follow_snap(self) -> float:
            """
            Lattice (m) the follow centre is snapped to; 0 = no snapping. Choose an integer number of density voxels when the field also carries a DensityRepr, or the haze visibly swims.
            """
        @follow_snap.setter
        def follow_snap(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def lifetime(self) -> float:
            """
            Slot repeat period (s). It has to CONTAIN the whole story: fall, land, rest/splash, fade.
            """
        @lifetime.setter
        def lifetime(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def lifetime_jitter(self) -> float:
            ...
        @lifetime_jitter.setter
        def lifetime_jitter(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def seed(self) -> int:
            ...
        @seed.setter
        def seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        @property
        def size(self) -> float:
            """
            Per-particle radius in METRES, written into w. Sizes here are metres, not pixels — a 2.4 cm proxy centimetres from the lens paints 150 px.
            """
        @size.setter
        def size(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def size_jitter(self) -> float:
            ...
        @size_jitter.setter
        def size_jitter(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def spawn_center(self) -> Vector3:
            """
            Birth region centre, FIELD-LOCAL.
            """
        @spawn_center.setter
        def spawn_center(self, arg0: Vector3) -> None:
            ...
        @property
        def spawn_half_extent(self) -> Vector3:
            """
            Birth region half-size. A thin slab is an emission plane — author snow/rain as a slab at the TOP with lifetime = height / speed, not as a box the size of the volume.
            """
        @spawn_half_extent.setter
        def spawn_half_extent(self, arg0: Vector3) -> None:
            ...
        @property
        def speed_spread(self) -> float:
            """
            Isotropic per-particle perturbation of the velocity (m/s).
            """
        @speed_spread.setter
        def speed_spread(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def surface(self) -> ParticleField.EmitterParams.Surface:
            """
            Analytic landing (rest / splash) against a baked height map.
            """
        @surface.setter
        def surface(self, arg0: ParticleField.EmitterParams.Surface) -> None:
            ...
        @property
        def velocity(self) -> Vector3:
            """
            Initial velocity (m/s).
            """
        @velocity.setter
        def velocity(self, arg0: Vector3) -> None:
            ...
        @property
        def wind(self) -> Vector3:
            """
            Uniform horizontal drift (m/s), summed with velocity. Usually animated.
            """
        @wind.setter
        def wind(self, arg0: Vector3) -> None:
            ...
    class MeshRepr:
        enabled: bool
        @property
        def lod_fade(self) -> float:
            """
            Metres of soft shrink below lod_far, so the swap cross-dissolves.
            """
        @lod_fade.setter
        def lod_fade(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def lod_far(self) -> float:
            """
            Metres beyond which the proxy collapses to zero area and the billboard is expected to take over (BillboardRepr.lod_near). 0 = no LOD.
            """
        @lod_far.setter
        def lod_far(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
        @property
        def near_cull(self) -> float:
            """
            Shrink particles closer than this to the camera; caps how big the nearest one may get. 0 = off.
            """
        @near_cull.setter
        def near_cull(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
            ...
    class Ownership:
        """
        Members:
        
          Interop : A foreign device API (CUDA/PhysX) writes the positions. NOT reproducible.
        
          HostRing : The CPU owns the positions; feed them with submit().
        
          Renderer : The GPU owns the positions, written from the closed-form EmitterParams. This is what weather uses.
        """
        HostRing: typing.ClassVar[ParticleField.Ownership]  # value = <Ownership.HostRing: 1>
        Interop: typing.ClassVar[ParticleField.Ownership]  # value = <Ownership.Interop: 0>
        Renderer: typing.ClassVar[ParticleField.Ownership]  # value = <Ownership.Renderer: 2>
        __members__: typing.ClassVar[dict[str, ParticleField.Ownership]]  # value = {'Interop': <Ownership.Interop: 0>, 'HostRing': <Ownership.HostRing: 1>, 'Renderer': <Ownership.Renderer: 2>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    class WSemantic:
        """
        Members:
        
          InvMass : w is PhysX's inverse mass; the radius comes from Config.uniform_radius.
        
          Radius : w IS the world radius (so emitter size_jitter is free per-particle variety).
        """
        InvMass: typing.ClassVar[ParticleField.WSemantic]  # value = <WSemantic.InvMass: 0>
        Radius: typing.ClassVar[ParticleField.WSemantic]  # value = <WSemantic.Radius: 1>
        __members__: typing.ClassVar[dict[str, ParticleField.WSemantic]]  # value = {'InvMass': <WSemantic.InvMass: 0>, 'Radius': <WSemantic.Radius: 1>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    @staticmethod
    def create(config: ParticleField.Config) -> ParticleField:
        """
        Create a field at its FINAL capacity (never resized — see Config.capacity). Raises ValueError on capacity == 0.
        """
    def __repr__(self) -> str:
        ...
    def set_attributes(self, rgba: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """
        Per-particle appearance as (n, 4) float32 (r, g, b, a): rgb is LINEAR HDR radiance in the same domain BillboardRepr.color_hot is authored in, a is reserved for the phase-2 alpha-over opacity. Requires Config.attributes.
        
        WRITE-ONCE by contract, exactly like set_orientations: the device buffer is not ringed, so rewriting it while frames are in flight is a host write to memory the GPU may be reading. A sim that needs colours EVERY frame wants the interop leg — enable_particle_field_interop hands back a second handle for this buffer and the foreign kernel writes it device-to-device.
        
        THROWS on an Ownership.Interop field that is not in host_fallback().
        """
    def set_billboard_repr(self, hot: Color, cool: Color, intensity: typing.SupportsFloat | typing.SupportsIndex = 1.0, size_scale: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        """
        Draw every live particle as one camera-facing ADDITIVE quad, composited after the upscalers (so it is outside TAA and is NOT exposed by auto-exposure — scale intensity with the scene exposure by hand). Vulkan only.
        """
    def set_density_repr(self, center: Vector3, half_extent: Vector3, sigma_per_particle: typing.SupportsFloat | typing.SupportsIndex, resolution: typing.SupportsInt | typing.SupportsIndex = 128) -> None:
        """
        Scatter the field once per frame into a world-anchored extinction volume every view's froxel pass then samples — the haze a snowfall or a rain curtain adds. `resolution` is LATCHED the frame the volume is allocated. Vulkan only.
        """
    def set_emitter(self, params: ParticleField.EmitterParams) -> None:
        """
        Install the closed-form trajectory the device emitter evaluates. Free to call every frame (O(1) bytes, published as push constants), so animating the wind costs nothing. Raises on a HostRing / Interop field.
        """
    def set_emitter_time(self, time_sec: typing.SupportsFloat | typing.SupportsIndex, dt_sec: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Advance the emitter to ABSOLUTE time (not a delta): the trajectory is closed form in t, so any t is valid in any order and a capture may seek with no warm-up. dt_sec is the interval the motion vectors are taken over — pass the frame's own delta, or 0 to freeze the field for a still.
        """
    def set_follow_center(self, world_center: Vector3) -> None:
        """
        Move the centre of the toroidal follow box — pass the CAMERA's world position (weather that follows anything else is not weather). Snapped to emitter.follow_snap here; read the snapped point back from follow_center.
        """
    def set_live_count(self, n: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Park a field with 0 (it stays in the scene and costs one entry, and the emit dispatch is skipped) or, on a HostRing field, cap the live prefix.
        """
    def set_mesh_repr(self, geometry: BufferGeometry, material: typing.Any) -> None:
        """
        Draw every live particle as a lit proxy in the G-buffer — ONE indirect draw. `material` also becomes the field's Mesh material, which is what keeps the shading in step. Vulkan only.
        """
    def set_orientations(self, quaternions: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """
        Per-particle orientation as (n, 4) float32 quaternions in (x, y, z, w) order. Requires Config.orientations. WRITE-ONCE by contract: the device buffer is not ringed, so this is authored with the field, not animated.
        """
    def submit(self, positions: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        """
        Point a HostRing field at n positions as an (n, 4) float32 array — xyz plus w, which is the radius under WSemantic.Radius and is the DEAD sentinel when negative under either. One memcpy, no per-particle loop. n > capacity is clamped; also sets the live count. Raises on a Renderer / Interop field. `dt` is the step this submit advanced the pool over, read only by the velocity stretch under Config.host_stable_slots (0 = assume 1/60 s).
        """
    @property
    def billboard_repr(self) -> ParticleField.BillboardRepr:
        ...
    @property
    def capacity(self) -> int:
        ...
    @property
    def config(self) -> ParticleField.Config:
        ...
    @property
    def density_repr(self) -> ParticleField.DensityRepr:
        ...
    @property
    def emitter(self) -> ParticleField.EmitterParams:
        """
        A COPY of the current parameters — mutate it and hand it back to set_emitter().
        """
    @property
    def emitter_dt(self) -> float:
        ...
    @property
    def emitter_time(self) -> float:
        ...
    @property
    def follow_center(self) -> Vector3:
        """
        The SNAPPED centre — use this, not the raw camera position, to place anything that must agree with the wrap box (above all density_repr.center).
        """
    @property
    def host_fallback(self) -> bool:
        """
        True when an Interop field had to fall back to the host path (this device cannot export memory to a foreign API), which makes submit() legal on it.
        """
    @property
    def live_count(self) -> int:
        ...
    @property
    def mesh_repr(self) -> ParticleField.MeshRepr:
        ...
class Path(Curve2):
    auto_close: bool
    current_point: Vector2
    def __init__(self, points: collections.abc.Sequence[Vector2] = []) -> None:
        ...
    def absarc(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex, start_angle: typing.SupportsFloat | typing.SupportsIndex, end_angle: typing.SupportsFloat | typing.SupportsIndex, clockwise: bool = False) -> Path:
        ...
    def bezier_curve_to(self, cp1x: typing.SupportsFloat | typing.SupportsIndex, cp1y: typing.SupportsFloat | typing.SupportsIndex, cp2x: typing.SupportsFloat | typing.SupportsIndex, cp2y: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex) -> Path:
        ...
    def close_path(self) -> None:
        ...
    def line_to(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex) -> Path:
        ...
    def move_to(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex) -> Path:
        ...
    def quadratic_curve_to(self, cpx: typing.SupportsFloat | typing.SupportsIndex, cpy: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex) -> Path:
        ...
    def set_from_points(self, points: collections.abc.Sequence[Vector2]) -> Path:
        ...
    def spline_thru(self, points: collections.abc.Sequence[Vector2]) -> Path:
        ...
class PathTracedLidarSensor(Object3D, Sensor):
    """
    Path-traced LIDAR: fires beams through the renderer's own acceleration structure and returns full radiometric hits (position, normal, distance, intensity from the GPU back-scatter BRDF, stable instance id, return number/kind). Vulkan only; render() the scene at least once first so the TLAS exists.
    
    `params` (a threepp.LidarParams, mutated in place) exposes the whole LIDAR equation: max/min range, laser power, reference range, detector threshold, atmospheric extinction, multi-return through transmissive surfaces, beam divergence sampling, a dedicated water-column/dust medium, and the paired clean/degraded trace (see LidarParams).
    
    Beam convention matches DepthSensor: beams leave along the sensor's LOCAL -Z, and look_at() honours it -- look_at(target) turns the beams toward the target, exactly as it would for a camera. (Older releases aimed local +Z at the target instead; mirror-point workarounds now aim backwards.)
    """
    @typing.overload
    def __init__(self, h_res: typing.SupportsInt | typing.SupportsIndex, v_res: typing.SupportsInt | typing.SupportsIndex, max_range: typing.SupportsFloat | typing.SupportsIndex = 100.0) -> None:
        """
        Dense grid: h_res x v_res beams over the full sphere (debug / ground truth).
        """
    @typing.overload
    def __init__(self, model: LidarModel, max_range: typing.SupportsFloat | typing.SupportsIndex = 100.0) -> None:
        """
        Real-sensor beam pattern, e.g. LidarModel.vlp16() / os1_64().
        """
    @typing.overload
    def __init__(self, fov_y: typing.SupportsFloat | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex, max_range: typing.SupportsFloat | typing.SupportsIndex = 100.0) -> None:
        """
        Depth-camera mode: a pinhole grid down local -Z, same mounting as DepthSensor.
        """
    def scan(self, renderer: VulkanRenderer) -> dict:
        """
        One scan from the current pose -> dict of numpy arrays keyed position (N,3), normal (N,3), distance, intensity, instance_id, return_no, return_kind (all length N = beams x samples_per_beam x max_returns; return_no > 0 is the 'real return' predicate). Adds key 'clean' when params.paired_clean_trace is set. Call after render(); never during it.
        """
    @property
    def beam_count(self) -> int:
        ...
    @property
    def noise(self) -> RangeNoiseModel:
        """
        Seeded RangeNoiseModel applied along each beam (default zero model: the tracer's own range is already physical). Same replay contract as DepthSensor.noise.
        """
    @noise.setter
    def noise(self, arg0: RangeNoiseModel) -> None:
        ...
    @property
    def params(self) -> LidarParams:
        """
        LidarParams, live-tweakable between scans; mutate in place.
        """
    @params.setter
    def params(self, arg0: LidarParams) -> None:
        ...
class PerspectiveCamera(Camera):
    def __init__(self, fov: typing.SupportsFloat | typing.SupportsIndex = 60.0, aspect: typing.SupportsFloat | typing.SupportsIndex = 1.0, near: typing.SupportsFloat | typing.SupportsIndex = 0.10000000149011612, far: typing.SupportsFloat | typing.SupportsIndex = 2000.0) -> None:
        ...
    def get_focal_length(self) -> float:
        ...
    def set_focal_length(self, focal_length: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def update_projection_matrix(self) -> None:
        ...
    @property
    def aspect(self) -> float:
        ...
    @aspect.setter
    def aspect(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def film_gauge(self) -> float:
        ...
    @film_gauge.setter
    def film_gauge(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def film_offset(self) -> float:
        ...
    @film_offset.setter
    def film_offset(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def focus(self) -> float:
        ...
    @focus.setter
    def focus(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fov(self) -> float:
        ...
    @fov.setter
    def fov(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PhysxGpuBatch:
    """
    Batched GPU-resident state I/O over many reduced-coordinate articulations in one direct-GPU scene. The read_*/write_* methods take a torch CUDA tensor (validated for cuda/float32/contiguous/correct-size) and move ALL robots' state in one call with no CPU readback; *_host variants stage through numpy for debugging. All articulations in a batch must share a DOF count. Requires PhysxWorld(direct_gpu=True) and finalized articulations.
    """
    def __init__(self, world: PhysxWorld, articulations: collections.abc.Iterable) -> None:
        """
        world must be created with direct_gpu=True and outlive this batch.
        """
    def gpu_indices(self) -> numpy.typing.NDArray[numpy.uint32]:
        """
        The K articulation GPU indices as a uint32 numpy array (upload once to build subset-index buffers for resets).
        """
    def read_joint_pos(self, tensor: typing.Any) -> None:
        """
        Fill the [n, max_dofs] float32 cuda tensor with joint positions.
        """
    def read_joint_pos_host(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
    def read_joint_vel(self, tensor: typing.Any) -> None:
        """
        Fill the [n, max_dofs] float32 cuda tensor with joint velocities.
        """
    def read_joint_vel_host(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
    def read_link_angvel(self, tensor: typing.Any) -> None:
        """
        Fill the [n, max_links*3] float32 cuda tensor with per-link angular velocities (world frame).
        """
    def read_link_angvel_host(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
    def read_link_linvel(self, tensor: typing.Any) -> None:
        """
        Fill the [n, max_links*3] float32 cuda tensor with per-link linear velocities (world frame).
        """
    def read_link_linvel_host(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
    def read_link_pose(self, tensor: typing.Any) -> None:
        """
        Fill the [n, max_links*7] float32 cuda tensor with per-link poses [qx,qy,qz,qw,px,py,pz].
        """
    def read_link_pose_host(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
    def read_root_angvel(self, tensor: typing.Any) -> None:
        """
        Fill the [n, 3] float32 cuda tensor with root angular velocity.
        """
    def read_root_angvel_host(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
    def read_root_linvel(self, tensor: typing.Any) -> None:
        """
        Fill the [n, 3] float32 cuda tensor with root linear velocity.
        """
    def read_root_linvel_host(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
    def read_root_pose(self, tensor: typing.Any) -> None:
        """
        Fill the [n, 7] float32 cuda tensor with root pose [qx,qy,qz,qw,px,py,pz].
        """
    def read_root_pose_host(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
    def step(self, dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Advance every articulation one substep on the GPU (no binding sync). Releases the GIL while stepping.
        """
    def write_joint_force(self, tensor: typing.Any) -> None:
        """
        Apply per-DOF joint forces/torques (effort control) from the [n, max_dofs] float32 cuda tensor. Re-apply each step (forces don't persist). Use for force-controlled joints.
        """
    def write_joint_pos(self, tensor: typing.Any) -> None:
        """
        Overwrite ALL joints' positions from the [n, max_dofs] float32 cuda tensor (full-batch reset).
        """
    def write_joint_target_pos(self, tensor: typing.Any) -> None:
        """
        Set all joints' PD position targets from the [n, max_dofs] float32 cuda tensor.
        """
    def write_joint_target_vel(self, tensor: typing.Any) -> None:
        """
        Set all joints' PD velocity targets from the [n, max_dofs] float32 cuda tensor.
        """
    def write_joint_vel(self, tensor: typing.Any) -> None:
        """
        Overwrite ALL joints' velocities from the [n, max_dofs] float32 cuda tensor (full-batch reset).
        """
    def write_link_force(self, tensor: typing.Any) -> None:
        """
        Apply an external force (N) to every link, from the [n, max_links, 3] float32 cuda tensor, in WORLD coordinates at each link's centre of mass. This is the only way to push a batched robot: ArticulationLink.add_force is a CPU-path call and PhysX rejects it outright under direct-GPU. Forces are consumed by the next step and cleared, so re-apply every substep you want them to act on — a random shove is one substep of impulse/dt on the base link.
        """
    def write_link_torque(self, tensor: typing.Any) -> None:
        """
        Apply an external torque (N*m) to every link, from the [n, max_links, 3] float32 cuda tensor, in WORLD coordinates. Cleared after each step, like write_link_force.
        """
    def write_subset_joint_pos(self, src: typing.Any, indices: typing.Any) -> None:
        ...
    def write_subset_joint_vel(self, src: typing.Any, indices: typing.Any) -> None:
        ...
    def write_subset_root_angvel(self, src: typing.Any, indices: typing.Any) -> None:
        ...
    def write_subset_root_linvel(self, src: typing.Any, indices: typing.Any) -> None:
        ...
    def write_subset_root_pose(self, src: typing.Any, indices: typing.Any) -> None:
        ...
    @property
    def count(self) -> int:
        ...
    @property
    def max_dofs(self) -> int:
        ...
    @property
    def max_links(self) -> int:
        ...
class PhysxMaterial:
    """
    A contact material (surface friction + restitution). Create via world.create_material(...), pass to add_link/add/add_static. The static_friction / dynamic_friction / restitution properties are mutable at runtime — re-roll them each reset for per-env friction domain randomization (a key sim-to-real robustness lever).
    """
    def set(self, static_friction: typing.SupportsFloat | typing.SupportsIndex, dynamic_friction: typing.SupportsFloat | typing.SupportsIndex, restitution: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set all three coefficients at once (the domain-randomization hot path).
        """
    @property
    def dynamic_friction(self) -> float:
        ...
    @dynamic_friction.setter
    def dynamic_friction(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def restitution(self) -> float:
        ...
    @restitution.setter
    def restitution(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def static_friction(self) -> float:
        ...
    @static_friction.setter
    def static_friction(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PhysxSoftBodyMaterial:
    """
    A deformable-volume material: Young's modulus (Pa), Poisson's ratio and surface friction. Create via world.create_soft_body_material(...) and pass the SAME handle to every add_soft_body that shares it — PhysX keeps one PxMaterial per call otherwise. Not interchangeable with PhysxMaterial (rigid bodies use a different PhysX type).
    """
    @property
    def damping(self) -> float:
        ...
    @damping.setter
    def damping(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dynamic_friction(self) -> float:
        ...
    @dynamic_friction.setter
    def dynamic_friction(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def poisson(self) -> float:
        ...
    @poisson.setter
    def poisson(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def young(self) -> float:
        ...
    @young.setter
    def young(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PhysxVehicle:
    """
    A drivable 4-wheel vehicle (PxVehicle2 direct drive) in a PhysxWorld. Feed it throttle/brake/steer each frame and copy its chassis + wheel poses onto your visuals; it advances itself inside world.step(dt). Valid only while its world lives. Wheel indices: 0=front-right, 1=front-left, 2=rear-right, 3=rear-left.
    """
    class Gear:
        """
        Members:
        
          REVERSE
        
          NEUTRAL
        
          FORWARD
        """
        FORWARD: typing.ClassVar[PhysxVehicle.Gear]  # value = <Gear.FORWARD: 2>
        NEUTRAL: typing.ClassVar[PhysxVehicle.Gear]  # value = <Gear.NEUTRAL: 1>
        REVERSE: typing.ClassVar[PhysxVehicle.Gear]  # value = <Gear.REVERSE: 0>
        __members__: typing.ClassVar[dict[str, PhysxVehicle.Gear]]  # value = {'REVERSE': <Gear.REVERSE: 0>, 'NEUTRAL': <Gear.NEUTRAL: 1>, 'FORWARD': <Gear.FORWARD: 2>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def __init__(self, world: PhysxWorld, chassis_width: typing.SupportsFloat | typing.SupportsIndex = 1.9500000476837158, chassis_height: typing.SupportsFloat | typing.SupportsIndex = 1.399999976158142, chassis_length: typing.SupportsFloat | typing.SupportsIndex = 4.400000095367432, chassis_mass: typing.SupportsFloat | typing.SupportsIndex = 1500.0, wheelbase: typing.SupportsFloat | typing.SupportsIndex = 2.6600000858306885, track_width: typing.SupportsFloat | typing.SupportsIndex = 1.649999976158142, wheel_radius: typing.SupportsFloat | typing.SupportsIndex = 0.4000000059604645, wheel_half_width: typing.SupportsFloat | typing.SupportsIndex = 0.15000000596046448, wheel_mass: typing.SupportsFloat | typing.SupportsIndex = 25.0, driven_wheels: typing.Annotated[collections.abc.Sequence[bool], "FixedSize(4)"] = [True, True, True, True], max_throttle_torque: typing.SupportsFloat | typing.SupportsIndex = 1500.0, max_brake_torque: typing.SupportsFloat | typing.SupportsIndex = 5000.0, max_steer_angle: typing.SupportsFloat | typing.SupportsIndex = 0.6000000238418579, tire_friction: typing.SupportsFloat | typing.SupportsIndex = 2.0, longitudinal_stiffness: typing.SupportsFloat | typing.SupportsIndex = 100000.0, lateral_stiffness: typing.SupportsFloat | typing.SupportsIndex = 80000.0, suspension_travel: typing.SupportsFloat | typing.SupportsIndex = 0.30000001192092896, suspension_stiffness: typing.SupportsFloat | typing.SupportsIndex = 35000.0, suspension_damping: typing.SupportsFloat | typing.SupportsIndex = 4500.0, suspension_attachment_y: typing.SupportsFloat | typing.SupportsIndex = -0.4000000059604645, wheel_damping_rate: typing.SupportsFloat | typing.SupportsIndex = 1.5, position: Vector3 = ..., rotation: Quaternion = ...) -> None:
        """
        Spawn a vehicle in `world`. Defaults are the Range Rover Evoque tuning of the C++ demo: 4WD direct drive, tire_friction 2.0 (dry asphalt). Dimensions are the chassis box PhysX simulates — match them to whatever body you draw on top. driven_wheels selects which wheels take throttle torque, in wheel-index order.
        """
    def add_force_at_pos(self, force: Vector3, world_pos: Vector3) -> None:
        """
        Apply a continuous force (N) to the chassis at a world-space point — the way to add something PhysX's vehicle knows nothing about, e.g. the bulldozing drag of a wheel ploughing through soil. Consumed by the next step().
        """
    def clear_road_override(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Give this wheel back to the scene query (the rigid fallback).
        """
    def respawn(self, position: Vector3, rotation: Quaternion = ...) -> None:
        """
        Teleport the chassis and kill its velocities (the wheels keep spinning down on their own). The suspension re-settles over the next few steps.
        """
    def road_override_active(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def set_brake(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Brake, 0..1 (all four wheels).
        """
    def set_road_override(self, wheel: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, v_road: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        """
        Hand this wheel's suspension a road of your own instead of what the PhysX scene query found: a horizontal plane at world y=`height` with friction `mu`. Set it per wheel, per frame, from whatever ground model you own — e.g. terrain grade minus the soil's equilibrium sinkage, so the wheel rides IN the ground by exactly the sinkage the load dictates, with mu from Mohr-Coulomb rather than the tire_friction ceiling. On a road PROFILE also pass `v_road`, the surface's vertical velocity under the wheel (v * slope, m/s, +up) — it reaches the tire slip terms. NOTE (probe-verified, PhysX 5.x): the suspension damper IGNORES it and still measures against a static plane, so suspension_force() reads biased low by damping*v_road on grades while the limit constraint quietly carries the difference; see the header for why compensating externally makes it worse. Everything else about the vehicle is untouched.
        """
    def set_steer(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Steer, -1..1 (front wheels; 1 = max_steer_angle to the right).
        """
    def set_throttle(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Throttle, 0..1. Direct drive: torque straight to the driven wheels.
        """
    def suspension_force(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Wheel load (N): the magnitude of the suspension force this wheel puts into the chassis. On level ground the four sum to the chassis weight and redistribute under braking/cornering — this is the W a soil model wants for sinkage and grip.
        """
    def suspension_jounce(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Suspension compression (m): 0 = full droop, suspension_travel = bottomed out.
        """
    def suspension_jounce_speed(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Compression rate (m/s) — spikes on landings and curb strikes.
        """
    def tire_lateral_slip(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Lateral slip (≈ tan of the slip angle; 0.1 ≈ 6° of drift).
        """
    def tire_longitudinal_slip(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Longitudinal slip ratio: 0 = pure rolling, +1 = the wheel spinning up under a stationary car, -1 = locked. The wheelspin readout.
        """
    def wheel_angular_speed(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Wheel spin rate (rad/s).
        """
    def wheel_grounded(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> bool:
        """
        True while this wheel has ground within suspension reach.
        """
    def wheel_local_pose(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> tuple[Vector3, Quaternion]:
        """
        (position, quaternion) of the wheel in CHASSIS space — steer, suspension travel and spin included. Put your wheel visual under the chassis group and assign both.
        """
    def wheel_rotation_angle(self, wheel: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Wheel spin angle (radians, wrapped to ±2π).
        """
    @property
    def forward_speed(self) -> float:
        """
        Speed along the chassis forward axis (m/s); negative in reverse.
        """
    @property
    def gear(self) -> PhysxVehicle.Gear:
        """
        Gear.FORWARD / Gear.NEUTRAL / Gear.REVERSE. Direct drive has no gearbox — this only picks the sign of the drive torque.
        """
    @gear.setter
    def gear(self, arg1: PhysxVehicle.Gear) -> None:
        ...
    @property
    def position(self) -> Vector3:
        """
        Chassis center, world space. Copy onto your visual each frame.
        """
    @property
    def quaternion(self) -> Quaternion:
        ...
class PhysxWorld:
    """
    A PhysX rigid-body world wired to the threepp scene graph. Add meshes as bodies, then call step(dt) each frame; every bound mesh's position/quaternion follows the simulation. Pure CPU — no canvas or renderer required.
    """
    def __init__(self, gravity: Vector3 = ..., fixed_timestep: typing.SupportsFloat | typing.SupportsIndex = 0.01666666753590107, max_substeps: typing.SupportsInt | typing.SupportsIndex = 4, num_threads: typing.SupportsInt | typing.SupportsIndex = 2, gpu_dynamics: bool = False, direct_gpu: bool = False, tgs_pcm: bool = False, cuda_context: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        gpu_dynamics requires a CUDA GPU (needed for soft bodies). direct_gpu also enables the PhysX direct-GPU API for batched GPU-resident articulation state I/O (PhysxGpuBatch) — the basis for GPU vectorized RL. Under direct_gpu the per-actor CPU getters and the binding-sync step() are NOT valid. tgs_pcm makes a CPU world use the TGS solver + PCM + stabilization (the GPU path always does) so its contact model MATCHES a GPU-trained policy for sim-to-sim deploy. cuda_context (an existing CUcontext as an int, e.g. torch's primary context) makes PhysX share that context instead of creating its own — required to mix PhysX GPU work with the framework's cuBLAS/cuDNN on the same device.
        """
    def add(self, mesh: Mesh, density: typing.SupportsFloat | typing.SupportsIndex = 1000.0, material: typing.Any = None) -> RigidBody:
        """
        Add a dynamic body whose shape is inferred from the mesh's Box/Sphere/Capsule geometry; the mesh is bound so it follows the sim. `material` (from create_material) overrides the contact friction/restitution. Returns a RigidBody.
        """
    def add_dynamic_convex(self, mesh: Mesh, density: typing.SupportsFloat | typing.SupportsIndex = 1000.0) -> RigidBody:
        """
        Add a dynamic body as the convex hull of the mesh's vertices (arbitrary shapes).
        """
    def add_instanced(self, mesh: InstancedMesh, density: typing.SupportsFloat | typing.SupportsIndex = 1000.0) -> list[RigidBody]:
        """
        Add one dynamic body per instance of an InstancedMesh. Returns a list of RigidBody.
        """
    def add_soft_body(self, mesh: Mesh, material: typing.Any = None, voxel_resolution: typing.SupportsInt | typing.SupportsIndex = 10, solver_iterations: typing.SupportsInt | typing.SupportsIndex = 20, self_collision: bool = False, cache_key: str = '', mass: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> SoftBody:
        """
        Cook `mesh` into a deformable volume and add it. Requires PhysxWorld(gpu_dynamics=True). The mesh's world matrix is baked into the cooked geometry and its local transform reset, so place the mesh first, then add it. voxel_resolution sets the SIMULATION mesh detail (higher = finer and slower); the collision mesh is conforming and follows the surface. cache_key reuses the (expensive) cook and per-vertex binding across every body built from the same source geometry at the same voxel_resolution — pass the model's filename to pay the cook once per species. mass in kg; 0 keeps the unit-density mass from the tet volume. Returns a SoftBody handle; the world owns the body.
        """
    def add_static(self, mesh: Mesh, material: typing.Any = None) -> RigidBody:
        """
        Add a static collider inferred from the mesh's Box/Sphere/Capsule geometry. `material` (from create_material) sets its friction/restitution — e.g. a grippy floor.
        """
    def add_static_heightfield(self, heights: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], cell: typing.SupportsFloat | typing.SupportsIndex, origin: Vector3, thickness: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> RigidBody:
        """
        Add a 2.5D height field as a static collider, in a Z-UP world.
        
        `heights` is a 2-D float32/float64 array of shape (ny, nx): ROWS ARE Y. heights[iy, ix] is the surface z at world (origin.x + ix*cell, origin.y + iy*cell), with `cell` the sample spacing in metres, the same in x and y. A (nx, ny) grid indexed the other way round must be passed transposed. Heights are quantised to int16 against the field's own z range (0.6 mm for a 20 m range), and a non-finite sample is pinned to the field's floor rather than poisoning the whole grid.
        
        Use this instead of add_static_trimesh wherever the collider is terrain: a height field cannot represent a HOLE or a near-vertical SPIKE, which is what a marching-cubes bake of a scan is full of, and it costs ~4 bytes a sample against ~100 for the same surface as triangles.
        
        `thickness` is how deep below the surface the field is meant to stay solid. PhysX 5 has no such knob (PxHeightFieldDesc::thickness was a PhysX 3 field) and a height field is a surface, not a volume: measured, a 4 cm ball at 6 m/s tunnels through it exactly as it tunnels through the same surface as a trimesh. Nothing tunnels while the body's diameter exceeds its per-substep travel. The argument is accepted and validated (> 0) but PhysX cannot honour it.
        
        Returns a RigidBody; the world owns the actor.
        """
    def add_static_trimesh(self, mesh: Mesh) -> RigidBody:
        """
        Add a static collider matching the mesh triangles exactly (static/kinematic only).
        """
    def add_static_trimesh_tree(self, root: typing.Any) -> list[RigidBody]:
        """
        Add every Mesh under `root` as its own static trimesh collider — e.g. turn an imported glTF environment straight into collision geometry. Returns a list.
        """
    def create_articulation(self, fixed_base: bool = False, solver_position_iterations: typing.SupportsInt | typing.SupportsIndex = 8, disable_self_collision: bool = False) -> Articulation:
        """
        Create a reduced-coordinate articulation (robot). fixed_base pins the root to the world (use for arms; leave false for free-floating bodies like a walking robot). Add links, then call finalize().
        """
    def create_material(self, static_friction: typing.SupportsFloat | typing.SupportsIndex = 0.5, dynamic_friction: typing.SupportsFloat | typing.SupportsIndex = 0.5, restitution: typing.SupportsFloat | typing.SupportsIndex = 0.0, friction_combine: str = 'average', restitution_combine: str = 'average') -> PhysxMaterial:
        """
        Create a contact material. Defaults: friction 0.5/0.5, restitution 0 (no bounce — right for feet/locomotion, unlike the world's shared 0.2 default). combine modes ('average'|'min'|'multiply'|'max') control how two contacting materials' coefficients mix — use 'min' so a clean material governs a contact against a different one. The returned PhysxMaterial is mutable (per-env friction randomization). Keeps the world alive.
        """
    def create_soft_body_material(self, young: typing.SupportsFloat | typing.SupportsIndex = 1000000.0, poisson: typing.SupportsFloat | typing.SupportsIndex = 0.44999998807907104, friction: typing.SupportsFloat | typing.SupportsIndex = 0.5, damping: typing.Any = None) -> PhysxSoftBodyMaterial:
        """
        Create a deformable-volume material (Young's modulus Pa, Poisson's ratio, surface friction). Requires gpu_dynamics=True. Create ONE and share it across every add_soft_body that uses the same flesh — each call allocates a PxMaterial that lives until the world dies.
        """
    def load_articulation(self, path: str, fixed_base: bool = False, base_position: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"] = [0.0, 0.0, 0.0], default_density: typing.SupportsFloat | typing.SupportsIndex = 1000.0, stiffness: typing.SupportsFloat | typing.SupportsIndex = 0.0, damping: typing.SupportsFloat | typing.SupportsIndex = 0.0, max_force: typing.SupportsFloat | typing.SupportsIndex = 1000000.0, self_collision: bool = False, solver_position_iterations: typing.SupportsInt | typing.SupportsIndex = 12, render_visuals: bool = True, scale: typing.SupportsFloat | typing.SupportsIndex = 1.0, args: collections.abc.Mapping[str, str] = {}) -> tuple[Articulation, list[Mesh], list[str]]:
        """
        Import a URDF/xacro as a finalized Articulation (one shared parser with the C++ URDFLoader — xacro supported). Returns (articulation, meshes, joint_names): the collider meshes are bound to the sim (add them to a scene to render), joint_names lists the actuated joints in drive-target order. Per-link handles are on the articulation itself: articulation.link('tool_link_name') resolves every URDF link name to its ArticulationLink (for add_force on a tool link, per-link poses...). Collision is primitive/bbox, mass from <inertial> (else default_density x volume); fixed joints are collapsed. stiffness/damping/max_force set a PD drive on every joint. scale reinterprets the file's length units (a millimetre URDF in a metre world is 0.001) - shapes, joint frames and prismatic limits are built scaled, masses stay as authored, and a prismatic DOF then reads and drives in the SCALED units. `args` are xacro argument overrides, the same name:=value pairs the xacro CLI takes - a parameterised description built without them expands to the FILE's defaults, which for many robots names config paths that do not exist.
        """
    def on_post_substep(self, callback: collections.abc.Callable) -> int:
        """
        Register callback(dt) fired after each fixed substep. Returns a handle for remove_substep_callback().
        """
    def on_pre_substep(self, callback: collections.abc.Callable) -> int:
        """
        Register callback(dt) fired before each fixed substep. Returns a handle for remove_substep_callback().
        """
    def register_sensor(self, sensor: Sensor) -> None:
        """
        Register a sensor (Imu, JointEncoder, ContactSensor, ...) to be sampled from the step loop once per fixed substep, the instant body states are fresh. Call AFTER adding the body the sensor is attached to; raises if the attachment has no managed rigid body. The world keeps the sensor alive.
        """
    def remove(self, body: RigidBody) -> None:
        """
        Remove a body (from add / add_static / add_dynamic_convex / add_static_trimesh) from the world and release it — e.g. to rebuild geometry without recreating the world. Any mesh binding is dropped; the RigidBody handle is INVALID afterwards (don't reuse it).
        """
    def remove_soft_body(self, body: SoftBody) -> None:
        """
        Destroy a soft body: releases the PhysX actor and its GPU/pinned buffers, and (for bodies added from a Mesh) detaches that mesh from its parent. The SoftBody handle is INVALID afterwards.
        """
    def remove_substep_callback(self, handle: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Unregister a pre/post substep callback by its handle. A stale or already-removed handle is a no-op.
        """
    def set_gravity(self, gravity: Vector3) -> None:
        ...
    def step(self, dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Advance the simulation by dt seconds (variable-rate caller, fixed-rate physics). After it returns, every bound mesh's transform reflects the new state. Releases the GIL while stepping.
        """
    def unregister_sensor(self, sensor: Sensor) -> None:
        """
        Stop sampling a previously registered sensor.
        """
    @property
    def sim_time(self) -> float:
        """
        Accumulated fixed-substep simulation time (s) — the clock stamped onto sensor samples.
        """
class PlaneGeometry(BufferGeometry):
    def __init__(self, width: typing.SupportsFloat | typing.SupportsIndex = 1.0, height: typing.SupportsFloat | typing.SupportsIndex = 1.0, width_segments: typing.SupportsInt | typing.SupportsIndex = 1, height_segments: typing.SupportsInt | typing.SupportsIndex = 1) -> None:
        ...
    @property
    def height(self) -> float:
        ...
    @property
    def width(self) -> float:
        ...
class PointLight(Light):
    def __init__(self, color: Color = ..., intensity: typing.SupportsFloat | typing.SupportsIndex = 1.0, distance: typing.SupportsFloat | typing.SupportsIndex = 0.0, decay: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    def get_power(self) -> float:
        ...
    def set_power(self, power: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def decay(self) -> float:
        ...
    @decay.setter
    def decay(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def distance(self) -> float:
        ...
    @distance.setter
    def distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PointLightHelper(Mesh):
    def __init__(self, light: PointLight, sphere_size: typing.SupportsFloat | typing.SupportsIndex = 1.0, color: Color | None = None) -> None:
        ...
    def update(self) -> None:
        ...
class Points(Object3D):
    cast_shadow: bool
    frustum_culled: bool
    layers: Layers
    matrix_auto_update: bool
    name: str
    position: Vector3
    quaternion: Quaternion
    receive_shadow: bool
    rotation: Euler
    scale: Vector3
    up: Vector3
    visible: bool
    def __init__(self, geometry: BufferGeometry = None, material: typing.Any = None) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def add(self, *args) -> None:
        ...
    def clear(self) -> None:
        ...
    def get_object_by_name(self, name: str) -> Object3D:
        ...
    def get_user_data(self, key: str) -> typing.Any:
        ...
    def get_world_direction(self) -> Vector3:
        ...
    def get_world_position(self) -> Vector3:
        ...
    def get_world_quaternion(self) -> Quaternion:
        ...
    def get_world_scale(self) -> Vector3:
        ...
    def local_to_world(self, vector: Vector3) -> Vector3:
        ...
    @typing.overload
    def look_at(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def look_at(self, vector: Vector3) -> None:
        ...
    def remove(self, object: typing.Any) -> None:
        ...
    def remove_from_parent(self) -> None:
        ...
    def rotate_on_axis(self, axis: Vector3, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_x(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_y(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_z(self, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_user_data(self, key: str, value: str) -> None:
        ...
    def translate_x(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def translate_y(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def translate_z(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def traverse(self, callback: collections.abc.Callable[[typing.Any], None]) -> None:
        ...
    def update_matrix(self) -> None:
        ...
    def update_matrix_world(self, force: bool = False) -> None:
        ...
    def world_to_local(self, vector: Vector3) -> Vector3:
        ...
    @property
    def children(self) -> list[Object3D]:
        ...
    @property
    def geometry(self) -> BufferGeometry:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def material(self) -> Material | None:
        ...
    @property
    def matrix_world(self) -> Matrix4:
        ...
    @property
    def parent(self) -> Object3D:
        ...
    @property
    def render_order(self) -> int:
        ...
    @render_order.setter
    def render_order(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def uuid(self) -> str:
        ...
class PointsMaterial(Material):
    alpha_map: Texture
    blending: Blending
    color: Color
    depth_test: bool
    depth_write: bool
    fog: bool
    map: Texture
    name: str
    premultiplied_alpha: bool
    side: Side
    size_attenuation: bool
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def size(self) -> float:
        ...
    @size.setter
    def size(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PolarGridHelper(LineSegments):
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex = 10.0, sectors: typing.SupportsInt | typing.SupportsIndex = 16, rings: typing.SupportsInt | typing.SupportsIndex = 8, divisions: typing.SupportsInt | typing.SupportsIndex = 64, color1: Color = ..., color2: Color = ...) -> None:
        ...
class PositionalAudio(Object3D):
    def __init__(self, listener: AudioListener, file: str) -> None:
        ...
    def play(self) -> None:
        ...
    def seek_to_start(self) -> None:
        ...
    def set_distance_model(self, model: AudioDistanceModel) -> None:
        ...
    def set_looping(self, loop: bool) -> None:
        ...
    def set_max_distance(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_min_distance(self, distance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_playback_rate(self, rate: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_rolloff_factor(self, rolloff: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_volume(self, volume: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def stop(self) -> None:
        ...
    def toggle_play(self) -> None:
        ...
    @property
    def is_playing(self) -> bool:
        ...
class Quaternion:
    def __init__(self, x: typing.SupportsFloat | typing.SupportsIndex = 0.0, y: typing.SupportsFloat | typing.SupportsIndex = 0.0, z: typing.SupportsFloat | typing.SupportsIndex = 0.0, w: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def invert(self) -> Quaternion:
        ...
    def normalize(self) -> Quaternion:
        ...
    def set(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex, w: typing.SupportsFloat | typing.SupportsIndex) -> Quaternion:
        ...
    def set_from_axis_angle(self, axis: Vector3, angle: typing.SupportsFloat | typing.SupportsIndex) -> Quaternion:
        ...
    def set_from_euler(self, euler: Euler) -> Quaternion:
        ...
    def slerp(self, qb: Quaternion, t: typing.SupportsFloat | typing.SupportsIndex) -> Quaternion:
        ...
    @property
    def w(self) -> float:
        ...
    @w.setter
    def w(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def z(self) -> float:
        ...
    @z.setter
    def z(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class QuaternionKeyframeTrack(KeyframeTrack):
    def __init__(self, name: str, times: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], values: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], interpolation: Interpolation | None = None) -> None:
        ...
class RGBELoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str, flip_y: bool = True) -> Texture:
        """
        Load a Radiance .hdr equirectangular environment as a float Texture.
        """
class RangeNoiseModel:
    """
    Range noise for a ranging sensor (LIDAR / depth camera). Per RETURN, not per second: sigma = hypot(stddev, r * stddev_per_metre) metres, plus a fixed `bias`. `seed` makes a scan reproducible — same seed and same beam order gives the same cloud on every run and machine. All-zero = a perfect sensor (clean ranges pass through untouched).
    """
    def __init__(self, stddev: typing.SupportsFloat | typing.SupportsIndex = 0.0, stddev_per_metre: typing.SupportsFloat | typing.SupportsIndex = 0.0, bias: typing.SupportsFloat | typing.SupportsIndex = 0.0, seed: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def bias(self) -> float:
        """
        Fixed offset [m]; positive reads long.
        """
    @bias.setter
    def bias(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def seed(self) -> int:
        ...
    @seed.setter
    def seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def stddev(self) -> float:
        """
        Constant sigma [m].
        """
    @stddev.setter
    def stddev(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def stddev_per_metre(self) -> float:
        """
        Range-proportional sigma [m/m], added in quadrature.
        """
    @stddev_per_metre.setter
    def stddev_per_metre(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Ray:
    direction: Vector3
    origin: Vector3
    def __init__(self, origin: Vector3 = ..., direction: Vector3 = ...) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def apply_matrix4(self, matrix: Matrix4) -> Ray:
        ...
    def at(self, t: typing.SupportsFloat | typing.SupportsIndex) -> Vector3:
        """
        The point `t` along the ray from its origin.
        """
    def distance_to_point(self, point: Vector3) -> float:
        ...
    def intersects_box(self, box: Box3) -> bool:
        ...
    def look_at(self, v: Vector3) -> Ray:
        ...
    def set(self, origin: Vector3, direction: Vector3) -> Ray:
        ...
class RayHit:
    def __repr__(self) -> str:
        ...
    @property
    def distance(self) -> float:
        ...
    @property
    def normal(self) -> Vector3:
        """
        Geometric face normal, flipped to face the ray origin.
        """
    @property
    def point(self) -> Vector3:
        ...
    @property
    def triangle_index(self) -> int:
        ...
class RectAreaLight(Light):
    def __init__(self, color: Color = ..., intensity: typing.SupportsFloat | typing.SupportsIndex = 1.0, width: typing.SupportsFloat | typing.SupportsIndex = 1.0, height: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    @property
    def height(self) -> float:
        ...
    @property
    def width(self) -> float:
        ...
class RigidBody:
    """
    Handle to a PhysX actor created via PhysxWorld. Valid only while its world is alive. Velocity/force/kinematic operations require a dynamic body.
    """
    angular_velocity: Vector3
    linear_velocity: Vector3
    def add_force(self, force: Vector3) -> None:
        """
        Apply a continuous force (N), consumed by the next step().
        """
    def add_impulse(self, impulse: Vector3) -> None:
        """
        Apply an instantaneous impulse (kg·m/s).
        """
    def set_angular_damping(self, d: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_angular_velocity(self, v: Vector3) -> None:
        ...
    def set_kinematic(self, kinematic: bool) -> None:
        """
        Toggle kinematic mode: the body is driven by set_kinematic_target and ignores forces/gravity.
        """
    def set_kinematic_target(self, position: Vector3, quaternion: Quaternion = ...) -> None:
        ...
    def set_linear_damping(self, d: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_linear_velocity(self, v: Vector3) -> None:
        ...
    def set_pose(self, position: Vector3, quaternion: Quaternion = ...) -> None:
        ...
    def wake_up(self) -> None:
        ...
    @property
    def is_dynamic(self) -> bool:
        ...
    @property
    def mass(self) -> float:
        ...
    @property
    def position(self) -> Vector3:
        ...
    @property
    def quaternion(self) -> Quaternion:
        ...
class RingGeometry(BufferGeometry):
    def __init__(self, inner_radius: typing.SupportsFloat | typing.SupportsIndex = 0.5, outer_radius: typing.SupportsFloat | typing.SupportsIndex = 1.0, theta_segments: typing.SupportsInt | typing.SupportsIndex = 16, phi_segments: typing.SupportsInt | typing.SupportsIndex = 2, theta_start: typing.SupportsFloat | typing.SupportsIndex = 0.0, theta_length: typing.SupportsFloat | typing.SupportsIndex = 6.2831854820251465) -> None:
        ...
class Robot(Object3D):
    def __init__(self) -> None:
        ...
    def compute_end_effector_transform(self, values: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], deg: bool = False, enforce_limits: bool = True) -> Matrix4:
        ...
    def get_articulated_joint_info(self) -> list[JointInfo]:
        ...
    def get_end_effector_transform(self) -> Matrix4:
        ...
    def get_joint_range(self, index: typing.SupportsInt | typing.SupportsIndex, deg: bool = False) -> JointRange:
        ...
    def get_joint_ranges(self, deg: bool = False) -> list[JointRange]:
        ...
    def get_joint_value(self, index: typing.SupportsInt | typing.SupportsIndex, deg: bool = False) -> float:
        ...
    def joint_values(self, deg: bool = False) -> list[float]:
        ...
    def set_end_effector(self, link_name: str) -> None:
        """
        Retarget FK/IK at the named link. Recomputes the root-to-tool path and therefore `chain_dofs`. Raises if the link is unknown.
        """
    def set_joint_value(self, index: typing.SupportsInt | typing.SupportsIndex, value: typing.SupportsFloat | typing.SupportsIndex, deg: bool = False) -> None:
        ...
    def set_joint_values(self, values: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], deg: bool = False) -> None:
        ...
    def show_colliders(self, flag: bool) -> None:
        ...
    @property
    def chain_dofs(self) -> list[int]:
        """
        DOF indices on the root-to-end-effector path, ascending — the only ones an IkSolver is allowed to move. A gripper's finger joints keep their slots in the joint vector but are not here, so closing the hand can never be mistaken for extra reach.
        """
    @property
    def end_effector_link(self) -> str:
        """
        Name of the link FK and IK currently drive.
        """
    @property
    def num_dof(self) -> int:
        ...
class RotationOrder:
    """
    Members:
    
      XYZ
    
      YZX
    
      ZXY
    
      XZY
    
      YXZ
    
      ZYX
    """
    XYZ: typing.ClassVar[RotationOrder]  # value = <RotationOrder.XYZ: 0>
    XZY: typing.ClassVar[RotationOrder]  # value = <RotationOrder.XZY: 3>
    YXZ: typing.ClassVar[RotationOrder]  # value = <RotationOrder.YXZ: 4>
    YZX: typing.ClassVar[RotationOrder]  # value = <RotationOrder.YZX: 1>
    ZXY: typing.ClassVar[RotationOrder]  # value = <RotationOrder.ZXY: 2>
    ZYX: typing.ClassVar[RotationOrder]  # value = <RotationOrder.ZYX: 5>
    __members__: typing.ClassVar[dict[str, RotationOrder]]  # value = {'XYZ': <RotationOrder.XYZ: 0>, 'YZX': <RotationOrder.YZX: 1>, 'ZXY': <RotationOrder.ZXY: 2>, 'XZY': <RotationOrder.XZY: 3>, 'YXZ': <RotationOrder.YXZ: 4>, 'ZYX': <RotationOrder.ZYX: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class STLLoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str) -> BufferGeometry:
        ...
class SVGLoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str) -> Group:
        """
        Load an .svg file as a Group of filled meshes.
        """
    def parse(self, text: str) -> Group:
        """
        Parse SVG XML into a Group of filled meshes.
        """
class ScalarField:
    def __init__(self) -> None:
        ...
    def at(self, x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex, z: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    def data_numpy(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Return field data as (nz, ny, nx) float32 numpy array.
        """
    @property
    def cell_size(self) -> float:
        ...
    @property
    def empty(self) -> bool:
        ...
    @property
    def nx(self) -> int:
        ...
    @property
    def ny(self) -> int:
        ...
    @property
    def nz(self) -> int:
        ...
    @property
    def origin(self) -> Vector3:
        ...
class Scene(Object3D):
    auto_update: bool
    environment: Texture
    override_material: Material
    def __init__(self) -> None:
        ...
    def clear_fog(self) -> None:
        ...
    def set_fog(self, color: Color, near: typing.SupportsFloat | typing.SupportsIndex = 1.0, far: typing.SupportsFloat | typing.SupportsIndex = 1000.0) -> None:
        ...
    def set_fog_exp2(self, color: Color, density: typing.SupportsFloat | typing.SupportsIndex = 0.019999999552965164) -> None:
        """
        Exponential participating-media fog. On Vulkan: Beer-Lambert + volumetric scattering. Call renderer.fog_anisotropy to tune the Henyey-Greenstein phase (0 = isotropic, +0.9 = forward god-rays, -0.9 = back-scatter halo).
        """
    @property
    def background(self) -> Background:
        ...
    @background.setter
    def background(self, arg1: typing.Any) -> None:
        ...
class Sensor:
    """
    Abstract base of the sensor suite. Register a concrete sensor with PhysxWorld.register_sensor to have it sampled from the step loop.
    """
    def advance_clock(self, dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Advance the sensor clock by dt seconds.
        """
    @property
    def rate_hz(self) -> float:
        """
        Target sample rate (Hz); 0 = every physics substep.
        """
    @rate_hz.setter
    def rate_hz(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sim_time(self) -> float:
        """
        The sensor's clock (s) — the time base every measurement is stamped with. Latched from the world automatically while registered with a PhysxWorld; drive it yourself (advance_clock / sim_time = t) for a sensor pulled from a render loop. Always sim time, never wall time, so a replayed run reproduces its timestamps.
        """
    @sim_time.setter
    def sim_time(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class ShadowMaterial(Material):
    blending: Blending
    color: Color
    depth_test: bool
    depth_write: bool
    fog: bool
    name: str
    premultiplied_alpha: bool
    side: Side
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Shape(Path):
    def __init__(self, points: collections.abc.Sequence[Vector2] = []) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def extract_points(self, divisions: typing.SupportsInt | typing.SupportsIndex = 12) -> tuple[list[Vector2], list[list[Vector2]]]:
        """
        (outline, holes) as point lists, sampling every curve segment with `divisions` steps.
        """
    @property
    def holes(self) -> list[Path]:
        ...
    @holes.setter
    def holes(self, arg0: collections.abc.Sequence[Path]) -> None:
        ...
    @property
    def uuid(self) -> str:
        ...
class ShapeGeometry(BufferGeometry):
    def __init__(self, shapes: typing.Any, curve_segments: typing.SupportsInt | typing.SupportsIndex = 12) -> None:
        ...
class Side:
    """
    Members:
    
      Front
    
      Back
    
      Double
    """
    Back: typing.ClassVar[Side]  # value = <Side.Back: 1>
    Double: typing.ClassVar[Side]  # value = <Side.Double: 2>
    Front: typing.ClassVar[Side]  # value = <Side.Front: 0>
    __members__: typing.ClassVar[dict[str, Side]]  # value = {'Front': <Side.Front: 0>, 'Back': <Side.Back: 1>, 'Double': <Side.Double: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SkeletonHelper(LineSegments):
    def __init__(self, skeleton: typing.Any) -> None:
        ...
class SoftBody:
    """
    Handle to a PhysX deformable volume created via world.add_soft_body. Valid only while its world lives (and until remove_soft_body). The simulation runs on two meshes: a CONFORMING collision tet mesh (what tet_mesh/sim_positions report, and what contact is resolved against) and a voxelised simulation mesh the solver integrates; voxel_resolution sizes the latter.
    """
    def enable_gpu_skinning(self) -> None:
        """
        Blend the visual mesh in the vertex shader from a small per-body tet texture instead of CPU-skinning and re-uploading the full-resolution mesh every step. Call once, right after add_soft_body. Also the cheap option when the visual mesh is NOT drawn at all (an external skinner owns the render surface): it reduces the per-step cost to one few-hundred-texel texture write.
        """
    def set_recompute_normals(self, enabled: bool) -> None:
        """
        Recompute the visual geometry's vertex normals each step (default on). Turn it off when something else owns the normals — it is a full pass over the visual mesh.
        """
    def sim_positions(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Current collision-mesh vertex positions as an (N, 3) float32 array, world space. One device->host copy per call — read it once per frame, not once per fish per query.
        """
    def tet_mesh(self) -> tuple:
        """
        ((V, 3) float32 rest vertices, (T, 4) int32 tets) of the cooked CONFORMING collision mesh. Rest vertices are in the template mesh's own local space, so they differ from sim_positions() by the spawn transform only — rest-relative metrics (volume ratio, edge stretch) compare directly. Feed the pair to another solver to run PhysX's tetrahedralisation elsewhere.
        """
    @property
    def num_tets(self) -> int:
        """
        Collision-mesh tetrahedron count.
        """
    @property
    def num_vertices(self) -> int:
        """
        Collision-mesh vertex count (the length of sim_positions()).
        """
class SogLoader:
    @staticmethod
    def describe(path: os.PathLike | str | bytes) -> dict:
        """
        What the asset holds, WITHOUT decoding a plane — json only, milliseconds against the gigabyte load() of level 0 costs. Returns a dict: lod_levels, sh_degree, counts (splats per level, finest first), levels (per-level dicts with lod/count/chunks) and bound_min/max. Levels are ALTERNATIVES, not a residual pyramid: each covers the whole scene at its own density, so the counts do not add up to a total.
        """
    @staticmethod
    def is_sog(path: os.PathLike | str | bytes) -> bool:
        """
        Is this a SOG / SuperSplat SSOG asset? Answers by CONTENT (the meta.json / lod-meta.json inside), not by name — a SOG asset is usually an extensionless directory. Never raises: a missing or malformed asset is simply False.
        """
    @staticmethod
    def load(path: os.PathLike | str | bytes, level: typing.SupportsInt | typing.SupportsIndex = 0) -> SplatData:
        """
        Read ONE detail level of a SOG asset into a SplatData. 0 is the finest (and the only legal value for a lone chunk, which declares no levels). Seconds and gigabytes at level 0 on a real scan — call describe() first and pick a coarser level if that is not what you want. Raises RuntimeError naming the offending member, file and numbers.
        
        FRAME: the file's own coordinates, untouched. A SOG re-encoded from a COLMAP .ply is +Y DOWN like its .ply, so the caller flips it the same way — cloud.rotation.x = math.pi is what the editor and the gaussian_splats example do.
        """
    def __init__(self) -> None:
        ...
class SphereGeometry(BufferGeometry):
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex = 1.0, width_segments: typing.SupportsInt | typing.SupportsIndex = 16, height_segments: typing.SupportsInt | typing.SupportsIndex = 12, phi_start: typing.SupportsFloat | typing.SupportsIndex = 0.0, phi_length: typing.SupportsFloat | typing.SupportsIndex = 6.2831854820251465, theta_start: typing.SupportsFloat | typing.SupportsIndex = 0.0, theta_length: typing.SupportsFloat | typing.SupportsIndex = 3.1415927410125732) -> None:
        ...
    @property
    def radius(self) -> float:
        ...
class SplatCloud(Mesh):
    @staticmethod
    def from_sog_lod(path: os.PathLike | str | bytes) -> SplatCloud:
        """
        Load a multi-level SOG asset for DYNAMIC LOD: every other level (0, 2, 4, ...) resident in one cloud, with the LOD table set on it. Every other because residency is the sum of the levels and it is paid twice (GL textures + the Vulkan pass); adjacent levels differ by only 2x, and keeping level 0 is what the close-up invariant needs.
        
        Then call select_lod(cloud, camera, viewport_h) once per frame: it writes submit_ranges for the level and chunks that frame should draw. A single-level asset comes back with an EMPTY table — it renders plainly and select_lod is a no-op returning 0.
        """
    def __init__(self, data: SplatData) -> None:
        """
        Create a renderable splat cloud. CONSUMES `data` (it is left empty) to avoid copying gigabyte-scale scans; load the file again if you need a second cloud.
        """
    def set_viewport_size(self, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Override the framebuffer pixel size used for splat scaling — only needed when rendering into a target whose size differs from the renderer's own.
        """
    def update(self, camera: Camera) -> None:
        """
        Sort splats back-to-front for this camera and refresh per-frame state. Cheap when nothing moved. The object's own pre-render hook also calls this, but one frame LATE for single-shot captures — call it explicitly before render() when grabbing a single frame headless.
        """
    @property
    def cpu_bytes(self) -> int:
        """
        Host memory this cloud holds (bytes): splat data + sort scratch, plus the GL data textures once a GL frame has built them. ~423 B/splat at SH degree 3 after a GL draw.
        """
    @property
    def debug_non_finite(self) -> bool:
        """
        Draw non-finite colour fragments magenta instead of discarding them, so a cloud quietly producing NaNs stops looking identical to a healthy one. Default off.
        """
    @debug_non_finite.setter
    def debug_non_finite(self, arg1: bool) -> None:
        ...
    @property
    def lod_held_level(self) -> int:
        """
        Index into lod_levels of the level select_lod is currently holding (hysteresis state lives on the cloud's own table). 0 without LOD. On the per-node path this is the FINEST level in use this frame, since there is no single level any more.
        """
    @property
    def lod_levels(self) -> list:
        """
        The resident detail levels as dicts (lod, base, count, chunks), finest first — empty unless the cloud came from from_sog_lod on a multi-level asset. `base` is the level's first splat index in this cloud.
        """
    @property
    def lod_node_count(self) -> int:
        """
        Leaves of the asset's SSOG tree resident in this cloud. 0 when the asset carries no tree (or no per-node offsets), which is exactly when select_lod falls back to whole-cloud selection.
        """
    @property
    def lod_node_levels(self) -> list:
        """
        The level index each tree node was submitted at by the last per-node select_lod, -1 for a node the frustum culled. Length lod_node_count; all -1 before the first per-node selection. A histogram of this is the readable form of 'what did LOD actually do this frame'.
        """
    @property
    def point_mix(self) -> float:
        """
        0 (default) renders Gaussians; 1 renders every splat as an opaque disc of point_size pixels at its centre, nearest wins — the point cloud view. Values between dissolve one into the other. Same depth sort and mesh occlusion on both backends.
        """
    @point_mix.setter
    def point_mix(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def point_size(self) -> float:
        """
        Disc diameter in pixels at point_mix 1. Floored at 1; default 2.
        """
    @point_size.setter
    def point_size(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def splat_count(self) -> int:
        """
        Number of splats in the cloud.
        """
    @property
    def submit_ranges(self) -> list[tuple[int, int]]:
        """
        Partial submission: a list of (offset, count) ranges into this cloud's splats to draw this frame, in order; empty (default) draws all. The chunk-LOD/culling mechanism. VULKAN ONLY today — the GL path draws every splat regardless. At most 64 ranges are honoured.
        """
    @submit_ranges.setter
    def submit_ranges(self, arg1: collections.abc.Sequence[tuple[typing.SupportsInt | typing.SupportsIndex, typing.SupportsInt | typing.SupportsIndex]]) -> None:
        ...
class SplatData:
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def count(self) -> int:
        """
        Number of splats held.
        """
class SplatLoader:
    @staticmethod
    def is_point_cloud_ply(path: os.PathLike | str | bytes) -> bool:
        """
        Does this .ply hold a colour-only point cloud (x/y/z, no f_dc_0, no faces)? Header only, never raises.
        """
    @staticmethod
    def is_splat_ply(path: os.PathLike | str | bytes) -> bool:
        """
        Does this .ply hold Gaussian splats rather than a mesh? Reads only the header and never raises — a missing or malformed file is simply False.
        """
    @staticmethod
    def load_ply(path: os.PathLike | str | bytes) -> SplatData:
        """
        Load a 3D-Gaussian-Splatting .ply (the format 3DGS optimisers emit) into a SplatData. Header-driven: files with/without normals, extra per-splat properties or any SH degree all parse. Raises RuntimeError with the offending property in the message on anything unrepresentable.
        """
    @staticmethod
    def load_point_cloud_ply(path: os.PathLike | str | bytes, sigma: typing.SupportsFloat | typing.SupportsIndex = 0.0, sigma_per_spacing: typing.SupportsFloat | typing.SupportsIndex = 1.0, opacity: typing.SupportsFloat | typing.SupportsIndex = 1.0, use_normals: bool = True, normal_thickness: typing.SupportsFloat | typing.SupportsIndex = 0.15000000596046448) -> SplatData:
        """
        Load a colour-only point-cloud .ply (binary or ascii; red/green/blue, nx/ny/nz and intensity honoured) as a SplatData of degree-0 Gaussians, one per point. sigma 0 sizes them from the cloud's median neighbour spacing times sigma_per_spacing; a point with normals becomes a disc facing them. Render with SplatCloud.point_mix = 1 for dots, 0 for a closed surface.
        """
    @staticmethod
    def write_ply(data: SplatData, path: os.PathLike | str | bytes) -> None:
        """
        Write a SplatData as a 3DGS .ply (the INRIA layout load_ply reads: channel-major f_rest, log scales, logit opacity, w-first rotation). Raises RuntimeError if the file cannot be written.
        """
    def __init__(self) -> None:
        ...
class SplatPoseSet:
    """
    Members:
    
      Orbit : Outside the fit sphere looking IN. Right for an object, a facade, an outdoor site seen from outside. The default.
    
      Interior : Inside the scan looking OUT, from the fit centre plus jittered stations. Right for a room, a canyon, anything the camera stands IN. Orbiting such a scan reconstructs the OUTSIDE of its walls and never observes the walkable volume at all.
    """
    Interior: typing.ClassVar[SplatPoseSet]  # value = <SplatPoseSet.Interior: 1>
    Orbit: typing.ClassVar[SplatPoseSet]  # value = <SplatPoseSet.Orbit: 0>
    __members__: typing.ClassVar[dict[str, SplatPoseSet]]  # value = {'Orbit': <SplatPoseSet.Orbit: 0>, 'Interior': <SplatPoseSet.Interior: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SplineCurve(Curve2):
    def __init__(self, points: collections.abc.Sequence[Vector2] = []) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def points(self) -> list[Vector2]:
        ...
    @points.setter
    def points(self, arg0: collections.abc.Sequence[Vector2]) -> None:
        ...
class SpotLight(Light):
    def __init__(self, color: Color = ..., intensity: typing.SupportsFloat | typing.SupportsIndex = 1.0, distance: typing.SupportsFloat | typing.SupportsIndex = 0.0, angle: typing.SupportsFloat | typing.SupportsIndex = 1.0471975803375244, penumbra: typing.SupportsFloat | typing.SupportsIndex = 0.0, decay: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    def get_target(self) -> Object3D:
        ...
    def set_target(self, target: Object3D) -> None:
        ...
    @property
    def angle(self) -> float:
        ...
    @angle.setter
    def angle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def decay(self) -> float:
        ...
    @decay.setter
    def decay(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def distance(self) -> float:
        ...
    @distance.setter
    def distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def penumbra(self) -> float:
        ...
    @penumbra.setter
    def penumbra(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SpotLightHelper(Object3D):
    def __init__(self, light: SpotLight, color: Color | None = None) -> None:
        ...
    def update(self) -> None:
        ...
class Sprite(Object3D):
    center: Vector2
    screen_anchor: Vector2
    screen_space: bool
    def __init__(self, material: SpriteMaterial = None) -> None:
        ...
    @property
    def material(self) -> Material | None:
        ...
class SpriteMaterial(Material):
    alpha_map: Texture
    blending: Blending
    color: Color
    depth_test: bool
    depth_write: bool
    fog: bool
    map: Texture
    name: str
    premultiplied_alpha: bool
    side: Side
    size_attenuation: bool
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    @property
    def alpha_test(self) -> float:
        ...
    @alpha_test.setter
    def alpha_test(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def opacity(self) -> float:
        ...
    @opacity.setter
    def opacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rotation(self) -> float:
        ...
    @rotation.setter
    def rotation(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def size(self) -> float:
        ...
    @size.setter
    def size(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SurfaceMesh:
    """
    The triangle surface bake_surface fused out of a splat cloud, in WORLD space (the cloud's transform is already in the vertices). Three consumers: make_sensor_mesh() for the renderer's sensors, PhysxWorld.add_static_trimesh for collision, and to_geometry() for anything else.
    """
    def __repr__(self) -> str:
        ...
    def to_geometry(self) -> BufferGeometry:
        """
        The surface as a BufferGeometry (position + index + computed normals), ready for tp.Mesh(geom, material). The hand route is the same thing spelled out: BufferGeometry().set_attribute('position', s.positions) then .set_index(s.indices.reshape(-1)). A tp.Mesh built either way is what PhysxWorld.add_static_trimesh takes — or skip both and hand it make_sensor_mesh(s), which is already a Mesh over these triangles.
        """
    @property
    def empty(self) -> bool:
        ...
    @property
    def indices(self) -> numpy.typing.NDArray[numpy.uint32]:
        """
        Triangles as an (M, 3) uint32 array of vertex indices. A copy.
        """
    @property
    def positions(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Vertices as an (N, 3) float32 array, world space. A copy.
        """
    @property
    def stats(self) -> dict:
        """
        What the bake did and what it dropped, as a dict. The ones that diagnose an empty or wrong bake: observed_voxels 0 means no voxel ever met weight_floor (too few poses agreed); refused_blocks > 0 means max_block_bytes bit; beyond_centre_samples a large fraction of depth_samples under PoseSet.Orbit means YOUR SCAN MAY BE AN INTERIOR — re-bake with PoseSet.Interior. render_ms/fuse_ms/mesh_ms split the cost.
        """
    @property
    def triangle_count(self) -> int:
        ...
    @property
    def vertex_count(self) -> int:
        ...
class TerrainFalloff:
    """
    Members:
    
      Off
    
      Radial
    """
    Off: typing.ClassVar[TerrainFalloff]  # value = <TerrainFalloff.Off: 0>
    Radial: typing.ClassVar[TerrainFalloff]  # value = <TerrainFalloff.Radial: 1>
    __members__: typing.ClassVar[dict[str, TerrainFalloff]]  # value = {'Off': <TerrainFalloff.Off: 0>, 'Radial': <TerrainFalloff.Radial: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TerrainGenerator:
    def __init__(self, seed: typing.SupportsInt | typing.SupportsIndex = 1337) -> None:
        ...
    def bake_splat_colors(self, params: TerrainParams) -> numpy.typing.NDArray[numpy.uint8]:
        """
        Bake slope/altitude splat into RGBA8 numpy array of shape (dim, dim, 4).
        """
    def bake_splat_texture(self, params: TerrainParams) -> Texture:
        """
        Bake splat colours into a sRGB DataTexture ready for material.map.
        """
    def build_field(self, params: TerrainParams) -> None:
        ...
    def create_geometry(self, params: TerrainParams, with_erosion: bool = False) -> BufferGeometry:
        ...
    def displace_to(self, geometry: BufferGeometry, params: TerrainParams) -> None:
        ...
    def erode(self, params: TerrainParams) -> None:
        ...
    def get_field(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Height field as float32 numpy array of shape (dim, dim), values in [0,1].
        """
    def height_at(self, wx: typing.SupportsFloat | typing.SupportsIndex, wz: typing.SupportsFloat | typing.SupportsIndex, params: TerrainParams) -> float:
        ...
    def make_geometry(self, params: TerrainParams) -> BufferGeometry:
        ...
    def reseed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def dim(self) -> int:
        ...
    @property
    def seed(self) -> int:
        ...
class TerrainParams:
    erosion: ErosionType
    falloff: TerrainFalloff
    noise_type: NoiseType
    def __init__(self) -> None:
        ...
    @property
    def amplitude(self) -> float:
        ...
    @amplitude.setter
    def amplitude(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ao_max(self) -> float:
        ...
    @ao_max.setter
    def ao_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ao_strength(self) -> float:
        ...
    @ao_strength.setter
    def ao_strength(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def band_edge(self) -> float:
        ...
    @band_edge.setter
    def band_edge(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def deposit_speed(self) -> float:
        ...
    @deposit_speed.setter
    def deposit_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def droplet_lifetime(self) -> int:
        ...
    @droplet_lifetime.setter
    def droplet_lifetime(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def droplets(self) -> int:
        ...
    @droplets.setter
    def droplets(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def erode_speed(self) -> float:
        ...
    @erode_speed.setter
    def erode_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def erosion_radius(self) -> int:
        ...
    @erosion_radius.setter
    def erosion_radius(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def evaporation(self) -> float:
        ...
    @evaporation.setter
    def evaporation(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def falloff_start(self) -> float:
        ...
    @falloff_start.setter
    def falloff_start(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def feature_scale(self) -> float:
        ...
    @feature_scale.setter
    def feature_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gain(self) -> float:
        ...
    @gain.setter
    def gain(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def grass_color(self) -> list[float]:
        ...
    @grass_color.setter
    def grass_color(self, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def gravity(self) -> float:
        ...
    @gravity.setter
    def gravity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def height_exponent(self) -> float:
        ...
    @height_exponent.setter
    def height_exponent(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def inertia(self) -> float:
        ...
    @inertia.setter
    def inertia(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lacunarity(self) -> float:
        ...
    @lacunarity.setter
    def lacunarity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def min_slope(self) -> float:
        ...
    @min_slope.setter
    def min_slope(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def octaves(self) -> int:
        ...
    @octaves.setter
    def octaves(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def resolution(self) -> int:
        ...
    @resolution.setter
    def resolution(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ridge_sharpness(self) -> float:
        ...
    @ridge_sharpness.setter
    def ridge_sharpness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rock_color(self) -> list[float]:
        ...
    @rock_color.setter
    def rock_color(self, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def scree_color(self) -> list[float]:
        ...
    @scree_color.setter
    def scree_color(self, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def sediment_capacity(self) -> float:
        ...
    @sediment_capacity.setter
    def sediment_capacity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def seed(self) -> int:
        ...
    @seed.setter
    def seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def slope_grass_max(self) -> float:
        ...
    @slope_grass_max.setter
    def slope_grass_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def slope_rock_min(self) -> float:
        ...
    @slope_rock_min.setter
    def slope_rock_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def snow_color(self) -> list[float]:
        ...
    @snow_color.setter
    def snow_color(self, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def snow_line(self) -> float:
        ...
    @snow_line.setter
    def snow_line(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def snow_noise_amp(self) -> float:
        ...
    @snow_noise_amp.setter
    def snow_noise_amp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def snow_slope_max(self) -> float:
        ...
    @snow_slope_max.setter
    def snow_slope_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def talus_angle(self) -> float:
        ...
    @talus_angle.setter
    def talus_angle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def terraces(self) -> int:
        ...
    @terraces.setter
    def terraces(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def thermal_iterations(self) -> int:
        ...
    @thermal_iterations.setter
    def thermal_iterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def thermal_rate(self) -> float:
        ...
    @thermal_rate.setter
    def thermal_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def warp(self) -> float:
        ...
    @warp.setter
    def warp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def world_size(self) -> float:
        ...
    @world_size.setter
    def world_size(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Text2D(Mesh):
    def __init__(self, font: Font, text: str = '', size: typing.SupportsFloat | typing.SupportsIndex = 1.0, curve_segments: typing.SupportsInt | typing.SupportsIndex = 3, material: typing.Any = None) -> None:
        ...
    def set_color(self, color: Color) -> None:
        ...
    def set_text(self, text: str) -> None:
        ...
class Text3D(Mesh):
    def __init__(self, font: Font, text: str = '', size: typing.SupportsFloat | typing.SupportsIndex = 1.0, height: typing.SupportsFloat | typing.SupportsIndex = 0.20000000298023224, bevel: bool = False, material: typing.Any = None) -> None:
        ...
    def set_color(self, color: Color) -> None:
        ...
class TextSprite(Sprite):
    def __init__(self, font: Font, world_scale: typing.Any = None) -> None:
        ...
    def get_text(self) -> str:
        ...
    def set_color(self, color: Color) -> None:
        ...
    def set_horizontal_alignment(self, alignment: HorizontalAlignment) -> None:
        ...
    def set_text(self, text: str) -> None:
        ...
    def set_vertical_alignment(self, alignment: VerticalAlignment) -> None:
        ...
    def set_world_scale(self, scale: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Texture:
    center: Vector2
    color_space: ColorSpace
    generate_mipmaps: bool
    mag_filter: Filter
    mapping: Mapping
    min_filter: Filter
    name: str
    offset: Vector2
    repeat: Vector2
    wrap_s: TextureWrapping
    wrap_t: TextureWrapping
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
    def needs_update(self) -> None:
        ...
    def update_data(self, data: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8]) -> None:
        """
        Rewrite a uint8 texture's pixels in place from a (height, width, 3|4) uint8 array of the SAME size, and mark it dirty. For per-frame panels (sensor readouts) without churning texture allocations.
        """
    def update_float(self, data: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """
        Rewrite a float texture's pixels in place from a (height, width, 3|4) float32 array of the SAME size, and mark it dirty. For ping-ponging a procedural sky.
        """
    def update_matrix(self) -> None:
        ...
    @property
    def anisotropy(self) -> int:
        ...
    @anisotropy.setter
    def anisotropy(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def rotation(self) -> float:
        ...
    @rotation.setter
    def rotation(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class TextureLoader:
    def __init__(self, use_cache: bool = True) -> None:
        ...
    def clear_cache(self) -> None:
        ...
    @typing.overload
    def load(self, path: str, flip_y: bool = True) -> Texture:
        ...
    @typing.overload
    def load(self, path: str, color_space: ColorSpace, flip_y: bool = True) -> Texture:
        ...
class TextureWrapping:
    """
    Members:
    
      Repeat
    
      ClampToEdge
    
      MirroredRepeat
    """
    ClampToEdge: typing.ClassVar[TextureWrapping]  # value = <TextureWrapping.ClampToEdge: 1001>
    MirroredRepeat: typing.ClassVar[TextureWrapping]  # value = <TextureWrapping.MirroredRepeat: 1002>
    Repeat: typing.ClassVar[TextureWrapping]  # value = <TextureWrapping.Repeat: 1000>
    __members__: typing.ClassVar[dict[str, TextureWrapping]]  # value = {'Repeat': <TextureWrapping.Repeat: 1000>, 'ClampToEdge': <TextureWrapping.ClampToEdge: 1001>, 'MirroredRepeat': <TextureWrapping.MirroredRepeat: 1002>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ToneMapping:
    """
    Members:
    
      NoToneMapping
    
      Linear
    
      Reinhard
    
      Cineon
    
      ACESFilmic
    
      Neutral
    
      AgX
    """
    ACESFilmic: typing.ClassVar[ToneMapping]  # value = <ToneMapping.ACESFilmic: 4>
    AgX: typing.ClassVar[ToneMapping]  # value = <ToneMapping.AgX: 7>
    Cineon: typing.ClassVar[ToneMapping]  # value = <ToneMapping.Cineon: 3>
    Linear: typing.ClassVar[ToneMapping]  # value = <ToneMapping.Linear: 1>
    Neutral: typing.ClassVar[ToneMapping]  # value = <ToneMapping.Neutral: 6>
    NoToneMapping: typing.ClassVar[ToneMapping]  # value = <ToneMapping.NoToneMapping: 0>
    Reinhard: typing.ClassVar[ToneMapping]  # value = <ToneMapping.Reinhard: 2>
    __members__: typing.ClassVar[dict[str, ToneMapping]]  # value = {'NoToneMapping': <ToneMapping.NoToneMapping: 0>, 'Linear': <ToneMapping.Linear: 1>, 'Reinhard': <ToneMapping.Reinhard: 2>, 'Cineon': <ToneMapping.Cineon: 3>, 'ACESFilmic': <ToneMapping.ACESFilmic: 4>, 'Neutral': <ToneMapping.Neutral: 6>, 'AgX': <ToneMapping.AgX: 7>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TorusGeometry(BufferGeometry):
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex = 1.0, tube: typing.SupportsFloat | typing.SupportsIndex = 0.4000000059604645, radial_segments: typing.SupportsInt | typing.SupportsIndex = 20, tubular_segments: typing.SupportsInt | typing.SupportsIndex = 64, arc: typing.SupportsFloat | typing.SupportsIndex = 6.2831854820251465) -> None:
        ...
class TorusKnotGeometry(BufferGeometry):
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex = 1.0, tube: typing.SupportsFloat | typing.SupportsIndex = 0.4000000059604645, tubular_segments: typing.SupportsInt | typing.SupportsIndex = 64, radial_segments: typing.SupportsInt | typing.SupportsIndex = 16, p: typing.SupportsInt | typing.SupportsIndex = 2, q: typing.SupportsInt | typing.SupportsIndex = 3) -> None:
        ...
class TransformControls(Object3D):
    enabled: bool
    show_x: bool
    show_y: bool
    show_z: bool
    def __init__(self, camera: Camera, canvas: Canvas) -> None:
        ...
    def attach(self, object: typing.Any) -> TransformControls:
        """
        Attach the gizmo to an Object3D. Add the TransformControls itself to the scene.
        """
    def detach(self) -> TransformControls:
        ...
    def get_space(self) -> str:
        ...
    def set_mode(self, mode: str) -> None:
        """
        Mode: 'translate' | 'rotate' | 'scale'
        """
    def set_rotation_snap(self, snap: typing.SupportsFloat | typing.SupportsIndex | None) -> None:
        ...
    def set_scale_snap(self, snap: typing.SupportsFloat | typing.SupportsIndex | None) -> None:
        ...
    def set_size(self, size: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_space(self, space: str) -> None:
        """
        Space: 'world' | 'local'
        """
    def set_translation_snap(self, snap: typing.SupportsFloat | typing.SupportsIndex | None) -> None:
        ...
    @property
    def dragging(self) -> bool:
        """
        True while the user is actively dragging the gizmo.
        """
class TreeGenerator:
    def __init__(self, seed: typing.SupportsInt | typing.SupportsIndex = 1337) -> None:
        ...
    def build_skeleton(self, params: TreeParams) -> None:
        ...
    def create_leaf_geometry(self, params: TreeParams) -> BufferGeometry:
        ...
    def create_trunk_geometry(self, params: TreeParams) -> BufferGeometry:
        ...
    def make_leaf_geometry(self, params: TreeParams) -> BufferGeometry:
        ...
    def make_trunk_geometry(self, params: TreeParams) -> BufferGeometry:
        ...
    def reseed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def node_count(self) -> int:
        ...
    @property
    def seed(self) -> int:
        ...
class TreeParams:
    bark_style: BarkStyle
    branching_mode: BranchingMode
    crown_shape: CrownShape
    leaf_shape: LeafShape
    leaf_style: LeafStyle
    def __init__(self) -> None:
        ...
    @property
    def attractor_count(self) -> int:
        ...
    @attractor_count.setter
    def attractor_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def bark_bump_amp(self) -> float:
        ...
    @bark_bump_amp.setter
    def bark_bump_amp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def bark_bump_amp2(self) -> float:
        ...
    @bark_bump_amp2.setter
    def bark_bump_amp2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def bark_bump_lobes(self) -> int:
        ...
    @bark_bump_lobes.setter
    def bark_bump_lobes(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def bark_bump_lobes2(self) -> int:
        ...
    @bark_bump_lobes2.setter
    def bark_bump_lobes2(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def bark_color(self) -> list[float]:
        ...
    @bark_color.setter
    def bark_color(self, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def branch_droop(self) -> float:
        ...
    @branch_droop.setter
    def branch_droop(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def branch_length(self) -> float:
        ...
    @branch_length.setter
    def branch_length(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def branch_tip_upturn(self) -> float:
        ...
    @branch_tip_upturn.setter
    def branch_tip_upturn(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def branches_per_whorl(self) -> int:
        ...
    @branches_per_whorl.setter
    def branches_per_whorl(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def crown_height(self) -> float:
        ...
    @crown_height.setter
    def crown_height(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def crown_profile_exponent(self) -> float:
        ...
    @crown_profile_exponent.setter
    def crown_profile_exponent(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def crown_radius_x(self) -> float:
        ...
    @crown_radius_x.setter
    def crown_radius_x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def crown_radius_z(self) -> float:
        ...
    @crown_radius_z.setter
    def crown_radius_z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def foliage_occlusion(self) -> float:
        ...
    @foliage_occlusion.setter
    def foliage_occlusion(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def influence_distance(self) -> float:
        ...
    @influence_distance.setter
    def influence_distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def kill_distance(self) -> float:
        ...
    @kill_distance.setter
    def kill_distance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def leaf_atlas_cells(self) -> int:
        ...
    @leaf_atlas_cells.setter
    def leaf_atlas_cells(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def leaf_clumping(self) -> float:
        ...
    @leaf_clumping.setter
    def leaf_clumping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def leaf_color(self) -> list[float]:
        ...
    @leaf_color.setter
    def leaf_color(self, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def leaf_density(self) -> float:
        ...
    @leaf_density.setter
    def leaf_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def leaf_droop(self) -> float:
        ...
    @leaf_droop.setter
    def leaf_droop(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def leaf_size(self) -> float:
        ...
    @leaf_size.setter
    def leaf_size(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def leaf_spread(self) -> float:
        ...
    @leaf_spread.setter
    def leaf_spread(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def leaves_per_cluster(self) -> int:
        ...
    @leaves_per_cluster.setter
    def leaves_per_cluster(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_iterations(self) -> int:
        ...
    @max_iterations.setter
    def max_iterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def min_branch_radius(self) -> float:
        ...
    @min_branch_radius.setter
    def min_branch_radius(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def pendant_density(self) -> float:
        ...
    @pendant_density.setter
    def pendant_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def pendant_length(self) -> float:
        ...
    @pendant_length.setter
    def pendant_length(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def radial_segments(self) -> int:
        ...
    @radial_segments.setter
    def radial_segments(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def radius_exponent(self) -> float:
        ...
    @radius_exponent.setter
    def radius_exponent(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def randomness(self) -> float:
        ...
    @randomness.setter
    def randomness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def root_flare_asym(self) -> float:
        ...
    @root_flare_asym.setter
    def root_flare_asym(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def seed(self) -> int:
        ...
    @seed.setter
    def seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def segment_length(self) -> float:
        ...
    @segment_length.setter
    def segment_length(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def side_twig_density(self) -> float:
        ...
    @side_twig_density.setter
    def side_twig_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def tropism(self) -> float:
        ...
    @tropism.setter
    def tropism(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def trunk_bend(self) -> float:
        ...
    @trunk_bend.setter
    def trunk_bend(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def trunk_height(self) -> float:
        ...
    @trunk_height.setter
    def trunk_height(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def trunk_lean(self) -> float:
        ...
    @trunk_lean.setter
    def trunk_lean(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def trunk_radius(self) -> float:
        ...
    @trunk_radius.setter
    def trunk_radius(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def trunk_twist(self) -> float:
        ...
    @trunk_twist.setter
    def trunk_twist(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def twig_shade(self) -> float:
        ...
    @twig_shade.setter
    def twig_shade(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def whorl_jitter(self) -> float:
        ...
    @whorl_jitter.setter
    def whorl_jitter(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def whorl_spacing(self) -> float:
        ...
    @whorl_spacing.setter
    def whorl_spacing(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class TubeGeometry(BufferGeometry):
    def __init__(self, path: Curve3, tubular_segments: typing.SupportsInt | typing.SupportsIndex = 64, radius: typing.SupportsFloat | typing.SupportsIndex = 1.0, radial_segments: typing.SupportsInt | typing.SupportsIndex = 8, closed: bool = False) -> None:
        ...
    def get_path(self) -> Curve3:
        ...
    @property
    def radius(self) -> float:
        ...
class URDFLoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str) -> Robot:
        """
        Load a .urdf/.xacro file into a Robot (meshes via ModelLoader).
        
        Raises RuntimeError carrying the parser's own explanation - the same text `last_error` holds, which for a xacro failure includes the file and LINE. Read `diagnostics` afterwards for the warnings too, which a load that SUCCEEDS can also produce.
        """
    def parse(self, base_dir: str, xml: str) -> Robot:
        """
        Parse URDF XML from a string; base_dir resolves relative mesh paths.
        
        Raises RuntimeError carrying the parser's own explanation, exactly as `load` does. This is the call a ROS node makes on /robot_description, where the XML came off a topic and there is no file to go and look at.
        """
    def set_args(self, args: collections.abc.Mapping[str, str]) -> None:
        """
        xacro arg overrides (equivalent to name:=value on the xacro CLI).
        """
    @property
    def diagnostics(self) -> list[str]:
        """
        Everything the most recent call had to say: warnings first, then errors, each group in the order it was produced.
        
        Not the same as `last_error`, and the difference is the point: the warnings come from the XACRO expansion - a redefined macro, an undeclared attribute being ignored, a name resolved as an arg rather than a property - and a document that produces them still loads. They only ever went to stderr, where a script cannot see them and a GUI has nowhere to show them.
        
        Cleared at the start of every call, so an empty list after a success means there was genuinely nothing to say.
        """
    @property
    def last_error(self) -> str:
        """
        Why the most recent load/parse/parse_articulation failed: the errors it produced, joined into one message. Empty after a call that succeeded, and never empty after one that did not.
        
        The raise already carries this. It is here for the caller that wants to report rather than propagate - a ROS node logging a bad /robot_description and carrying on, say.
        """
class Vector2:
    __hash__: typing.ClassVar[None] = None
    def __add__(self, arg0: Vector2) -> Vector2:
        ...
    def __eq__(self, arg0: Vector2) -> bool:
        ...
    def __init__(self, x: typing.SupportsFloat | typing.SupportsIndex = 0.0, y: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    def __mul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Vector2:
        ...
    def __repr__(self) -> str:
        ...
    def __sub__(self, arg0: Vector2) -> Vector2:
        ...
    def add(self, v: Vector2) -> Vector2:
        ...
    def clone(self) -> Vector2:
        ...
    def copy(self, v: Vector2) -> Vector2:
        ...
    def distance_to(self, v: Vector2) -> float:
        ...
    def dot(self, v: Vector2) -> float:
        ...
    def length(self) -> float:
        ...
    def multiply_scalar(self, s: typing.SupportsFloat | typing.SupportsIndex) -> Vector2:
        ...
    def normalize(self) -> Vector2:
        ...
    def set(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex) -> Vector2:
        ...
    def sub(self, v: Vector2) -> Vector2:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Vector3:
    __hash__: typing.ClassVar[None] = None
    def __add__(self, arg0: Vector3) -> Vector3:
        ...
    def __eq__(self, arg0: Vector3) -> bool:
        ...
    def __init__(self, x: typing.SupportsFloat | typing.SupportsIndex = 0.0, y: typing.SupportsFloat | typing.SupportsIndex = 0.0, z: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    def __mul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Vector3:
        ...
    def __repr__(self) -> str:
        ...
    def __sub__(self, arg0: Vector3) -> Vector3:
        ...
    def add(self, v: Vector3) -> Vector3:
        ...
    def add_scaled_vector(self, v: Vector3, s: typing.SupportsFloat | typing.SupportsIndex) -> Vector3:
        ...
    def apply_matrix4(self, m: ...) -> Vector3:
        ...
    def apply_quaternion(self, q: ...) -> Vector3:
        ...
    def clone(self) -> Vector3:
        ...
    def copy(self, v: Vector3) -> Vector3:
        ...
    def cross(self, v: Vector3) -> Vector3:
        ...
    def distance_to(self, v: Vector3) -> float:
        ...
    def dot(self, v: Vector3) -> float:
        ...
    def length(self) -> float:
        ...
    def length_sq(self) -> float:
        ...
    def lerp(self, v: Vector3, alpha: typing.SupportsFloat | typing.SupportsIndex) -> Vector3:
        ...
    def multiply_scalar(self, s: typing.SupportsFloat | typing.SupportsIndex) -> Vector3:
        ...
    def negate(self) -> Vector3:
        ...
    def normalize(self) -> Vector3:
        ...
    def project(self, camera: ...) -> Vector3:
        ...
    def set(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> Vector3:
        ...
    def sub(self, v: Vector3) -> Vector3:
        ...
    def unproject(self, camera: ...) -> Vector3:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def z(self) -> float:
        ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Vector4:
    def __init__(self, x: typing.SupportsFloat | typing.SupportsIndex = 0.0, y: typing.SupportsFloat | typing.SupportsIndex = 0.0, z: typing.SupportsFloat | typing.SupportsIndex = 0.0, w: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def set(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex, w: typing.SupportsFloat | typing.SupportsIndex) -> Vector4:
        ...
    @property
    def w(self) -> float:
        ...
    @w.setter
    def w(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def z(self) -> float:
        ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class VectorKeyframeTrack(KeyframeTrack):
    def __init__(self, name: str, times: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], values: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], interpolation: Interpolation | None = None) -> None:
        ...
class VerticalAlignment:
    """
    Members:
    
      Above
    
      Center
    
      Below
    """
    Above: typing.ClassVar[VerticalAlignment]  # value = <VerticalAlignment.Above: 0>
    Below: typing.ClassVar[VerticalAlignment]  # value = <VerticalAlignment.Below: 2>
    Center: typing.ClassVar[VerticalAlignment]  # value = <VerticalAlignment.Center: 1>
    __members__: typing.ClassVar[dict[str, VerticalAlignment]]  # value = {'Above': <VerticalAlignment.Above: 0>, 'Center': <VerticalAlignment.Center: 1>, 'Below': <VerticalAlignment.Below: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class VoxelGrid:
    """
    Voxel-hash spatial index. O(1) insert, nearest-neighbour queries over the 27 surrounding voxels. Use voxelSize >= search radius for exact results.
    """
    def __init__(self, voxel_size: typing.SupportsFloat | typing.SupportsIndex, max_points_per_voxel: typing.SupportsInt | typing.SupportsIndex = 20, min_spacing: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    def clear(self) -> None:
        ...
    def collect(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Return all stored points as (N,3) float32 numpy array.
        """
    def collect_voxel_centers(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Return one centre point per occupied voxel as (N,3) float32 numpy array.
        """
    def insert(self, point: Vector3) -> bool:
        """
        Insert a single Vector3. Returns True if stored (passed cap + spacing).
        """
    def insert_array(self, points: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> int:
        """
        Insert an (N,3) float32 array. Returns number of points actually stored.
        """
    def nearest(self, query: Vector3, max_dist: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        """
        Nearest stored point within max_dist. Returns Vector3 or None.
        """
    @property
    def empty(self) -> bool:
        ...
    @property
    def size(self) -> int:
        ...
    @property
    def voxel_count(self) -> int:
        ...
    @property
    def voxel_size(self) -> float:
        ...
class VulkanRenderer:
    shadow_map_enabled: bool
    tone_mapping: ToneMapping
    volumetric_fog: bool
    def __init__(self, canvas: Canvas, flush_frames: typing.SupportsInt | typing.SupportsIndex = 3) -> None:
        """
        Deferred (RasterFirst) Vulkan renderer. Pass a headless Canvas created with vsync=False.
        """
    def add_view(self, camera: Camera, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        Attach a persistent extra view. Every render() then produces the primary AND every added view from one scene build, in a single queue submission — N viewpoints of the SAME simulated instant, which N render() calls cannot give. Each view has its own G-buffer, temporal history and camera state; acceleration structures, lights, materials and probe GI are shared.
        
        Views are PERSISTENT: this call drains the device and allocates a full deferred chain, while rendering an existing view every frame is cheap. Do NOT add and remove per frame.
        
        Secondary views are deliberately plainer than the primary — native resolution with the built-in temporal resolve, no DLSS/FSR, no occlusion culling, no UI overlay, no depth of field, no lens or sensor model. They are measurement cameras, not the display.
        
        Returns a handle (> 0), or 0 if the view could not be created — notably when render() has not run yet, since a view shares the primary's render pass and pipelines.
        """
    def disable_clouds(self) -> None:
        """
        Turn the volumetric cloud layer off (default).
        """
    def disable_frame_interop(self, view: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        Release this view's frame-interop exports. Close the importing tensors FIRST (FrameTensors.close() does both in the right order) — freeing the Vulkan allocation under a live CUDA mapping reports as nothing at all.
        """
    def disable_height_fog(self) -> None:
        """
        Turn near-field height fog off (default).
        """
    def disable_vertex_interop(self, mesh: Mesh) -> None:
        """
        Release the exports and return the mesh to the CPU attribute path. STOP the foreign writes first — nothing here can wait on a CUDA stream. Close the importing VkInteropArrays before calling this.
        """
    def enable_frame_interop(self, view: typing.SupportsInt | typing.SupportsIndex = 0, channels: collections.abc.Iterable = ('color', 'depth')) -> list:
        """
        Export this view's per-frame images as CUDA-importable buffers and arm the device-to-device copies that fill them.
        
        channels: any of 'color', 'depth', 'normal', 'motion', 'ids', 'albedo', 'splat_depth' (or FrameChannel values). view=0 is the primary; anything else must be a live add_view handle.
        
        Returns a list of dicts, one per EXPORTABLE channel — duplicates collapse and unexportable ones are skipped ('splat_depth' while the AOV is unallocated — neither set_splat_depth_aov nor the overlay-occlusion latch), so match on the 'channel' key. Each dict carries handle, size_bytes, width, height, bytes_per_pixel and (for 'color') bgra. An EMPTY list means nothing could be exported: before the first render(), on a stale view handle, or on a device with no external-memory extension (one line on stderr) — the fallback is the read_* host readback path.
        
        CALL IT AFTER THE FIRST render(): the exports are sized from the G-buffer / swapchain extents, which exist only once a frame has run.
        
        SYNC: render() -> sync_frame_interop() -> read the tensors -> next render(). Host ordering is the only cross-API synchronization; there is no shared semaphore.
        
        SINGLE-BUFFERED: the tensors are live views of renderer memory that the next render() overwrites. Clone what you need to keep.
        
        INVALIDATION: a resize, render_scale, gbuffer MSAA or splat_depth_aov toggle, or removing the view DISABLES interop for that view (one warning on stderr) rather than reallocating under a live import — re-enable and re-import. The handles are OS handles owned by the RENDERER: import them, never CloseHandle them from Python.
        """
    def enable_particle_field_interop(self, field: ParticleField, device_copy: collections.abc.Callable[[], None]) -> typing.Any:
        """
        Export an Ownership.Interop ParticleField's positions allocation and arm the per-frame device-to-device copy that fills it.
        
        Returns (os_handle, size_bytes) — or, when the field was created with Config.attributes, (os_handle, size_bytes, attr_handle, attr_size_bytes): the second allocation is the per-particle vec4 appearance buffer (rgb = linear HDR radiance), imported and written exactly like the positions and snapshotted by the same per-frame copy. Or None when the device has no external-memory extension — in which case the field is left in host_fallback() and submit() is legal on it, so the caller drops to the HostRing path rather than failing.
        
        CALL IT AFTER THE FIRST render(): the field's device state and this renderer's field pass are both created on the frame the field is first seen, so this returns None until then — render once, then enable.
        
        device_copy() runs INSIDE render(), once per frame, pre-record, and MUST BE SYNCHRONOUS (wp.copy(...) then wp.synchronize_device(device) as the last statement, or cuMemcpyDtoDAsync + cuStreamSynchronize). That host ordering is the only thing sequencing the foreign write against the frame that reads it — there is no shared semaphore.
        
        The handle is an OS handle owned by the RENDERER (a Win32 NT handle on Windows): import it, but never CloseHandle it from Python. The layout is ParticlePos — 16-byte xyzw slots, byte-identical to wp.vec4 and to PxVec4 — and w < 0 is the DEAD sentinel every consumer tests. size_bytes is the ALLOCATION size and may exceed capacity*16.
        
        Liveness is the sim's job here: set_live_count(capacity) once and let the kernel write w < 0 for dead slots. An Interop field forfeits reproducibility and every emitter-derived feature (age fade, size taper, colour ramp, surface landing) — it is positions, a radius and an orientation set, and that is the whole model.
        """
    def enable_vertex_interop(self, mesh: Mesh, on_frame: collections.abc.Callable[[], None], validate: bool = True, stable_correspondence: bool = True) -> typing.Any:
        """
        Export mesh.geometry's position + normal buffers for a foreign GPU producer and arm the per-frame device write that fills them.
        
        Returns ((pos_handle, pos_bytes), (nrm_handle, nrm_bytes)) or None.
        
        POLL IT: the renderer's record for a mesh is created on the frame the mesh is first drawn, so this returns None until after the first render() — call render() once, then enable.
        
        FIXED-CAPACITY ALLOCATION, VARIABLE DRAW: a producer whose triangle count varies allocates its maximum once and publishes the live count with set_draw_range — the raster path clamps to the range, the BLAS is built over [0, start + count), and the interop copies are trimmed to match, so no degenerate tail is needed. Changing an attribute's count after enabling DISABLES interop for that mesh (with a warning on stderr) rather than tearing down memory CUDA has imported.
        
        on_frame() runs INSIDE render(), once per frame while the mesh is visible, post-fence and pre-record, and MUST BE SYNCHRONOUS: every kernel writing the exported buffers has to have completed when it returns (wp.synchronize_device(device) as the last statement). That host ordering is the only thing sequencing the foreign write against the frame that reads it — there is no shared semaphore.
        
        The handles are OS handles owned by the RENDERER (Win32 NT handles on Windows): import them, but never CloseHandle them from Python — disable_vertex_interop / renderer teardown releases them. The layout is tightly-packed float xyz (12-byte stride, wp.vec3), and *_bytes is the ALLOCATION size, which may exceed count*12 — write only the real range.
        
        validate=True (default) runs a GPU finiteness pass over the exported positions each frame, rewriting non-finite vertices as degenerates. Leave it on unless the producer is trusted: a NaN reaching the BLAS build is a device-lost (GPU reset) on NVIDIA, not an error return.
        
        stable_correspondence=True (default) means vertex i is the SAME surface point every frame (a deforming fixed-topology mesh — cloth, a soft body); per-vertex motion vectors then come from the previous frame's positions. Pass False for a producer that RE-TRIANGULATES each frame (a marching-cubes soup: one changed cell shifts every later vertex slot) — that history is noise there, so the mesh reprojects as world-static and the temporal passes (TAA/upscaler, reflection denoiser) stop flickering on the regions that changed.
        """
    def frame_interop_active(self, view: typing.SupportsInt | typing.SupportsIndex = 0) -> bool:
        """
        Is this view's frame interop still armed? False after an invalidation (a resize, render_scale, ...) tore the exports down — the one way to notice without reading stderr. FrameTensors.stale is this, negated.
        """
    def hide_view(self, handle: typing.SupportsInt | typing.SupportsIndex) -> bool:
        """
        Back to a measurement camera: still rendered, still readable, no longer drawn into the frame.
        """
    def read_albedo(self, scene: Object3D, camera: Camera) -> numpy.typing.NDArray[numpy.uint8]:
        ...
    def read_aovs_typed(self, scene: Object3D, camera: Camera, aovs: collections.abc.Sequence[str] = ['rgb', 'depth', 'normals', 'instance_ids']) -> dict:
        """
        Render ONCE and read every requested AOV from that same frame, each as its natural dtype: depth (H,W) f32 · normals (H,W,3) f32 · instance_ids (H,W) u32 · motion (H,W,2) f32 · rgb (H,W,3) u8 · albedo (H,W,4) u8 (linear rgb + metalness). The efficient multi-AOV entry point.
        """
    def read_class_ids(self, scene: Object3D, camera: Camera) -> numpy.typing.NDArray[numpy.uint32]:
        """
        Semantic class ids as (H, W) uint32 (0..255; 0 = unset). Tag objects with set_class_id() to get semantic segmentation alongside the instance ids.
        """
    def read_depth(self, scene: Object3D, camera: Camera) -> numpy.typing.NDArray[numpy.float32]:
        """
        Metric depth as (H, W) float32 — distance from the camera in scene units. Background reads as the camera far plane. Full 32-bit precision (native D32 read; supersedes the old 24-bit-packed path).
        """
    def read_event_camera_visualisation(self) -> numpy.typing.NDArray[numpy.uint8]:
        """
        The detector's accumulator image as (H, W) uint8 at the SENSOR
        resolution, row 0 = top: 255 = positive (brightening) event,
        0 = negative, 128 = no event, decaying back toward 128 at the
        `decay` rate. Empty array while the event camera is off. Two
        renderer frames of latency (it reads the oldest ring slot, which
        the in-flight fences guarantee complete — no device wait).
        """
    def read_event_stream(self, max_events: typing.SupportsInt | typing.SupportsIndex = 1000000) -> tuple[numpy.typing.NDArray[numpy.int64], bool]:
        """
        The sparse event stream of the last completed detector frame as
        ((N, 4) int64, overflowed): columns are x, y (image convention,
        row 0 = top), polarity (+1 brightening / -1 darkening) and t_us.
        `overflowed` is True when the GPU append list saturated and events
        were dropped — the same failure mode a real sensor's readout has.
        
        t_us is sub-frame interpolated: log intensity is taken to ramp
        linearly between two consecutive detector frames and each threshold
        crossing is stamped where it crosses, so one frame's events span
        (previous frame_time_us, this one] rather than sharing a value.
        Rows come back sorted ascending by (t_us, y, x, polarity) — time
        order like a real readout, and deterministic despite the GPU's
        atomic append order being scheduler-dependent.
        """
    def read_gbuffer_aov_raw(self, aov: str, view: typing.SupportsInt | typing.SupportsIndex = 0) -> typing.Any:
        """
        The raw bytes of one G-buffer attachment of the last rendered frame as (H, W, bytes_per_pixel) uint8, or None. No decode: this is the host readback the zero-copy frame-interop path is checked byte-for-byte against. Applications want read_depth / read_instance_ids / read_aovs_typed instead.
        """
    def read_instance_ids(self, scene: Object3D, camera: Camera) -> numpy.typing.NDArray[numpy.uint32]:
        """
        Stable per-pixel instance ids as (H, W) uint32. 0 = sky / no hit; otherwise a per-object id that persists across frames and visible-set changes (no hashing, no collisions). Auto-assigned; override with set_instance_id().
        """
    def read_motion(self, scene: Object3D, camera: Camera) -> numpy.typing.NDArray[numpy.float32]:
        """
        Screen-space motion vectors as (H, W, 2) float32, in pixels (previous - current surface position; +x rightward, y down-positive).
        """
    def read_normals(self, scene: Object3D, camera: Camera) -> numpy.typing.NDArray[numpy.uint8]:
        ...
    def read_normals_float(self, scene: Object3D, camera: Camera) -> numpy.typing.NDArray[numpy.float32]:
        """
        World-space unit normals as (H, W, 3) float32, components in [-1, 1] (full precision; read_normals() stays the 8-bit visualisation).
        """
    def read_pixels(self) -> numpy.typing.NDArray[numpy.uint8]:
        """
        Final shaded RGB of the last render as (H, W, 3) uint8.
        """
    def read_scene_pixels(self) -> numpy.typing.NDArray[numpy.uint8]:
        """
        Last captured scene-only RGB (post-TAA, pre-overlay; no sprite/ImGui) as (H, W, 3) uint8. Requires scene_capture=True.
        """
    def read_segmentation(self, scene: Object3D, camera: Camera) -> numpy.typing.NDArray[numpy.uint8]:
        ...
    def read_view_rgb_pixels(self, handle: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.uint8]:
        """
        This view's most recent frame as (H, W, 3) uint8, TOP-LEFT origin — the same convention as read_pixels. Reads the view's own colour image, never the swapchain. An unknown handle gives an empty (0, 0, 3) array.
        """
    def remove_view(self, handle: typing.SupportsInt | typing.SupportsIndex) -> bool:
        """
        Destroy the view and free everything it owns. False for an unknown handle. Handles are never reused, so a stale one is inert rather than dangerous.
        """
    def render(self, scene: Object3D, camera: Camera) -> None:
        ...
    def render_aov(self, scene: Object3D, camera: Camera, aov: str) -> numpy.typing.NDArray[numpy.uint8]:
        """
        Render and return a G-buffer AOV as (H, W, 3) uint8: 'rgb' | 'normals' | 'segmentation' | 'albedo' | 'motion'.
        """
    def render_aovs(self, scene: Object3D, camera: Camera, aovs: collections.abc.Sequence[str] = ['rgb', 'normals', 'segmentation']) -> dict:
        """
        Render the requested AOVs and return {name: (H, W, 3) uint8}.
        """
    def reset_sensor_noise(self) -> None:
        """
        Restart the noise sequence -- call on episode reset so two episodes with the same seed produce the same frames.
        """
    def save_frame(self, scene: Object3D, camera: Camera, path: str) -> None:
        ...
    def scan_lidar(self, origins: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], directions: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], params: LidarParams = ...) -> dict:
        """
        Trace an arbitrary beam table in ONE dispatch: origins (N,3) + unit directions (N,3) -> the same dict of numpy arrays as PathTracedLidarSensor.scan(), row i belonging to beam i (x samples_per_beam x max_returns when those are raised; return_no > 0 is the real-return predicate). render() the scene once first. Use this when the beams do not follow a single pose - e.g. scoring an object from a ring of viewpoints in one round trip.
        """
    def set_auto_exposure_range(self, min_ev: typing.SupportsFloat | typing.SupportsIndex, max_ev: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        EV clamp for auto-exposure relative to linear 1.0 (default -3 to +3).
        """
    def set_auto_exposure_speed(self, ev_per_second: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Adaptation speed in EV/s (default 2.0). Darkening is applied at 0.5× speed.
        """
    def set_camera_exposure(self, aperture: typing.SupportsFloat | typing.SupportsIndex = 16.0, shutter: typing.SupportsFloat | typing.SupportsIndex = 0.00800000037997961, iso: typing.SupportsFloat | typing.SupportsIndex = 100.0) -> None:
        """
        Camera exposure triplet (used while physical_camera is on): f-number, shutter seconds, ISO.
        """
    def set_class_id(self, object: Object3D, class_id: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Tag an object with a semantic class id (0..255) for read_class_ids(). Objects sharing a class id share a semantic label.
        """
    def set_clear_color(self, color: Color, alpha: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    def set_clouds(self, coverage: typing.SupportsFloat | typing.SupportsIndex = 0.44999998807907104, density: typing.SupportsFloat | typing.SupportsIndex = 1.0, bottom_y: typing.SupportsFloat | typing.SupportsIndex = 600.0, top_y: typing.SupportsFloat | typing.SupportsIndex = 1400.0, wind: Vector3 = ..., evolve_speed: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        """
        Enable the volumetric cloud layer. coverage 0..1 (0=clear, 1=overcast); density multiplier; bottom_y/top_y world-Y shell; wind m/s xz drift; evolve_speed shape churn rate.
        """
    def set_color_grade(self, lift: Vector3 = ..., gamma: Vector3 = ..., gain: Vector3 = ..., saturation: typing.SupportsFloat | typing.SupportsIndex = 1.0, contrast: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        """
        Lift/gamma/gain wheels + saturation + contrast, baked into a 33^3 LUT applied after the tone map. Defaults = identity = off.
        """
    def set_event_camera_params(self, threshold: typing.SupportsFloat | typing.SupportsIndex = 0.15000000596046448, decay: typing.SupportsFloat | typing.SupportsIndex = 0.8500000238418579, min_luma: typing.SupportsFloat | typing.SupportsIndex = 0.004999999888241291, max_events_per_pixel: typing.SupportsInt | typing.SupportsIndex = 5, frame_time_us: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        Contrast threshold in log-intensity units (0.15 fires on almost any
        edge, 0.30 only on hard ones), the visualisation's per-frame decay
        toward mid-grey, the luma floor that stops log() exploding in the
        shadows, the per-pixel event cap, and the microsecond clock for THIS
        frame's sample.
        
        Drive `frame_time_us` EVERY frame from a monotone sim clock (a
        wall-clock stamp is not reproducible). The detector remembers the
        previous value and interpolates each event's timestamp linearly
        between the two — the ESIM model — so a frame's events spread across
        (previous, this] instead of sharing one stamp. Left at its default the
        interval is empty and every event carries 0, an obviously-unstamped
        stream rather than a plausible wrong one. The call only stores a
        struct, so per-frame is cheap.
        """
    def set_event_camera_resolution(self, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Pin the sensor's native resolution (0, 0 tracks the swapchain).
        Clamped to [16, swapchain]. A real DVS is coarse — 640x480 is
        Prophesee Gen3/4 territory — and a coarser detector is also less
        readback and less compute.
        """
    def set_flush_frames(self, n: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Frames driven per render() to flush the MAILBOX swapchain (default 3; raise to 4+ for fast-moving dynamic scenes).
        """
    def set_fog_water_surface_y(self, y: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        World-Y of the water surface: murk applies below it, the air medium above it. 1e30 (default) = no waterline.
        """
    def set_height_fog(self, density: typing.SupportsFloat | typing.SupportsIndex = 0.019999999552965164, base_y: typing.SupportsFloat | typing.SupportsIndex = 0.0, falloff: typing.SupportsFloat | typing.SupportsIndex = 80.0, noise_amount: typing.SupportsFloat | typing.SupportsIndex = 0.6000000238418579) -> None:
        """
        Enable near-field heterogeneous height fog. density = sigma_t at base_y; base_y world-Y; falloff exponential height scale (m); noise_amount 0=smooth..1=fully noise-modulated.
        """
    def set_instance_id(self, object: Object3D, instance_id: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Assign a specific stable instance id (0..65535) to an object for the ids AOV. Overrides the auto-assigned id. Takes effect on the next render.
        """
    def set_lens_distortion(self, model: str = 'none', k1: typing.SupportsFloat | typing.SupportsIndex = 0.0, k2: typing.SupportsFloat | typing.SupportsIndex = 0.0, k3: typing.SupportsFloat | typing.SupportsIndex = 0.0, k4: typing.SupportsFloat | typing.SupportsIndex = 0.0, p1: typing.SupportsFloat | typing.SupportsIndex = 0.0, p2: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        """
        OpenCV-compatible lens distortion. 'brown_conrady' takes cv2's (k1, k2, p1, p2, k3); 'fisheye' takes cv2.fisheye's (k1..k4). Applied to BOTH the displayed image and the AOV readback, so segmentation/depth labels stay aligned with the distorted pixels. Default 'none' (pinhole, zero cost).
        """
    def set_scissor(self, x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_scissor_test(self, enabled: bool) -> None:
        ...
    def set_sensor_noise(self, enabled: bool = True, full_well: typing.SupportsFloat | typing.SupportsIndex = 20000.0, read_noise: typing.SupportsFloat | typing.SupportsIndex = 3.0, dark_current: typing.SupportsFloat | typing.SupportsIndex = 5.0, prnu_percent: typing.SupportsFloat | typing.SupportsIndex = 0.5, seed: typing.SupportsInt | typing.SupportsIndex = 1) -> None:
        """
        Shot/read/dark-current/PRNU sensor noise, in ELECTRONS, applied after the temporal resolve (TAA would otherwise average it away). Noise scales with the ISO from set_camera_exposure, as on a real sensor. Deterministic: the same seed replays the same frames. Default off.
        """
    def set_sensor_only_surfaces(self, enabled: bool) -> None:
        """
        Scene master switch for sensor-only surfaces (meshes from splats.make_sensor_mesh). OFF, the default, nothing at all sees them. ON, lidar beams hit them and the secondary views that ALSO ask (set_view_sensor_surfaces) rasterize them. The primary view never draws them either way, and no radiance trace — reflection, shadow, GI — has them in its cull mask: the real splats are what the picture shows.
        """
    def set_size(self, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Resize the renderer's framebuffer/swapchain — call this from canvas.on_window_resize together with updating the camera aspect.
        """
    def set_splat_depth_aov(self, mode: str) -> None:
        """
        Ask for the Gaussian-splat depth AOV: 'off', 'expected' (the transmittance-weighted mean view distance) or 'median' (the front of the cloud — what an occlusion test wants). A SETUP knob: turning it on or off reallocates the render targets, so call it once before the loop, not per frame. Changing only the statistic reallocates nothing. Primary view only.
        """
    def set_underwater_murk(self, density: typing.SupportsFloat | typing.SupportsIndex, color: Color = ...) -> None:
        """
        Enable underwater murk (below fog_water_surface_y). density = sigma_t (1/m; 0 disables); color = inscatter tint.
        """
    def set_view_camera(self, handle: typing.SupportsInt | typing.SupportsIndex, camera: Camera) -> bool:
        """
        Repoint a view at a different camera. Treated as a CUT: the view's temporal history is dropped rather than reprojected across a discontinuity that never happened in world space.
        """
    def set_view_display_rect(self, handle: typing.SupportsInt | typing.SupportsIndex, x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> bool:
        """
        Picture-in-picture: show this view inside the primary's frame with its top-left corner at (x, y) in window pixels. The image is already resolved, on the device and in the swapchain's format, so this is a single image copy in the frame's own command buffer — no readback, no upload, no texture, no second submission.
        
        1:1 ONLY: width/height must equal the size the view was added at, and a mismatch draws NOTHING rather than a filtered rescale. A rect running off the window edge is clipped. Composited after the scene capture and before the UI overlay, so ImGui and screen-space sprites still draw on top.
        """
    def set_view_sensor_surfaces(self, handle: typing.SupportsInt | typing.SupportsIndex, enabled: bool) -> bool:
        """
        Let ONE secondary view rasterize sensor-only surfaces. OFF by default for every view, because 'secondary' does not mean 'sensor': an RGB camera preview showing an untextured bake shell in front of the splats it approximates is a defect. A DEPTH consumer is the caller that turns it on. Takes effect only if set_sensor_only_surfaces(True) as well. False for an unknown handle, and for handle 0 — the primary never draws them.
        """
    def set_viewport(self, x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_white_balance(self, temperature: typing.SupportsFloat | typing.SupportsIndex = 6500.0, tint: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        """
        Scene-illuminant white balance: Kelvin on the Planckian locus (6500 = neutral = off), tint green(-)/magenta(+).
        """
    def size(self) -> tuple[int, int]:
        ...
    def sync_frame_interop(self) -> bool:
        """
        Block until the last submitted frame's interop copies have completed. Waits ONE frame fence, not the whole device: a single queue signals fences in submission order, so this retires every earlier frame too. False before the first frame.
        """
    def view_sensor_surfaces(self, handle: typing.SupportsInt | typing.SupportsIndex) -> bool:
        """
        Whether this view has asked for sensor-only surfaces.
        """
    def view_size(self, handle: typing.SupportsInt | typing.SupportsIndex) -> typing.Any:
        """
        Pixel size of a view's output as (width, height), or None if the handle is unknown.
        """
    @property
    def auto_exposure(self) -> bool:
        """
        Toggle automatic exposure / eye adaptation (default off). Drives tone-mapping exposure toward 18% gray for the scene's weighted-average luminance. tone_mapping_exposure is ignored while this is True.
        """
    @auto_exposure.setter
    def auto_exposure(self, arg1: bool) -> None:
        ...
    @property
    def auto_lod(self) -> bool:
        """
        Toggle automatic mesh LOD (background-simplified chains chosen by projected screen-space error). Default ON; False pins every mesh to full detail.
        """
    @auto_lod.setter
    def auto_lod(self, arg1: bool) -> None:
        ...
    @property
    def auto_lod_stats(self) -> dict:
        """
        Auto-LOD counters: resident index/BLAS bytes, chains ready/queued, and the per-level entry histogram.
        """
    @property
    def bloom_clamp(self) -> float:
        """
        Bloom input clamp to stabilise flickery ultra-bright highlights. <=0 disables (default); typical 8-32.
        """
    @bloom_clamp.setter
    def bloom_clamp(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def bloom_intensity(self) -> float:
        """
        Bloom strength. 0 disables; typical 0.2-0.8.
        """
    @bloom_intensity.setter
    def bloom_intensity(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def bloom_threshold(self) -> float:
        """
        Bright-pass cutoff (linear-HDR luma); higher = only the brightest glow. Typical 0.8-2.0.
        """
    @bloom_threshold.setter
    def bloom_threshold(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def camera_intrinsics(self) -> dict:
        """
        Pinhole intrinsics in RENDER-extent pixels (top-left origin, OpenCV convention) as a dict fx/fy/cx/cy/width/height. Derived from the camera's own film gauge and focal length -- set a real camera with `cam.film_gauge = 6.3; cam.set_focal_length(4.8)`. Valid after the first render.
        """
    @property
    def deferred_ao(self) -> bool:
        """
        Toggle ray-traced ambient occlusion / diffuse GI. Default on.
        """
    @deferred_ao.setter
    def deferred_ao(self, arg1: bool) -> None:
        ...
    @property
    def denoise(self) -> bool:
        """
        Toggle the deferred denoiser (SVGF + temporal). Default on. Off uses the deterministic 64-ray AO (noise-free, no GI colour).
        """
    @denoise.setter
    def denoise(self, arg1: bool) -> None:
        ...
    @property
    def depth_of_field(self) -> bool:
        """
        Thin-lens bokeh on the HDR scene (before bloom/TAA). CoC comes from the camera: set_camera_exposure's f-number (independent of physical_camera), FOV-derived focal length, focus_distance. Default off (zero cost).
        """
    @depth_of_field.setter
    def depth_of_field(self, arg1: bool) -> None:
        ...
    @property
    def env_sun_color(self) -> typing.Annotated[list[float], "FixedSize(3)"]:
        """
        Integrated sun-disc energy (linear RGB irradiance, valid when env_sun_found).
        """
    @property
    def env_sun_direction(self) -> typing.Annotated[list[float], "FixedSize(3)"]:
        """
        Unit direction TOWARD the detected env sun (valid when env_sun_found). Use to align an explicit DirectionalLight with the HDRI.
        """
    @property
    def env_sun_extraction(self) -> bool:
        """
        Extract the HDRI sun into an analytic light (default on). Toggling rebuilds the environment on the next frame.
        """
    @env_sun_extraction.setter
    def env_sun_extraction(self, arg1: bool) -> None:
        ...
    @property
    def env_sun_found(self) -> bool:
        """
        True when the current environment has a detected sun disc.
        """
    @property
    def env_sun_policy(self) -> str:
        """
        'auto' (a scene DirectionalLight claims the sun role), 'always', or 'off'.
        """
    @env_sun_policy.setter
    def env_sun_policy(self, arg1: str) -> None:
        ...
    @property
    def event_camera_enabled(self) -> bool:
        """
        GPU DVS event detector on/off. Costs nothing while off. Toggling
        does a device-idle + image resize, so do it between shots, never
        mid-sequence.
        """
    @event_camera_enabled.setter
    def event_camera_enabled(self, arg1: bool) -> None:
        ...
    @property
    def event_camera_params(self) -> dict:
        """
        The detector's current parameters as a dict.
        """
    @property
    def event_camera_resolution(self) -> tuple[int, int]:
        """
        (width, height) the detector is actually running at.
        """
    @property
    def event_camera_source(self) -> str:
        """
        What the detector looks at. 'shaded' (default): a deterministic
        Lambert proxy of the raster G-buffer — directional lights +
        ambient + emissive, no specular / transmission / point lights /
        GI. Noise-free and jitter-free (a static scene emits nothing),
        but only silhouettes and diffuse texture fire: water glitter,
        backlit sails and light flashes are invisible to it. 'final': the
        presented frame — the same pixels read_pixels() returns, post
        TAA / upscale / tonemap — box-averaged to the sensor resolution
        (a DVS pixel integrates its photodiode area). Everything the
        picture shows fires; it inherits the picture's temporal residue
        (denoiser, auto-exposure drift) and TAA jitter stays on. A switch
        while enabled re-latches the per-pixel reference on the next
        frame (no burst). Ignored under events_only_mode.
        """
    @event_camera_source.setter
    def event_camera_source(self, arg1: str) -> None:
        ...
    @property
    def events_only_mode(self) -> bool:
        """
        Present the event visualisation INSTEAD of the shaded scene.
        Leave it off if you also want read_pixels()/AOVs from the same
        frames — the detector runs either way. Forces the 'shaded'
        event_camera_source (no final frame exists to read).
        """
    @events_only_mode.setter
    def events_only_mode(self, arg1: bool) -> None:
        ...
    @property
    def exposure_compensation(self) -> float:
        """
        EV compensation while physical_camera is on (+1 doubles brightness).
        """
    @exposure_compensation.setter
    def exposure_compensation(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def focus_distance(self) -> float:
        """
        Focus plane distance in scene units/meters (default 10).
        """
    @focus_distance.setter
    def focus_distance(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fog_anisotropy(self) -> float:
        ...
    @fog_anisotropy.setter
    def fog_anisotropy(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def frame_timings(self) -> dict:
        ...
    @property
    def fsr(self) -> bool:
        """
        AMD FSR 3.1 upscaler on/off (no-op / False if unavailable — see fsr_available).
        """
    @fsr.setter
    def fsr(self, arg1: bool) -> None:
        ...
    @property
    def fsr_available(self) -> bool:
        """
        True when FSR was compiled in and its context created on this GPU.
        """
    @property
    def gbuffer_msaa(self) -> int:
        """
        G-buffer MSAA sample count (1, 2 or 4; default 1 = off). Stabilizes silhouette/edge flicker in the deferred renderer.
        """
    @gbuffer_msaa.setter
    def gbuffer_msaa(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def lens_distortion(self) -> dict:
        """
        Current lens distortion as a dict (see set_lens_distortion).
        """
    @property
    def lens_overscan(self) -> float:
        """
        Render the scene with the frustum widened by this factor so the lens warp has real geometry for the output corners instead of a clamped, smeared border. Barrel distortion (k1 < 0) needs it; 1.15-1.3 covers typical wide lenses. Costs effective resolution (the same pixels cover a wider field). camera_intrinsics still reports the OUTPUT camera. Default 1 (off).
        """
    @lens_overscan.setter
    def lens_overscan(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def occlusion_culling(self) -> bool:
        """
        Two-phase GPU occlusion culling: hidden objects stop paying raster cost (phase-2 same-frame recovery, no popping). Works with gbuffer_msaa. Wins scale with occlusion (interiors, city blocks). Default off.
        """
    @occlusion_culling.setter
    def occlusion_culling(self, arg1: bool) -> None:
        ...
    @property
    def physical_camera(self) -> bool:
        """
        Derive exposure from aperture/shutter/ISO (EV100) instead of tone_mapping_exposure; the HDR target is pre-exposed so 100k-lux daylight survives fp16. Defaults = sunny-16 (f/16, 1/125 s, ISO 100). Pair with physical_light_units. Default off.
        """
    @physical_camera.setter
    def physical_camera(self, arg1: bool) -> None:
        ...
    @property
    def physical_light_units(self) -> bool:
        """
        Interpret light intensities photometrically: directional = lux (sun ~100000), point/spot = lumens, rect/emissive = nits. Default off.
        """
    @physical_light_units.setter
    def physical_light_units(self, arg1: bool) -> None:
        ...
    @property
    def probe_gi(self) -> bool:
        """
        Toggle the world-space irradiance probe grid (multi-bounce GI + occlusion-correct ambient). Default ON; needs deferred_ao + denoise on. Interiors read physically dark — pair with auto_exposure or a raised tone_mapping_exposure. False restores the legacy cosmetic ambient.
        """
    @probe_gi.setter
    def probe_gi(self, arg1: bool) -> None:
        ...
    @property
    def render_scale(self) -> float:
        ...
    @render_scale.setter
    def render_scale(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def restir_di(self) -> bool:
        """
        ReSTIR DI (streaming RIS + temporal/spatial reuse at primary
        surfaces) for the deferred shade's next-event estimation.
        Default on. Off falls back to the legacy per-light NEE loops:
        cheaper with a handful of lights, markedly noisier with many.
        """
    @restir_di.setter
    def restir_di(self, arg1: bool) -> None:
        ...
    @property
    def scene_capture(self) -> bool:
        """
        Toggle scene-only swapchain capture (post-TAA, pre-overlay). When on, read it via read_scene_pixels(); off = no cost.
        """
    @scene_capture.setter
    def scene_capture(self, arg1: bool) -> None:
        ...
    @property
    def sensor_noise(self) -> dict:
        """
        Current sensor-noise settings as a dict.
        """
    @property
    def sensor_only_surfaces(self) -> bool:
        """
        Whether sensor-only surfaces are perceivable at all (the scene master switch).
        """
    @property
    def sim_time(self) -> float | None:
        """
        Simulation time in seconds that pins EVERY wall-clock read the frame
        path makes, or None while the renderer runs on the wall clock (the
        default). Pinned, it drives every renderer-side animated field: the
        ocean/DisplacedMesh FFT deform and its foam decay, grass wind, the
        clouds, the shared shader timeSec (water, particle lights, splats),
        the TAA blend dt and the DLSS/FSR frame deltas. Set it once per frame
        BEFORE render(), monotonically non-decreasing; stepping it by a fixed
        dt makes ocean/grass/particle animation deterministic and frame-rate
        independent, and the output replayable bit-for-bit. Unpinned, an
        offline render whose frames take 80 ms of wall time animates the sea
        ~5x faster than the dt the app integrates its own physics with (the
        hull can no longer follow the waves). None or a negative value
        returns to the wall clock, which is what a live window wants.
        """
    @sim_time.setter
    def sim_time(self, arg1: typing.SupportsFloat | typing.SupportsIndex | None) -> None:
        ...
    @property
    def splat_depth_aov(self) -> bool:
        """
        Whether the APPLICATION asked for the splat depth AOV. Note the renderer also turns it on by itself — the first frame a scene holds both splat clouds and overlay content (lines, wireframe, world sprites, unlit transparent meshes), so the cloud can occlude them — and that does NOT show up here. A read_gbuffer_aov_raw('splat_depth') can therefore succeed while this is False; it returns None only when the AOV is genuinely unallocated (a 1x1 placeholder).
        """
    @property
    def splat_depth_aov_mode(self) -> str:
        """
        Which statistic set_splat_depth_aov asked for: 'off', 'expected' or 'median'. 'off' with a readable AOV means the renderer latched it on for overlay occlusion, in which case the image carries the median.
        """
    @property
    def starfield(self) -> float:
        """
        Procedural star field drawn on sky pixels. 0 disables (default); ~1.0 is a night sky. Ramp it with the daylight rather than snapping it on.
        """
    @starfield.setter
    def starfield(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sun_angular_radius(self) -> float:
        """
        Directional-light angular radius in degrees for soft sun shadows (default 0.5; 0 = hard shadow).
        """
    @sun_angular_radius.setter
    def sun_angular_radius(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def tone_mapping_exposure(self) -> float:
        ...
    @tone_mapping_exposure.setter
    def tone_mapping_exposure(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class WrenchSample:
    """
    One six-component wrench reading, in the measured joint's child frame. force is N, torque is N*m.
    """
    def __repr__(self) -> str:
        ...
    @property
    def force(self) -> Vector3:
        ...
    @property
    def t(self) -> float:
        ...
    @property
    def torque(self) -> Vector3:
        ...
def apply_terrain_preset(preset: typing.SupportsInt | typing.SupportsIndex, params: TerrainParams) -> None:
    """
    Apply named preset: 0=Alpine, 1=Rolling Hills, 2=Desert Mesa, 3=Volcanic.
    """
def apply_tree_preset(preset: typing.SupportsInt | typing.SupportsIndex, params: TreeParams) -> None:
    """
    Apply species preset: 0=Oak, 1=Pine/Spruce (whorl conifer, Frond leaves), 2=Birch, 3=Willow.
    """
def bake_surface(renderer: typing.Any, cloud: SplatCloud, voxel_size: typing.SupportsFloat | typing.SupportsIndex = 0.0, truncation: typing.SupportsFloat | typing.SupportsIndex = 0.0, truncation_voxels: typing.SupportsFloat | typing.SupportsIndex = 4.0, max_weight: typing.SupportsFloat | typing.SupportsIndex = 32.0, weight_floor: typing.SupportsFloat | typing.SupportsIndex = 2.0, max_depth: typing.SupportsFloat | typing.SupportsIndex = 0.0, carve_fast_paths: bool = True, max_block_bytes: typing.SupportsInt | typing.SupportsIndex = 1073741824, pose_set: SplatPoseSet = SplatPoseSet.Orbit, poses: collections.abc.Sequence[BakePose] = [], pose_count: typing.SupportsInt | typing.SupportsIndex = 26, pose_distance: typing.SupportsFloat | typing.SupportsIndex = 0.0, min_component_voxels: typing.SupportsInt | typing.SupportsIndex = 256, fringe_erode: typing.SupportsInt | typing.SupportsIndex = 1, outlier_tolerance: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> SurfaceMesh:
    """
    Fuse a Gaussian-splat cloud into a triangle SurfaceMesh, by rendering it from a set of poses and integrating the median-depth AOV into a TSDF. VULKAN ONLY. Deterministic: the same cloud baked twice gives the same vertices and indices, bit for bit.
    
    Bake AFTER the cloud has its final transform — the vertices come out in WORLD space, so a later rotation of the cloud does not move them.
    
    The knobs that matter first:
      pose_set    Orbit (default) stands OUTSIDE looking in; Interior stands inside looking out. A canyon, a room, anything the camera is IN needs Interior — Orbit reconstructs the outside of its walls instead.
      poses       an explicit list of BakePose overrides pose_set entirely; the answer for replaying a real capture trajectory.
      voxel_size  0 derives it from the cloud's robust fit (radius / 256, clamped to 5 mm .. 10 cm).
      weight_floor is COUNTED IN POSES: 2 means two viewpoints had to agree. It can never be met by fewer poses than its own value.
      min_component_voxels drops islands smaller than this many surface CELLS (voxel_size^2 each) — photogrammetry floaters that survived carving.
    
    An empty result is diagnosed from .stats: see its docstring.
    """
def clamp(value: typing.SupportsFloat | typing.SupportsIndex, low: typing.SupportsFloat | typing.SupportsIndex, high: typing.SupportsFloat | typing.SupportsIndex) -> float:
    ...
def damp(x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, lambda_: typing.SupportsFloat | typing.SupportsIndex, dt: typing.SupportsFloat | typing.SupportsIndex) -> float:
    ...
def data_texture(data: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], srgb: bool = True) -> Texture:
    """
    Build a Texture from a (height, width, 3|4) uint8 numpy array. srgb=True for colour maps (map, emissive_map); srgb=False for data maps (roughness_map, metalness_map, ao_map).
    """
def deg_to_rad(degrees: typing.SupportsFloat | typing.SupportsIndex) -> float:
    ...
def euclidean_modulo(n: typing.SupportsFloat | typing.SupportsIndex, m: typing.SupportsFloat | typing.SupportsIndex) -> float:
    ...
def float_texture(data: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> Texture:
    """
    Build a float (HDR) Texture from a (height, width, 3|4) float32 numpy array: RGBA, linear, equirectangular. This is the sky/IBL texture type — assign it to scene.environment and scene.background. Row 0 is v = 0 (the zenith end of an equirect). Use Texture.update_float to re-bake one in place.
    """
def generate_uuid() -> str:
    ...
def icp_point_to_point(source: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], target: VoxelGrid, pose: Matrix4, opts: IcpOptions = ...) -> IcpResult:
    """
    Register source (N,3) float32 array against a VoxelGrid target. pose (Matrix4) is updated in place; seed it with an initial guess first.
    """
def inverse_lerp(x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, value: typing.SupportsFloat | typing.SupportsIndex) -> float:
    ...
def is_power_of_two(value: typing.SupportsInt | typing.SupportsIndex) -> bool:
    ...
def iso_mesh_to_geometry(iso_mesh: IsoMesh) -> BufferGeometry:
    """
    Convert an IsoMesh to a BufferGeometry ready for tp.Mesh(geom, material).
    """
def lerp(x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, t: typing.SupportsFloat | typing.SupportsIndex) -> float:
    ...
def make_bark_textures(size: typing.SupportsInt | typing.SupportsIndex = 256, seed: typing.SupportsInt | typing.SupportsIndex = 1337, base_color: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] = [0.3400000035762787, 0.23999999463558197, 0.1599999964237213], style: BarkStyle = BarkStyle.Furrowed) -> tuple:
    """
    Returns (albedo, normal) tiling bark Textures. `style` selects furrowed (oak), plated (conifer) or papery-with-lenticels (birch).
    """
def make_flower_texture(size: typing.SupportsInt | typing.SupportsIndex = 128, seed: typing.SupportsInt | typing.SupportsIndex = 1337) -> Texture:
    """
    RGBA wildflower alpha-cutout Texture. seed % 5 selects petal colour.
    """
def make_leaf_texture(size: typing.SupportsInt | typing.SupportsIndex = 256, seed: typing.SupportsInt | typing.SupportsIndex = 1337, base_color: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] = [0.25999999046325684, 0.44999998807907104, 0.14000000059604645], shape: LeafShape = LeafShape.Ovate, leaflets_per_twig: typing.SupportsInt | typing.SupportsIndex = 8, variants: typing.SupportsInt | typing.SupportsIndex = 2) -> Texture:
    """
    RGBA leaf-sprig alpha-cutout DataTexture: a branchlet of small leaflets with the given blade outline. `variants` is the atlas grid side and must match TreeParams.leaf_atlas_cells. Use mat.alpha_test = 0.4.
    """
def make_needle_frond_texture(size: typing.SupportsInt | typing.SupportsIndex = 256, seed: typing.SupportsInt | typing.SupportsIndex = 1337, base_color: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] = [0.10999999940395355, 0.28999999165534973, 0.10000000149011612], variants: typing.SupportsInt | typing.SupportsIndex = 2) -> Texture:
    """
    RGBA conifer needle-frond alpha-cutout DataTexture. Pair with LeafStyle.Frond + BranchingMode.Whorl. `variants` is the atlas grid side and must match TreeParams.leaf_atlas_cells. Use mat.alpha_test = 0.5.
    """
def make_sensor_mesh(surface: SurfaceMesh) -> Mesh:
    """
    The baked surface as a Mesh that ONLY THE SENSORS perceive. Add it at the SCENE ROOT, not under the cloud: the vertices are world space already.
    
    It is inert until the scene opts in with VulkanRenderer.set_sensor_only_surfaces(True); after that a lidar scan hits it, and so does any secondary view that also asks (set_view_sensor_surfaces(handle, True)). The primary camera NEVER draws it and no radiance trace sees it — the real splats render there, which is the whole point. Returns None for an empty surface.
    
    It is also an ordinary triangle Mesh, so PhysxWorld.add_static_trimesh(mesh) takes it directly: one bake is the ground for the feet and the ground for the sensors.
    """
def map_linear(x: typing.SupportsFloat | typing.SupportsIndex, a1: typing.SupportsFloat | typing.SupportsIndex, a2: typing.SupportsFloat | typing.SupportsIndex, b1: typing.SupportsFloat | typing.SupportsIndex, b2: typing.SupportsFloat | typing.SupportsIndex) -> float:
    ...
def marching_cubes(field: ScalarField, isolevel: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> IsoMesh:
    """
    Extract an isosurface mesh from a ScalarField. Returns an IsoMesh.
    """
def merge_buffer_geometries(geometries: collections.abc.Sequence[BufferGeometry], use_groups: bool = False) -> BufferGeometry:
    ...
def merge_vertices(geometry: BufferGeometry, tolerance: typing.SupportsFloat | typing.SupportsIndex = 9.999999747378752e-05) -> BufferGeometry:
    ...
def rad_to_deg(radians: typing.SupportsFloat | typing.SupportsIndex) -> float:
    ...
def rand_float() -> float:
    ...
def rand_float_range(min: typing.SupportsFloat | typing.SupportsIndex, max: typing.SupportsFloat | typing.SupportsIndex) -> float:
    ...
def rand_float_spread(range: typing.SupportsFloat | typing.SupportsIndex) -> float:
    ...
def rand_int(low: typing.SupportsInt | typing.SupportsIndex, high: typing.SupportsInt | typing.SupportsIndex) -> int:
    ...
def rotate_equirect(texture: Texture, x_deg: typing.SupportsFloat | typing.SupportsIndex = 0.0, y_deg: typing.SupportsFloat | typing.SupportsIndex = 0.0, z_deg: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> Texture:
    """
    Resample an equirect float texture by a 3D rotation (Euler XYZ degrees). For a Z-up scene with a Y-up HDRI: rotate_equirect(env, -90, 0, 0).
    """
def select_lod(cloud: SplatCloud, camera: Camera, viewport_height_px: typing.SupportsInt | typing.SupportsIndex, target_splats_per_pixel: typing.SupportsFloat | typing.SupportsIndex = 1.0, hysteresis: typing.SupportsFloat | typing.SupportsIndex = 1.25, per_node: bool = True) -> int:
    """
    Choose what this frame draws, and write it into the cloud's submit_ranges. Returns a level index into cloud.lod_levels (not the asset's own lod number).
    
    PER NODE by default, whenever the asset carried an SSOG tree (cloud.lod_node_count > 0): every leaf of that tree gets its own level, so a near wall stays fine while the far end of the canyon coarsens. Without a tree this is the whole-cloud rule — one level for everything, then its chunks against the frustum — which is also what per_node=False forces, for A/B.
    
    target_splats_per_pixel means the same thing on both paths: splats per SCREEN pixel for the whole visible cloud. The per-node rule derives its own per-leaf threshold by dividing it by the frame's overdraw factor (the visible leaves' footprints summed over the screen area), so the total submitted count lands near target_splats_per_pixel * screen pixels by construction and the SAME argument value keeps its meaning across the switch. Raise it when the camera stands INSIDE the scan; the calico demo runs 8.
    
    Call once per frame, before render(), with the RENDER resolution's height. The cloud carries its own hysteresis state (per node on the per-node path), so nothing has to be kept on the Python side. A cloud with no LOD table is left alone and 0 comes back.
    """
def simplify_geometry(geometry: BufferGeometry, ratio: typing.SupportsFloat | typing.SupportsIndex, error: typing.SupportsFloat | typing.SupportsIndex = 0.009999999776482582) -> BufferGeometry:
    ...
def splat_points_to_field(points: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], cell_size: typing.SupportsFloat | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex, max_nodes: typing.SupportsInt | typing.SupportsIndex = 8000000) -> ScalarField:
    """
    Build a union-of-balls scalar field from an (N,3) point array.
    """
def terrain_from_json(json: str, params: TerrainParams) -> bool:
    """
    Deserialise TerrainParams from a JSON string. Unknown keys keep current value.
    """
def terrain_load_config(path: str, params: TerrainParams) -> bool:
    ...
def terrain_save_config(path: str, params: TerrainParams) -> bool:
    ...
def terrain_to_json(params: TerrainParams) -> str:
    """
    Serialise TerrainParams to a JSON string.
    """
def voxel_downsample(points: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], voxel_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float32]:
    """
    Voxel-downsample an (N,3) float32 point array. Returns (M,3) float32.
    """
def vulkan_available() -> bool:
    """
    True when the Vulkan loader is present at runtime. HAS_VULKAN=True + vulkan_available()=False means the wheel carries the backend but this machine has no Vulkan runtime — use GLRenderer.
    """
def vulkan_validation_active() -> bool:
    """
    True once a renderer has actually installed the validation-layer messenger (layer requested via THREEPP_VULKAN_VALIDATION=1 or a debug build, AND found on the machine).
    """
def vulkan_validation_error_count() -> int:
    """
    Validation-layer ERROR messages counted since process start. Always 0 unless the layer is active — check vulkan_validation_active() first, or a 'no errors' assertion passes vacuously.
    """
def write_wav(path: str, samples: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], sample_rate: typing.SupportsInt | typing.SupportsIndex = 44100) -> None:
    """
    Write a mono 16-bit PCM WAV file from normalised float samples in [-1, 1].
    """
Approach: BirdState  # value = <BirdState.Approach: 1>
Blob: LeafStyle  # value = <LeafStyle.Blob: 3>
Cluster: LeafStyle  # value = <LeafStyle.Cluster: 1>
Colonise: BranchingMode  # value = <BranchingMode.Colonise: 0>
Cone: CrownShape  # value = <CrownShape.Cone: 2>
CrossQuad: LeafStyle  # value = <LeafStyle.CrossQuad: 2>
Cruise: BirdState  # value = <BirdState.Cruise: 0>
Cylinder: CrownShape  # value = <CrownShape.Cylinder: 4>
EXPONENTIAL: AudioDistanceModel  # value = <AudioDistanceModel.EXPONENTIAL: 3>
Ellipsoid: CrownShape  # value = <CrownShape.Ellipsoid: 1>
Evade: BirdState  # value = <BirdState.Evade: 5>
Flare: BirdState  # value = <BirdState.Flare: 2>
Follower: BirdRole  # value = <BirdRole.Follower: 0>
Frond: LeafStyle  # value = <LeafStyle.Frond: 4>
Furrowed: BarkStyle  # value = <BarkStyle.Furrowed: 0>
HAS_AUDIO: bool = True
HAS_IMGUI: bool = True
HAS_PHYSX: bool = True
HAS_VULKAN: bool = True
Hemisphere: CrownShape  # value = <CrownShape.Hemisphere: 3>
Hop: Gait  # value = <Gait.Hop: 0>
INVERSE: AudioDistanceModel  # value = <AudioDistanceModel.INVERSE: 1>
LINEAR: AudioDistanceModel  # value = <AudioDistanceModel.LINEAR: 2>
Lanceolate: LeafShape  # value = <LeafShape.Lanceolate: 3>
Launch: BirdState  # value = <BirdState.Launch: 4>
Leader: BirdRole  # value = <BirdRole.Leader: 1>
Lobed: LeafShape  # value = <LeafShape.Lobed: 1>
Loner: BirdRole  # value = <BirdRole.Loner: 2>
NONE: AudioDistanceModel  # value = <AudioDistanceModel.NONE: 0>
Ovate: LeafShape  # value = <LeafShape.Ovate: 0>
Papery: BarkStyle  # value = <BarkStyle.Papery: 2>
Perched: BirdState  # value = <BirdState.Perched: 3>
Plated: BarkStyle  # value = <BarkStyle.Plated: 1>
Quad: LeafStyle  # value = <LeafStyle.Quad: 0>
Serrate: LeafShape  # value = <LeafShape.Serrate: 2>
Sphere: CrownShape  # value = <CrownShape.Sphere: 0>
Walk: Gait  # value = <Gait.Walk: 1>
Whorl: BranchingMode  # value = <BranchingMode.Whorl: 1>
