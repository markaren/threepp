"""
threepp — a cross-platform C++ 3D library with the high-level API of three.js, exposed to Python (scene graph, geometries, materials, cameras, lights, OpenGL renderer + headless render-to-numpy).
"""
from __future__ import annotations
import numpy
import typing
__all__: list[str] = ['AmbientLight', 'Background', 'Blending', 'Box3', 'BoxGeometry', 'BufferGeometry', 'Camera', 'Canvas', 'CapsuleGeometry', 'CircleGeometry', 'Clock', 'Color', 'ColorSpace', 'CombineOperation', 'ConeGeometry', 'CylinderGeometry', 'DirectionalLight', 'Euler', 'Filter', 'Fog', 'GLRenderer', 'GLTFLoader', 'Group', 'HAS_VULKAN', 'HemisphereLight', 'IcosahedronGeometry', 'InstancedMesh', 'Light', 'Line', 'LineBasicMaterial', 'LineSegments', 'Mapping', 'Material', 'Matrix3', 'Matrix4', 'Mesh', 'MeshBasicMaterial', 'MeshLambertMaterial', 'MeshNormalMaterial', 'MeshPhongMaterial', 'MeshStandardMaterial', 'ModelLoader', 'OBJLoader', 'Object3D', 'OctahedronGeometry', 'OrbitControls', 'OrthographicCamera', 'PerspectiveCamera', 'PlaneGeometry', 'PointLight', 'Points', 'PointsMaterial', 'Quaternion', 'RectAreaLight', 'RingGeometry', 'RotationOrder', 'STLLoader', 'Scene', 'ShadowMaterial', 'Side', 'SphereGeometry', 'SpotLight', 'Sprite', 'SpriteMaterial', 'Texture', 'TextureLoader', 'TextureWrapping', 'TorusGeometry', 'TorusKnotGeometry', 'Vector2', 'Vector3', 'Vector4', 'VulkanRenderer']
class AmbientLight(Light):
    def __init__(self, color: Color = ..., intensity: float = 1.0) -> None:
        ...
class Background:
    @typing.overload
    def __init__(self, color: int) -> None:
        ...
    @typing.overload
    def __init__(self, color: Color) -> None:
        ...
    def is_color(self) -> bool:
        ...
    def is_texture(self) -> bool:
        ...
class Blending:
    """
    Members:
    
      None
    
      Normal
    
      Additive
    
      Subtractive
    
      Multiply
    
      Custom
    """
    Additive: typing.ClassVar[Blending]  # value = <Blending.Additive: 2>
    Custom: typing.ClassVar[Blending]  # value = <Blending.Custom: 5>
    Multiply: typing.ClassVar[Blending]  # value = <Blending.Multiply: 4>
    None: typing.ClassVar[Blending]  # value = <Blending.None: 0>
    Normal: typing.ClassVar[Blending]  # value = <Blending.Normal: 1>
    Subtractive: typing.ClassVar[Blending]  # value = <Blending.Subtractive: 3>
    __members__: typing.ClassVar[dict[str, Blending]]  # value = {'None': <Blending.None: 0>, 'Normal': <Blending.Normal: 1>, 'Additive': <Blending.Additive: 2>, 'Subtractive': <Blending.Subtractive: 3>, 'Multiply': <Blending.Multiply: 4>, 'Custom': <Blending.Custom: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
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
    def set_from_object(self, object: ..., precise: bool = False) -> Box3:
        ...
class BoxGeometry(BufferGeometry):
    def __init__(self, width: float = 1.0, height: float = 1.0, depth: float = 1.0, width_segments: int = 1, height_segments: int = 1, depth_segments: int = 1) -> None:
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
class BufferGeometry:
    name: str
    def __init__(self) -> None:
        ...
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
    def rotate_x(self, angle: float) -> BufferGeometry:
        ...
    def rotate_y(self, angle: float) -> BufferGeometry:
        ...
    def rotate_z(self, angle: float) -> BufferGeometry:
        ...
    def scale(self, x: float, y: float, z: float) -> BufferGeometry:
        ...
    def set_from_points(self, points: list[Vector3]) -> BufferGeometry:
        ...
    def translate(self, x: float, y: float, z: float) -> BufferGeometry:
        ...
class Camera(Object3D):
    far: float
    far_plane: float
    near: float
    near_plane: float
    zoom: float
    def update_projection_matrix(self) -> None:
        ...
class Canvas:
    def __init__(self, title: str = 'threepp', width: int = -1, height: int = -1, antialiasing: int = 4, vsync: bool = True, resizable: bool = True, headless: bool = False) -> None:
        ...
    def animate(self, callback: typing.Callable[[], None]) -> None:
        """
        Run the render loop, calling callback() every frame until the window closes.
        """
    def animate_once(self, callback: typing.Callable[[], None]) -> bool:
        """
        Render a single frame; returns False when the app should quit.
        """
    def aspect(self) -> float:
        ...
    def close(self) -> None:
        ...
    def is_open(self) -> bool:
        ...
    def set_size(self, width: int, height: int) -> None:
        ...
    def size(self) -> tuple:
        ...
class CapsuleGeometry(BufferGeometry):
    def __init__(self, radius: float = 0.5, length: float = 1.0, cap_segments: int = 8, radial_segments: int = 16) -> None:
        ...
    @property
    def length(self) -> float:
        ...
    @property
    def radius(self) -> float:
        ...
class CircleGeometry(BufferGeometry):
    def __init__(self, radius: float = 1.0, segments: int = 16, theta_start: float = 0.0, theta_length: float = 6.2831854820251465) -> None:
        ...
class Clock:
    elapsed_time: float
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
class Color:
    __hash__: typing.ClassVar[None] = None
    b: float
    g: float
    r: float
    def __eq__(self, arg0: Color) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, hex: int) -> None:
        ...
    @typing.overload
    def __init__(self, r: float, g: float, b: float) -> None:
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
    def lerp(self, color: Color, alpha: float) -> Color:
        ...
    def set_hex(self, hex: int) -> Color:
        ...
    def set_rgb(self, r: float, g: float, b: float) -> Color:
        ...
    def set_style(self, style: str) -> Color:
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
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
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
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
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
    def __init__(self, radius: float = 1.0, height: float = 1.0, radial_segments: int = 16, height_segments: int = 1, open_ended: bool = False, theta_start: float = 0.0, theta_length: float = 6.2831854820251465) -> None:
        ...
class CylinderGeometry(BufferGeometry):
    def __init__(self, radius_top: float = 1.0, radius_bottom: float = 1.0, height: float = 1.0, radial_segments: int = 16, height_segments: int = 1, open_ended: bool = False, theta_start: float = 0.0, theta_length: float = 6.2831854820251465) -> None:
        ...
class DirectionalLight(Light):
    def __init__(self, color: Color = ..., intensity: float = 1.0) -> None:
        ...
    def set_target(self, target: Object3D) -> None:
        ...
class Euler:
    order: RotationOrder
    x: float
    y: float
    z: float
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def set(self, x: float, y: float, z: float) -> Euler:
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
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Fog:
    color: Color
    far: float
    near: float
    def __init__(self, color: Color, near: float = 1.0, far: float = 1000.0) -> None:
        ...
class GLRenderer:
    auto_clear: bool
    check_shader_errors: bool
    shadow_map_enabled: bool
    sort_objects: bool
    tone_mapping_exposure: float
    def __init__(self, canvas: Canvas) -> None:
        ...
    def clear(self, color: bool = True, depth: bool = True, stencil: bool = True) -> None:
        ...
    def read_pixels(self, flip: bool = True) -> numpy.ndarray[numpy.uint8]:
        ...
    def render(self, scene: Object3D, camera: Camera) -> None:
        ...
    def save_frame(self, path: str) -> None:
        ...
    def set_clear_color(self, color: Color, alpha: float = 1.0) -> None:
        ...
    def set_pixel_ratio(self, value: float) -> None:
        ...
    def set_size(self, width: int, height: int) -> None:
        ...
    def size(self) -> tuple:
        ...
class GLTFLoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str) -> Group:
        ...
class Group(Object3D):
    def __init__(self) -> None:
        ...
class HemisphereLight(Light):
    ground_color: Color
    def __init__(self, sky_color: Color = ..., ground_color: Color = ..., intensity: float = 1.0) -> None:
        ...
class IcosahedronGeometry(BufferGeometry):
    def __init__(self, radius: float = 1.0, detail: int = 0) -> None:
        ...
class InstancedMesh(Mesh):
    def __init__(self, geometry: BufferGeometry, material: typing.Any, count: int) -> None:
        ...
    def get_matrix_at(self, index: int) -> Matrix4:
        ...
    def instance_color_needs_update(self) -> None:
        ...
    def instance_matrix_needs_update(self) -> None:
        ...
    def set_color_at(self, index: int, color: Color) -> None:
        ...
    def set_count(self, count: int) -> None:
        ...
    def set_matrix_at(self, index: int, matrix: Matrix4) -> None:
        ...
    @property
    def count(self) -> int:
        ...
class Light(Object3D):
    color: Color
    intensity: float
class Line(Object3D):
    cast_shadow: bool
    frustum_culled: bool
    matrix_auto_update: bool
    name: str
    position: Vector3
    quaternion: Quaternion
    receive_shadow: bool
    render_order: int
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
    def get_world_direction(self) -> Vector3:
        ...
    def get_world_position(self) -> Vector3:
        ...
    @typing.overload
    def look_at(self, x: float, y: float, z: float) -> None:
        ...
    @typing.overload
    def look_at(self, vector: Vector3) -> None:
        ...
    def remove(self, object: typing.Any) -> None:
        ...
    def remove_from_parent(self) -> None:
        ...
    def rotate_on_axis(self, axis: Vector3, angle: float) -> None:
        ...
    def rotate_x(self, angle: float) -> None:
        ...
    def rotate_y(self, angle: float) -> None:
        ...
    def rotate_z(self, angle: float) -> None:
        ...
    def translate_x(self, distance: float) -> None:
        ...
    def translate_y(self, distance: float) -> None:
        ...
    def translate_z(self, distance: float) -> None:
        ...
    def traverse(self, callback: typing.Callable[[typing.Any], None]) -> None:
        ...
    def update_matrix(self) -> None:
        ...
    def update_matrix_world(self, force: bool = False) -> None:
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
    def material(self) -> typing.Any:
        ...
    @property
    def parent(self) -> Object3D:
        ...
    @property
    def uuid(self) -> str:
        ...
class LineBasicMaterial(Material):
    alpha_test: float
    blending: Blending
    color: Color
    depth_test: bool
    depth_write: bool
    fog: bool
    linewidth: float
    name: str
    opacity: float
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
class LineSegments(Line):
    def __init__(self, geometry: BufferGeometry = None, material: typing.Any = None) -> None:
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
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
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
    pass
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
    def make_rotation_x(self, theta: float) -> Matrix4:
        ...
    def make_rotation_y(self, theta: float) -> Matrix4:
        ...
    def make_rotation_z(self, theta: float) -> Matrix4:
        ...
    def make_scale(self, x: float, y: float, z: float) -> Matrix4:
        ...
    def make_translation(self, x: float, y: float, z: float) -> Matrix4:
        ...
    def set_position(self, x: float, y: float, z: float) -> Matrix4:
        ...
    def transpose(self) -> Matrix4:
        ...
class Mesh(Object3D):
    cast_shadow: bool
    frustum_culled: bool
    matrix_auto_update: bool
    name: str
    position: Vector3
    quaternion: Quaternion
    receive_shadow: bool
    render_order: int
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
    def get_world_direction(self) -> Vector3:
        ...
    def get_world_position(self) -> Vector3:
        ...
    @typing.overload
    def look_at(self, x: float, y: float, z: float) -> None:
        ...
    @typing.overload
    def look_at(self, vector: Vector3) -> None:
        ...
    def remove(self, object: typing.Any) -> None:
        ...
    def remove_from_parent(self) -> None:
        ...
    def rotate_on_axis(self, axis: Vector3, angle: float) -> None:
        ...
    def rotate_x(self, angle: float) -> None:
        ...
    def rotate_y(self, angle: float) -> None:
        ...
    def rotate_z(self, angle: float) -> None:
        ...
    def set_geometry(self, geometry: BufferGeometry) -> None:
        ...
    def set_material(self, material: typing.Any) -> None:
        ...
    def translate_x(self, distance: float) -> None:
        ...
    def translate_y(self, distance: float) -> None:
        ...
    def translate_z(self, distance: float) -> None:
        ...
    def traverse(self, callback: typing.Callable[[typing.Any], None]) -> None:
        ...
    def update_matrix(self) -> None:
        ...
    def update_matrix_world(self, force: bool = False) -> None:
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
    def material(self) -> typing.Any:
        ...
    @property
    def parent(self) -> Object3D:
        ...
    @property
    def uuid(self) -> str:
        ...
class MeshBasicMaterial(Material):
    alpha_map: Texture
    alpha_test: float
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
    opacity: float
    premultiplied_alpha: bool
    reflectivity: float
    refraction_ratio: float
    side: Side
    specular_map: Texture
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    wireframe: bool
    wireframe_linewidth: float
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
class MeshLambertMaterial(Material):
    alpha_map: Texture
    alpha_test: float
    ao_map: Texture
    blending: Blending
    color: Color
    depth_test: bool
    depth_write: bool
    emissive: Color
    emissive_intensity: float
    emissive_map: Texture
    env_map: Texture
    fog: bool
    map: Texture
    name: str
    opacity: float
    premultiplied_alpha: bool
    reflectivity: float
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
class MeshNormalMaterial(Material):
    alpha_test: float
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
    opacity: float
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
class MeshPhongMaterial(Material):
    alpha_map: Texture
    alpha_test: float
    ao_map: Texture
    blending: Blending
    bump_map: Texture
    color: Color
    combine: CombineOperation
    depth_test: bool
    depth_write: bool
    emissive: Color
    emissive_intensity: float
    emissive_map: Texture
    env_map: Texture
    flat_shading: bool
    fog: bool
    map: Texture
    name: str
    normal_map: Texture
    opacity: float
    premultiplied_alpha: bool
    reflectivity: float
    shininess: float
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
class MeshStandardMaterial(Material):
    alpha_map: Texture
    alpha_test: float
    ao_map: Texture
    blending: Blending
    bump_map: Texture
    color: Color
    depth_test: bool
    depth_write: bool
    displacement_map: Texture
    emissive: Color
    emissive_intensity: float
    emissive_map: Texture
    env_map: Texture
    env_map_intensity: float
    flat_shading: bool
    fog: bool
    map: Texture
    metalness: float
    metalness_map: Texture
    name: str
    normal_map: Texture
    normal_scale: Vector2
    opacity: float
    premultiplied_alpha: bool
    roughness: float
    roughness_map: Texture
    side: Side
    tone_mapped: bool
    transparent: bool
    vertex_colors: bool
    visible: bool
    wireframe: bool
    wireframe_linewidth: float
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def dispose(self) -> None:
        ...
class ModelLoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str) -> Group:
        """
        Load a model (.obj/.gltf/.glb/.stl/.dae) as a Group.
        """
    def set_ignore_up_direction(self, ignore: bool) -> None:
        ...
class OBJLoader:
    def __init__(self) -> None:
        ...
    def load(self, path: str, try_load_mtl: bool = True) -> Group:
        ...
class Object3D:
    cast_shadow: bool
    frustum_culled: bool
    matrix_auto_update: bool
    name: str
    position: Vector3
    quaternion: Quaternion
    receive_shadow: bool
    render_order: int
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
    def get_object_by_name(self, name: str) -> Object3D:
        ...
    def get_world_direction(self) -> Vector3:
        ...
    def get_world_position(self) -> Vector3:
        ...
    @typing.overload
    def look_at(self, x: float, y: float, z: float) -> None:
        ...
    @typing.overload
    def look_at(self, vector: Vector3) -> None:
        ...
    def remove(self, object: typing.Any) -> None:
        ...
    def remove_from_parent(self) -> None:
        ...
    def rotate_on_axis(self, axis: Vector3, angle: float) -> None:
        ...
    def rotate_x(self, angle: float) -> None:
        ...
    def rotate_y(self, angle: float) -> None:
        ...
    def rotate_z(self, angle: float) -> None:
        ...
    def translate_x(self, distance: float) -> None:
        ...
    def translate_y(self, distance: float) -> None:
        ...
    def translate_z(self, distance: float) -> None:
        ...
    def traverse(self, callback: typing.Callable[[typing.Any], None]) -> None:
        ...
    def update_matrix(self) -> None:
        ...
    def update_matrix_world(self, force: bool = False) -> None:
        ...
    @property
    def children(self) -> list[Object3D]:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def parent(self) -> Object3D:
        ...
    @property
    def uuid(self) -> str:
        ...
class OctahedronGeometry(BufferGeometry):
    def __init__(self, radius: float = 1.0, detail: int = 0) -> None:
        ...
class OrbitControls:
    auto_rotate: bool
    auto_rotate_speed: float
    damping_factor: float
    enable_damping: bool
    enable_pan: bool
    enable_rotate: bool
    enable_zoom: bool
    enabled: bool
    max_distance: float
    min_distance: float
    rotate_speed: float
    target: Vector3
    zoom_speed: float
    def __init__(self, camera: Camera, canvas: Canvas) -> None:
        ...
    def update(self) -> bool:
        ...
class OrthographicCamera(Camera):
    bottom: float
    left: float
    right: float
    top: float
    def __init__(self, left: float = -1.0, right: float = 1.0, top: float = 1.0, bottom: float = -1.0, near: float = 0.10000000149011612, far: float = 2000.0) -> None:
        ...
    def update_projection_matrix(self) -> None:
        ...
class PerspectiveCamera(Camera):
    aspect: float
    film_gauge: float
    film_offset: float
    focus: float
    fov: float
    def __init__(self, fov: float = 60.0, aspect: float = 1.0, near: float = 0.10000000149011612, far: float = 2000.0) -> None:
        ...
    def get_focal_length(self) -> float:
        ...
    def set_focal_length(self, focal_length: float) -> None:
        ...
    def update_projection_matrix(self) -> None:
        ...
class PlaneGeometry(BufferGeometry):
    def __init__(self, width: float = 1.0, height: float = 1.0, width_segments: int = 1, height_segments: int = 1) -> None:
        ...
    @property
    def height(self) -> float:
        ...
    @property
    def width(self) -> float:
        ...
class PointLight(Light):
    decay: float
    distance: float
    def __init__(self, color: Color = ..., intensity: float = 1.0, distance: float = 0.0, decay: float = 1.0) -> None:
        ...
    def get_power(self) -> float:
        ...
    def set_power(self, power: float) -> None:
        ...
class Points(Object3D):
    cast_shadow: bool
    frustum_culled: bool
    matrix_auto_update: bool
    name: str
    position: Vector3
    quaternion: Quaternion
    receive_shadow: bool
    render_order: int
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
    def get_world_direction(self) -> Vector3:
        ...
    def get_world_position(self) -> Vector3:
        ...
    @typing.overload
    def look_at(self, x: float, y: float, z: float) -> None:
        ...
    @typing.overload
    def look_at(self, vector: Vector3) -> None:
        ...
    def remove(self, object: typing.Any) -> None:
        ...
    def remove_from_parent(self) -> None:
        ...
    def rotate_on_axis(self, axis: Vector3, angle: float) -> None:
        ...
    def rotate_x(self, angle: float) -> None:
        ...
    def rotate_y(self, angle: float) -> None:
        ...
    def rotate_z(self, angle: float) -> None:
        ...
    def translate_x(self, distance: float) -> None:
        ...
    def translate_y(self, distance: float) -> None:
        ...
    def translate_z(self, distance: float) -> None:
        ...
    def traverse(self, callback: typing.Callable[[typing.Any], None]) -> None:
        ...
    def update_matrix(self) -> None:
        ...
    def update_matrix_world(self, force: bool = False) -> None:
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
    def material(self) -> typing.Any:
        ...
    @property
    def parent(self) -> Object3D:
        ...
    @property
    def uuid(self) -> str:
        ...
class PointsMaterial(Material):
    alpha_map: Texture
    alpha_test: float
    blending: Blending
    color: Color
    depth_test: bool
    depth_write: bool
    fog: bool
    map: Texture
    name: str
    opacity: float
    premultiplied_alpha: bool
    side: Side
    size: float
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
class Quaternion:
    w: float
    x: float
    y: float
    z: float
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def invert(self) -> Quaternion:
        ...
    def normalize(self) -> Quaternion:
        ...
    def set(self, x: float, y: float, z: float, w: float) -> Quaternion:
        ...
    def set_from_axis_angle(self, axis: Vector3, angle: float) -> Quaternion:
        ...
    def set_from_euler(self, euler: Euler) -> Quaternion:
        ...
    def slerp(self, qb: Quaternion, t: float) -> Quaternion:
        ...
class RectAreaLight(Light):
    def __init__(self, color: Color = ..., intensity: float = 1.0, width: float = 1.0, height: float = 1.0) -> None:
        ...
    @property
    def height(self) -> float:
        ...
    @property
    def width(self) -> float:
        ...
class RingGeometry(BufferGeometry):
    def __init__(self, inner_radius: float = 0.5, outer_radius: float = 1.0, theta_segments: int = 16, phi_segments: int = 2, theta_start: float = 0.0, theta_length: float = 6.2831854820251465) -> None:
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
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
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
class Scene(Object3D):
    auto_update: bool
    background: Background
    override_material: Material
    def __init__(self) -> None:
        ...
    def clear_fog(self) -> None:
        ...
    def set_fog(self, color: Color, near: float = 1.0, far: float = 1000.0) -> None:
        ...
class ShadowMaterial(Material):
    alpha_test: float
    blending: Blending
    color: Color
    depth_test: bool
    depth_write: bool
    fog: bool
    name: str
    opacity: float
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
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SphereGeometry(BufferGeometry):
    def __init__(self, radius: float = 1.0, width_segments: int = 16, height_segments: int = 12, phi_start: float = 0.0, phi_length: float = 6.2831854820251465, theta_start: float = 0.0, theta_length: float = 3.1415927410125732) -> None:
        ...
    @property
    def radius(self) -> float:
        ...
class SpotLight(Light):
    angle: float
    decay: float
    distance: float
    penumbra: float
    def __init__(self, color: Color = ..., intensity: float = 1.0, distance: float = 0.0, angle: float = 1.0471975803375244, penumbra: float = 0.0, decay: float = 1.0) -> None:
        ...
    def set_target(self, target: Object3D) -> None:
        ...
class Sprite(Object3D):
    center: Vector2
    screen_space: bool
    def __init__(self, material: SpriteMaterial = None) -> None:
        ...
    @property
    def material(self) -> typing.Any:
        ...
class SpriteMaterial(Material):
    alpha_map: Texture
    alpha_test: float
    blending: Blending
    color: Color
    depth_test: bool
    depth_write: bool
    fog: bool
    map: Texture
    name: str
    opacity: float
    premultiplied_alpha: bool
    rotation: float
    side: Side
    size: float
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
class Texture:
    anisotropy: int
    center: Vector2
    color_space: ColorSpace
    generate_mipmaps: bool
    mag_filter: Filter
    mapping: Mapping
    min_filter: Filter
    name: str
    offset: Vector2
    repeat: Vector2
    rotation: float
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
    def update_matrix(self) -> None:
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
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
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
    def __init__(self, radius: float = 1.0, tube: float = 0.4000000059604645, radial_segments: int = 20, tubular_segments: int = 64, arc: float = 6.2831854820251465) -> None:
        ...
class TorusKnotGeometry(BufferGeometry):
    def __init__(self, radius: float = 1.0, tube: float = 0.4000000059604645, tubular_segments: int = 64, radial_segments: int = 16, p: int = 2, q: int = 3) -> None:
        ...
class Vector2:
    __hash__: typing.ClassVar[None] = None
    x: float
    y: float
    def __add__(self, arg0: Vector2) -> Vector2:
        ...
    def __eq__(self, arg0: Vector2) -> bool:
        ...
    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        ...
    def __mul__(self, arg0: float) -> Vector2:
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
    def multiply_scalar(self, s: float) -> Vector2:
        ...
    def normalize(self) -> Vector2:
        ...
    def set(self, x: float, y: float) -> Vector2:
        ...
    def sub(self, v: Vector2) -> Vector2:
        ...
class Vector3:
    __hash__: typing.ClassVar[None] = None
    x: float
    y: float
    z: float
    def __add__(self, arg0: Vector3) -> Vector3:
        ...
    def __eq__(self, arg0: Vector3) -> bool:
        ...
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        ...
    def __mul__(self, arg0: float) -> Vector3:
        ...
    def __repr__(self) -> str:
        ...
    def __sub__(self, arg0: Vector3) -> Vector3:
        ...
    def add(self, v: Vector3) -> Vector3:
        ...
    def add_scaled_vector(self, v: Vector3, s: float) -> Vector3:
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
    def lerp(self, v: Vector3, alpha: float) -> Vector3:
        ...
    def multiply_scalar(self, s: float) -> Vector3:
        ...
    def negate(self) -> Vector3:
        ...
    def normalize(self) -> Vector3:
        ...
    def set(self, x: float, y: float, z: float) -> Vector3:
        ...
    def sub(self, v: Vector3) -> Vector3:
        ...
class Vector4:
    w: float
    x: float
    y: float
    z: float
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def set(self, x: float, y: float, z: float, w: float) -> Vector4:
        ...
class VulkanRenderer:
    def __init__(self, canvas: Canvas, flush_frames: int = 3) -> None:
        """
        Deferred (RasterFirst) Vulkan renderer. Pass a headless Canvas created with vsync=False.
        """
    def read_albedo(self, scene: Object3D, camera: Camera) -> numpy.ndarray[numpy.uint8]:
        ...
    def read_normals(self, scene: Object3D, camera: Camera) -> numpy.ndarray[numpy.uint8]:
        ...
    def read_pixels(self) -> numpy.ndarray[numpy.uint8]:
        """
        Final shaded RGB of the last render as (H, W, 3) uint8.
        """
    def read_segmentation(self, scene: Object3D, camera: Camera) -> numpy.ndarray[numpy.uint8]:
        ...
    def render(self, scene: Object3D, camera: Camera) -> None:
        ...
    def render_aov(self, scene: Object3D, camera: Camera, aov: str) -> numpy.ndarray[numpy.uint8]:
        """
        Render and return a G-buffer AOV as (H, W, 3) uint8: 'rgb' | 'normals' | 'segmentation' | 'albedo' | 'motion'.
        """
    def render_aovs(self, scene: Object3D, camera: Camera, aovs: list[str] = ['rgb', 'normals', 'segmentation']) -> dict:
        """
        Render the requested AOVs and return {name: (H, W, 3) uint8}.
        """
    def save_frame(self, scene: Object3D, camera: Camera, path: str) -> None:
        ...
    def set_clear_color(self, color: Color, alpha: float = 1.0) -> None:
        ...
    def set_flush_frames(self, n: int) -> None:
        """
        Frames driven per render() to flush the MAILBOX swapchain (default 3; raise to 4+ for fast-moving dynamic scenes).
        """
    def size(self) -> tuple[int, int]:
        ...
HAS_VULKAN: bool = True
