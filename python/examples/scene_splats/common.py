"""What the scene-to-splats scripts share: the scene, the camera rig and the dataset format.

The dataset is nerfstudio's transforms.json, so other trainers can read it too:
`transform_matrix` is camera-to-world in the OpenGL convention (camera looks down
-Z, +Y up), which is exactly a threepp camera's matrix_world; intrinsics are
pinhole pixels with a top-left origin; depth is a 16-bit PNG in millimetres
(0 = no surface), planar along the view axis.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import threepp as tp  # noqa: E402

# Light grey-blue: nothing in the scene is this colour, so the trainer cannot
# explain a dark part of the model as "background showing through".
BACKGROUND = (0.62, 0.68, 0.76)
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), ".cache", "threepp", "scene_splats", "rover")


def _ground_texture(size=1024, seed=7):
    """Sand with pebbles: enough texture for multi-view matching, all from a seed."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for cells in (8, 16, 32, 64, 128, 256):
        grid = rng.random((cells + 1, cells + 1)).astype(np.float32)
        x = np.linspace(0, cells, size, endpoint=False, dtype=np.float32)
        i = x.astype(int)
        f = x - i
        f = f * f * (3 - 2 * f)
        a = grid[i][:, i] * (1 - f)[None, :] + grid[i][:, i + 1] * f[None, :]
        b = grid[i + 1][:, i] * (1 - f)[None, :] + grid[i + 1][:, i + 1] * f[None, :]
        acc += amp * (a * (1 - f)[:, None] + b * f[:, None])
        total += amp
        amp *= 0.6
    n = acc / total
    base = np.array([0.55, 0.47, 0.38], np.float32)
    img = base[None, None, :] * (0.65 + 0.7 * n[..., None])
    # pebbles: small dark and light discs
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(900):
        cx, cy = rng.integers(0, size, 2)
        r = rng.uniform(1.5, 5.0)
        tone = rng.choice([0.45, 0.6, 1.25, 1.4])
        y0, y1 = max(0, cy - 6), min(size, cy + 7)
        x0, x1 = max(0, cx - 6), min(size, cx + 7)
        m = (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2 <= r * r
        img[y0:y1, x0:x1][m] *= tone
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def build_scene():
    """The rover (our own model, rover.glb) on a textured sand square under a sun.

    Returns (scene, info) with info = {"center", "radius", "rover", "ground"}; the
    radius is half the rover's bounding-box diagonal.
    """
    scene = tp.Scene()
    scene.background = tp.Color(*BACKGROUND)

    gl = tp.GLTFLoader().load(os.path.join(os.path.dirname(HERE), "rover", "rover.glb"))
    rover = gl.scene

    def shadows(o):
        if isinstance(o, tp.Mesh):
            o.cast_shadow = True
            o.receive_shadow = True

    rover.traverse(shadows)
    scene.add(rover)
    rover.update_matrix_world(True)
    box = tp.Box3().set_from_object(rover, True)
    size, center = box.get_size(), box.get_center()
    # stand it on y = 0
    floor = box.min().y
    rover.position.y -= floor
    center.y -= floor
    radius = 0.5 * math.sqrt(size.x ** 2 + size.y ** 2 + size.z ** 2)

    side = 8.0 * radius
    ground = tp.Mesh(tp.PlaneGeometry(side, side),
                     tp.MeshStandardMaterial())
    ground.material.map = tp.data_texture(_ground_texture())
    ground.material.roughness = 0.95
    ground.material.metalness = 0.0
    ground.rotation.x = -math.pi / 2
    ground.receive_shadow = True
    ground.name = "ground"
    scene.add(ground)

    sun = tp.DirectionalLight(0xfff2e0, 3.2)
    sun.position.set(center.x + 4 * radius, 6 * radius, center.z + 2.5 * radius)
    aim = tp.Object3D()
    aim.position.set(center.x, center.y, center.z)
    scene.add(aim)
    sun.set_target(aim)
    sun.cast_shadow = True
    scene.add(sun)
    scene.add(tp.HemisphereLight(0xb8c8e0, 0x4a3c30, 0.9))

    scene.update_matrix_world(True)
    return scene, {"center": (center.x, center.y, center.z), "radius": radius,
                   "rover": rover, "ground": ground}


def setup_renderer(renderer):
    """The deferred frame as a camera would deliver it, and the same every run."""
    renderer.dlss = False
    renderer.fsr = False
    renderer.render_scale = 1.0
    renderer.auto_exposure = False
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = 1.0
    renderer.sim_time = 0.0  # the scene is still; pin every clock the frame reads


def rig_poses(center, radius, count, seed_phase=0.0, elev_deg=(8.0, 68.0), dist=(2.1, 3.0)):
    """Cameras on a golden-angle spiral around the subject, all looking at it.

    Each pose is (position, target). Elevation runs evenly through elev_deg; the
    distance alternates between the two shells in `dist` (units of the radius).
    A different seed_phase gives held-out positions between the training ones.
    """
    golden = math.pi * (3.0 - math.sqrt(5.0))
    cx, cy, cz = center
    poses = []
    for i in range(count):
        t = (i + 0.5) / count
        elev = math.radians(elev_deg[0] + t * (elev_deg[1] - elev_deg[0]))
        az = i * golden + seed_phase
        d = radius * (dist[0] if i % 2 == 0 else dist[1])
        px = cx + d * math.cos(elev) * math.cos(az)
        py = cy + d * math.sin(elev)
        pz = cz + d * math.cos(elev) * math.sin(az)
        poses.append(((px, py, pz), (cx, cy * 0.6, cz)))
    return poses


def place(camera, pose):
    (px, py, pz), (tx, ty, tz) = pose
    camera.position.set(px, py, pz)
    camera.look_at(tx, ty, tz)
    camera.update_matrix_world(True)


def matrix_world_np(obj):
    """An Object3D's matrix_world as a row-major 4x4 numpy array."""
    m = obj.matrix_world
    m = m() if callable(m) else m
    e = np.array(m.elements(), np.float64)  # column-major, as three.js
    out = e.reshape(4, 4).T
    p = obj.position
    assert np.allclose(out[:3, 3], (p.x, p.y, p.z), atol=1e-4) or obj.parent is not None
    return out


def load_transforms(root):
    with open(os.path.join(root, "transforms.json"), encoding="utf-8") as f:
        return json.load(f)


def write_points_ply(path, xyz, rgb):
    """A colour point cloud (x y z float, red green blue uchar), binary little-endian."""
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())
