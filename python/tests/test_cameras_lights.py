import math

import numpy as np
import pytest

import threepp as tp


def test_perspective_camera():
    cam = tp.PerspectiveCamera(60, 1.5, 0.1, 100)
    assert cam.fov == 60
    assert cam.aspect == 1.5
    assert cam.near == pytest.approx(0.1)  # three.js alias for near_plane
    assert cam.far == 100
    cam.fov = 75
    cam.update_projection_matrix()  # should not raise
    assert cam.fov == 75


def test_perspective_defaults():
    cam = tp.PerspectiveCamera()
    assert cam.fov == 60
    assert cam.near == pytest.approx(0.1)
    assert cam.far == 2000


def test_orthographic_camera():
    cam = tp.OrthographicCamera(-2, 2, 2, -2, 0.1, 50)
    assert (cam.left, cam.right, cam.top, cam.bottom) == (-2, 2, 2, -2)
    cam.update_projection_matrix()


def test_camera_inherits_object3d():
    cam = tp.PerspectiveCamera()
    cam.position.set(1, 2, 3)
    assert cam.position.z == 3
    cam.look_at(0, 0, 0)  # should not raise


def test_camera_matrices_shapes_and_inverse():
    cam = tp.PerspectiveCamera(60, 16 / 9, 0.1, 100)
    cam.position.set(0, 0, 5)
    cam.look_at(0, 0, 0)
    cam.update_projection_matrix()
    cam.update_matrix_world(True)
    P = cam.projection_matrix.to_numpy()       # intrinsics (clip-from-view)
    V = cam.matrix_world_inverse.to_numpy()    # view (world -> camera)
    M = cam.matrix_world.to_numpy()            # extrinsics (camera -> world)
    assert P.shape == V.shape == M.shape == (4, 4)
    # view is the inverse of the camera pose
    assert np.allclose(V @ M, np.eye(4), atol=1e-4)
    # extrinsics translation == camera world position
    assert np.allclose(M[:3, 3], [0, 0, 5], atol=1e-4)
    # perspective intrinsics: fy = cot(fov/2), fx = fy / aspect
    fy = 1.0 / math.tan(math.radians(60) / 2)
    assert P[1, 1] == pytest.approx(fy, abs=1e-3)
    assert P[0, 0] == pytest.approx(fy / (16 / 9), abs=1e-3)


def test_project_unproject_roundtrip():
    cam = tp.PerspectiveCamera(60, 1.0, 0.1, 100)
    cam.position.set(0, 0, 5)
    cam.look_at(0, 0, 0)
    cam.update_projection_matrix()
    cam.update_matrix_world(True)
    # world -> NDC (in front of the camera, so inside the [-1,1] frustum)
    ndc = tp.Vector3(1.0, 0.5, 0.0).project(cam)
    assert -1 <= ndc.x <= 1 and -1 <= ndc.y <= 1
    # NDC -> world round-trips back to the original point
    back = tp.Vector3(ndc.x, ndc.y, ndc.z).unproject(cam)
    assert back.x == pytest.approx(1.0, abs=1e-3)
    assert back.y == pytest.approx(0.5, abs=1e-3)
    assert back.z == pytest.approx(0.0, abs=1e-3)


def test_lights_color_intensity():
    light = tp.DirectionalLight(0xff0000, 2.5)
    assert light.color == tp.Color(0xff0000)
    assert light.intensity == 2.5
    light.color = 0x00ff00
    light.intensity = 1.0
    assert light.color == tp.Color(0x00ff00)


def test_ambient_default_white():
    assert tp.AmbientLight().color == tp.Color(0xffffff)


def test_point_light_fields():
    p = tp.PointLight(0xffffff, 1.0, 10.0, 2.0)
    assert p.distance == 10.0
    assert p.decay == 2.0
    p.distance = 5.0
    assert p.distance == 5.0


def test_spot_light_fields():
    s = tp.SpotLight(0xffffff, 1.0)
    assert abs(s.angle - math.pi / 3) < 1e-5  # default angle
    s.angle = 0.5
    s.penumbra = 0.2
    assert s.angle == 0.5


def test_hemisphere_ground_color():
    h = tp.HemisphereLight(0xffffff, 0x444444, 1.0)
    assert h.ground_color == tp.Color(0x444444)


def test_directional_set_target():
    light = tp.DirectionalLight()
    target = tp.Object3D()
    light.set_target(target)  # should not raise


def test_light_position():
    light = tp.DirectionalLight()
    light.position.set(3, 5, 2)
    assert (light.position.x, light.position.y, light.position.z) == (3, 5, 2)
