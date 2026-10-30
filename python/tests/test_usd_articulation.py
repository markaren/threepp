"""threepp.usd: a hand-written USD arm through the Python binding into PhysX.

The importer's coverage is tests/extras/UsdArticulation_test.cpp; this is the binding's: arguments
reach it, None takes the file's value, and the result drives. Skips on a build without USD. CPU, headless.
"""
import gc
import math

import numpy as np
import pytest
import threepp as tp

pytestmark = pytest.mark.skipif(not getattr(tp, "HAS_USD", False), reason="built without USD and PhysX")

from threepp.usd import UsdArticulation, load_articulation  # noqa: E402

CUBE = """
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0, 1, 3, 2, 2, 3, 7, 6, 6, 7, 5, 4, 4, 5, 1, 0, 2, 6, 4, 0, 7, 3, 1, 5]
            point3f[] points = [(-0.05, -0.05, -0.05), (0.05, -0.05, -0.05), (-0.05, 0.05, -0.05), (0.05, 0.05, -0.05), (-0.05, -0.05, 0.05), (0.05, -0.05, 0.05), (-0.05, 0.05, 0.05), (0.05, 0.05, 0.05)]
"""

ARM = f"""#usda 1.0
(
    defaultPrim = "arm"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "arm" {{
    def PhysicsFixedJoint "root_joint" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI"]
    ) {{
        rel physics:body1 = </arm/base>
    }}

    def Xform "base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    ) {{
        float physics:mass = 5
        def Mesh "visual" {{{CUBE}        }}
        def Mesh "collider" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        ) {{
            uniform token physics:approximation = "convexHull"
            uniform token purpose = "guide"{CUBE}        }}
    }}

    def Xform "link" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    ) {{
        float physics:mass = 2
        double3 xformOp:translate = (0, 0, 0.3)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "visual" {{{CUBE}        }}
        def Mesh "collider" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        ) {{
            uniform token physics:approximation = "convexHull"
            uniform token purpose = "guide"{CUBE}        }}
    }}

    def PhysicsRevoluteJoint "hinge" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    ) {{
        rel physics:body0 = </arm/base>
        rel physics:body1 = </arm/link>
        uniform token physics:axis = "Y"
        point3f physics:localPos0 = (0, 0, 0.1)
        point3f physics:localPos1 = (0, 0, -0.2)
        float physics:lowerLimit = -90
        float physics:upperLimit = 90
        float drive:angular:physics:stiffness = 100
        float drive:angular:physics:damping = 10
    }}

    def PhysicsSphericalJoint "ball" {{
        rel physics:body0 = </arm/base>
        rel physics:body1 = </arm/link>
    }}
}}
"""


@pytest.fixture
def path(tmp_path):
    file = tmp_path / "arm.usda"
    file.write_text(ARM)
    return file


def world():
    return tp.PhysxWorld(fixed_timestep=1 / 240)


def test_imports_what_the_file_says(path):
    w = world()
    robot = load_articulation(w, path)
    assert isinstance(robot, UsdArticulation)
    assert robot.joint_names == ["hinge"]
    assert robot.num_dof == 1
    assert len(robot.positions()) == 1
    assert len(robot.meshes) == 2
    assert isinstance(robot.scene, tp.Group)
    assert any("PhysicsSphericalJoint" in d for d in robot.diagnostics)


def test_link_resolves_prim_path_and_leaf_name(path):
    w = world()
    robot = load_articulation(w, path)
    # Z-up in the file, Y-up here.
    assert robot.articulation.link("/arm/link").position.y == pytest.approx(0.3, abs=1e-5)
    assert robot.articulation.link("link").position.y == pytest.approx(0.3, abs=1e-5)


def test_drive_reaches_a_target_under_gravity(path):
    w = world()
    robot = load_articulation(w, path)
    robot.set_targets([0.7])
    for _ in range(480):
        w.step(1 / 240)
    assert robot.positions()[0] == pytest.approx(0.7, abs=1e-3)


def test_none_keeps_the_file_and_arguments_override_it(path):
    w = world()
    robot = load_articulation(w, path, stiffness=0.0, damping=0.0)
    robot.set_targets([0.7])
    for _ in range(240):
        w.step(1 / 240)
    assert abs(robot.positions()[0]) < 1e-4  # no gain, so the target moves nothing


def test_fixed_base_override_lets_the_robot_fall(path):
    w = world()
    robot = load_articulation(w, path, fixed_base=False)
    start = robot.articulation.link("base").position.y
    for _ in range(24):
        w.step(1 / 240)
    assert robot.articulation.link("base").position.y < start - 0.01


def test_base_pose_moves_the_whole_robot(path):
    w = world()
    half = math.sqrt(0.5)
    robot = load_articulation(w, path, base_position=(1, 2, 3), base_rotation=(0, half, 0, half))
    p = robot.articulation.link("link").position
    assert np.allclose([p.x, p.y, p.z], [1, 2.3, 3], atol=1e-4)


def test_an_articulation_may_outlive_its_world(path):
    # A collected reference cycle destroys the two in no fixed order. The importer
    # sets the authored pose, so the articulation holds a PhysX cache to release.
    w = world()
    articulation, *_ = w.load_usd_articulation(str(path))
    del w
    gc.collect()
    del articulation
    gc.collect()


def test_a_file_without_a_robot_raises(tmp_path):
    w = world()
    with pytest.raises(RuntimeError, match="load_usd_articulation"):
        load_articulation(w, tmp_path / "missing.usda")
