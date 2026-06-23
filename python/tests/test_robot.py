"""URDFLoader + Robot: parse a URDF string, articulate joints, run FK.

Uses inline primitive geometry (box/cylinder) so no external mesh files are
needed — URDFLoader.parse resolves those itself.
"""
import numpy as np
import pytest

import threepp as tp

URDF = """<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base">
    <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
  </link>
  <link name="arm">
    <visual><geometry><cylinder radius="0.05" length="0.5"/></geometry></visual>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="1"/>
  </joint>
</robot>
"""


def _robot():
    return tp.URDFLoader().parse(".", URDF)


def test_urdf_parse_returns_robot():
    robot = _robot()
    assert isinstance(robot, tp.Robot)
    assert isinstance(robot, tp.Object3D)  # inherits the scene-graph base
    assert robot.num_dof == 1


def test_urdf_joint_range_and_clamp():
    robot = _robot()
    rng = robot.get_joint_range(0)
    assert rng.min == pytest.approx(-1.0, abs=1e-4)
    assert rng.max == pytest.approx(1.0, abs=1e-4)
    # values outside the limit are clamped
    robot.set_joint_value(0, 5.0)
    assert robot.get_joint_value(0) == pytest.approx(1.0, abs=1e-4)


def test_urdf_forward_kinematics():
    robot = _robot()
    ee0 = robot.get_end_effector_transform().to_numpy()
    robot.set_joint_value(0, 0.5)
    assert robot.get_joint_value(0) == pytest.approx(0.5, abs=1e-4)
    ee1 = robot.get_end_effector_transform().to_numpy()
    assert ee1.shape == (4, 4)
    assert not np.allclose(ee0, ee1)  # moving the joint moves the end-effector


def test_urdf_compute_fk_is_pure():
    robot = _robot()
    a = robot.compute_end_effector_transform([0.0]).to_numpy()
    b = robot.compute_end_effector_transform([1.0]).to_numpy()
    assert a.shape == (4, 4) and not np.allclose(a, b)  # input changes the pose
    assert robot.get_joint_value(0) == 0.0  # ...without mutating the robot


def test_urdf_joint_info():
    robot = _robot()
    info = robot.get_articulated_joint_info()
    assert len(info) == 1
    assert info[0].type == tp.JointType.Revolute
    assert info[0].name == "j1"
    assert info[0].range is not None


def test_urdf_set_joint_values_vector():
    robot = _robot()
    robot.set_joint_values([0.25])
    assert robot.joint_values() == pytest.approx([0.25], abs=1e-4)
