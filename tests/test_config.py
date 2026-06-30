"""Tests for SOArmConfig / LeRobotPolicyConfig."""

from __future__ import annotations

import pytest
from robolens.spaces import CameraSpec

from robolens_soarm.config import (
    DEFAULT_CAMERAS,
    LeRobotPolicyConfig,
    SOArmConfig,
    camera_specs,
)


def test_soarm_defaults() -> None:
    cfg = SOArmConfig()
    assert cfg.port == "/dev/ttyACM0"
    assert cfg.robot_type == "so101_follower"
    assert cfg.control_hz == 30.0
    assert cfg.cameras == DEFAULT_CAMERAS
    assert cfg.low.shape == (6,)
    assert cfg.high.shape == (6,)
    # gripper slot (index 5) bounded [0, 100]; joints bounded by +/-180 degrees.
    assert cfg.low[5] == 0.0 and cfg.high[5] == 100.0
    assert cfg.low[0] == pytest.approx(-180.0)


def test_policy_defaults() -> None:
    cfg = LeRobotPolicyConfig()
    assert cfg.policy_type == "smolvla"
    assert cfg.pretrained_path == "lerobot/smolvla_base"
    assert cfg.state_key == "joint_pos"
    assert cfg.cameras == DEFAULT_CAMERAS
    assert cfg.chunk_size == 50


def test_from_kwargs_populates_scalars() -> None:
    cfg = LeRobotPolicyConfig.from_kwargs(pretrained_path="my/ckpt", policy_type="act")
    assert cfg.pretrained_path == "my/ckpt"
    assert cfg.policy_type == "act"


def test_soarm_from_kwargs() -> None:
    cfg = SOArmConfig.from_kwargs(port="/dev/ttyUSB0", control_hz=25.0)
    assert cfg.port == "/dev/ttyUSB0"
    assert cfg.control_hz == 25.0


def test_from_kwargs_rejects_unknown() -> None:
    with pytest.raises(TypeError, match="unexpected config keys"):
        LeRobotPolicyConfig.from_kwargs(nope=1)


def test_soarm_rejects_bad_joint_limits() -> None:
    with pytest.raises(ValueError, match="joint_low must have 6 entries"):
        SOArmConfig(joint_low=(0.0,) * 5)


def test_soarm_rejects_bad_home_pose() -> None:
    with pytest.raises(ValueError, match="home_pose must have 6 entries"):
        SOArmConfig(home_pose=(0.0,) * 4)


def test_soarm_accepts_valid_home_pose() -> None:
    cfg = SOArmConfig(home_pose=(0.0,) * 6)
    assert cfg.home_pose is not None and len(cfg.home_pose) == 6


def test_camera_specs() -> None:
    specs = camera_specs(480, 640, ("front", "wrist"))
    assert len(specs) == 2
    assert all(isinstance(s, CameraSpec) for s in specs)
    assert specs[0].name == "front"
    assert specs[0].height == 480 and specs[0].width == 640
