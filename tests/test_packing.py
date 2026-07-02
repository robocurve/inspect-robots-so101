"""Tests for the pure 6-D SO-ARM packing module."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_so101 import packing


def test_constants() -> None:
    assert packing.TOTAL_DIM == 6
    assert packing.NUM_JOINTS == 5
    assert packing.GRIPPER_IDX == 5
    assert packing.STATE_KEY == "joint_pos"
    assert packing.MOTORS[0] == "shoulder_pan"
    assert packing.MOTORS[-1] == "gripper"


def test_state_spec_keys_match_state_key() -> None:
    assert packing.STATE_SPEC.keys == frozenset({"joint_pos"})


def test_motor_keys() -> None:
    keys = packing.motor_keys()
    assert keys == (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )


def test_validate_dim_accepts_correct_length() -> None:
    out = packing.validate_dim(list(range(6)))
    assert out.shape == (6,)
    assert out.dtype == np.float64


def test_validate_dim_flattens() -> None:
    out = packing.validate_dim(np.zeros((1, 6)))
    assert out.shape == (6,)


def test_validate_dim_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected a 6-D vector"):
        packing.validate_dim(np.zeros(8))


def test_to_action_dict_maps_motors_in_order() -> None:
    out = packing.to_action_dict(np.arange(6, dtype=float))
    assert out == {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 1.0,
        "elbow_flex.pos": 2.0,
        "wrist_flex.pos": 3.0,
        "wrist_roll.pos": 4.0,
        "gripper.pos": 5.0,
    }


def test_to_action_dict_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected a 6-D vector"):
        packing.to_action_dict(np.zeros(5))


def test_from_obs_dict_reads_named_positions() -> None:
    obs = {f"{m}.pos": float(i) for i, m in enumerate(packing.MOTORS)}
    obs["front"] = np.zeros((4, 4, 3))  # cameras ignored
    vec = packing.from_obs_dict(obs)
    assert np.array_equal(vec, np.arange(6, dtype=float))


def test_from_obs_dict_roundtrips_to_action_dict() -> None:
    vec = np.array([10.0, -20.0, 30.0, -40.0, 50.0, 60.0])
    assert np.array_equal(packing.from_obs_dict(packing.to_action_dict(vec)), vec)


def test_from_obs_dict_missing_motor_raises() -> None:
    obs = {f"{m}.pos": 0.0 for m in packing.MOTORS[:-1]}  # drop gripper
    with pytest.raises(KeyError, match="missing motor position"):
        packing.from_obs_dict(obs)
