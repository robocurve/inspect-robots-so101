"""Canonical 6-D joint packing for an SO-ARM follower (SO-100 / SO-101).

A LeRobot SO follower is a single 6-motor arm. LeRobot names the motors and keys
its observations / actions by ``"<motor>.pos"``; Inspect Robots, like the rest of this
package, works in a flat **6-D** vector. This module is the *one* place that
defines how those 6 numbers map to the named motors, so the policy (a LeRobot
model) and the embodiment (the LeRobot driver) can never disagree.

Convention (6-D): ``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper]`` — the five revolute joints in order, gripper last. This is
exactly the motor order of :class:`lerobot.robots.so_follower.SOFollower`.

This module is pure NumPy with no optional/hardware dependencies (no torch, no
``lerobot``), so it imports and tests anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import StateField, StateSpec

# The SO follower motor names, in LeRobot's canonical order (see
# lerobot/robots/so_follower/so_follower.py). The five arm joints then the gripper.
MOTORS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
NUM_JOINTS = 5  # revolute joints
GRIPPER_DOF = 1  # one gripper
TOTAL_DIM = NUM_JOINTS + GRIPPER_DOF  # 6-D
GRIPPER_IDX = TOTAL_DIM - 1  # index 5

# LeRobot keys motor positions as "<motor>.pos" in observations and actions.
POS_SUFFIX = ".pos"

# The canonical proprioception key. The vector is one field so
# ``StateSpec.keys == {"joint_pos"}`` stays consistent with the ``state_keys``
# both components declare for compatibility. Its unit depends on the LeRobot
# driver's ``use_degrees`` mode and is therefore built by :func:`state_spec`.
STATE_KEY = "joint_pos"

Vec = npt.NDArray[np.float64]


def state_spec(*, use_degrees: bool) -> StateSpec:
    """Describe packed motor positions in the configured LeRobot native units.

    Degree mode uses degrees for the five arm joints and LeRobot's 0..100
    normalized gripper position. Normalized mode uses LeRobot's normalized
    positions for every motor (arm joints -100..100, gripper 0..100).
    """
    unit = "deg+normalized" if use_degrees else "normalized"
    return StateSpec(fields=(StateField(key=STATE_KEY, shape=(TOTAL_DIM,), unit=unit),))


def motor_keys() -> tuple[str, ...]:
    """The LeRobot ``"<motor>.pos"`` keys, in canonical order."""
    return tuple(f"{m}{POS_SUFFIX}" for m in MOTORS)


def validate_dim(vec: npt.ArrayLike, n: int = TOTAL_DIM) -> Vec:
    """Return ``vec`` as a 1-D float array, raising ``ValueError`` if not length ``n``."""
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(
            f"expected a {n}-D vector, got shape {np.shape(vec)} ({arr.shape[0]} elems)"
        )
    return arr


def to_action_dict(vec: npt.ArrayLike) -> dict[str, float]:
    """Turn a flat 6-D vector into LeRobot's ``{"<motor>.pos": value}`` action dict."""
    arr = validate_dim(vec)
    return {f"{m}{POS_SUFFIX}": float(arr[i]) for i, m in enumerate(MOTORS)}


def from_obs_dict(obs: Mapping[str, Any]) -> Vec:
    """Extract the flat 6-D joint vector from a LeRobot observation/action dict.

    Reads ``"<motor>.pos"`` for each motor in canonical order. Raises ``KeyError``
    if any motor position is missing (a misconfigured driver).
    """
    try:
        return np.asarray([float(obs[f"{m}{POS_SUFFIX}"]) for m in MOTORS], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"observation missing motor position {exc} (need {motor_keys()})") from exc
