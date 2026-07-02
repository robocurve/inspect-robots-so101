"""Tests for LeRobotPolicy (the inference seam is injected — no torch, no lerobot)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from inspect_robots_so101 import packing
from inspect_robots_so101.config import LeRobotPolicyConfig
from inspect_robots_so101.policy import LeRobotPolicy


def _obs(instruction: str | None = "do it") -> Observation:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return Observation(
        images={"front": img},
        state={"joint_pos": np.zeros(6)},
        instruction=instruction,
    )


def _fake_predict(actions: np.ndarray):
    captured: dict[str, Any] = {}

    def _predict(obs: Any) -> np.ndarray:
        captured["obs"] = obs
        return actions

    return _predict, captured


def test_info_and_config_zero_arg() -> None:
    pol = LeRobotPolicy()
    assert pol.info.name == "lerobot"
    assert pol.info.action_space.dim == 6
    assert pol.info.action_space.semantics is not None
    assert pol.info.action_space.semantics.control_mode == "joint_pos"
    assert pol.info.control_hz is None  # load-bearing: keeps compat warning-free
    assert pol.info.observation_space.state_keys == frozenset({"joint_pos"})
    assert pol.config.action_horizon == 50


def test_act_builds_lerobot_obs_and_chunk() -> None:
    actions = np.arange(2 * 6, dtype=float).reshape(2, 6)
    predict, captured = _fake_predict(actions)
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction="pick up the cube"))
    chunk = pol.act(_obs())

    assert len(chunk) == 2
    assert np.array_equal(chunk.actions[0].data, actions[0])
    assert chunk.control_hz is None  # in-process: embodiment paces
    assert chunk.inference_latency_s is not None
    obs = captured["obs"]
    assert obs["task"] == "pick up the cube"
    assert obs["observation.state"].dtype == np.float32
    assert "observation.images.front" in obs
    assert pol.num_inferences == 1


def test_act_uses_empty_instruction_when_none() -> None:
    predict, captured = _fake_predict(np.zeros((1, 6)))
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction=None))
    pol.act(_obs(instruction=None))
    assert captured["obs"]["task"] == ""


def test_act_empty_actions_raises() -> None:
    predict, _ = _fake_predict(np.zeros((0, 6)))
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="empty action chunk"):
        pol.act(_obs())


def test_act_wrong_action_width_raises() -> None:
    predict, _ = _fake_predict(np.zeros((2, 7)))
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match=r"expected \(N, 6\)"):
        pol.act(_obs())


def test_act_missing_camera_raises() -> None:
    predict, _ = _fake_predict(np.zeros((1, 6)))
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    obs = Observation(images={}, state={"joint_pos": np.zeros(6)})
    with pytest.raises(ValueError, match="missing camera"):
        pol.act(obs)


def test_act_missing_state_raises() -> None:
    predict, _ = _fake_predict(np.zeros((1, 6)))
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    obs = Observation(images={"front": np.zeros((4, 4, 3), np.uint8)}, state={})
    with pytest.raises(ValueError, match="missing state key"):
        pol.act(obs)


def test_config_object_overrides_flat() -> None:
    pol = LeRobotPolicy(LeRobotPolicyConfig(chunk_size=3))
    assert pol.config.action_horizon == 3
    assert packing.TOTAL_DIM == 6  # sanity
