"""Tests for SOArmEmbodiment (all hardware/IO seams injected — no serial, motors, stdin)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from inspect_robots.embodiment import SELF_PACED
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_so101 import packing
from inspect_robots_so101.config import SOArmConfig
from inspect_robots_so101.embodiment import SOArmEmbodiment
from inspect_robots_so101.operator import OperatorIO


class FakeDriver:
    """Stand-in for a LeRobot SO follower: dict obs of '<motor>.pos' + cameras."""

    def __init__(self, state: np.ndarray | None = None) -> None:
        self.state = np.zeros(6) if state is None else np.asarray(state, dtype=float)
        self.commands: list[np.ndarray] = []
        self.disconnected = False

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = packing.to_action_dict(self.state)
        obs["front"] = np.zeros((4, 4, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.commands.append(packing.from_obs_dict(action))
        return action

    def disconnect(self) -> None:
        self.disconnected = True


def _operator(answers: list[str] | None = None) -> OperatorIO:
    seq = list(answers or [""])
    return OperatorIO(input_fn=lambda _p: seq.pop(0), output_fn=lambda _m: None)


def _build(
    cfg: SOArmConfig | None = None,
    *,
    driver: FakeDriver | None = None,
    poll_end_seq: list[bool] | None = None,
    operator: OperatorIO | None = None,
):
    drv = driver or FakeDriver()
    polls = list(poll_end_seq or [False])
    sleeps: list[float] = []
    emb = SOArmEmbodiment(
        cfg or SOArmConfig(),
        driver_factory=lambda _c: drv,
        operator=operator or _operator(),
        poll_end=lambda: polls.pop(0) if polls else False,
        sleep_fn=sleeps.append,
        clock=lambda: 0.0,
    )
    return emb, drv, sleeps


def test_zero_arg_info_no_hardware() -> None:
    emb = SOArmEmbodiment()  # nothing mocked: construction must not touch hardware
    assert emb.info.name == "so_arm"
    assert emb.info.action_space.dim == 6
    assert emb.info.action_space.low is not None and emb.info.action_space.high is not None
    assert emb.info.control_hz == 30.0
    assert SELF_PACED in emb.info.capabilities
    assert emb.info.observation_space.camera_names == frozenset({"front"})
    assert emb.info.observation_space.state_keys == frozenset({"joint_pos"})


def test_reset_returns_observation_and_homes() -> None:
    cfg = SOArmConfig(home_pose=(5.0,) * 6)
    emb, drv, _ = _build(cfg)
    obs = emb.reset(Scene(id="s", instruction="reach"))
    assert set(obs.images) == {"front"}
    assert obs.state["joint_pos"].shape == (6,)
    assert obs.instruction == "reach"
    assert len(drv.commands) == 1  # homing command issued


def test_reset_without_home_pose_issues_no_command() -> None:
    emb, drv, _ = _build()
    emb.reset(Scene(id="s", instruction="x"))
    assert drv.commands == []


def test_step_clamps_to_limits() -> None:
    emb, drv, _ = _build()
    emb.reset(Scene(id="s", instruction="x"))
    # Way out of bounds; joints clip to +/-180, gripper to [0, 100].
    emb.step(Action(data=np.full(6, 1000.0)))
    cmd = drv.commands[-1]
    assert cmd[0] == pytest.approx(180.0)  # joint clamped
    assert cmd[5] == pytest.approx(100.0)  # gripper clamped


def test_step_clamps_low_side() -> None:
    emb, drv, _ = _build()
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(6, -1000.0)))
    cmd = drv.commands[-1]
    assert cmd[0] == pytest.approx(-180.0)
    assert cmd[5] == pytest.approx(0.0)  # gripper floor


def test_step_delta_mode_adds_current() -> None:
    drv = FakeDriver(state=np.full(6, 10.0))
    cfg = SOArmConfig(joints_are_delta=True)
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(6, 1.0)))
    # current 10 + delta 1 = 11 (within limits)
    assert drv.commands[-1][0] == pytest.approx(11.0)


def test_reset_twice_reuses_driver() -> None:
    calls = {"n": 0}

    def _factory(_c):
        calls["n"] += 1
        return FakeDriver()

    emb = SOArmEmbodiment(
        SOArmConfig(),
        driver_factory=_factory,
        operator=_operator(["", ""]),
        poll_end=lambda: False,
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )
    emb.reset(Scene(id="s", instruction="x"))
    emb.reset(Scene(id="s", instruction="x"))
    assert calls["n"] == 1  # driver built once, reused on the second reset


def test_step_terminates_success_on_operator_yes() -> None:
    emb, _, _ = _build(poll_end_seq=[True], operator=_operator(["", "y"]))
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(6)))
    assert result.terminated is True
    assert result.termination_reason == "success"
    assert result.info["operator_confirmed"] is True


def test_step_terminates_failure_on_operator_no() -> None:
    emb, _, _ = _build(poll_end_seq=[True], operator=_operator(["", "n"]))
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(6)))
    assert result.terminated is True
    assert result.termination_reason == "failure"


def test_step_continues_when_no_end_signal() -> None:
    emb, _, _ = _build(poll_end_seq=[False])
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(6)))
    assert result.terminated is False
    assert emb.num_steps == 1


def test_pacing_sleeps_to_control_rate() -> None:
    emb, _, sleeps = _build()  # control_hz=30 -> period ~0.0333, clock constant 0
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.zeros(6)))
    assert sleeps and sleeps[-1] == pytest.approx(1.0 / 30.0)


def test_pacing_skipped_when_hz_zero() -> None:
    cfg = SOArmConfig(control_hz=0.0)
    emb, _, sleeps = _build(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.zeros(6)))
    assert sleeps == []  # no sleep attempted at hz=0


def test_close_idempotent_and_releases() -> None:
    emb, drv, _ = _build()
    emb.close()  # before connect: no error
    emb.reset(Scene(id="s", instruction="x"))
    emb.close()
    assert drv.disconnected is True
    emb.close()  # second close: no error
