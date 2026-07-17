"""Goal-closing test: Inspect Robots is provably compatible with an SO-ARM follower +
a LeRobot policy — zero errors, zero warnings — and built-in tasks are realizable."""

from __future__ import annotations

from inspect_robots.compat import check_compatibility
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.spaces import ActionSemantics, Box

from inspect_robots_so101.config import (
    LeRobotPolicyConfig,
    SOArmConfig,
    action_box,
    observation_space,
)
from inspect_robots_so101.embodiment import SOArmEmbodiment
from inspect_robots_so101.policy import LeRobotPolicy


def test_lerobot_soarm_compatible_no_errors_no_warnings() -> None:
    report = check_compatibility(LeRobotPolicy(), SOArmEmbodiment())
    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []


def test_normalized_lerobot_soarm_pair_declares_matching_contract() -> None:
    policy = LeRobotPolicy(LeRobotPolicyConfig(use_degrees=False))
    embodiment = SOArmEmbodiment(SOArmConfig(use_degrees=False))
    report = check_compatibility(policy, embodiment)
    assert report.ok is True
    assert report.issues == []
    assert policy.info.observation_space.state == embodiment.info.observation_space.state
    assert policy.info.observation_space.state is not None
    assert policy.info.observation_space.state.fields[0].unit == "normalized"


def test_builtin_task_is_realizable() -> None:
    task = resolve("task", "cubepick-reach")
    report = check_compatibility(LeRobotPolicy(), SOArmEmbodiment(), task)
    assert report.ok is True, [i.message for i in report.errors]
    assert report.errors == []


def test_negative_wrong_dim_policy_trips_action_dim_error() -> None:
    bad = PolicyInfo(
        name="bad",
        action_space=Box(shape=(8,), semantics=ActionSemantics(control_mode="joint_pos")),
    )

    class _Bad:
        info = bad
        config = PolicyConfig()

        def reset(self, scene: object) -> None: ...
        def act(self, obs: object) -> object: ...  # pragma: no cover

    report = check_compatibility(_Bad(), SOArmEmbodiment())  # type: ignore[arg-type]
    assert report.ok is False
    assert any(i.code == "action_dim" for i in report.errors)


def test_negative_policy_advertising_rate_trips_control_rate_warning() -> None:
    # Locks the load-bearing detail: leaving PolicyInfo.control_hz=None is the only
    # reason the real pairing is warning-free. A policy that advertises a rate warns.
    info = PolicyInfo(
        name="rated",
        action_space=action_box(),
        observation_space=observation_space(480, 640, ("front",)),
        control_hz=100.0,
    )

    class _Rated:
        config = PolicyConfig()

        def __init__(self) -> None:
            self.info = info

        def reset(self, scene: object) -> None: ...
        def act(self, obs: object) -> object: ...  # pragma: no cover

    report = check_compatibility(_Rated(), SOArmEmbodiment())  # type: ignore[arg-type]
    assert report.ok is True  # only a warning, not an error
    assert any(i.code == "control_rate" for i in report.warnings)
