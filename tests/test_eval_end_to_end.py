"""End-to-end: full eval() rollouts on a mocked SO-ARM and LeRobot policy.

The test proves the judgement-based wiring (operator end -> before_scoring ->
operator judgement -> operator scorer) and chunk replay compose. The inline
task keeps the suite self-contained without hardware, lerobot, or torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from inspect_robots import Scene, Task, TrialRecord, operator_scorer
from inspect_robots import eval as rl_eval

from inspect_robots_so101 import packing
from inspect_robots_so101.config import LeRobotPolicyConfig, SOArmConfig
from inspect_robots_so101.embodiment import SOArmEmbodiment
from inspect_robots_so101.operator import OperatorIO
from inspect_robots_so101.policy import LeRobotPolicy


class _FakeDriver:
    def __init__(self) -> None:
        self.state = np.zeros(6)

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = packing.to_action_dict(self.state)
        obs["front"] = np.zeros((4, 4, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.state = packing.from_obs_dict(action)
        return action

    def disconnect(self) -> None: ...


def _predict(_obs: Any) -> np.ndarray:
    return np.zeros((1, 6), dtype=np.float32)  # one-action chunk of zeros


def _grade_yes(record: TrialRecord, scene: Scene) -> None:
    del scene
    assert record.termination_reason == "operator_end"
    record.operator_judgement = "y"


@pytest.mark.parametrize("use_degrees", [True, False])
def test_eval_scores_success_end_to_end(use_degrees: bool) -> None:
    policy = LeRobotPolicy(
        LeRobotPolicyConfig(
            cam_height=4,
            cam_width=4,
            chunk_size=1,
            use_degrees=use_degrees,
        ),
        predict_fn=_predict,
    )
    embodiment = SOArmEmbodiment(
        SOArmConfig(cam_height=4, cam_width=4, use_degrees=use_degrees),
        driver_factory=lambda _c: _FakeDriver(),
        operator=OperatorIO(input_fn=lambda _p: "", output_fn=lambda _m: None),
        poll_end=lambda: True,  # operator ends every episode immediately
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )
    task = Task(
        name="so101-operator-e2e",
        scenes=[Scene(id="operator-e2e", instruction="reach")],
        scorer=operator_scorer(),
        max_steps=1,
    )

    logs = rl_eval(
        task,
        policy,
        embodiment,
        sinks=[],
        seed=0,
        before_scoring=_grade_yes,
    )

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.results.metrics["operator"] == 1.0
