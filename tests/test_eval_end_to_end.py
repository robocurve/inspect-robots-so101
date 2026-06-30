"""End-to-end: a full eval() rollout on a mocked SO-ARM + LeRobot policy actually
scores success — proving the termination_reason -> scorer wiring and chunk replay
compose (the static compat test cannot show this).

Uses RoboLens's built-in ``cubepick-reach`` task (``success_at_end`` scorer), so
the suite stays self-contained: no kitchenbench, no lerobot, no torch, no hardware.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from robolens import eval as rl_eval

from robolens_soarm import packing
from robolens_soarm.config import LeRobotPolicyConfig, SOArmConfig
from robolens_soarm.embodiment import SOArmEmbodiment
from robolens_soarm.operator import OperatorIO
from robolens_soarm.policy import LeRobotPolicy


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


def _always_yes_operator() -> OperatorIO:
    return OperatorIO(input_fn=lambda _p: "y", output_fn=lambda _m: None)


def test_eval_scores_success_end_to_end() -> None:
    policy = LeRobotPolicy(LeRobotPolicyConfig(chunk_size=1), predict_fn=_predict)
    embodiment = SOArmEmbodiment(
        SOArmConfig(),
        driver_factory=lambda _c: _FakeDriver(),
        operator=_always_yes_operator(),
        poll_end=lambda: True,  # operator ends every episode immediately
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )

    logs = rl_eval("cubepick-reach", policy, embodiment, sinks=[], seed=0)

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.results.metrics["success_at_end"] == 1.0
