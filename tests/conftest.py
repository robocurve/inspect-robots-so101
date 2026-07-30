"""Shared SO-ARM driver, clock, operator, and embodiment test helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np

from inspect_robots_so101 import packing
from inspect_robots_so101.config import SOArmConfig
from inspect_robots_so101.embodiment import SOArmEmbodiment
from inspect_robots_so101.operator import OperatorIO


class Clock:
    """Fake monotonic clock advanced explicitly by test collaborators."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.now += seconds


class FakeDriver:
    """Immediate SO follower stand-in with motor dictionaries and one camera."""

    def __init__(
        self,
        state: np.ndarray | None = None,
        *,
        frame_shape: tuple[int, ...] = (4, 4, 3),
    ) -> None:
        self.state = np.zeros(6) if state is None else np.asarray(state, dtype=float)
        self.frame_shape = frame_shape
        self.commands: list[np.ndarray] = []
        self.observation_reads = 0
        self.disconnected = False

    def get_observation(self) -> dict[str, Any]:
        """Return the current motor state and a synthetic camera frame."""
        self.observation_reads += 1
        obs: dict[str, Any] = packing.to_action_dict(self.state)
        obs["front"] = np.zeros(self.frame_shape, dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Accept, echo, and immediately reach the requested action."""
        accepted = packing.from_obs_dict(action)
        self.state = accepted.copy()
        self.commands.append(accepted.copy())
        return packing.to_action_dict(accepted)

    def disconnect(self) -> None:
        """Mark the fake hardware as disconnected."""
        self.disconnected = True


class SettleDriver(FakeDriver):
    """Driver that reaches an accepted target after a configured read count.

    ``offset`` models steady-state error after convergence. ``max_relative_target``
    models LeRobot's internal slew truncation and is reflected in the action
    mapping returned by :meth:`send_action`.
    """

    def __init__(
        self,
        *,
        state: np.ndarray | None = None,
        converge_after: int = 1,
        offset: np.ndarray | None = None,
        clock: Clock | None = None,
        read_advance: float = 0.0,
        max_relative_target: float | None = None,
    ) -> None:
        super().__init__(state)
        self.offset = np.zeros(6) if offset is None else np.asarray(offset, dtype=float)
        self._converge_after = converge_after
        self._clock = clock
        self._read_advance = read_advance
        self._max_relative_target = max_relative_target
        self._target: np.ndarray | None = None
        self._reads_since_command = 0

    def get_observation(self) -> dict[str, Any]:
        """Advance convergence, charge read time, and return an observation."""
        self.observation_reads += 1
        if self._clock is not None and self._read_advance:
            self._clock.advance(self._read_advance)
        if self._target is not None:
            self._reads_since_command += 1
            if self._reads_since_command >= self._converge_after:
                self.state = self._target + self.offset
        obs: dict[str, Any] = packing.to_action_dict(self.state)
        obs["front"] = np.zeros((4, 4, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Record and echo the possibly slew-truncated action the driver accepts."""
        requested = packing.from_obs_dict(action)
        accepted = requested
        if self._max_relative_target is not None:
            delta = np.clip(
                requested - self.state,
                -self._max_relative_target,
                self._max_relative_target,
            )
            accepted = self.state + delta
        self.commands.append(accepted.copy())
        self._target = accepted.copy()
        self._reads_since_command = 0
        return packing.to_action_dict(accepted)


def _operator(*, prompts: list[str] | None = None) -> OperatorIO:
    def _input(prompt: str) -> str:
        if prompts is not None:
            prompts.append(prompt)
        return ""

    return OperatorIO(input_fn=_input, output_fn=lambda _m: None)


def _build(
    cfg: SOArmConfig | None = None,
    *,
    driver: FakeDriver | None = None,
    poll_end_seq: list[bool] | None = None,
    operator: OperatorIO | None = None,
    clock: Callable[[], float] | None = None,
):
    cfg = dataclasses.replace(
        cfg if cfg is not None else SOArmConfig(),
        cam_height=4,
        cam_width=4,
    )
    drv = driver or FakeDriver()
    polls = list(poll_end_seq or [False])
    sleeps: list[float] = []
    emb = SOArmEmbodiment(
        cfg,
        driver_factory=lambda _c: drv,
        operator=operator or _operator(),
        poll_end=lambda: polls.pop(0) if polls else False,
        sleep_fn=sleeps.append,
        clock=clock or (lambda: 0.0),
    )
    return emb, drv, sleeps
