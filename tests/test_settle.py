"""Settle-before-observe tests for the single-arm SO follower."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Action, StepResult

from conftest import Clock, SettleDriver, _build
from inspect_robots_so101 import packing
from inspect_robots_so101.config import SOArmConfig
from inspect_robots_so101.embodiment import SOArmEmbodiment
from inspect_robots_so101.operator import OperatorIO

POLL_S = 0.01
PACE_S = 1.0 / 30.0
MAX_POLLS = 100
READ_ADVANCE_S = 0.05
TARGET = np.full(6, 0.125)
ARM_SLOTS = tuple(range(5))


def _far(index: int = 0) -> np.ndarray:
    offset = np.zeros(6)
    offset[index] = 0.5
    return offset


def _settled_cfg(**kwargs: Any) -> SOArmConfig:
    kwargs.setdefault("cam_height", 4)
    kwargs.setdefault("cam_width", 4)
    kwargs.setdefault("settle_tolerance", 0.05)
    return SOArmConfig(**kwargs)


def _step(emb: SOArmEmbodiment, sleeps: list[float]) -> StepResult:
    sleeps.clear()
    return emb.step(Action(data=TARGET.copy()))


def test_settle_state_is_initialized_during_construction() -> None:
    emb = SOArmEmbodiment(SOArmConfig(cam_height=4, cam_width=4, settle_tolerance=0.05))
    assert emb.settle_timeouts == 0
    assert emb._settle_disabled is False


def test_disabled_by_default_adds_no_reads_or_info() -> None:
    driver = SettleDriver(converge_after=3)
    emb, _driver, sleeps = _build(driver=driver)
    emb.reset(Scene(id="s", instruction="go"))
    reads_before = driver.observation_reads

    result = _step(emb, sleeps)

    assert driver.observation_reads - reads_before == 1
    assert sleeps == [PACE_S]
    assert result.info == {}


def test_settle_runs_before_pace_and_observe() -> None:
    driver = SettleDriver(converge_after=3)
    emb, _driver, sleeps = _build(_settled_cfg(), driver=driver)
    emb.reset(Scene(id="s", instruction="go"))
    reads_before = driver.observation_reads

    _step(emb, sleeps)

    assert sleeps == [POLL_S, POLL_S, PACE_S]
    assert driver.observation_reads - reads_before == 4


def test_already_converged_step_has_zero_settle_sleeps() -> None:
    driver = SettleDriver(converge_after=1)
    emb, _driver, sleeps = _build(
        _settled_cfg(control_hz=0.0),
        driver=driver,
    )
    emb.reset(Scene(id="s", instruction="go"))

    result = _step(emb, sleeps)

    assert sleeps == []
    assert result.info["settled"] is True
    assert result.info["settle_residual"] == pytest.approx(0.0)
    assert result.info["settle_timeouts"] == 0
    assert "settle_disabled" not in result.info


@pytest.mark.parametrize("index", ARM_SLOTS)
def test_frozen_clock_timeout_uses_poll_cap_for_every_arm_slot(index: int) -> None:
    driver = SettleDriver(converge_after=1, offset=_far(index))
    emb, _driver, sleeps = _build(_settled_cfg(), driver=driver)
    emb.reset(Scene(id="s", instruction="go"))
    reads_before = driver.observation_reads

    result = _step(emb, sleeps)

    assert driver.observation_reads - reads_before == MAX_POLLS + 1
    assert sleeps.count(POLL_S) == MAX_POLLS - 1
    assert result.info["settled"] is False
    assert result.info["settle_residual"] == pytest.approx(0.5)
    assert result.info["settle_timeouts"] == 1


def test_clock_advancing_reads_timeout_by_elapsed_time() -> None:
    clock = Clock()
    driver = SettleDriver(
        converge_after=1,
        offset=_far(),
        clock=clock,
        read_advance=READ_ADVANCE_S,
    )
    emb, _driver, sleeps = _build(_settled_cfg(), driver=driver, clock=clock)
    emb.reset(Scene(id="s", instruction="go"))
    reads_before = driver.observation_reads

    result = _step(emb, sleeps)

    elapsed_bound_polls = int(1.0 / READ_ADVANCE_S)
    assert elapsed_bound_polls < MAX_POLLS
    assert driver.observation_reads - reads_before == elapsed_bound_polls + 1
    assert sleeps.count(POLL_S) == elapsed_bound_polls - 1
    assert result.info["settled"] is False


def test_reset_settles_before_readiness_and_first_observation() -> None:
    driver = SettleDriver(converge_after=2)
    reads_at_ready: list[int] = []

    def _ready(_prompt: str) -> str:
        reads_at_ready.append(driver.observation_reads)
        return ""

    cfg = _settled_cfg(
        control_hz=0.0,
        home_pose=(1.0,) * 6,
        max_relative_target=10.0,
    )
    emb, _driver, sleeps = _build(
        cfg,
        driver=driver,
        operator=OperatorIO(input_fn=_ready, output_fn=lambda _m: None),
    )

    emb.reset(Scene(id="s", instruction="go"))

    # _home() reads the current pose (1 read) before interpolating, then settle
    # polls converge_after=2 times (2 reads). Total before wait_ready = 3.
    # After wait_ready, _observe() adds 1 more read.
    assert reads_at_ready == [3]
    assert driver.observation_reads == 4
    assert sleeps == [POLL_S]


def test_gripper_divergence_does_not_block_settling() -> None:
    gripper_offset = np.zeros(6)
    gripper_offset[5] = 50.0
    driver = SettleDriver(converge_after=1, offset=gripper_offset)
    emb, _driver, sleeps = _build(
        _settled_cfg(control_hz=0.0),
        driver=driver,
    )
    emb.reset(Scene(id="s", instruction="go"))

    result = _step(emb, sleeps)

    assert result.info["settled"] is True
    assert result.info["settle_residual"] == pytest.approx(0.0)
    assert sleeps == []


def test_arm_mask_excludes_only_the_gripper() -> None:
    from inspect_robots_so101.embodiment import _ARM_SLOTS

    assert set(_ARM_SLOTS.tolist()) == set(range(6)) - {5}


def test_settle_targets_driver_clipped_echo_in_delta_mode() -> None:
    driver = SettleDriver(converge_after=1, max_relative_target=1.0)
    emb, _driver, sleeps = _build(
        _settled_cfg(
            control_hz=0.0,
            joints_are_delta=True,
            settle_timeout_s=0.02,
        ),
        driver=driver,
    )
    emb.reset(Scene(id="s", instruction="go"))

    result = emb.step(Action(data=np.full(6, 10.0)))

    assert np.array_equal(driver.commands[-1], np.ones(6))
    assert result.info["settled"] is True
    assert result.info["settle_residual"] == pytest.approx(0.0)
    assert sleeps == []


def test_budget_exhaustion_disables_trial_and_reset_rearms(
    caplog: pytest.LogCaptureFixture,
) -> None:
    driver = SettleDriver(converge_after=1)
    cfg = _settled_cfg(
        control_hz=0.0,
        home_pose=(0.0,) * 6,
        max_relative_target=10.0,
        settle_timeout_s=0.01,
        settle_timeout_budget=2,
    )
    emb, _driver, sleeps = _build(cfg, driver=driver)
    emb.reset(Scene(id="one", instruction="go"))

    driver.offset = _far()
    first = _step(emb, sleeps)
    assert first.info["settle_timeouts"] == 1

    driver.offset = np.zeros(6)
    good = _step(emb, sleeps)
    assert good.info["settled"] is True
    assert good.info["settle_timeouts"] == 1

    driver.offset = _far()
    with caplog.at_level(logging.WARNING, logger="inspect_robots_so101.embodiment"):
        tripped = _step(emb, sleeps)
    assert tripped.info["settled"] is False
    assert tripped.info["settle_timeouts"] == 2
    assert tripped.info["settle_disabled"] is True

    reads_before = driver.observation_reads
    after = _step(emb, sleeps)
    assert driver.observation_reads - reads_before == 1
    assert after.info == {"settle_timeouts": 2, "settle_disabled": True}
    assert sleeps == []

    driver.offset = np.zeros(6)
    reads_before = driver.observation_reads
    emb.reset(Scene(id="two", instruction="again"))
    # _home() reads current pose (1 read) + settle converges in 1 read = 2,
    # plus _observe() at end of reset = 1. Total: 3 reads.
    assert driver.observation_reads - reads_before == 3
    assert emb.settle_timeouts == 0
    assert emb._settle_disabled is False

    rearmed = _step(emb, sleeps)
    assert rearmed.info["settled"] is True
    assert rearmed.info["settle_timeouts"] == 0


@pytest.mark.parametrize(
    ("use_degrees", "unit"),
    [(True, "degrees"), (False, "normalized")],
)
def test_budget_warning_names_worst_motor_residual_and_units(
    use_degrees: bool,
    unit: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    driver = SettleDriver(converge_after=1, offset=_far(2))
    emb, _driver, sleeps = _build(
        _settled_cfg(
            control_hz=0.0,
            use_degrees=use_degrees,
            settle_timeout_s=0.01,
            settle_timeout_budget=1,
        ),
        driver=driver,
    )
    emb.reset(Scene(id="s", instruction="go"))

    with caplog.at_level(logging.WARNING, logger="inspect_robots_so101.embodiment"):
        result = _step(emb, sleeps)
        _step(emb, sleeps)

    records = [record for record in caplog.records if "budget exhausted" in record.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert packing.MOTORS[2] in message
    assert "residual=0.5000" in message
    assert unit in message
    assert result.info["settle_disabled"] is True


def test_residual_exactly_at_tolerance_is_settled() -> None:
    tolerance = 0.0625
    driver = SettleDriver(converge_after=1, offset=_far() * (tolerance / 0.5))
    emb, _driver, sleeps = _build(
        _settled_cfg(control_hz=0.0, settle_tolerance=tolerance),
        driver=driver,
    )
    emb.reset(Scene(id="s", instruction="go"))

    result = _step(emb, sleeps)

    assert result.info["settle_residual"] == tolerance
    assert result.info["settled"] is True


def test_terminal_result_keeps_settle_info() -> None:
    driver = SettleDriver(converge_after=1)
    emb, _driver, sleeps = _build(
        _settled_cfg(control_hz=0.0),
        driver=driver,
        poll_end_seq=[True],
    )
    emb.reset(Scene(id="s", instruction="go"))

    result = _step(emb, sleeps)

    assert result.terminated is True
    assert result.info["settled"] is True
    assert result.info["settle_residual"] == pytest.approx(0.0)
    assert result.info["settle_timeouts"] == 0
