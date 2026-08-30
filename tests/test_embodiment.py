"""Tests for SOArmEmbodiment (all hardware/IO seams injected — no serial, motors, stdin)."""

from __future__ import annotations

import numpy as np
import pytest
from inspect_robots.embodiment import SELF_PACED
from inspect_robots.scene import Scene
from inspect_robots.types import OPERATOR_END, Action

from conftest import FakeDriver, _build, _operator
from inspect_robots_so101.config import SOArmConfig
from inspect_robots_so101.embodiment import SOArmEmbodiment, _check_calibrated


def test_zero_arg_info_no_hardware() -> None:
    emb = SOArmEmbodiment()  # nothing mocked: construction must not touch hardware
    assert emb.info.name == "so_arm"
    assert emb.info.action_space.dim == 6
    assert emb.info.action_space.low is not None and emb.info.action_space.high is not None
    assert emb.info.control_hz == 30.0
    assert SELF_PACED in emb.info.capabilities
    assert emb.info.observation_space.camera_names == frozenset({"front"})
    assert emb.info.observation_space.state_keys == frozenset({"joint_pos"})


def test_normalized_info_uses_normalized_state_and_limits() -> None:
    emb = SOArmEmbodiment(SOArmConfig(use_degrees=False))
    state = emb.info.observation_space.state
    assert state is not None and state.fields[0].unit == "normalized"
    assert emb.info.action_space.low is not None
    assert emb.info.action_space.high is not None
    assert np.array_equal(emb.info.action_space.low, [-100.0] * 5 + [0.0])
    assert np.array_equal(emb.info.action_space.high, [100.0] * 6)


def test_reset_returns_observation_and_homes() -> None:
    cfg = SOArmConfig(home_pose=(5.0,) * 6, max_relative_target=10.0)
    emb, drv, _ = _build(cfg)
    obs = emb.reset(Scene(id="s", instruction="reach"))
    assert set(obs.images) == {"front"}
    assert obs.state["joint_pos"].shape == (6,)
    assert obs.instruction == "reach"
    assert len(drv.commands) == 1  # homing command issued


def test_reset_interpolates_homing_when_far() -> None:
    """Each intermediate command differs from the previous by at most max_relative_target."""
    # Mixed directions: joints 0-2 start below home, joint 3 is already at home,
    # joints 4-5 start above home (negative direction).
    home = (10.0, 10.0, 10.0, 5.0, 0.0, 0.0)
    start = np.array([0.0, 0.0, 0.0, 5.0, 8.0, 8.0])
    cfg = SOArmConfig(
        home_pose=home,
        max_relative_target=4.0,
        joint_low=(-20.0,) * 6,
        joint_high=(20.0,) * 6,
    )
    driver = FakeDriver(state=start)
    emb, drv, _ = _build(cfg, driver=driver)
    emb.reset(Scene(id="s", instruction="reach"))

    commands = np.stack(drv.commands)  # shape (n_steps, 6)
    # Every step must stay within max_relative_target of the prior step.
    for i in range(1, len(commands)):
        per_joint_delta = np.abs(commands[i] - commands[i - 1])
        assert np.all(per_joint_delta <= 4.0 + 1e-9), (
            f"step {i}: delta {per_joint_delta} exceeded max_relative_target"
        )
    # Final command must land exactly on home_pose.
    assert commands[-1] == pytest.approx(list(home))


def test_reset_homing_raises_on_non_finite_observation() -> None:
    cfg = SOArmConfig(home_pose=(5.0,) * 6, max_relative_target=2.0)
    driver = FakeDriver(state=np.full(6, np.nan))
    emb = SOArmEmbodiment(cfg, driver_factory=lambda _c: driver, operator=_operator())
    with pytest.raises(RuntimeError, match="non-finite values"):
        emb.reset(Scene(id="s", instruction="reach"))


def test_reset_homing_raises_when_start_out_of_limits() -> None:
    """_home() raises if the observed start pose is outside limits by more than step_limit."""
    cfg = SOArmConfig(
        home_pose=(5.0,) * 6,
        max_relative_target=1.0,
        joint_low=(0.0,) * 6,
        joint_high=(10.0,) * 6,
    )
    # Start far above joint_high (150 >> 10 + 1.0).
    driver = FakeDriver(state=np.full(6, 150.0))
    emb = SOArmEmbodiment(cfg, driver_factory=lambda _c: driver, operator=_operator())
    with pytest.raises(RuntimeError, match="outside joint limits"):
        emb.reset(Scene(id="s", instruction="reach"))


def test_homing_first_step_is_paced() -> None:
    """The first homing step must be paced like every other step.

    Regression test: before the fix, _t_last was 0.0 from __init__.  With a
    real clock that had advanced (e.g. startup cost), elapsed >> period so
    _pace() slept 0.0 for the very first step — doubling the intended slew rate.
    After the fix, _t_last is reset to clock() immediately before the homing
    loop, so elapsed ≈ 0 and sleep ≈ period for the first step.
    """
    control_hz = 30.0
    period = 1.0 / control_hz

    # Simulate a clock that has already advanced 100 s since __init__.
    # We return T0=100.0 for the _t_last reset call inside _home(), then
    # T0 + period for every subsequent call so that each _pace() pair sees
    # elapsed = period and sleeps max(0, period - period) = 0.
    # But the key regression: WITHOUT the _t_last reset, elapsed on the first
    # _pace() would be (100 + period) - 0 >> period, yielding sleep = 0.
    # WITH the fix, elapsed = (100 + period) - 100 = period, sleep = 0.
    #
    # To make the test directly observable, we use a strictly-increasing clock
    # where consecutive calls differ by period/2 so each _pace() pair spans
    # exactly period/2, and sleep = period - period/2 = period/2 > 0.
    _t = [100.0]

    def _clock() -> float:
        t = _t[0]
        _t[0] += period / 2
        return t

    cfg = SOArmConfig(
        home_pose=(10.0,) * 6,
        max_relative_target=2.0,
        joint_low=(-20.0,) * 6,
        joint_high=(20.0,) * 6,
        control_hz=control_hz,
        cam_height=4,
        cam_width=4,
    )
    sleeps: list[float] = []
    emb = SOArmEmbodiment(
        cfg,
        driver_factory=lambda _c: FakeDriver(),
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=sleeps.append,
        clock=_clock,
    )
    emb.reset(Scene(id="s", instruction="reach"))

    # Homing 0→10 in steps of 2 → 5 commands → 5 _pace() calls.
    assert len(sleeps) >= 5, f"expected ≥5 sleeps, got {len(sleeps)}"
    # With the _t_last reset before the loop, elapsed per pace = period/2,
    # so sleep = period - period/2 = period/2 for every step including the first.
    # Without the fix, sleep[0] = max(0, period - (100+period/2 - 0.0)) ≈ 0.
    expected_sleep = period / 2
    for i, s in enumerate(sleeps[:5]):
        assert s == pytest.approx(expected_sleep, abs=1e-9), (
            f"sleep[{i}] = {s:.9f}s; expected ~{expected_sleep:.9f}s — "
            "first step must be paced (regression: was 0.0 before _t_last reset fix)"
        )


def test_observation_records_monotonic_capture_times() -> None:
    times = iter([10.0, 10.25])
    emb = SOArmEmbodiment(
        SOArmConfig(cam_height=4, cam_width=4),
        driver_factory=lambda _c: FakeDriver(),
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _d: None,
        clock=lambda: next(times),
    )

    obs = emb.reset(Scene(id="s", instruction="reach"))

    assert obs.image_times == {"front": 10.25}
    assert obs.state_time == 10.25


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


def test_step_clamps_in_normalized_mode() -> None:
    emb, drv, _ = _build(SOArmConfig(use_degrees=False))
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.array([1000.0, -1000.0, 0.0, 0.0, 0.0, -1.0])))
    assert np.array_equal(drv.commands[-1], [100.0, -100.0, 0.0, 0.0, 0.0, 0.0])


def test_step_delta_mode_adds_current() -> None:
    drv = FakeDriver(state=np.full(6, 10.0))
    cfg = SOArmConfig(joints_are_delta=True)
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(6, 1.0)))
    emb.step(Action(data=np.full(6, 1.0)))
    # current 10 + delta 1 = 11 (within limits)
    assert drv.commands[0][0] == pytest.approx(11.0)
    assert drv.commands[1][0] == pytest.approx(12.0)
    # One read at reset and one post-action read per step; no extra delta-mode read.
    assert drv.observation_reads == 3


def test_reset_twice_reuses_driver() -> None:
    calls = {"n": 0}

    def _factory(_c):
        calls["n"] += 1
        return FakeDriver()

    emb = SOArmEmbodiment(
        SOArmConfig(cam_height=4, cam_width=4),
        driver_factory=_factory,
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )
    emb.reset(Scene(id="s", instruction="x"))
    emb.reset(Scene(id="s", instruction="x"))
    assert calls["n"] == 1  # driver built once, reused on the second reset


def test_step_terminates_operator_end_without_grading_prompt() -> None:
    prompts: list[str] = []
    emb, _, _ = _build(poll_end_seq=[True], operator=_operator(prompts=prompts))
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(6)))
    assert result.terminated is True
    assert result.termination_reason == OPERATOR_END
    assert result.info == {}
    assert prompts == ["Position the scene, then press Enter to start..."]


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


def test_close_releases_driver_even_if_disconnect_raises() -> None:
    class ExplodingDriver(FakeDriver):
        def disconnect(self) -> None:
            raise RuntimeError("serial port yanked")

    drv = ExplodingDriver()
    emb, _, _ = _build(driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    with pytest.raises(RuntimeError, match="serial port yanked"):
        emb.close()
    emb.close()  # handle was cleared despite the raise: now a no-op


def test_context_manager_closes_on_exit() -> None:
    emb, drv, _ = _build()
    with emb as entered:
        assert entered is emb
        emb.reset(Scene(id="s", instruction="x"))
    assert drv.disconnected is True


def test_context_manager_closes_on_exception() -> None:
    emb, drv, _ = _build()
    with pytest.raises(RuntimeError, match="boom"), emb:
        emb.reset(Scene(id="s", instruction="x"))
        raise RuntimeError("boom")
    assert drv.disconnected is True


@pytest.mark.parametrize("shape", [(2, 4, 3), (4, 4, 1)])
def test_observe_rejects_wrong_camera_shape(shape: tuple[int, ...]) -> None:
    driver = FakeDriver(frame_shape=shape)
    emb, _, _ = _build(driver=driver)

    expected = (4, 4, 3)
    with pytest.raises(ValueError) as exc:
        emb.reset(Scene(id="s", instruction="x"))
    assert str(exc.value) == f"camera 'front' returned shape {shape}, expected {expected}"


def test_check_calibrated_passes_when_calibrated() -> None:
    robot = type("R", (), {"is_calibrated": True, "calibration_fpath": "/tmp/my_arm.json"})()
    _check_calibrated(robot, SOArmConfig(robot_id="my_arm"))  # no raise


def test_check_calibrated_raises_actionable_error() -> None:
    robot = type("R", (), {"is_calibrated": False, "calibration_fpath": "/cal/my_arm.json"})()
    with pytest.raises(RuntimeError, match="lerobot-calibrate") as exc:
        _check_calibrated(robot, SOArmConfig(robot_id="my_arm"))
    msg = str(exc.value)
    assert "/cal/my_arm.json" in msg  # names the calibration file it looked for
    assert "robot_id='my_arm'" in msg
    assert "--robot.type=so101_follower" in msg
