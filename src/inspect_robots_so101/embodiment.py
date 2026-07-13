"""``SOArmEmbodiment`` — Inspect Robots embodiment for a LeRobot SO-ARM follower.

Wraps the LeRobot SO follower driver (SO-100 / SO-101). Designed for real-robot
reality:

* **Safety backstop** — every command is clamped to the configured joint limits
  inside :meth:`step`, *independently* of any Inspect Robots ``Approver`` (so unclamped
  model outputs can never reach the motors). This is layered on top of LeRobot's
  own ``max_relative_target`` slew limit, which the driver applies.
* **Operator-in-the-loop success** — there is no privileged oracle; when the
  operator signals end-of-episode the embodiment returns
  ``StepResult(terminated=True, termination_reason="success"|"failure")``, which is
  the only path that reaches the scorer.
* **Self-paced** — declares ``SELF_PACED`` and sleeps to the control rate inside
  :meth:`step` (the framework does not pace for us).
* **No interactive calibration** — the default driver connects with
  ``calibrate=False`` and fails fast (:func:`_check_calibrated`) when the arm
  isn't calibrated for ``SOArmConfig.robot_id``, instead of dropping into
  lerobot's blocking, arm-moving calibration prompt mid-:meth:`reset`.

The driver is injected (``driver_factory``) and so are the clock / sleep / operator
seams, so the whole embodiment runs in tests with no serial port, no motors, no
cameras, and no stdin. The real driver — a connected ``lerobot`` SO follower whose
``get_observation`` already returns both motor positions *and* camera frames — is
built in a pragma'd default that only executes on hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_so101 import packing
from inspect_robots_so101.config import SOArmConfig, action_box, observation_space
from inspect_robots_so101.operator import OperatorIO, default_poll_end

if TYPE_CHECKING:
    from types import TracebackType

ImageMap = Mapping[str, npt.NDArray[np.uint8]]
Vec = npt.NDArray[np.float64]


@runtime_checkable
class SOArmDriver(Protocol):
    """The minimal LeRobot-robot surface the embodiment needs.

    Satisfied directly by ``lerobot.robots.so_follower.SOFollower`` (and any other
    LeRobot ``Robot``): observations are dicts of ``"<motor>.pos"`` floats plus
    camera frames keyed by camera name; actions are dicts of ``"<motor>.pos"``.
    """

    def get_observation(self) -> Mapping[str, Any]: ...

    def send_action(self, action: Mapping[str, float]) -> Mapping[str, Any]: ...

    def disconnect(self) -> None: ...


DriverFactory = Callable[[SOArmConfig], SOArmDriver]


def _check_calibrated(robot: Any, cfg: SOArmConfig) -> None:
    """Refuse to run with an uncalibrated arm, loudly and before any motion.

    ``connect(calibrate=False)`` does **not** raise on missing/mismatched
    calibration — without this guard the failure would surface mid-``reset()`` as
    an opaque error at the first normalized read/write. This also catches a
    calibration file that exists but no longer matches the motors' EEPROM.
    """
    if not robot.is_calibrated:
        raise RuntimeError(
            f"SO-ARM follower (robot_id={cfg.robot_id!r}) is not calibrated: no calibration "
            f"matching the motors at {getattr(robot, 'calibration_fpath', '<unknown>')}. "
            f"Calibrate first with `lerobot-calibrate --robot.type={cfg.robot_type} "
            f"--robot.port={cfg.port} --robot.id=<your-id>`, then set "
            "SOArmConfig.robot_id (and calibration_dir if you used a custom one) to match."
        )


def _default_driver_factory(cfg: SOArmConfig) -> SOArmDriver:  # pragma: no cover - real hardware
    from pathlib import Path

    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    robot = SOFollower(
        SOFollowerRobotConfig(
            id=cfg.robot_id,
            calibration_dir=Path(cfg.calibration_dir) if cfg.calibration_dir else None,
            port=cfg.port,
            cameras=dict(cfg.camera_configs or {}),
            max_relative_target=cfg.max_relative_target,
            use_degrees=cfg.use_degrees,
            disable_torque_on_disconnect=cfg.disable_torque_on_disconnect,
        )
    )
    # NEVER auto-calibrate: lerobot's default connect(calibrate=True) drops into a
    # BLOCKING interactive calibration that moves the arm — mid-reset(), unprompted.
    robot.connect(calibrate=False)
    _check_calibrated(robot, cfg)
    return cast(SOArmDriver, robot)


class SOArmEmbodiment:
    """Inspect Robots embodiment for a single SO-ARM follower (joint-position control)."""

    def __init__(
        self,
        config: SOArmConfig | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else SOArmConfig.from_kwargs(**flat)
        self._driver_factory: DriverFactory = driver_factory or _default_driver_factory
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end: Callable[[], bool] = poll_end or default_poll_end
        self._sleep: Callable[[float], None] = sleep_fn or time.sleep
        self._clock: Callable[[], float] = clock or time.perf_counter

        self._driver: SOArmDriver | None = None
        self._instruction: str | None = None
        self._last_state: Vec | None = None
        self._t_last = 0.0
        self.num_steps = 0

        self.info = EmbodimentInfo(
            name="so_arm",
            action_space=action_box(low=self._cfg.low, high=self._cfg.high),
            observation_space=observation_space(
                self._cfg.cam_height, self._cfg.cam_width, self._cfg.cameras
            ),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect (if needed), drive to home, and block on operator readiness."""
        self._last_state = None
        if self._driver is None:
            self._driver = self._driver_factory(self._cfg)
        if self._cfg.home_pose is not None:
            self._send(np.asarray(self._cfg.home_pose, dtype=np.float64))
        self._operator.wait_ready()
        self._instruction = scene.instruction
        self.num_steps = 0
        self._t_last = self._clock()
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Clamp + command one action, pace to the control rate, then maybe end."""
        self._require_driver()
        self.num_steps += 1
        cmd = packing.validate_dim(action.data)
        if self._cfg.joints_are_delta:
            assert self._last_state is not None
            cmd = self._last_state + cmd
        self._send(cmd)
        self._pace()

        obs = self._observe(self._instruction)
        if self._poll_end():
            success = self._operator.confirm_success()
            return StepResult(
                observation=obs,
                terminated=True,
                termination_reason="success" if success else "failure",
                info={"operator_confirmed": success},
            )
        return StepResult(observation=obs, terminated=False)

    def close(self) -> None:
        """Disconnect and release the driver (no-op if never connected).

        The handle is cleared even if ``disconnect()`` raises, so a failed
        shutdown can't leave a half-dead driver behind (``close()`` stays
        idempotent either way).
        """
        if self._driver is not None:
            try:
                self._driver.disconnect()
            finally:
                self._driver = None
                self._last_state = None

    def __enter__(self) -> SOArmEmbodiment:
        """Support ``with SOArmEmbodiment(...) as emb:`` for guaranteed shutdown."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _require_driver(self) -> SOArmDriver:
        if self._driver is None:  # pragma: no cover - reset() always connects first
            raise RuntimeError("step() called before reset()")
        return self._driver

    def _send(self, cmd: Vec) -> None:
        """Clamp to joint limits (safety backstop) and command the motors."""
        clamped = np.clip(cmd, self._cfg.low, self._cfg.high)
        self._require_driver().send_action(packing.to_action_dict(clamped))

    def _pace(self) -> None:
        hz = self._cfg.control_hz
        if hz and hz > 0:
            elapsed = self._clock() - self._t_last
            self._sleep(max(0.0, 1.0 / hz - elapsed))
        self._t_last = self._clock()

    def _observe(self, instruction: str | None) -> Observation:
        raw = self._require_driver().get_observation()
        state = packing.from_obs_dict(raw)
        self._last_state = state
        images = {cam: np.asarray(raw[cam], dtype=np.uint8) for cam in self._cfg.cameras}
        return Observation(
            images=images,
            state={packing.STATE_KEY: state},
            instruction=instruction,
        )
