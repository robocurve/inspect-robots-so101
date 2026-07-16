"""Configuration for the SO-ARM embodiment and the LeRobot policy client.

Both configs are frozen dataclasses with defaults that match a stock LeRobot SO
follower (SO-100 / SO-101) and a SmolVLA checkpoint, so zero-arg construction
"just works" for `.info` / preflight. Each exposes :meth:`from_kwargs` so the
adapters accept flat scalar keyword arguments — this is what lets
``inspect-robots run -P pretrained_path=... -E port=...`` configure them, since the
Inspect Robots CLI only forwards scalar ``key=value`` pairs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
)

from inspect_robots_so101 import packing
from inspect_robots_so101.packing import NUM_JOINTS, STATE_KEY, TOTAL_DIM

_T = TypeVar("_T", bound="_FromKwargs")

# Conservative default action limits: the five revolute joints in [-180, 180]
# degrees, the gripper in [0, 100]. These are SAFETY limits — override with your
# real, calibrated SO-ARM joint limits before trusting them on hardware.
_DEGREE_LOW: tuple[float, ...] = (-180.0,) * NUM_JOINTS + (0.0,)
_DEGREE_HIGH: tuple[float, ...] = (180.0,) * NUM_JOINTS + (100.0,)
_NORMALIZED_LOW: tuple[float, ...] = (-100.0,) * NUM_JOINTS + (0.0,)
_NORMALIZED_HIGH: tuple[float, ...] = (100.0,) * NUM_JOINTS + (100.0,)

DEFAULT_CAMERAS: tuple[str, ...] = ("front",)

# The SO follower variants lerobot ships. At lerobot v0.5.x both names alias the
# SAME driver + config classes (SO100Follower = SO101Follower = SOFollower), so
# `robot_type` is a documented, validated label — it does not change runtime
# behavior. It exists so configs stay self-describing and so a future lerobot
# that splits the classes has an obvious wiring point.
VALID_ROBOT_TYPES: tuple[str, ...] = ("so101_follower", "so100_follower")


class _FromKwargs:
    """Mixin: build a frozen dataclass from flat scalar kwargs (CLI-friendly)."""

    @classmethod
    def from_kwargs(cls: type[_T], **flat: Any) -> _T:
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(flat) - names
        if unknown:
            raise TypeError(f"{cls.__name__} got unexpected config keys: {sorted(unknown)}")
        return cls(**flat)


@dataclass(frozen=True)
class SOArmConfig(_FromKwargs):
    """Static configuration for a single SO-ARM follower embodiment."""

    port: str = "/dev/ttyACM0"
    robot_type: str = "so101_follower"  # or "so100_follower" (see VALID_ROBOT_TYPES)
    # The lerobot robot id: selects the calibration file
    # (<calibration_dir>/<robot_id>.json, written by `lerobot-calibrate`). Without
    # it lerobot would look for "None.json" — set it to the id you calibrated with.
    robot_id: str | None = None
    # Where lerobot stores calibration files. None -> lerobot's default
    # (~/.cache/huggingface/lerobot/calibration/robots/<robot_type>/).
    calibration_dir: str | None = None
    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    control_hz: float = 30.0
    cam_height: int = 480
    cam_width: int = 640
    # The degree tuples are identity sentinels as well as the public defaults.
    # __post_init__ replaces only omitted defaults in normalized mode, preserving
    # explicit calibrated tuples in either mode.
    joint_low: tuple[float, ...] = _DEGREE_LOW
    joint_high: tuple[float, ...] = _DEGREE_HIGH
    home_pose: tuple[float, ...] | None = None
    joints_are_delta: bool = False
    use_degrees: bool = True
    max_relative_target: float | None = None
    disable_torque_on_disconnect: bool = True
    # LeRobot CameraConfig objects keyed by camera name, used only by the default
    # (hardware) driver factory. Opaque to this package; not CLI-settable. Their
    # keys should match ``cameras``. ``None`` means "build the robot with no
    # cameras" — fine for preflight, but a real run needs camera streams.
    camera_configs: Any = None

    def __post_init__(self) -> None:
        if not self.use_degrees:
            if self.joint_low is _DEGREE_LOW:
                object.__setattr__(self, "joint_low", _NORMALIZED_LOW)
            if self.joint_high is _DEGREE_HIGH:
                object.__setattr__(self, "joint_high", _NORMALIZED_HIGH)
        for name in ("joint_low", "joint_high"):
            if len(getattr(self, name)) != TOTAL_DIM:
                raise ValueError(f"{name} must have {TOTAL_DIM} entries")
        if self.home_pose is not None and len(self.home_pose) != TOTAL_DIM:
            raise ValueError(f"home_pose must have {TOTAL_DIM} entries")
        if self.robot_type not in VALID_ROBOT_TYPES:
            raise ValueError(
                f"robot_type must be one of {VALID_ROBOT_TYPES}, got {self.robot_type!r}"
            )
        if self.home_pose is not None and self.max_relative_target is None:
            # Homing sends ONE absolute command; without lerobot's
            # max_relative_target slew limit the arm would slam to home at full
            # speed from wherever it is. Interpolated homing is tracked as an issue.
            raise ValueError(
                "home_pose without max_relative_target would command a full-speed "
                "jump to the home pose; set SOArmConfig.max_relative_target (native "
                "motor units per step) to slew-limit it, or unset home_pose"
            )

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Return the configured lower motor limits as a float64 array."""
        return np.asarray(self.joint_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Return the configured upper motor limits as a float64 array."""
        return np.asarray(self.joint_high, dtype=np.float64)


@dataclass(frozen=True)
class LeRobotPolicyConfig(_FromKwargs):
    """Static configuration for a LeRobot policy loaded from a checkpoint."""

    pretrained_path: str = "lerobot/smolvla_base"
    policy_type: str = "smolvla"  # act, smolvla, pi0, pi05, diffusion, ...
    device: str = "cuda"
    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    state_key: str = STATE_KEY
    # Max actions consumed per inference: ``act()`` truncates the model's chunk to
    # its first ``chunk_size`` actions (mirrors the async policy server's
    # ``actions_per_chunk``) and advertises it as ``PolicyConfig.action_horizon``.
    # Distinct from the framework-side ``DefaultController.replan_interval``, which
    # caps how many actions of an already-returned chunk get executed per replan.
    chunk_size: int = 50
    cam_height: int = 480
    cam_width: int = 640
    # Must match SOArmConfig.use_degrees so the policy's declared observation
    # contract reflects the motor positions it receives.
    use_degrees: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")


# The action *semantics* both the policy and the embodiment declare. Compatibility
# checking compares control_mode + rotation_repr (errors) and gripper + frame
# (warnings); declaring this single constant on both sides guarantees a clean check.
ACTION_SEMANTICS = ActionSemantics(
    control_mode="joint_pos",
    rotation_repr="none",
    gripper="continuous",
    frame="base",
)


def camera_specs(height: int, width: int, names: tuple[str, ...]) -> tuple[CameraSpec, ...]:
    """Build CameraSpecs for the given names at one resolution (single source of truth)."""
    return tuple(CameraSpec(name=n, height=height, width=width, channels=3) for n in names)


def action_box(
    low: npt.NDArray[np.float64] | None = None,
    high: npt.NDArray[np.float64] | None = None,
) -> Box:
    """The shared 6-D joint-position action space. ``low``/``high`` are optional
    safety limits (the embodiment supplies them; the policy leaves them unset)."""
    return Box(shape=(TOTAL_DIM,), low=low, high=high, semantics=ACTION_SEMANTICS)


def observation_space(
    height: int,
    width: int,
    names: tuple[str, ...],
    *,
    use_degrees: bool = True,
) -> ObservationSpace:
    """Build cameras and packed motor state in the configured native units."""
    return ObservationSpace(
        cameras=camera_specs(height, width, names),
        state_keys=frozenset({STATE_KEY}),
        state=packing.state_spec(use_degrees=use_degrees),
    )
