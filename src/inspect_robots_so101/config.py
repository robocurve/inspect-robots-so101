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

from inspect_robots_so101.packing import NUM_JOINTS, STATE_KEY, STATE_SPEC, TOTAL_DIM

_T = TypeVar("_T", bound="_FromKwargs")

# Conservative default action limits: the five revolute joints in [-180, 180]
# degrees, the gripper in [0, 100]. These are SAFETY limits — override with your
# real, calibrated SO-ARM joint limits before trusting them on hardware.
_DEFAULT_LOW: tuple[float, ...] = (-180.0,) * NUM_JOINTS + (0.0,)
_DEFAULT_HIGH: tuple[float, ...] = (180.0,) * NUM_JOINTS + (100.0,)

DEFAULT_CAMERAS: tuple[str, ...] = ("front",)


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
    robot_type: str = "so101_follower"  # or "so100_follower"
    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    control_hz: float = 30.0
    cam_height: int = 480
    cam_width: int = 640
    joint_low: tuple[float, ...] = _DEFAULT_LOW
    joint_high: tuple[float, ...] = _DEFAULT_HIGH
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
        for name in ("joint_low", "joint_high"):
            if len(getattr(self, name)) != TOTAL_DIM:
                raise ValueError(f"{name} must have {TOTAL_DIM} entries")
        if self.home_pose is not None and len(self.home_pose) != TOTAL_DIM:
            raise ValueError(f"home_pose must have {TOTAL_DIM} entries")

    @property
    def low(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.joint_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
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


def observation_space(height: int, width: int, names: tuple[str, ...]) -> ObservationSpace:
    """The shared observation space: the configured cameras + packed 6-D ``joint_pos``."""
    return ObservationSpace(
        cameras=camera_specs(height, width, names),
        state_keys=frozenset({STATE_KEY}),
        state=STATE_SPEC,
    )
