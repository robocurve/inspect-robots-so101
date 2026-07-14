"""``LeRobotPolicy`` — a Inspect Robots policy backed by a LeRobot checkpoint.

LeRobot policies (ACT, SmolVLA, π0, diffusion, …) are ordinary ``nn.Module``\\ s
loaded from the Hugging Face Hub and run **in process** on the GPU. Unlike the
YAM/MolmoAct2 stack (where the model owns its own HTTP server), LeRobot models are
a library you import — so the heavy, GPU-bound dependencies (``torch`` +
``lerobot`` + the checkpoint) live behind a single injectable seam: a
``predict_fn`` that maps a **raw robot-style observation dict** (``"<motor>.pos"``
floats, camera frames keyed by camera name, and ``"task"``) to an action chunk.

The default ``predict_fn`` (``_default_predict``) lazily builds the policy and its
pre/post-processors from a pretrained path and prepares each observation with
lerobot's own ``raw_observation_to_observation`` helper — the same path
``lerobot.async_inference.policy_server`` uses — so torch conversion, HWC→CHW,
0..1 scaling, resizing, and batching stay lerobot's job, not ours. Tests exercise
this wiring with ``sys.modules`` fakes (no torch, no lerobot, no network), and the
``lerobot-seam`` CI job re-imports the real symbols on Python 3.12 to catch
upstream drift. ``import inspect_robots_so101`` never imports torch.

The action chunk this policy returns is already in the robot's **native motor
units** (degrees for the joints, 0..100 for the gripper): LeRobot's postprocessor
unnormalizes for us, so the embodiment commands the values verbatim (after its
safety clamp). There is therefore no gripper renormalization here.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_so101 import packing
from inspect_robots_so101.config import LeRobotPolicyConfig, action_box, observation_space

# A RAW robot-style observation dict -> an (N, action_dim) action-chunk array.
# Keys: "<motor>.pos" floats (packing.motor_keys()), camera frames keyed by the
# camera name, and "task" (the instruction) — exactly what a lerobot Robot's
# get_observation() returns, plus the task string. The default predict_fn feeds
# this to lerobot's raw_observation_to_observation, which packs it into
# "observation.state" / "observation.images.<cam>" tensors itself.
LeRobotObs = Mapping[str, Any]
PredictFn = Callable[[LeRobotObs], npt.NDArray[np.floating[Any]]]

# LeRobot key conventions (see lerobot/utils/constants.py). Hardcoded so this
# module needs no lerobot import.
OBS_STR = "observation"
OBS_IMAGE_PREFIX = "observation.images."
TASK_KEY = "task"


def _import_hw_to_dataset_features() -> Callable[..., dict[str, Any]]:
    """Import lerobot's ``hw_to_dataset_features`` across supported versions.

    lerobot moved it from ``lerobot.datasets.utils`` (v0.5.0) to
    ``lerobot.datasets.feature_utils`` (v0.5.1) to
    ``lerobot.utils.feature_utils`` (v0.6.0); the ``lerobot-seam`` CI job
    guards this against further drift.
    """
    try:
        from lerobot.utils.feature_utils import hw_to_dataset_features
    except ImportError:
        try:  # lerobot 0.5.1 - 0.5.x
            from lerobot.datasets.feature_utils import hw_to_dataset_features
        except ImportError:  # lerobot == 0.5.0
            from lerobot.datasets.utils import hw_to_dataset_features
    return hw_to_dataset_features  # type: ignore[no-any-return]


def _hw_observation_features(cfg: LeRobotPolicyConfig) -> dict[str, Any]:
    """Reconstruct the SO follower's ``observation_features`` mapping.

    The policy never holds a lerobot ``Robot`` object (the embodiment owns the
    driver), so we synthesize the same mapping the driver would report —
    ``"<motor>.pos" -> float`` for each motor in canonical order plus
    ``"<camera>" -> (height, width, 3)`` for each configured camera — for
    ``hw_to_dataset_features`` to turn into lerobot dataset features.
    """
    features: dict[str, Any] = dict.fromkeys(packing.motor_keys(), float)
    for cam in cfg.cameras:
        features[cam] = (cfg.cam_height, cfg.cam_width, 3)
    return features


def _default_predict(cfg: LeRobotPolicyConfig) -> PredictFn:
    """Build an in-process LeRobot inference closure from a pretrained checkpoint.

    Mirrors ``lerobot.async_inference.policy_server``: load the policy class from
    the factory, build the matching pre/post-processor pipelines, then per call
    ``raw_observation_to_observation`` → preprocess → ``predict_action_chunk`` →
    postprocess back to native motor units.
    """
    import torch
    from lerobot.async_inference.helpers import raw_observation_to_observation
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    hw_to_dataset_features = _import_hw_to_dataset_features()

    policy = get_policy_class(cfg.policy_type).from_pretrained(cfg.pretrained_path)
    policy.to(cfg.device)
    policy.eval()  # torch nn.Module inference mode (not Python's eval builtin)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
        postprocessor_overrides={"device_processor": {"device": cfg.device}},
    )

    # The dataset-feature spec raw_observation_to_observation uses to pack raw
    # "<motor>.pos" keys into "observation.state" and camera frames into
    # "observation.images.<cam>" (use_video=False: these are live frames).
    lerobot_features = hw_to_dataset_features(
        _hw_observation_features(cfg), OBS_STR, use_video=False
    )
    policy_image_features = dict(policy.config.image_features)
    missing = [
        cam for cam in cfg.cameras if f"{OBS_IMAGE_PREFIX}{cam}" not in policy_image_features
    ]
    if missing:
        raise ValueError(
            f"checkpoint {cfg.pretrained_path!r} was trained with image features "
            f"{sorted(policy_image_features)} which do not cover configured camera(s) "
            f"{missing} (expected {[OBS_IMAGE_PREFIX + cam for cam in missing]}); "
            "set LeRobotPolicyConfig.cameras to the camera names the checkpoint expects"
        )

    def _predict(obs: LeRobotObs) -> npt.NDArray[np.floating[Any]]:
        observation = raw_observation_to_observation(
            dict(obs), lerobot_features, policy_image_features
        )
        batch = preprocessor(observation)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        out = torch.stack(
            [postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1
        ).squeeze(0)
        return out.detach().cpu().numpy()  # type: ignore[no-any-return]

    return _predict


class LeRobotPolicy:
    """Inspect Robots policy wrapping a LeRobot checkpoint for the SO-ARM action space."""

    def __init__(
        self,
        config: LeRobotPolicyConfig | None = None,
        *,
        predict_fn: PredictFn | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else LeRobotPolicyConfig.from_kwargs(**flat)
        self._predict_fn = predict_fn
        self._instruction: str | None = None
        self.num_inferences = 0
        self.info = PolicyInfo(
            name="lerobot",
            action_space=action_box(),  # semantics only; the embodiment owns limits
            observation_space=observation_space(
                self._cfg.cam_height, self._cfg.cam_width, self._cfg.cameras
            ),
            # Intentionally None: advertising a rate would trip a (harmless) compat
            # control_rate warning. The embodiment paces the rollout.
            control_hz=None,
        )
        self.config = PolicyConfig(action_horizon=self._cfg.chunk_size)

    def _predict(self) -> PredictFn:
        """Lazily build the real inference closure on first use."""
        if self._predict_fn is None:
            self._predict_fn = _default_predict(self._cfg)
        return self._predict_fn

    def reset(self, scene: Scene) -> None:
        """Stash the scene's instruction (fed to the VLA verbatim)."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Build the raw robot-style observation, run inference, return the chunk.

        The chunk is truncated to the first ``chunk_size`` actions (see
        ``LeRobotPolicyConfig.chunk_size``).
        """
        cfg = self._cfg
        try:
            images = {cam: observation.images[cam] for cam in cfg.cameras}
        except KeyError as exc:
            raise ValueError(f"observation missing camera {exc} for lerobot policy") from exc
        if cfg.state_key not in observation.state:
            raise ValueError(f"observation missing state key {cfg.state_key!r}")
        state = packing.validate_dim(observation.state[cfg.state_key])

        # The raw robot format lerobot's raw_observation_to_observation expects:
        # "<motor>.pos" floats, camera frames keyed by camera name, and "task".
        lerobot_obs: dict[str, Any] = {
            **packing.to_action_dict(state),
            TASK_KEY: self._instruction or "",
            **images,
        }

        t0 = time.perf_counter()
        raw = self._predict()(lerobot_obs)
        elapsed = time.perf_counter() - t0

        actions = np.asarray(raw, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != packing.TOTAL_DIM:
            raise ValueError(
                f"policy returned actions of shape {actions.shape}; "
                f"expected (N, {packing.TOTAL_DIM})"
            )
        if actions.shape[0] == 0:
            raise ValueError("policy returned an empty action chunk")
        actions = actions[: cfg.chunk_size]

        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in actions],
            control_hz=None,
            inference_latency_s=elapsed,
        )
