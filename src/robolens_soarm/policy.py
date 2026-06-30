"""``LeRobotPolicy`` — a RoboLens policy backed by a LeRobot checkpoint.

LeRobot policies (ACT, SmolVLA, π0, diffusion, …) are ordinary ``nn.Module``\\ s
loaded from the Hugging Face Hub and run **in process** on the GPU. Unlike the
YAM/MolmoAct2 stack (where the model owns its own HTTP server), LeRobot models are
a library you import — so the heavy, GPU-bound dependencies (``torch`` +
``lerobot`` + the checkpoint) live behind a single injectable seam: a
``predict_fn`` that maps a LeRobot-style observation dict to an action chunk.

The default ``predict_fn`` (``_default_predict``) lazily builds the policy and its
pre/post-processors from a pretrained path; it is a ``# pragma: no cover`` seam
that only runs with real weights on a real device. Tests inject a fake
``predict_fn``, so the whole policy is exercised with no torch, no lerobot, and no
network — and ``import robolens_soarm`` never imports torch.

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
from robolens.policy import PolicyConfig, PolicyInfo
from robolens.scene import Scene
from robolens.types import Action, ActionChunk, Observation

from robolens_soarm import packing
from robolens_soarm.config import LeRobotPolicyConfig, action_box, observation_space

# A LeRobot-style observation dict -> an (N, action_dim) action-chunk array. Keys:
# "observation.state", "observation.images.<cam>", and "task" (the instruction).
LeRobotObs = Mapping[str, Any]
PredictFn = Callable[[LeRobotObs], npt.NDArray[np.floating[Any]]]

# LeRobot observation key conventions (see lerobot/utils/constants.py). Hardcoded
# so this module needs no lerobot import.
OBS_STATE = "observation.state"
OBS_IMAGE_PREFIX = "observation.images."
TASK_KEY = "task"


def _default_predict(cfg: LeRobotPolicyConfig) -> PredictFn:  # pragma: no cover - real model/GPU
    """Build an in-process LeRobot inference closure from a pretrained checkpoint.

    Mirrors ``lerobot.async_inference.policy_server``: load the policy class, build
    the matching pre/post-processor pipelines, then per call preprocess → run
    ``predict_action_chunk`` → postprocess back to native motor units.
    """
    import torch
    from lerobot.policies import get_policy_class, make_pre_post_processors

    policy = get_policy_class(cfg.policy_type).from_pretrained(cfg.pretrained_path)
    policy.to(cfg.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
        postprocessor_overrides={"device_processor": {"device": cfg.device}},
    )

    def _predict(obs: LeRobotObs) -> npt.NDArray[np.floating[Any]]:
        batch = preprocessor(dict(obs))
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
    """RoboLens policy wrapping a LeRobot checkpoint for the SO-ARM action space."""

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
        """Lazily build the real inference closure on first use (pragma'd seam)."""
        if self._predict_fn is None:  # pragma: no cover - loads torch + the real model
            self._predict_fn = _default_predict(self._cfg)
        return self._predict_fn

    def reset(self, scene: Scene) -> None:
        """Stash the scene's instruction (fed to the VLA verbatim)."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Build the LeRobot observation, run inference, return the action chunk."""
        cfg = self._cfg
        try:
            images = {f"{OBS_IMAGE_PREFIX}{cam}": observation.images[cam] for cam in cfg.cameras}
        except KeyError as exc:
            raise ValueError(f"observation missing camera {exc} for lerobot policy") from exc
        if cfg.state_key not in observation.state:
            raise ValueError(f"observation missing state key {cfg.state_key!r}")
        state = packing.validate_dim(observation.state[cfg.state_key]).astype(np.float32)

        lerobot_obs: dict[str, Any] = {
            OBS_STATE: state,
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

        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in actions],
            control_hz=None,
            inference_latency_s=elapsed,
        )
