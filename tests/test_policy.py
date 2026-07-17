"""Tests for LeRobotPolicy.

The injected-``predict_fn`` tests need no torch and no lerobot. The
``_default_predict`` seam tests install ``sys.modules`` fakes for torch and the
lerobot modules the seam imports, and assert the *wiring* (which lerobot helper is
called with which arguments) — real tensor behaviour is lerobot's contract,
drift-guarded by the ``lerobot-seam`` CI job that imports the real symbols.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from inspect_robots_so101 import packing
from inspect_robots_so101.config import LeRobotPolicyConfig
from inspect_robots_so101.policy import LeRobotPolicy


def _obs(instruction: str | None = "do it") -> Observation:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return Observation(
        images={"front": img},
        state={"joint_pos": np.zeros(6)},
        instruction=instruction,
    )


def _fake_predict(actions: np.ndarray):
    captured: dict[str, Any] = {}

    def _predict(obs: Any) -> np.ndarray:
        captured["obs"] = obs
        return actions

    return _predict, captured


def test_info_and_config_zero_arg() -> None:
    pol = LeRobotPolicy()
    assert pol.info.name == "lerobot"
    assert pol.info.action_space.dim == 6
    assert pol.info.action_space.semantics is not None
    assert pol.info.action_space.semantics.control_mode == "joint_pos"
    assert pol.info.control_hz is None  # load-bearing: keeps compat warning-free
    assert pol.info.observation_space.state_keys == frozenset({"joint_pos"})
    assert pol.config.action_horizon == 50


def test_normalized_info_declares_normalized_state() -> None:
    pol = LeRobotPolicy(LeRobotPolicyConfig(use_degrees=False))
    state = pol.info.observation_space.state
    assert state is not None and state.fields[0].unit == "normalized"


def test_act_builds_raw_lerobot_obs_and_chunk() -> None:
    actions = np.arange(2 * 6, dtype=float).reshape(2, 6)
    predict, captured = _fake_predict(actions)
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction="pick up the cube"))
    chunk = pol.act(_obs())

    assert len(chunk) == 2
    assert np.array_equal(chunk.actions[0].data, actions[0])
    assert chunk.control_hz is None  # in-process: embodiment paces
    assert chunk.inference_latency_s is not None
    # The payload is the RAW robot format (what a lerobot Robot.get_observation
    # returns, plus "task"): "<motor>.pos" floats and frames keyed by camera name.
    obs = captured["obs"]
    assert obs["task"] == "pick up the cube"
    assert [obs[key] for key in packing.motor_keys()] == [0.0] * 6
    assert isinstance(obs["shoulder_pan.pos"], float)
    assert obs["front"].shape == (4, 4, 3)
    assert "observation.state" not in obs  # packing is lerobot's job, not ours
    assert pol.num_inferences == 1


def test_act_uses_empty_instruction_when_none() -> None:
    predict, captured = _fake_predict(np.zeros((1, 6)))
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction=None))
    pol.act(_obs(instruction=None))
    assert captured["obs"]["task"] == ""


def test_act_truncates_chunk_to_chunk_size() -> None:
    predict, _ = _fake_predict(np.arange(5 * 6, dtype=float).reshape(5, 6))
    pol = LeRobotPolicy(LeRobotPolicyConfig(chunk_size=2), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    chunk = pol.act(_obs())
    assert len(chunk) == 2  # model returned 5; consume only the first chunk_size
    assert np.array_equal(chunk.actions[1].data, np.arange(6, 12, dtype=float))


def test_act_empty_actions_raises() -> None:
    predict, _ = _fake_predict(np.zeros((0, 6)))
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="empty action chunk"):
        pol.act(_obs())


def test_act_wrong_action_width_raises() -> None:
    predict, _ = _fake_predict(np.zeros((2, 7)))
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match=r"expected \(N, 6\)"):
        pol.act(_obs())


def test_act_missing_camera_raises() -> None:
    predict, _ = _fake_predict(np.zeros((1, 6)))
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    obs = Observation(images={}, state={"joint_pos": np.zeros(6)})
    with pytest.raises(ValueError, match="missing camera"):
        pol.act(obs)


def test_act_missing_state_raises() -> None:
    predict, _ = _fake_predict(np.zeros((1, 6)))
    pol = LeRobotPolicy(predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    obs = Observation(images={"front": np.zeros((4, 4, 3), np.uint8)}, state={})
    with pytest.raises(ValueError, match="missing state key"):
        pol.act(obs)


def test_config_object_overrides_flat() -> None:
    pol = LeRobotPolicy(LeRobotPolicyConfig(chunk_size=3))
    assert pol.config.action_horizon == 3
    assert packing.TOTAL_DIM == 6  # sanity


# ---------------------------------------------------------------------------
# _default_predict seam: sys.modules fakes for torch + the lerobot modules.
# ---------------------------------------------------------------------------


class FakeTensor:
    """Minimal torch.Tensor stand-in: ndim/shape/slicing/unsqueeze/squeeze/numpy."""

    def __init__(self, arr: Any) -> None:
        self.arr = np.asarray(arr, dtype=np.float64)

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    def unsqueeze(self, dim: int) -> FakeTensor:
        return FakeTensor(np.expand_dims(self.arr, dim))

    def squeeze(self, dim: int) -> FakeTensor:
        return FakeTensor(np.squeeze(self.arr, dim))

    def __getitem__(self, idx: Any) -> FakeTensor:
        return FakeTensor(self.arr[idx])

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.arr


def _install_lerobot_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chunk: np.ndarray,
    image_features: dict[str, Any] | None = None,
    feature_layout: str = "v060",
) -> dict[str, Any]:
    """Install fakes for the v0.6.0, v0.5.1, or v0.5.0 feature layout."""
    calls: dict[str, Any] = {}
    if image_features is None:
        image_features = {"observation.images.front": SimpleNamespace(shape=(3, 4, 4))}

    class _FakePolicy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(image_features=image_features)
            calls["policy_config"] = self.config

        def to(self, device: str) -> None:
            calls["to_device"] = device

        def eval(self) -> None:
            calls["eval"] = True

        def predict_action_chunk(self, batch: Any) -> FakeTensor:
            calls["predict_batch"] = batch
            return FakeTensor(chunk)

    class _FakePolicyClass:
        @staticmethod
        def from_pretrained(path: str) -> _FakePolicy:
            calls["from_pretrained"] = path
            return _FakePolicy()

    def get_policy_class(name: str) -> type[_FakePolicyClass]:
        calls.setdefault("policy_types", []).append(name)
        return _FakePolicyClass

    def make_pre_post_processors(policy_cfg: Any, pretrained_path: Any = None, **kw: Any) -> Any:
        calls["processors_cfg"] = policy_cfg
        calls["processors_path"] = pretrained_path
        calls["processors_kwargs"] = kw

        def preprocessor(obs: Any) -> Any:
            calls["preprocessed"] = obs
            return {"batch": obs}

        def postprocessor(tensor: FakeTensor) -> FakeTensor:
            calls.setdefault("postprocessed", []).append(tensor)
            return tensor

        return preprocessor, postprocessor

    def hw_to_dataset_features(hw: Any, prefix: str, use_video: bool = True) -> Any:
        calls["hw_features"] = hw
        calls["hw_prefix"] = prefix
        calls["hw_use_video"] = use_video
        return {"observation.state": {"names": [k for k in hw if k.endswith(".pos")]}}

    def raw_observation_to_observation(raw: Any, feats: Any, img_feats: Any) -> Any:
        calls["raw_obs"] = raw
        calls["raw_features"] = feats
        calls["raw_image_features"] = img_feats
        return {"prepared": True}

    torch_mod = ModuleType("torch")
    torch_mod.no_grad = nullcontext  # type: ignore[attr-defined]

    def _stack(tensors: list[FakeTensor], dim: int = 0) -> FakeTensor:
        return FakeTensor(np.stack([t.arr for t in tensors], axis=dim))

    torch_mod.stack = _stack  # type: ignore[attr-defined]

    lerobot = ModuleType("lerobot")
    policies = ModuleType("lerobot.policies")
    factory = ModuleType("lerobot.policies.factory")
    factory.get_policy_class = get_policy_class  # type: ignore[attr-defined]
    factory.make_pre_post_processors = make_pre_post_processors  # type: ignore[attr-defined]
    async_inference = ModuleType("lerobot.async_inference")
    helpers = ModuleType("lerobot.async_inference.helpers")
    helpers.raw_observation_to_observation = raw_observation_to_observation  # type: ignore[attr-defined]
    datasets = ModuleType("lerobot.datasets")
    lerobot.policies = policies  # type: ignore[attr-defined]
    policies.factory = factory  # type: ignore[attr-defined]
    lerobot.async_inference = async_inference  # type: ignore[attr-defined]
    async_inference.helpers = helpers  # type: ignore[attr-defined]
    lerobot.datasets = datasets  # type: ignore[attr-defined]

    modules = {
        "torch": torch_mod,
        "lerobot": lerobot,
        "lerobot.policies": policies,
        "lerobot.policies.factory": factory,
        "lerobot.async_inference": async_inference,
        "lerobot.async_inference.helpers": helpers,
        "lerobot.datasets": datasets,
    }
    if feature_layout == "v060":
        # lerobot >= 0.6.0: hw_to_dataset_features lives in lerobot.utils.feature_utils.
        monkeypatch.delitem(sys.modules, "lerobot.datasets.feature_utils", raising=False)
        monkeypatch.delitem(sys.modules, "lerobot.datasets.utils", raising=False)
        utils = ModuleType("lerobot.utils")
        feature_utils_mod = ModuleType("lerobot.utils.feature_utils")
        feature_utils_mod.hw_to_dataset_features = hw_to_dataset_features  # type: ignore[attr-defined]
        utils.feature_utils = feature_utils_mod  # type: ignore[attr-defined]
        lerobot.utils = utils  # type: ignore[attr-defined]
        modules["lerobot.utils"] = utils
        modules["lerobot.utils.feature_utils"] = feature_utils_mod
    elif feature_layout == "v051":
        # lerobot 0.5.1 - 0.5.x: the helper moved to datasets.feature_utils.
        monkeypatch.delitem(sys.modules, "lerobot.utils.feature_utils", raising=False)
        monkeypatch.delitem(sys.modules, "lerobot.utils", raising=False)
        feature_utils_mod = ModuleType("lerobot.datasets.feature_utils")
        feature_utils_mod.hw_to_dataset_features = hw_to_dataset_features  # type: ignore[attr-defined]
        datasets.feature_utils = feature_utils_mod  # type: ignore[attr-defined]
        modules["lerobot.datasets.feature_utils"] = feature_utils_mod
    elif feature_layout == "v050":
        # lerobot == 0.5.0 layout: hw_to_dataset_features in lerobot.datasets.utils
        # and NO feature_utils module (the import must raise ImportError).
        monkeypatch.delitem(sys.modules, "lerobot.utils.feature_utils", raising=False)
        monkeypatch.delitem(sys.modules, "lerobot.utils", raising=False)
        monkeypatch.delitem(sys.modules, "lerobot.datasets.feature_utils", raising=False)
        utils_mod = ModuleType("lerobot.datasets.utils")
        utils_mod.hw_to_dataset_features = hw_to_dataset_features  # type: ignore[attr-defined]
        datasets.utils = utils_mod  # type: ignore[attr-defined]
        modules["lerobot.datasets.utils"] = utils_mod
    else:
        raise ValueError(f"unknown feature layout: {feature_layout}")

    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _seam_cfg(**over: Any) -> LeRobotPolicyConfig:
    return LeRobotPolicyConfig(device="cpu", cam_height=4, cam_width=4, **over)


def test_default_predict_wires_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = np.arange(1 * 2 * 6, dtype=float).reshape(1, 2, 6)
    calls = _install_lerobot_fakes(monkeypatch, chunk=chunk)
    pol = LeRobotPolicy(_seam_cfg())  # no predict_fn: exercises the real seam
    pol.reset(Scene(id="s", instruction="grab it"))
    out = pol.act(_obs(instruction="grab it"))

    # Model load: factory class + from_pretrained + device + eval.
    assert calls["policy_types"] == ["smolvla"]
    assert calls["from_pretrained"] == "lerobot/smolvla_base"
    assert calls["to_device"] == "cpu"
    assert calls["eval"] is True
    # Processor pipelines built from the loaded policy's config with device overrides.
    assert calls["processors_cfg"] is calls["policy_config"]
    assert calls["processors_path"] == "lerobot/smolvla_base"
    assert calls["processors_kwargs"] == {
        "preprocessor_overrides": {"device_processor": {"device": "cpu"}},
        "postprocessor_overrides": {"device_processor": {"device": "cpu"}},
    }
    # lerobot_features synthesized from packing.MOTORS + cfg cameras (no robot object).
    hw = calls["hw_features"]
    assert [k for k in hw if k.endswith(".pos")] == list(packing.motor_keys())
    assert all(hw[k] is float for k in packing.motor_keys())
    assert hw["front"] == (4, 4, 3)
    assert calls["hw_prefix"] == "observation"
    assert calls["hw_use_video"] is False
    # Obs prep delegated to lerobot's raw_observation_to_observation with the raw
    # payload, the synthesized features, and the checkpoint's image features.
    raw = calls["raw_obs"]
    assert raw["task"] == "grab it"
    assert [raw[k] for k in packing.motor_keys()] == [0.0] * 6
    assert raw["front"].shape == (4, 4, 3)
    assert calls["raw_features"] == {"observation.state": {"names": list(packing.motor_keys())}}
    assert calls["raw_image_features"] == calls["policy_config"].image_features
    # Prepared obs -> preprocessor -> predict_action_chunk -> per-step postprocess.
    assert calls["preprocessed"] == {"prepared": True}
    assert calls["predict_batch"] == {"batch": {"prepared": True}}
    assert len(calls["postprocessed"]) == 2
    assert len(out) == 2
    assert np.array_equal(out.actions[0].data, chunk[0, 0])
    assert np.array_equal(out.actions[1].data, chunk[0, 1])


def test_default_predict_unsqueezes_2d_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = np.arange(2 * 6, dtype=float).reshape(2, 6)  # no batch dim from the model
    _install_lerobot_fakes(monkeypatch, chunk=chunk)
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    out = pol.act(_obs())
    assert len(out) == 2
    assert np.array_equal(out.actions[1].data, chunk[1])


def test_default_predict_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, 6)))
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    pol.act(_obs())
    pol.act(_obs())
    assert calls["policy_types"] == ["smolvla"]  # lazy closure built exactly once
    assert pol.num_inferences == 2


def test_default_predict_rejects_camera_feature_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lerobot_fakes(
        monkeypatch,
        chunk=np.ones((1, 2, 6)),
        image_features={"observation.images.top": SimpleNamespace(shape=(3, 4, 4))},
    )
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match=r"do not cover configured camera\(s\) \['front'\]"):
        pol.act(_obs())


def test_default_predict_hw_features_fallback_import(monkeypatch: pytest.MonkeyPatch) -> None:
    # lerobot 0.5.0 layout: no lerobot.datasets.feature_utils -> fall back to
    # lerobot.datasets.utils.
    calls = _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, 6)), feature_layout="v050")
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    out = pol.act(_obs())
    assert len(out) == 2
    assert calls["hw_prefix"] == "observation"


def test_default_predict_hw_features_v051_fallback_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_lerobot_fakes(
        monkeypatch,
        chunk=np.ones((1, 2, 6)),
        feature_layout="v051",
    )
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    out = pol.act(_obs())
    assert len(out) == 2
    assert calls["hw_prefix"] == "observation"
