"""Guard the public API surface so changes to __all__ are deliberate."""

from __future__ import annotations

import robolens_soarm

EXPECTED_API = {
    "MOTORS",
    "STATE_KEY",
    "TOTAL_DIM",
    "LeRobotPolicy",
    "LeRobotPolicyConfig",
    "OperatorIO",
    "SOArmConfig",
    "SOArmEmbodiment",
    "build",
    "from_obs_dict",
    "run_preflight",
    "to_action_dict",
}


def test_public_api_matches_all() -> None:
    assert set(robolens_soarm.__all__) == EXPECTED_API


def test_all_names_are_importable() -> None:
    for name in robolens_soarm.__all__:
        assert hasattr(robolens_soarm, name), name


def test_version() -> None:
    assert robolens_soarm.__version__ == "0.1.0"


def test_entry_points_resolve_via_registry() -> None:
    # The installed entry points must resolve to our classes.
    from robolens.registry import resolve

    pol = resolve("policy", "lerobot")
    emb = resolve("embodiment", "so_arm")
    assert pol.info.name == "lerobot"
    assert emb.info.name == "so_arm"
