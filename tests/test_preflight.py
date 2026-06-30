"""Tests for the preflight compatibility check + CLI."""

from __future__ import annotations

import json

import pytest
from robolens.compat import CompatibilityReport, CompatIssue
from robolens.policy import PolicyConfig, PolicyInfo
from robolens.spaces import ActionSemantics, Box

from robolens_soarm import preflight
from robolens_soarm.embodiment import SOArmEmbodiment


class _WrongDimPolicy:
    """An 8-D policy — deliberately incompatible with the 6-D SO-ARM embodiment."""

    def __init__(self) -> None:
        self.info = PolicyInfo(
            name="wrong",
            action_space=Box(shape=(8,), semantics=ActionSemantics(control_mode="joint_pos")),
        )
        self.config = PolicyConfig()

    def reset(self, scene: object) -> None: ...

    def act(self, obs: object) -> object: ...  # pragma: no cover - never called


def test_build_returns_pair_without_connecting() -> None:
    pol, emb = preflight.build()
    assert pol.info.name == "lerobot"
    assert emb.info.name == "so_arm"


def test_run_preflight_default_is_compatible() -> None:
    report = preflight.run_preflight()
    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []  # the whole point: clean compat by construction


def test_run_preflight_with_builtin_task() -> None:
    report = preflight.run_preflight("cubepick-reach")
    assert report.ok is True
    assert report.errors == []


def test_run_preflight_incompatible_pair_reports_errors() -> None:
    report = preflight.run_preflight(
        policy=_WrongDimPolicy(),  # type: ignore[arg-type]
        embodiment=SOArmEmbodiment(),
    )
    assert report.ok is False
    assert any(i.code == "action_dim" for i in report.errors)


def test_run_preflight_uses_injected_check() -> None:
    sentinel = CompatibilityReport(issues=[CompatIssue("warning", "x", "y")])
    out = preflight.run_preflight(check=lambda *a, **k: sentinel)
    assert out is sentinel


def _ok() -> CompatibilityReport:
    return CompatibilityReport()


def _warn() -> CompatibilityReport:
    return CompatibilityReport(issues=[CompatIssue("warning", "frame", "frames differ")])


def _err() -> CompatibilityReport:
    return CompatibilityReport(issues=[CompatIssue("error", "action_dim", "8 != 6")])


def test_main_ok_human(capsys: pytest.CaptureFixture[str]) -> None:
    code = preflight.main([], run=lambda *_a, **_k: _ok())
    out = capsys.readouterr().out
    assert code == 0
    assert "OK:" in out


def test_main_warning_human_exit_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = preflight.main([], run=lambda *_a, **_k: _warn())
    out = capsys.readouterr().out
    assert code == 0  # warnings do not fail preflight
    assert "WARNING" in out


def test_main_error_human_exit_one(capsys: pytest.CaptureFixture[str]) -> None:
    code = preflight.main([], run=lambda *_a, **_k: _err())
    out = capsys.readouterr().out
    assert code == 1
    assert "INCOMPATIBLE" in out and "ERROR" in out


def test_main_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = preflight.main(["--json"], run=lambda *_a, **_k: _err())
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"][0]["code"] == "action_dim"


def test_main_dry_run_note(capsys: pytest.CaptureFixture[str]) -> None:
    preflight.main(["--dry-run"], run=lambda *_a, **_k: _ok())
    assert "dry-run" in capsys.readouterr().out


def test_main_default_run_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    # Exercise the real default run path (no injection) end-to-end.
    code = preflight.main([])
    assert code == 0
    assert "OK:" in capsys.readouterr().out
