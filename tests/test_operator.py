"""Tests for operator readiness and end-of-episode signaling."""

from __future__ import annotations

import pytest
from inspect_robots.errors import EmbodimentFault

from inspect_robots_so101.operator import OperatorIO, default_poll_end


def _scripted(answers: list[str]):
    seen: list[str] = []

    def _input(prompt: str) -> str:
        seen.append(prompt)
        return answers.pop(0)

    return _input, seen


def test_wait_ready_calls_input() -> None:
    inp, seen = _scripted([""])
    io = OperatorIO(input_fn=inp, output_fn=lambda _m: None)
    io.wait_ready("ready?")
    assert seen == ["ready?"]


def test_wait_ready_drain_is_noop_without_tty() -> None:
    inp, seen = _scripted([""])
    io = OperatorIO(input_fn=inp, output_fn=lambda _m: None)
    io.wait_ready()
    assert len(seen) == 1


def test_wait_ready_eof_raises_embodiment_fault() -> None:
    def _dead_stdin(_prompt: str) -> str:
        raise EOFError("stdin closed")

    io = OperatorIO(input_fn=_dead_stdin, output_fn=lambda _m: None)
    with pytest.raises(EmbodimentFault, match="real TTY") as exc:
        io.wait_ready()
    assert "OperatorIO(input_fn=...)" in str(exc.value)


def test_wait_ready_oserror_raises_embodiment_fault() -> None:
    def _dead_stdin(_prompt: str) -> str:
        raise OSError("stdin closed")

    io = OperatorIO(input_fn=_dead_stdin, output_fn=lambda _m: None)
    with pytest.raises(EmbodimentFault, match="real TTY") as exc:
        io.wait_ready()
    assert "OperatorIO(input_fn=...)" in str(exc.value)


def test_default_poll_end_is_callable() -> None:
    # The body is TTY-bound (pragma: no cover); just assert it's wired and callable.
    assert callable(default_poll_end)
