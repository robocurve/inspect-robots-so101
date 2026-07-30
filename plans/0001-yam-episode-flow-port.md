# 0001: Port yam's episode-flow fixes (operator_end, settle-before-observe, frame validation)

## Motivation

inspect-robots-yam shipped three fixes to the episode-flow contract that
inspect-robots-so101 shares structurally but never received. All three defects
are present in so101 today:

1. **Definitive operator verdicts suppress framework grading** (yam #81,
   inspect-robots#194). `SOArmEmbodiment.step()` prompts y/N via
   `OperatorIO.confirm_success()` and returns
   `termination_reason="success"|"failure"`. The framework (>= 0.25) owns a
   single operator prompt that collects the verdict, partial credit, skip, and
   grader notes; a definitive reason from the embodiment suppresses it. yam now
   terminates with the non-definitive `OPERATOR_END` and ships no grading UI.
2. **Chunked policies plan from a mid-motion pose** (yam #65). `step()`
   commands, paces one control period, and observes. LeRobot's
   `send_action()` returns immediately, so the observation reflects whatever
   the arm was doing ~33 ms later. A VLA replanning every step tolerates this;
   a chunked policy (MolmoAct2 chunk 30, the LLM-agent's interpolated tool
   calls) plans its next motion from a pose the arm may never have reached.
3. **Any frame shape passes silently** (yam #25). `_observe()` and
   `LeRobotPolicy.act()` accept camera frames of any resolution even though
   both declare `(cam_height, cam_width, 3)` in their observation contract. A
   misconfigured camera (or a driver that silently renegotiates resolution)
   corrupts inference instead of failing loudly at the boundary.

Evaluated and **not** ported, with reasons:

- yam #66 (camera drain): fixes yam's own buffered OpenCV reader. so101's
  frames come from lerobot's threaded cameras: `get_observation` calls
  `cam.read_latest()` (a non-blocking freshest-frame peek) across the
  supported `>=0.5,<0.6` range (verified in lerobot v0.5.0 and v0.5.1 tags,
  `so_follower.py get_observation`; v0.6.0 keeps `read_latest` too, as
  forward evidence).
- yam #84 (concurrent arm bring-up): bimanual only; so101 is single-arm by
  charter (root CLAUDE.md "Out of scope").
- yam #71 (RealSense depth), #73/#76 (health CLI), #82 (wizard-config
  defaults): tied to yam's device stack; lerobot has native RealSense camera
  configs, and so101 rig tooling lives outside the package.
- yam #86 (collision guardrail): valuable, but a straight port is unsound —
  lerobot's degree zero is defined by each rig's calibration homing offsets,
  so mapping actions onto a MuJoCo model needs its own design. Tracked as a
  follow-up issue instead.

## Implementation order

Commit-sized stages, in order. Framework-drift risk is isolated in stage 0;
each later stage lands with its tests.

### Stage 0: dependency floor bump, no feature code

Raise `inspect-robots>=0.3` to `inspect-robots>=0.25` in `pyproject.toml`
(the `OPERATOR_END` grading contract; verified present at v0.25.0 with the
`before_scoring` eval hook and the CLI operator prompt). Run `uv lock`,
commit the lockfile (CI installs `--locked`), and run the **full existing
suite** green before any feature work: the lock currently resolves 0.3.1, and
the jump to whatever the relock resolves (0.28.x today) carries framework
behavior changes (pacing revision, eval closing registry-resolved
embodiments, compat task-horizon errors) that must not be conflated with
port regressions. Fix any drift breakage in this stage. While in the README:
its install note ("Inspect Robots isn't on PyPI yet; uv resolves it from
git") is already false — the lock resolves it from PyPI — fix it here.

### Stage A: operator_end grading handoff (port of yam #81)

**`src/inspect_robots_so101/operator.py`**:
- Delete `confirm_success()` and `_AFFIRMATIVE`. Grading belongs to the
  framework's operator prompt; this module keeps readiness + end-poll only.
- Add `_drain_stdin()` (yam's TTY-guarded implementation, including its
  `# pragma: no cover` placement) and call it at the end of `wait_ready()` so
  a stale buffered newline doesn't trip `default_poll_end()` on step 1.
- Port yam's `wait_ready` fault wrapper as well: `EOFError`/`OSError` from
  `input_fn` raises `inspect_robots.errors.EmbodimentFault`. **Reword the
  message, do not copy it**: yam's points at `YamConfig(unattended=True)`,
  which so101 does not have (`from_kwargs` would reject the key). so101's
  message offers the remedies that exist: run from a real TTY, or inject
  `OperatorIO(input_fn=...)`. Tested with an `input_fn` raising `EOFError`
  and another raising `OSError` (one test each, so both arms are exercised).
- Update the module docstring: the operator readies scenes and signals
  end-of-episode; the verdict is collected afterwards by the framework.

**`src/inspect_robots_so101/embodiment.py`**:
- `from inspect_robots.types import OPERATOR_END`.
- In `step()`, the poll-end branch returns
  `StepResult(observation=obs, terminated=True, termination_reason=OPERATOR_END, info={})`
  (stage B retrofits settle info into both return paths) — no prompt, no
  `operator_confirmed` key.
- Update the class/module docstrings: retitle the "Operator-in-the-loop
  success" bullet to "Operator-in-the-loop episode end" with yam's body
  wording adapted for so101 (so the removed-phrase greps in Verification
  stay meaningful).

**Docs**: root `CLAUDE.md` safety invariant "Success reaches the scorer only
via `termination_reason="success"`" is rewritten: the embodiment terminates
with the non-definitive `operator_end`; the human verdict is captured by the
framework's operator prompt and read by judgement-based scorers.
`src/inspect_robots_so101/CLAUDE.md`: **both** the operator.py row and the
embodiment.py row ("operator-keypress success" becomes "operator-keypress
episode end (`operator_end`)", mirroring yam's row). README: the feature
bullet at ~line 29 ("operator-in-the-loop success") plus the quickstart.
**The quickstart is load-bearing**: README.md ~100-121 documents the y/N
prompt and says `success_at_end` reads the resulting `termination_reason` —
that chain no longer exists. Rewrite the scoring story precisely:
- The CLI owns the grading **prompt** for any attended run, but the verdict
  is *scored* only when the task's scorer reads judgements. Adhoc
  `inspect-robots run --instruction ...` runs default to the `operator`
  scorer and Just Work. Registered tasks bring their own scorers (`--scorer`
  is rejected for them), and `cubepick-reach`'s is `success_at_end`, which
  never reads judgements — so after this port a `--task cubepick-reach`
  attended run would collect a verdict and score 0.0.
- Therefore: switch the README's CLI example (~line 37, echoed in
  `__init__.py`'s module docstring) to an adhoc `--instruction` invocation
  (or keep `cubepick-reach` with an explicit "compat smoke only; scores via
  `success_at_end`" caveat), and replace the Python quickstart's
  `eval("cubepick-reach", ...)` with an inline `Task` using the framework's
  `operator_scorer()` plus a `before_scoring` hook that records the verdict
  — `eval()` has no scorer override, so both halves are needed:
  `operator_scorer()` alone scores 0.0 when no judgement was recorded, and
  a hook alone feeds a scorer that never reads judgements.
- Mirror yam's one-line warning: don't pair `success_at_end` with attended
  operator-graded runs — it scores them as failures.
Root `CLAUDE.md`'s tests bullet ("The end-to-end test uses Inspect Robots's
built-in `cubepick-reach` task so it stays self-contained") is also updated:
the e2e now builds an inline `Task` with `operator_scorer()` — still
self-contained, no longer via `cubepick-reach`.

**Tests**:
- `test_operator.py`: drop `confirm_success` cases; add `wait_ready` drain
  behavior (non-TTY no-op path) and both fault cases (EOFError →
  EmbodimentFault, OSError → EmbodimentFault; one test each). Update its
  module docstring ("operator-in-the-loop confirmation" is stale).
- `test_embodiment.py`: assert `termination_reason == OPERATOR_END` and that
  no success prompt runs (scripted `input_fn` records calls).
- `test_eval_end_to_end.py`: **cannot keep asserting
  `metrics["success_at_end"] == 1.0`** — `cubepick-reach`'s `success_at_end`
  scorer never reads operator judgements, so it would score 0.0 by
  construction. Rework the e2e like yam's: build an inline `Task` (eval() at
  0.25 accepts a Task object) using the framework's builtin
  `operator_scorer()`, pass `before_scoring=` a hook that asserts
  `record.termination_reason == "operator_end"` and stamps the verdict, and
  assert the operator metric is 1.0. The inline `Task` must declare exactly
  one of `max_steps`/`max_seconds` (`Task.__post_init__` raises otherwise).
  Keep the existing `use_degrees` True/False parametrization — the rework
  must not silently narrow the e2e — and update the file's module docstring
  (it names `cubepick-reach`).

**API note**: removing public `OperatorIO.confirm_success` is a breaking
change to a released package — next release is a **minor** bump (0.x line),
called out in the PR description. `__all__`/`test_api_snapshot.py` are
unaffected (`OperatorIO` itself stays exported).

### Stage B: settle-before-observe (port of yam #65)

Opt-in and off by default, so VLA cadence is untouched.

**`src/inspect_robots_so101/config.py`** — new `SOArmConfig` fields:
- `settle_tolerance: float | None = None` — when set, `step()`/`reset()`
  poll the driver until every **non-gripper** joint is within this tolerance
  (action units: degrees or normalized, matching `use_degrees`) of the
  command the driver actually accepted.
- `settle_timeout_s: float = 1.0`, `settle_timeout_budget: int = 20`.
- `__post_init__` validation mirroring yam's: tolerance finite and > 0,
  timeout finite and > 0, budget an int >= 1 with the bool guard on the
  budget (as in yam — `isinstance(x, int)` alone accepts `True`). All three
  parse from CLI scalars via the existing `from_kwargs` path (framework
  `_parse_value` coerces float/int/none).

**`src/inspect_robots_so101/embodiment.py`**:
- **Settle targets the driver's returned action, not the pre-slew clamp.**
  lerobot's `send_action()` applies `max_relative_target` truncation
  *inside the driver* and returns the action actually sent; the
  `SOArmDriver` protocol already declares that return. `_send()` therefore
  returns `packing.from_obs_dict(driver.send_action(...))` and `_settle`
  converges on that. This keeps settle meaningful under slew truncation
  (large chunked-policy motions would otherwise burn the whole timeout
  budget on poses the driver never commanded) and makes the reset settle an
  honest converged anchor for `_last_state` even though the single
  `home_pose` command is truncated (interpolated homing remains a
  pre-existing tracked issue). Test fakes must echo the (possibly clipped)
  action back the way the real driver does.
- `_settle(target) -> tuple[bool, float] | None` polls the driver until
  `max(|pos - target|)` over slots 0..4 (gripper slot 5 excluded — a gripper
  closing on an object never reaches its target) is `<= settle_tolerance`
  (inclusive, tested exactly on the boundary). Poll via
  `driver.get_observation()` + `packing.from_obs_dict` (no joints-only read
  exists on the driver; the camera cost is a non-blocking `read_latest`
  peek, verified for the pinned lerobot range). **Read before sleeping**, as
  yam does: an already-converged step returns without any sleep, so the
  settle check costs a converged 30 Hz loop nothing (a sleep-first loop
  would tax every step 10 ms). Poll spacing `_SETTLE_POLL_S = 0.01` via
  `self._sleep`; the loop is bounded by **both** elapsed `self._clock()`
  time and a max poll count (`ceil(settle_timeout_s / _SETTLE_POLL_S)`) so a
  frozen test clock cannot wedge it.
- `settle_timeouts` / `_settle_disabled` are initialized in `__init__` (as
  yam does), not only cleared in `reset()`, keeping construction inert and
  attribute access safe pre-reset.
- Timeouts are not failures: increment `self.settle_timeouts`; when it
  reaches `settle_timeout_budget`, set `self._settle_disabled = True` and
  emit ONE `logging.warning` naming the worst motor by name
  (`packing.MOTORS[i]`) and its residual in the configured units
  (degrees/normalized per `use_degrees`) — logging, not `warnings.warn`,
  because the warnings registry dedupes on message text and a joint parked
  against a hard stop repeats its residual.
- `step()`: settle runs after `self._send(cmd)` and **before `_pace()`**, so
  settle time is absorbed by the control period rather than added to it.
  Every `StepResult` (terminated or not, including the stage-A operator_end
  branch) carries `info=self._settle_info(...)`: `{}` when the feature is
  off; otherwise `settle_timeouts` always, `settled`/`settle_residual` when
  a settle ran, `settle_disabled: True` once disabled.
- `reset()`: clear `settle_timeouts`/`_settle_disabled` at entry (not next
  to `num_steps` — a trial that exhausted its budget must not suppress the
  next trial's settle), and settle **immediately after the `home_pose` send,
  before `_operator.wait_ready()`** (settling after the ready gate would
  trivially pass — the arm converges while the operator stages the scene —
  and measure nothing), which also puts it before the first `_observe`, so
  the delta-command anchor is a converged pose.
- Update the `SELF_PACED` docstring bullet: with `settle_tolerance` set,
  `control_hz` becomes a floor on step duration, not a fixed rate.

**Tests**: new `tests/test_settle.py` adapted from yam's. The injected
driver/clock/sleep fakes currently live as private helpers inside
`tests/test_embodiment.py` — promote `FakeDriver` and the `_build` helper
into a new `tests/conftest.py` (as yam's 9d184c0 did) so `test_settle.py`
shares them without cross-importing a test module. Cases: settle runs before
pace; a converged step performs zero **settle** sleeps (no `_SETTLE_POLL_S`
entries in the sleep log — `_pace()` still sleeps its control period, so
assert on the settle spacing or use `control_hz=0`); ordering in
`reset()` (settle before the first `_observe`); gripper-slot exclusion
(divergence parametrized across arm slots, plus a direct mask assertion);
settle converges on the driver-clipped return, not the raw command (this
also covers delta mode — `joints_are_delta` resolves to an absolute command
before `_send`); budget exhaustion disables settling for the trial and reset
re-arms it; boundary case exactly at tolerance; frozen-clock runs terminate
via the poll-count bound; **elapsed-time timeout with a clock-advancing
driver fake** (each `get_observation` advances the injected clock, so the
elapsed-time break — not the poll cap — ends the loop; both loop bounds get
their own test or branch coverage fails); terminated StepResult still
carries the settle keys;
`settle_tolerance=None` leaves `info` empty and adds zero extra driver reads.

**Config tests** (the new `__post_init__` branches must be covered or the
100% gate fails): extend `tests/test_config.py` with rejection cases for
each invalid value (non-finite / zero / negative tolerance and timeout,
budget 0, budget 1.5, `settle_timeout_budget=True`) and acceptance of
CLI-coerced scalars through `from_kwargs` (yam's 9d184c0 did the same).

**Docs** (mirror yam 9d184c0's doc surface): README gains a settle section
adapted from yam's (what it fixes, the three fields, off-by-default, the
gripper exclusion, timeout-budget semantics); root `CLAUDE.md` timing note —
`control_hz` is a fixed step rate only while `settle_tolerance is None`,
a floor otherwise; `src/inspect_robots_so101/CLAUDE.md` embodiment.py and
config.py rows mention settle.

### Stage C: frame-resolution validation (port of yam #25)

**`embodiment.py` `_observe()`**: after building `images`, raise
`ValueError(f"camera {name!r} returned shape {img.shape}, expected {expected}")`
on any frame whose shape != `(cam_height, cam_width, 3)`.

**`policy.py` `act()`**: same check over the frames it packs into the raw
payload, against `LeRobotPolicyConfig.cam_height/cam_width`.

**Test migration (required, suite-wide)**: existing fixtures serve 4x4x3
frames against default 480x640 configs (`test_embodiment.py` `FakeDriver` +
`_build`, `test_eval_end_to_end.py`, `test_policy.py`), so the new checks
would redden most of the suite. Thread `cam_height=4, cam_width=4` through
every fixture config, exactly as yam's #25 did — including the tests in
`test_embodiment.py` that build `SOArmConfig()` directly rather than via the
`_build` helper. Then add the negative tests:
wrong resolution and wrong channel count raise at both boundaries, message
names the camera.

**Docs**: `src/inspect_robots_so101/CLAUDE.md` embodiment.py and policy.py
rows note the boundary validation; README's camera bullet mentions that
frames are validated against the configured resolution.

## Non-goals

- No `unattended` config flag. Note honestly: so101's `wait_ready()` still
  blocks on `input()`; with a closed/dead stdin the stage-A wrapper turns
  the EOFError/OSError into an `EmbodimentFault` instead of an opaque crash,
  but an open-yet-silent pipe still hangs. A real unattended mode is a
  separate feature if ever needed.
- No collision guardrail (follow-up issue), no camera-drain logic, no
  bimanual work.

## Gates

All existing repo gates hold: `ruff check` + `ruff format --check`,
`mypy --strict`, `pytest --cov` at 100% (hardware/TTY-only lines pragma'd per
the repo's coverage discipline), docstrings on every public symbol (Ruff D1),
`uv lock` committed. `import inspect_robots_so101` must still work with only
`inspect-robots` + `numpy` (import-hygiene job).

## Verification

1. Stage 0 alone: full pre-existing suite green on inspect-robots 0.25+.
2. `uv run pytest --cov` → 100%, all green; `uv run mypy` strict clean;
   `uv run ruff check .` and `ruff format --check .` clean.
3. Case-insensitive grep: no remaining `confirm_success` /
   `operator_confirmed` / `Did the robot succeed` / `operator-keypress
   success` / `operator-in-the-loop success` / `operator-in-the-loop
   confirmation` references anywhere (src, tests, README, both CLAUDE.md
   files) — the stage-A retitling to "episode end" is what makes these
   greps clean. (`y/N` itself is not grep-banned: the rewritten README may
   legitimately describe the *framework's* `[y/n/partial/skip]` prompt.)
4. e2e test drives a scripted episode to operator-end and asserts the
   framework-visible termination reason is `operator_end` and the operator
   metric scores the stamped verdict.
