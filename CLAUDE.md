# inspect-robots-so101 — agent guide

Inspect Robots adapters that let evals run on real **LeRobot SO-ARM followers**
(SO-100 / SO-101) driven by **LeRobot policies**. This is a **plugin package** in
the Inspect Robots ecosystem — the framework lives in
[inspect-robots](https://github.com/robocurve/inspect-robots); benchmarks are separate repos
indexed by [WorldEvals](https://github.com/robocurve/worldevals). It is the
SO-ARM/LeRobot sibling of
[inspect-robots-yam](https://github.com/robocurve/inspect-robots-yam) (bimanual YAM +
MolmoAct2).

## The one big idea

Inspect Robots evals swap two inputs: a `Policy` (VLA brain) and an `Embodiment` (robot
body + world). We ship both for one real stack:

- **`lerobot` policy** — wraps a LeRobot checkpoint (ACT / SmolVLA / π0 / …) and
  runs it **in process** (`raw_observation_to_observation` + the model's pre/post
  processors + `predict_action_chunk`). Unlike YAM/MolmoAct2 (a separate HTTP
  server), LeRobot models are a library you import — so torch + lerobot + the
  checkpoint live behind **one injectable seam**, `LeRobotPolicy(predict_fn=...)`,
  which receives the RAW robot payload (`"<motor>.pos"` floats, frames keyed by
  camera name, `"task"`). We never import torch at module top; the seam's wiring
  is unit-tested with `sys.modules` fakes and its real imports are validated by
  the `lerobot-seam` CI job (py3.12).
- **`so_arm` embodiment** — the LeRobot SO follower driver. Its `get_observation`
  already returns both `"<motor>.pos"` floats *and* camera frames, so there is no
  separate camera reader (a small simplification over inspect-robots-yam).

Both declare the **same 6-D `joint_pos` contract** (5 revolute joints + gripper,
the configured cameras, packed `joint_pos` state). That makes
`inspect_robots.compat.check_compatibility` pass with zero errors **and** zero warnings
— the property `tests/test_compat.py` locks down.

## Layout

- `src/inspect_robots_so101/` — the package (see `src/inspect_robots_so101/CLAUDE.md`).
- `tests/` — pytest; everything (driver, cameras, model inference, clock, operator
  stdin) is injected, so the suite needs **no hardware, no GPU, no torch, no
  lerobot, no stdin**. The end-to-end test uses Inspect Robots's built-in
  `cubepick-reach` task so it stays self-contained.

## Working here

- Dev loop: `uv venv && uv pip install -e ".[dev]"`, `uv run pre-commit install`,
  then `uv run pytest --cov`.
- **Local install gotcha:** `uv pip install -e ".[dev]"` resolves `inspect-robots` from a
  git tag. To work against a sibling checkout instead:
  `uv pip install -e ../inspect-robots` (then `uv pip install -e . --no-deps`).
- Gates (all blocking in CI): `ruff check .`, `ruff format --check .`,
  `mypy` (strict), `pytest --cov` at **100%**.
- **mypy + numpy:** numpy 2.5's stubs use 3.12-only syntax that mypy (py3.10
  target) rejects; the dev extra pins `numpy<2.5` and CI runs mypy on 3.11.
- **No torch / no lerobot at import.** The model and driver live behind optional
  `lerobot` extras, lazily imported behind `# pragma: no cover` seams; the
  `import-hygiene` CI job enforces that `import inspect_robots_so101` works with only
  `inspect-robots` + `numpy`.

## Safety invariants (do not weaken)

- `SOArmEmbodiment.step()` **always clamps** to `SOArmConfig.joint_low/high` before
  commanding, independent of any `Approver` (and on top of LeRobot's
  `max_relative_target`). This is the last line of defense.
- LeRobot's postprocessor already unnormalizes to native motor units, so the
  embodiment does **not** renormalize the gripper — it commands clamped values
  verbatim. Keep it that way (units honesty: train and run in the same units).
- The declared `control_mode` is `joint_pos` (absolute). Delta checkpoints are
  converted to absolute *inside* `step()` (`joints_are_delta=True`) so the declared
  semantics stay honest. Compat cannot verify abs-vs-delta — that's a hardware check.
- Success reaches the scorer **only** via `StepResult.termination_reason="success"`
  (stock `rollout` never sets `operator_judgement`).

## Out of scope

Training/fine-tuning LeRobot policies (that's `huggingface/lerobot`), serving the
async gRPC `PolicyServer` (we run in process), and **bimanual** SO arms
(`bi_so_follower`) or non-SO LeRobot robots — this package is single-arm SO-ARM.
A bimanual sibling would mirror inspect-robots-yam's 14-D packing.

## CI, merging, and releases

- **main is PR-only** — a branch ruleset (admins included) blocks direct pushes,
  force pushes, and deletion. Merging requires the `ci-ok` check green and the
  branch up to date with main.
- **`ci-ok` is the single required status check** — an aggregate job at the end
  of `ci.yml`. When adding a CI job, add it to `ci-ok`'s `needs` list, or it
  will not gate merges.
- **Red main is stop-the-line**: if CI fails on a push to main, the
  `alert-red-main` job opens an issue. Fix forward or revert before merging
  anything else; if the failure was transient, re-run the failed jobs and close
  the issue.
- **CI installs from `uv.lock`** (`uv sync --locked`). After changing
  dependencies in `pyproject.toml`, run `uv lock` and commit the lockfile —
  otherwise CI fails with "the lockfile needs to be updated".
- A weekly **canary** (`canary.yml`) does the opposite: it installs the latest
  dependency versions the pyproject ranges allow (ignoring the lockfile), runs
  the tests, and opens an issue on failure — catching ecosystem breakage that
  locked CI can't see. A green canary means `uv lock --upgrade` is safe.
- Exception to locked installs: the `lerobot-seam` job installs with `uv pip`
  and `UV_TORCH_BACKEND=cpu` (uv sync cannot select the CPU torch wheel); its
  resolution is deliberately unlocked.
- **Releases are one-click**: Actions → Release → Run workflow → pick
  patch/minor/major. The version is derived from the git tag by hatch-vcs —
  never add a static `version =` back to pyproject (`__version__` comes from importlib.metadata). The same
  run publishes to PyPI via trusted publishing; nothing is pushed to main.
- **PyPI readme is transformed at build time** — `hatch-fancy-pypi-readme`
  rewrites GitHub-only alert syntax (`> [!NOTE]` etc.) in README.md into bold
  blockquotes (`> **Note:**`) that PyPI renders; keep using alert syntax in the
  README itself. Config lives at the bottom of pyproject.toml.
