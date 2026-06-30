# robolens-soarm — agent guide

RoboLens adapters that let evals run on real **LeRobot SO-ARM followers**
(SO-100 / SO-101) driven by **LeRobot policies**. This is a **plugin package** in
the RoboLens ecosystem — the framework lives in
[robolens](https://github.com/robocurve/robolens); benchmarks are separate repos
indexed by [WorldEvals](https://github.com/robocurve/worldevals). It is the
SO-ARM/LeRobot sibling of
[robolens-yam](https://github.com/robocurve/robolens-yam) (bimanual YAM +
MolmoAct2).

## The one big idea

RoboLens evals swap two inputs: a `Policy` (VLA brain) and an `Embodiment` (robot
body + world). We ship both for one real stack:

- **`lerobot` policy** — wraps a LeRobot checkpoint (ACT / SmolVLA / π0 / …) and
  runs it **in process** (`predict_action_chunk` + the model's pre/post
  processors). Unlike YAM/MolmoAct2 (a separate HTTP server), LeRobot models are a
  library you import — so torch + lerobot + the checkpoint live behind **one
  injectable seam**, `LeRobotPolicy(predict_fn=...)`. We never import torch at
  module top.
- **`so_arm` embodiment** — the LeRobot SO follower driver. Its `get_observation`
  already returns both `"<motor>.pos"` floats *and* camera frames, so there is no
  separate camera reader (a small simplification over robolens-yam).

Both declare the **same 6-D `joint_pos` contract** (5 revolute joints + gripper,
the configured cameras, packed `joint_pos` state). That makes
`robolens.compat.check_compatibility` pass with zero errors **and** zero warnings
— the property `tests/test_compat.py` locks down.

## Layout

- `src/robolens_soarm/` — the package (see `src/robolens_soarm/CLAUDE.md`).
- `tests/` — pytest; everything (driver, cameras, model inference, clock, operator
  stdin) is injected, so the suite needs **no hardware, no GPU, no torch, no
  lerobot, no stdin**. The end-to-end test uses RoboLens's built-in
  `cubepick-reach` task so it stays self-contained.

## Working here

- Dev loop: `uv venv && uv pip install -e ".[dev]"`, `uv run pre-commit install`,
  then `uv run pytest --cov`.
- **Local install gotcha:** `uv pip install -e ".[dev]"` resolves `robolens` from a
  git tag. To work against a sibling checkout instead:
  `uv pip install -e ../robolens` (then `uv pip install -e . --no-deps`).
- Gates (all blocking in CI): `ruff check .`, `ruff format --check .`,
  `mypy` (strict), `pytest --cov` at **100%**.
- **mypy + numpy:** numpy 2.5's stubs use 3.12-only syntax that mypy (py3.10
  target) rejects; the dev extra pins `numpy<2.5` and CI runs mypy on 3.11.
- **No torch / no lerobot at import.** The model and driver live behind optional
  `lerobot` extras, lazily imported behind `# pragma: no cover` seams; the
  `import-hygiene` CI job enforces that `import robolens_soarm` works with only
  `robolens` + `numpy`.

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
A bimanual sibling would mirror robolens-yam's 14-D packing.
