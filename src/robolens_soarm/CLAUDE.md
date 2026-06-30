# `robolens_soarm` package — module map

Two RoboLens components + the glue to make them an honest, testable, safe pair.
The package is `mypy --strict` clean, ships `py.typed`, and is 100%-covered.

## Modules

| Module | Responsibility |
|--------|----------------|
| `packing.py` | **Pure** 6-D SO-ARM packing — the single source of truth for the motor order (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`) and how the flat vector maps to LeRobot's `"<motor>.pos"` dicts. `to_action_dict`/`from_obs_dict`/`validate_dim`, `STATE_KEY`, `STATE_SPEC`, `MOTORS`. No optional deps. |
| `config.py` | `SOArmConfig` / `LeRobotPolicyConfig` (frozen, `from_kwargs` for CLI scalars) + shared `action_box()` / `observation_space()` / `ACTION_SEMANTICS` so both components declare an **identical** contract. |
| `operator.py` | `OperatorIO` (injectable stdin/stdout) for readiness + success prompts; `default_poll_end` (real TTY poll, `# pragma: no cover`). |
| `policy.py` | `LeRobotPolicy` — wraps a LeRobot checkpoint. `act()` builds the LeRobot observation (`observation.state`, `observation.images.<cam>`, `task`), runs the injectable `predict_fn`, returns an `ActionChunk`. Real in-process inference is the pragma'd `_default_predict` (lazy torch + lerobot). |
| `embodiment.py` | `SOArmEmbodiment` — LeRobot SO follower driver. Clamp backstop, optional delta→abs, `SELF_PACED` pacing, operator-keypress success. The driver's `get_observation` yields motor positions **and** cameras. Hardware seam (`_default_driver_factory`) is injected/pragma'd. |
| `preflight.py` | `build` / `run_preflight` + the `robolens-soarm-preflight` CLI: run the compat check, print, exit non-zero on errors. |
| `__init__.py` | Public API fenced by `__all__` (guarded by `tests/test_api_snapshot.py`). |

## Key invariants

- **Contract symmetry:** policy and embodiment build their `action_space` /
  `observation_space` from the *same* `config.py` helpers. If you change the dim,
  semantics, camera names, or state key, change them there once — not in two
  places — or compat breaks.
- **Construction is inert:** `__init__` touches no hardware/model/stdin (only
  `.info`). The driver connects, and the model loads, lazily on first use. This is
  what lets the registry (`factories[name]()`) and preflight construct components
  freely — and keeps `import robolens_soarm` free of torch.
- **Coverage discipline:** the only uncoverable code is hardware/model/TTY I/O,
  isolated in `# pragma: no cover` seams (`_default_predict`,
  `_default_driver_factory`, `default_poll_end`, the `_require_driver` pre-reset
  guard, `__main__`). Keep new hardware/model access inside such seams so the 100%
  gate stays meaningful.
- **Safety lives in `step()`**, not in an optional Approver — see the root
  `CLAUDE.md`.
