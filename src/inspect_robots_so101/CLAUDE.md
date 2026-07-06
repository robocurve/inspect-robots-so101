# `inspect_robots_so101` package — module map

Two Inspect Robots components + the glue to make them an honest, testable, safe pair.
The package is `mypy --strict` clean, ships `py.typed`, and is 100%-covered.

## Modules

| Module | Responsibility |
|--------|----------------|
| `packing.py` | **Pure** 6-D SO-ARM packing — the single source of truth for the motor order (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`) and how the flat vector maps to LeRobot's `"<motor>.pos"` dicts. `to_action_dict`/`from_obs_dict`/`validate_dim`, `STATE_KEY`, `STATE_SPEC`, `MOTORS`. No optional deps. |
| `config.py` | `SOArmConfig` / `LeRobotPolicyConfig` (frozen, `from_kwargs` for CLI scalars) + shared `action_box()` / `observation_space()` / `ACTION_SEMANTICS` so both components declare an **identical** contract. |
| `operator.py` | `OperatorIO` (injectable stdin/stdout) for readiness + success prompts; `default_poll_end` (real TTY poll, `# pragma: no cover`). |
| `policy.py` | `LeRobotPolicy` — wraps a LeRobot checkpoint. `act()` builds the RAW robot payload (`"<motor>.pos"` floats, frames keyed by camera name, `task`), runs the injectable `predict_fn`, truncates to `chunk_size`, returns an `ActionChunk`. Real in-process inference is `_default_predict` (lazy torch + lerobot; obs prep via lerobot's own `raw_observation_to_observation`) — NOT pragma'd: it is covered by `sys.modules`-fake tests, and its real import paths are validated by the `lerobot-seam` CI job. |
| `embodiment.py` | `SOArmEmbodiment` — LeRobot SO follower driver. Clamp backstop, optional delta→abs, `SELF_PACED` pacing, operator-keypress success, context manager (`with emb:` guarantees `close()`; `close()` clears the handle even if disconnect raises). The driver's `get_observation` yields motor positions **and** cameras. Hardware seam (`_default_driver_factory`) is injected/pragma'd; it connects with `calibrate=False` and fails fast via the tested `_check_calibrated` (never lerobot's blocking interactive calibration). |
| `preflight.py` | `build` / `run_preflight` + the `inspect-robots-so101-preflight` CLI: run the compat check, print, exit non-zero on errors. |
| `__init__.py` | Public API fenced by `__all__` (guarded by `tests/test_api_snapshot.py`). |

## Key invariants

- **Contract symmetry:** policy and embodiment build their `action_space` /
  `observation_space` from the *same* `config.py` helpers. If you change the dim,
  semantics, camera names, or state key, change them there once — not in two
  places — or compat breaks.
- **Construction is inert:** `__init__` touches no hardware/model/stdin (only
  `.info`). The driver connects, and the model loads, lazily on first use. This is
  what lets the registry (`factories[name]()`) and preflight construct components
  freely — and keeps `import inspect_robots_so101` free of torch.
- **Coverage discipline:** the only uncoverable code is hardware/TTY I/O,
  isolated in `# pragma: no cover` seams (`_default_driver_factory`,
  `default_poll_end`, the `_require_driver` pre-reset guard, `__main__`).
  `_default_predict` is NOT pragma'd — it is exercised with `sys.modules` fakes
  (see `tests/test_policy.py`), and the `lerobot-seam` CI job imports the real
  lerobot symbols it relies on. Prefer that pattern (fake the modules, assert the
  wiring, guard the imports in CI) over new pragmas.
- **Safety lives in `step()`**, not in an optional Approver — see the root
  `CLAUDE.md`.
