# 0002: Support lerobot 0.7 to clear the Dependabot alerts

## Motivation

~55 open Dependabot alerts on `uv.lock` (Pillow critical/high, virtualenv high,
setuptools, torch). None can be fixed by relocking alone: every patched version
is excluded by lerobot 0.6.1's own pins, and we cap `lerobot<0.7`.

| package | patched | blocked by (lerobot 0.6.1) |
|---|---|---|
| Pillow | 13.0.0 | `Pillow<13` |
| torch | 2.13.0 | `torch<2.12` |
| setuptools | 84.1.0 | `setuptools<82` |
| virtualenv (via pre-commit) | 21.14.9 | needs `packaging>=26.3`; lerobot pins `packaging<26` |

lerobot 0.7.0 relaxes all four (`Pillow<14`, `torch<2.16`, `setuptools<85`,
`packaging<27`). Raising the cap to `<0.8` resolves the lock to
lerobot 0.7.0, Pillow 13.0.0, torch 2.15.0, setuptools 84.2.1, virtualenv 21.15.6.
Dependabot PR #33 (Pillow only) cannot resolve for the same reason; close it.

## What breaks on 0.7

Checked by installing lerobot 0.7.0 (py3.12, CPU torch) and running the
`lerobot-seam` CI script:

1. `lerobot.async_inference` is gone, so
   `from lerobot.async_inference.helpers import raw_observation_to_observation`
   raises `ModuleNotFoundError`. 0.7 has no drop-in replacement: the closest,
   `lerobot.policies.utils.build_inference_frame`, does not resize images to the
   policy's image-feature shape and moves tensors to the device itself.
2. The `async` extra no longer exists (0.7 has `grpcio-dep`). We only needed it
   because `async_inference` imported grpc at module load.

Everything else the seams use is unchanged in 0.7 (verified by diffing the
0.6.1 and 0.7.0 wheels):

- `lerobot.utils.feature_utils.build_dataset_frame`: byte-identical.
- `lerobot.utils.feature_utils.hw_to_dataset_features`: type hints only.
- `get_policy_class`, `make_pre_post_processors(..., preprocessor_overrides=,
  postprocessor_overrides=)`: same signature.
- `SOFollower` / `SOFollowerRobotConfig`: same import paths; `get_observation`
  still uses `cam.read_latest()`, `send_action` / `connect` / `disconnect`
  unchanged in signature. **But the motor stack underneath is new**:
  `SOFollower` now builds a `SerialMotorsBus` backed by the `rustypot` Rust
  package instead of `FeetechMotorsBus`, and `is_calibrated` was
  reimplemented. Every read, write, unit conversion and calibration check on
  the arm runs through new upstream code.

## Change

### `src/inspect_robots_so101/policy.py`

Replace the `raw_observation_to_observation` import with a private
`_raw_to_policy_observation(raw, lerobot_features, policy_image_features)` that
reproduces lerobot 0.6.1's helper exactly (it is ~25 lines):

1. `frame = build_dataset_frame(lerobot_features, dict(raw), prefix="observation")`
2. `state = torch.tensor(frame["observation.state"])`, unsqueeze to `(1, D)`.
3. For each `observation.images.*` key in `frame`: `torch.tensor(img)` (HWC
   uint8) → permute to CHW → bilinear `interpolate` (`align_corners=False`) to
   `policy_image_features[key].shape[1:]` → `/255` to float32 if uint8 →
   `.contiguous()` → unsqueeze batch dim → `(1, C, H, W)`.
4. Carry `raw["task"]` through as `"task"` when present.

Ordering note: 0.6.1 resizes first (bilinear `interpolate` directly on the
uint8 tensor, which torch supports on CPU), then scales to float. Mirror that
order exactly rather than "fixing" it, so outputs match 0.6.1 bit for bit;
the local verification step compares both paths on the same input under
lerobot 0.6.1. Device placement stays with the preprocessor's
`device_processor` (as today).

Import `build_dataset_frame` through a version-tolerant helper mirroring
`_import_hw_to_dataset_features` (same three module paths for 0.5.0 / 0.5.1 /
>=0.6), or fold both into one `_import_feature_utils()` returning both
functions. Keep everything lazy (no torch/lerobot at module import).

Update the module docstring, the `LeRobotObs` comment, and `_default_predict`'s
docstring: obs prep is now ours, mirroring 0.6.1's helper, because lerobot 0.7
removed it.

### `pyproject.toml`

`lerobot = ["lerobot[feetech]>=0.5,<0.8; python_version >= '3.12'"]` (drop
`async`; update the comment block: the cap reason now cites 0.7's removal of
`async_inference`, and the `async` sentence goes).

### `uv.lock`

`uv lock --upgrade-package lerobot --upgrade-package pillow --upgrade-package torch
--upgrade-package torchvision --upgrade-package setuptools --upgrade-package virtualenv`.

### `.github/workflows/ci.yml` (`lerobot-seam`)

- Drop the `async_inference` import and its print; instead import
  `build_dataset_frame` via the new helper.
- Add a real-lerobot smoke of `_raw_to_policy_observation` on CPU: fake raw obs
  (6 `.pos` floats, one 480x640x3 uint8 frame, task), features from
  `hw_to_dataset_features`, a `PolicyFeature`-like shape `(3, 224, 224)`; assert
  state shape `(1, 6)`, image shape `(1, 3, 224, 224)`, float32 in [0, 1],
  `task` carried. This is the drift guard the removed import used to provide.
- Same smoke with a float32 frame, to cover the non-uint8 branch.
- Construct `SOFollower(SOFollowerRobotConfig(port="/dev/null"))` without
  `connect()`: the bus constructor resolves each motor model through
  `rustypot`, so a missing wheel or unknown `sts3215` fails here.
- Update the comment at line ~63.

### Tests (`tests/test_policy.py`)

- Remove the `lerobot.async_inference` fake modules; the fake `lerobot` now
  needs `build_dataset_frame` in each `feature_utils` layout.
- The existing `_default_predict` wiring tests install a fake torch without
  `tensor`/`permute`/`interpolate`, so they monkeypatch
  `policy._raw_to_policy_observation` with a recording stub and assert it got
  `(raw, lerobot_features, policy_image_features)` and that its return value is
  what reached the preprocessor.
- The import helper is tested in all three layouts (0.5.0 `datasets.utils`,
  0.5.1 `datasets.feature_utils`, >=0.6 `utils.feature_utils`); verified that
  `build_dataset_frame` sits beside `hw_to_dataset_features` in each wheel.
- `_raw_to_policy_observation` needs real torch, which tests don't have. Mark
  it `# pragma: no cover` (a fake torch with permute/interpolate/dtype would
  only test itself) and rely on the CI smoke above, which runs it against real
  torch + lerobot for both uint8 and float32 frames.

### Docs

`config.py:42` comment: "v0.5 through v0.6" → through v0.7.

Root `CLAUDE.md` and `src/inspect_robots_so101/CLAUDE.md`: replace
"lerobot's own `raw_observation_to_observation`" with the new helper and why.
`src/inspect_robots_so101/CLAUDE.md` "Coverage discipline": add the new
pragma to the allowed list (needs real torch; covered by the `lerobot-seam`
smoke). README.md:205: "lerobot v0.5–0.6" → "v0.5–0.7" (0.7 still aliases
`SO100Follower = SO101Follower = SOFollower`). Changelog: repo
has no `changelog.d/`; none.

## Out of scope

- Bumping the lerobot floor. 0.5.x/0.6.x keep working (helper only uses
  `build_dataset_frame`, present in all three layouts).
- Real-hardware validation: no SO-101 rig in CI. Obs prep is a line-for-line
  port checked by the CI smoke, but the 0.7 motor bus rewrite is only
  constructor-tested. Before the next release, a rig check is required:
  `connect()` against existing calibration files (`is_calibrated` true, no
  recalibration prompt), a few small moves, positions read back in the same
  units as on 0.6.1.

## Verification

- `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov` locally.
- Local py3.12 venv with `lerobot[feetech]==0.7.0` (CPU torch): run the updated
  seam script. Same with `lerobot==0.6.1` to confirm the floor still works.
- CI green on the PR; Dependabot alerts close after merge.
