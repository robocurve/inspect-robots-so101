<div align="center">

# inspect-robots-so101

Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals on real
[SO-ARM](https://github.com/TheRobotStudio/SO-ARM100) followers (SO-100 / SO-101)
driven by [LeRobot](https://github.com/huggingface/lerobot) policies.

![Status: alpha](https://img.shields.io/badge/status-alpha-blue)
[![CI](https://github.com/robocurve/inspect-robots-so101/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots-so101/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/inspect-robots-so101/actions/workflows/ci.yml)
[![Docs coverage](https://img.shields.io/badge/public%20docstrings-100%25-brightgreen)](https://github.com/robocurve/inspect-robots-so101/actions/workflows/ci.yml)
[![Built on Inspect Robots](https://img.shields.io/badge/built%20on-Inspect%20Robots-indigo)](https://github.com/robocurve/inspect-robots)

</div>

> [!NOTE]
> This project is in early development. The API may change between releases, so pin a version before depending on it.

Inspect Robots has two swappable inputs: a `Policy` (the VLA brain) and an
`Embodiment` (the robot body + world). This package provides both for the
SO-ARM + LeRobot stack, so any embodiment-agnostic Inspect Robots task runs on a real
arm:

- **`lerobot` policy**: wraps a LeRobot checkpoint (ACT, SmolVLA, π0, diffusion…)
  and runs it in process on the GPU, returning an action chunk per inference.
- **`so_arm` embodiment**: the LeRobot SO follower driver (Feetech bus), with a
  hard safety clamp, operator-in-the-loop success, and self-paced control.

Both declare the same 6-D joint-position contract (`shoulder_pan`,
`shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`; the cameras
you configure; packed `joint_pos` state), so Inspect Robots's compatibility check passes
with zero errors and zero warnings, verifiable before any motion.

```bash
inspect-robots run --task cubepick-reach --policy lerobot --embodiment so_arm
```

> This is the SO-ARM/LeRobot sibling of
> [inspect-robots-yam](https://github.com/robocurve/inspect-robots-yam) (bimanual I2RT YAM +
> MolmoAct2). Same Inspect Robots contract, different body and brain.

## Install (on the robot/GPU machine)

```bash
# Inspect Robots isn't on PyPI yet; uv resolves it from git. The `lerobot` extra pulls
# torch + lerobot + the Feetech motor bus the SO follower uses.
uv pip install "inspect-robots-so101[lerobot] @ git+https://github.com/robocurve/inspect-robots-so101"
```

- `lerobot` → `lerobot[feetech]` (torch, the policy, and the SO-ARM driver).
- The `lerobot` extra needs Python ≥ 3.12 (lerobot ≥ 0.5's floor). On
  3.10/3.11 the extra silently resolves to *nothing*: the core package still
  imports, but no torch/lerobot is installed and hardware runs will fail.

Then pick a checkpoint. Any LeRobot policy trained on your SO-ARM works, e.g. the
public `lerobot/smolvla_base`, or your own ACT/π0 checkpoint on the Hub or a path.

## Preflight: prove compatibility before any motion

```bash
inspect-robots-so101-preflight                          # dims/semantics/cameras/state
inspect-robots-so101-preflight --task cubepick-reach    # + scene realizability
inspect-robots-so101-preflight --dry-run                # affirm no motion
```

A green preflight means action dim (6), control mode (`joint_pos`), cameras, and
state keys all line up. It does not prove the joint values are interpreted the
same way (see *Safety* below).

## Calibrate first (once, with lerobot)

The embodiment never runs lerobot's interactive calibration: connecting with
an uncalibrated arm would otherwise drop into a *blocking* prompt that moves the
arm mid-eval. Calibrate once with lerobot's own tool, then tell the config which
identity you used:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
```

`SOArmConfig(robot_id="my_follower")` selects that calibration file
(`<calibration_dir>/<robot_id>.json`; leave `calibration_dir=None` for lerobot's
default location). If the arm isn't calibrated, or the file no longer matches
the motors, `reset()` fails fast with an actionable error instead of prompting.

## Run on hardware

You must point the embodiment at your serial port, calibration id, and camera
config, and the policy at a checkpoint:

```python
from inspect_robots import eval
from inspect_robots.approver import ClampApprover
from inspect_robots_so101 import LeRobotPolicy, SOArmEmbodiment, SOArmConfig, LeRobotPolicyConfig
from lerobot.cameras.opencv import OpenCVCameraConfig  # your camera backend

emb = SOArmEmbodiment(SOArmConfig(
    port="/dev/ttyACM0",
    robot_type="so101_follower",
    robot_id="my_follower",    # the id you ran `lerobot-calibrate` with
    max_relative_target=10.0,  # native-unit slew limit; required for home_pose
    cameras=("front",),
    camera_configs={"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)},
))
pol = LeRobotPolicy(LeRobotPolicyConfig(
    pretrained_path="lerobot/smolvla_base", policy_type="smolvla", device="cuda",
))

with emb:  # guarantees disconnect (and torque-off) even if the eval raises
    (log,) = eval("cubepick-reach", pol, emb,
                  approver=ClampApprover(emb.info.action_space))  # defense in depth
print(log.status, log.results.metrics)
```

(Equivalently, wrap the `eval(...)` in `try: ... finally: emb.close()`.)

At each episode end the embodiment asks the operator (y/N); a `yes` records
`termination_reason="success"`, which the task's `success_at_end` scorer reads.
Unattended runs simply run to `max_steps` and score as failures.

## Safety

- **Hard clamp backstop.** Every command is clipped to `SOArmConfig.joint_low/high`
  *inside* `step()`, independent of any Inspect Robots `Approver` and on top of LeRobot's
  own `max_relative_target` slew limit. Unclamped model outputs can never reach
  the motors. **Set these to your real, calibrated SO-ARM joint limits** (the
  defaults are conservative placeholders: joints ±180° in degree mode or ±100
  in normalized mode, with the gripper at 0–100 in both modes).
- **Use `ClampApprover`** on hardware for a second layer.
- **Native units, no renormalization.** The embodiment commands the policy
  output verbatim after the clamp. Set the same `use_degrees` value on
  `SOArmConfig` and `LeRobotPolicyConfig`: `True` uses degrees for arm joints,
  while `False` uses LeRobot's normalized ±100 joint positions. The gripper is
  0–100 in both modes. The declared state specification and automatic clamp
  bounds follow that value. Inspect Robots compatibility currently compares
  state keys, not units, so it will not flag different values across components
  or unit mismatches with third-party counterparts. Verify units when mixing
  stacks.
- **Homing is slew-limited or refused.** `home_pose` sends a single absolute
  command, so the config requires `max_relative_target` (LeRobot's per-step slew
  limit) whenever `home_pose` is set. Otherwise the arm would slam to home at
  full speed from wherever it happens to be.
- **Absolute vs. delta joints: verify first.** Actions are treated as absolute
  joint targets by default. If your checkpoint emits deltas, set
  `SOArmConfig(joints_are_delta=True)` (the embodiment converts to absolute
  internally so the declared `joint_pos` stays honest). The compat check *cannot*
  tell these apart. Confirm with `--dry-run` and a single slow jog before a task.

## Configuration

`SOArmConfig`: `port`, `robot_type`, `robot_id`, `calibration_dir`, `cameras`,
`camera_configs`, `control_hz`, `cam_height/width`, `joint_low/high`,
`home_pose` (requires `max_relative_target`), `joints_are_delta`, `use_degrees`
(defaults to `True`), `max_relative_target`, `disable_torque_on_disconnect`.
`robot_type` is validated (`so101_follower` / `so100_follower`) but is a label:
at lerobot v0.5.x both names alias the same driver class, so it changes no
runtime behavior.
`LeRobotPolicyConfig`: `pretrained_path`, `policy_type`, `device`, `cameras`,
`state_key`, `chunk_size`, `cam_height/width`, `use_degrees`. Set its
`use_degrees` value to match the embodiment.

Scalar knobs are settable from the CLI:
`inspect-robots run -P pretrained_path=lerobot/smolvla_base -E port=/dev/ttyACM0 ...`.

## Development

> **Dependency changes:** after editing dependencies in `pyproject.toml`, run
> `uv lock` and commit the updated lockfile: CI installs with
> `uv sync --locked` and fails with "the lockfile needs to be updated" if you
> forget. Day-to-day conventions (PR-only `main`, the required `ci-ok` check,
> one-click releases) are documented in [`CLAUDE.md`](CLAUDE.md).

Every public module, class, and function needs a docstring, enforced by Ruff D1;
state the contract, do not restate the name.

```bash
uv venv && uv pip install -e ".[dev]"     # inspect-robots from a git tag
uv run pre-commit install
uv run pytest --cov                        # 100% coverage required
uv run ruff check . && uv run mypy
```

The whole suite runs with no hardware, no GPU, no torch, no lerobot, and no
stdin: the SO-ARM driver, the policy inference, the clock, and operator I/O are
all injected. The real model seam (`_default_predict`) is covered via
`sys.modules` fakes, and a dedicated `lerobot-seam` CI job (py3.12) imports the
real lerobot symbols it uses; only direct hardware/TTY I/O keeps
`# pragma: no cover`.

## License

[MIT](LICENSE)
