<div align="center">

# 🦾 inspect-robots-so101

**Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals on real
[SO-ARM](https://github.com/TheRobotStudio/SO-ARM100) followers (SO-100 / SO-101)
driven by [LeRobot](https://github.com/huggingface/lerobot) policies.**

[![CI](https://github.com/robocurve/inspect-robots-so101/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots-so101/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/inspect-robots-so101/actions/workflows/ci.yml)
[![Built on Inspect Robots](https://img.shields.io/badge/built%20on-Inspect%20Robots-indigo)](https://github.com/robocurve/inspect-robots)

</div>

Inspect Robots has **two** swappable inputs: a `Policy` (the VLA brain) and an
`Embodiment` (the robot body + world). This package provides both for the
SO-ARM + LeRobot stack, so any embodiment-agnostic Inspect Robots task runs on a real
arm:

- **`lerobot` policy** — wraps a LeRobot checkpoint (ACT, SmolVLA, π0, diffusion…)
  and runs it **in process** on the GPU, returning an action chunk per inference.
- **`so_arm` embodiment** — the LeRobot SO follower driver (Feetech bus), with a
  hard safety clamp, operator-in-the-loop success, and self-paced control.

Both declare the **same 6-D joint-position contract** (`shoulder_pan`,
`shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`; the cameras
you configure; packed `joint_pos` state), so Inspect Robots's compatibility check passes
with **zero errors and zero warnings** — verifiable before any motion.

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

Then pick a checkpoint. Any LeRobot policy trained on your SO-ARM works — e.g. the
public `lerobot/smolvla_base`, or your own ACT/π0 checkpoint on the Hub or a path.

## Preflight — *prove compatibility before any motion*

```bash
inspect-robots-so101-preflight                          # dims/semantics/cameras/state
inspect-robots-so101-preflight --task cubepick-reach    # + scene realizability
inspect-robots-so101-preflight --dry-run                # affirm no motion
```

A green preflight means action dim (6), control mode (`joint_pos`), cameras, and
state keys all line up. **It does not prove the joint values are interpreted the
same way** — see *Safety* below.

## Run on hardware

You must point the embodiment at your serial port and camera config, and the
policy at a checkpoint:

```python
from inspect_robots import eval
from inspect_robots.approver import ClampApprover
from inspect_robots_so101 import LeRobotPolicy, SOArmEmbodiment, SOArmConfig, LeRobotPolicyConfig
from lerobot.cameras.opencv import OpenCVCameraConfig  # your camera backend

emb = SOArmEmbodiment(SOArmConfig(
    port="/dev/ttyACM0",
    robot_type="so101_follower",
    cameras=("front",),
    camera_configs={"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)},
))
pol = LeRobotPolicy(LeRobotPolicyConfig(
    pretrained_path="lerobot/smolvla_base", policy_type="smolvla", device="cuda",
))

(log,) = eval("cubepick-reach", pol, emb,
              approver=ClampApprover(emb.info.action_space))  # defense in depth
print(log.status, log.results.metrics)
```

At each episode end the embodiment asks the operator (y/N); a `yes` records
`termination_reason="success"`, which the task's `success_at_end` scorer reads.
Unattended runs simply run to `max_steps` and score as failures.

## Safety

- **Hard clamp backstop.** Every command is clipped to `SOArmConfig.joint_low/high`
  *inside* `step()`, independent of any Inspect Robots `Approver` and on top of LeRobot's
  own `max_relative_target` slew limit — unclamped model outputs can never reach
  the motors. **Set these to your real, calibrated SO-ARM joint limits** (the
  defaults are conservative placeholders: joints ±180°, gripper 0–100).
- **Use `ClampApprover`** on hardware for a second layer.
- **Native units, no renormalization.** LeRobot's postprocessor unnormalizes the
  policy output to the robot's native motor units (degrees for joints, 0–100 for
  the gripper), so the embodiment commands them verbatim after the clamp. Train
  and run the policy in the *same* units (`use_degrees` must match your dataset).
- **Absolute vs. delta joints — verify first.** Actions are treated as **absolute**
  joint targets by default. If your checkpoint emits deltas, set
  `SOArmConfig(joints_are_delta=True)` (the embodiment converts to absolute
  internally so the declared `joint_pos` stays honest). The compat check *cannot*
  tell these apart — confirm with `--dry-run` and a single slow jog before a task.

## Configuration

`SOArmConfig`: `port`, `robot_type`, `cameras`, `camera_configs`, `control_hz`,
`cam_height/width`, `joint_low/high`, `home_pose`, `joints_are_delta`,
`use_degrees`, `max_relative_target`, `disable_torque_on_disconnect`.
`LeRobotPolicyConfig`: `pretrained_path`, `policy_type`, `device`, `cameras`,
`state_key`, `chunk_size`, `cam_height/width`.

Scalar knobs are settable from the CLI:
`inspect-robots run -P pretrained_path=lerobot/smolvla_base -E port=/dev/ttyACM0 ...`.

## Development

```bash
uv venv && uv pip install -e ".[dev]"     # inspect-robots from a git tag
uv run pre-commit install
uv run pytest --cov                        # 100% coverage required
uv run ruff check . && uv run mypy
```

The whole suite runs with **no hardware, no GPU, no torch, no lerobot, and no
stdin** — the SO-ARM driver, the policy inference, the clock, and operator I/O are
all injected. The real model seam (`_default_predict`) is covered via
`sys.modules` fakes, and a dedicated `lerobot-seam` CI job (py3.12) imports the
real lerobot symbols it uses; only direct hardware/TTY I/O keeps
`# pragma: no cover`.

## License

[MIT](LICENSE)
