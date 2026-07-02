"""inspect-robots-so101 — Inspect Robots adapters for LeRobot SO-ARM followers + LeRobot policies.

Registers two Inspect Robots components via entry points:

* embodiment ``so_arm`` — :class:`~inspect_robots_so101.embodiment.SOArmEmbodiment`
* policy ``lerobot`` — :class:`~inspect_robots_so101.policy.LeRobotPolicy`

so ``inspect-robots run --task cubepick-reach --policy lerobot --embodiment so_arm``
works once both packages are installed. Use
:func:`~inspect_robots_so101.preflight.run_preflight` (or the ``inspect-robots-so101-preflight``
CLI) to verify compatibility before any motion.
"""

from __future__ import annotations

from inspect_robots_so101.config import LeRobotPolicyConfig, SOArmConfig
from inspect_robots_so101.embodiment import SOArmEmbodiment
from inspect_robots_so101.operator import OperatorIO
from inspect_robots_so101.packing import MOTORS, STATE_KEY, TOTAL_DIM, from_obs_dict, to_action_dict
from inspect_robots_so101.policy import LeRobotPolicy
from inspect_robots_so101.preflight import build, run_preflight

__version__ = "0.3.0"

__all__ = [
    "MOTORS",
    "STATE_KEY",
    "TOTAL_DIM",
    "LeRobotPolicy",
    "LeRobotPolicyConfig",
    "OperatorIO",
    "SOArmConfig",
    "SOArmEmbodiment",
    "build",
    "from_obs_dict",
    "run_preflight",
    "to_action_dict",
]
