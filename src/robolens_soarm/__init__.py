"""robolens-soarm — RoboLens adapters for LeRobot SO-ARM followers + LeRobot policies.

Registers two RoboLens components via entry points:

* embodiment ``so_arm`` — :class:`~robolens_soarm.embodiment.SOArmEmbodiment`
* policy ``lerobot`` — :class:`~robolens_soarm.policy.LeRobotPolicy`

so ``robolens run --task cubepick-reach --policy lerobot --embodiment so_arm``
works once both packages are installed. Use
:func:`~robolens_soarm.preflight.run_preflight` (or the ``robolens-soarm-preflight``
CLI) to verify compatibility before any motion.
"""

from __future__ import annotations

from robolens_soarm.config import LeRobotPolicyConfig, SOArmConfig
from robolens_soarm.embodiment import SOArmEmbodiment
from robolens_soarm.operator import OperatorIO
from robolens_soarm.packing import MOTORS, STATE_KEY, TOTAL_DIM, from_obs_dict, to_action_dict
from robolens_soarm.policy import LeRobotPolicy
from robolens_soarm.preflight import build, run_preflight

__version__ = "0.1.0"

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
