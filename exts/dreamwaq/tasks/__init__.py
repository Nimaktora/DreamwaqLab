# exts/flamingo/tasks/__init__.py

from dataclasses import dataclass, field
from typing import List, Type
import gymnasium as gym

from exts.dreamwaq.envs import wrappers
from exts.dreamwaq.learning import runners

from exts.dreamwaq.tasks.envs.env_cfg import SingleLegRoughEnvCfg, SingleLegRoughEnvEvalCfg
from exts.dreamwaq.tasks.agents.rsl_rl_cfg import SingleLegWaqPpoRunnerCfg
from exts.dreamwaq.learning.runners.dwaq_on_policy_runner import LegOnPolicyRunner
from exts.dreamwaq.tasks.algorithms import RslRlOnPolicyRunnerCfg

@dataclass
class TaskConfig:
    env_cfg_entry_point: Type
    rsl_rl_cfg_entry_point: Type
    agent_cfg: Type = RslRlOnPolicyRunnerCfg
    runner: Type = runners.OnPolicyRunner
    env_wrappers: List[Type] = field(default_factory=lambda: [wrappers.RslRlVecEnvWrapper])


registry = {

    "one_leg": TaskConfig(
        env_cfg_entry_point=SingleLegRoughEnvCfg,
        rsl_rl_cfg_entry_point=SingleLegWaqPpoRunnerCfg,
        runner=LegOnPolicyRunner,
    ),
    "one_leg_eval": TaskConfig(
        env_cfg_entry_point=SingleLegRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=SingleLegWaqPpoRunnerCfg,
        runner=LegOnPolicyRunner,
    ),
}

# Register each environment
for env_id, config in registry.items():
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": config.env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": config.rsl_rl_cfg_entry_point,
        },
    )

__all__ = ["registry"]