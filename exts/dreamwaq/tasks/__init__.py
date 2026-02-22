# exts/flamingo/tasks/__init__.py

from dataclasses import dataclass, field
from typing import List, Type
import gymnasium as gym

from exts.dreamwaq.envs import wrappers
from exts.dreamwaq.learning import runners

from exts.dreamwaq.tasks.envs.env_cfg import RobotEnvCfg, EvalRobotEnvCfg, PlayRobotEnvCfg
from exts.dreamwaq.tasks.agents.rsl_rl_cfg import DwaqPpoRunnerCfg
from exts.dreamwaq.learning.runners.dwaq_on_policy_runner import DwaqOnPolicyRunner
from exts.dreamwaq.tasks.algorithms import RslRlOnPolicyRunnerCfg

@dataclass
class TaskConfig:
    env_cfg_entry_point: Type
    rsl_rl_cfg_entry_point: Type
    agent_cfg: Type = RslRlOnPolicyRunnerCfg
    runner: Type = runners.OnPolicyRunner
    env_wrappers: List[Type] = field(default_factory=lambda: [wrappers.RslRlVecEnvWrapper])


registry = {

    "dwaq": TaskConfig(
        env_cfg_entry_point=RobotEnvCfg,
        rsl_rl_cfg_entry_point=DwaqPpoRunnerCfg,
        runner=DwaqOnPolicyRunner,
    ),
    "dwaq_eval": TaskConfig(
        env_cfg_entry_point=EvalRobotEnvCfg,
        rsl_rl_cfg_entry_point=DwaqPpoRunnerCfg,
        runner=DwaqOnPolicyRunner,
    ),
    "dwaq_play": TaskConfig(
        env_cfg_entry_point=PlayRobotEnvCfg,
        rsl_rl_cfg_entry_point=DwaqPpoRunnerCfg,
        runner=DwaqOnPolicyRunner,
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