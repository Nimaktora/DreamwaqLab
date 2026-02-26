# exts/flamingo/envs/env_cfg.py

from isaaclab.utils import configclass
from .base import BaseRobotEnvCfg

# base environment configuration class for training
@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    def __init__(self, num_envs=4096, **kwargs):
        super().__init__(**kwargs)
        self.num_envs = num_envs  # Environment Count for Training
        self.scene.num_envs = num_envs
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.num_cols = 20
        self.scene.terrain.terrain_generator.seed = 0
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.use_graphics = True  # Ensure graphics are enabled for training
        self.base_velocity = 0.5  # Some default value
        self.debug_vis = True  # Turn on debug visualization for training purposes

    def __post_init__(self):
        super().__post_init__()


# Separate configuration class for play environment (e.g., with fewer envs and less rendering overhead)
@configclass
class PlayRobotEnvCfg(RobotEnvCfg):
    def __init__(self, num_envs=4, **kwargs):
        super().__init__(**kwargs)
        self.num_envs = num_envs  # Fewer environments for play testing
        self.scene.num_envs = num_envs
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.num_cols = 20
        self.use_graphics = True  # Ensure graphics are enabled for training
        self.base_velocity = 3.0  # Some default value
        self.debug_vis = False  # Turn on debug visualization for training purposes

    def __post_init__(self):
        super().__post_init__()


# Separate configuration class for evaluation (for benchmarking or final tests)
@configclass
class EvalRobotEnvCfg(RobotEnvCfg):
    def __init__(self, num_envs=1, **kwargs):
        super().__init__(**kwargs)
        self.num_envs = num_envs  # Only one environment for evaluation
        self.scene.num_envs = num_envs
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.use_graphics = False  # Disable graphics during evaluation for speed
        self.base_velocity = 0.2  # Slower speeds during evaluation
        self.debug_vis = False  # Disable debug visuals in evaluation


    def __post_init__(self):
        super().__post_init__()


