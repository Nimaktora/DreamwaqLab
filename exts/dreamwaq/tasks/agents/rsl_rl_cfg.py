# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the isaaclab repository:
# https://github.com/isaac-sim/IsaacLab/
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications by Amr Mousa (2025).

from isaaclab.utils import configclass

from exts.dreamwaq.tasks.algorithms import *

# =============================
#         MLP POLICIES
# =============================

ppo_algo_cfg = RslRlPpoAlgorithmCfg(
    class_name="PPO",
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    # lr_max=1.0e-3,
    # lr_min=1.0e-5,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    optimizer="Adam",
)

@configclass
class DwaqPpoRunnerCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "dwaq"
    empirical_normalization = False  # Runner 내부에서 RMS를 직접 관리하므로 False
    
    policy = RslRlPpoPolicyCfg(
        class_name="ActorCriticDwaq", 
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOWAQ",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        # lr_max=1.0e-3,
        # lr_min=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
    )