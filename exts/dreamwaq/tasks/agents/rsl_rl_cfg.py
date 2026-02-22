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
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    optimizer="Adam",
)

#  TODO : agent, train, play 수정필요

# =============================
#         SINGLE LEG WAQ
# =============================
@configclass
class DwaqPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for Single Leg WAQ (PPO + CENet) runner."""
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "dwaq"
    empirical_normalization = False  # Runner 내부에서 RMS를 직접 관리하므로 False
    
    policy = RslRlPpoPolicyCfg(
        class_name="ActorCriticDwaq",  # 문서화용 (실제 Runner에선 hardcoded class 사용)
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
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
    )