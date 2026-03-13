# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES


import argparse
import os
import sys

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Custom Leg Runner.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==============================================================================
# [Path Setup]
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../../exts"))
sys.path.append(os.path.join(current_dir, "../../"))

import gymnasium as gym
import torch
import numpy as np

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

try:
    from exts.dreamwaq.learning.runners.dwaq_on_policy_runner import DwaqOnPolicyRunner
except ImportError:
    from exts.dreamwaq.learning.runners.dwaq_on_policy_runner import DwaqOnPolicyRunner


# ==============================================================================
# [CRITICAL FIX] PlayWrapper
# ==============================================================================
class PlayWrapper:
    """
    IsaacLab Env가 TensorDict를 반환할 때, 이를 (obs, extras) 튜플로 변환하여
    LegOnPolicyRunner와의 호환성 문제를 해결하는 래퍼입니다.
    """
    def __init__(self, env):
        self.env = env
        
    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        ret = self.env.get_observations()
        
        # [Case 1] TensorDict(혹은 dict)가 반환된 경우 (로그에 찍힌 상황)
        # 'policy' 키가 있다면 그 텐서를 obs로 추출합니다.
        if hasattr(ret, "keys") and "policy" in ret.keys():
            obs = ret["policy"]
            extras = {} # Runner 초기화를 위한 빈 딕셔너리
            return obs, extras
            
        # [Case 2] 튜플이 반환된 경우 (기존 호환성 유지)
        if isinstance(ret, tuple):
            # (obs, priv_obs, extras) -> priv 버리고 (obs, extras)
            if len(ret) == 3:
                return ret[0], ret[2]
            return ret
            
        # [Case 3] 그 외의 경우 (단일 텐서 등)
        return ret, {}
    
    def step(self, actions):
        ret = self.env.step(actions)
        
        # step 반환값 처리 (obs, rew, done, info) 등을 기대함
        # 만약 obs가 TensorDict라면 여기서도 추출해줘야 함
        
        if isinstance(ret, tuple):
            # 반환 값 개수에 따른 분기
            if len(ret) == 4:
                obs, rew, done, info = ret
            elif len(ret) == 5:
                # (obs, priv, rew, done, info) or (obs, rew, term, trunc, info)
                # 보통 RSL-RL 래퍼는 4개를 리턴하지만, 만약 5개라면:
                obs = ret[0]
                rew = ret[2]
                done = ret[3]
                info = ret[4]
            else:
                obs = ret[0]
                rew = ret[1]
                done = ret[2]
                info = ret[-1]

            # obs가 TensorDict면 policy 추출
            if hasattr(obs, "keys") and "policy" in obs.keys():
                obs = obs["policy"]
                
            return obs, rew, done, info

        return ret
    
    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()


def main():
    """Play with Custom Leg Agent (RMS)."""
    
    # 1. Config Setup
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 2. Checkpoint Loading
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # 3. Environment Creation
    from exts.dreamwaq.tasks import registry
    task_config = registry[args_cli.task]
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(resume_path), "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # # RSL-RL 호환 래퍼 적용
    # env = RslRlVecEnvWrapper(env)
    
    # # [핵심] PlayWrapper 적용 (TensorDict -> Tuple 변환)
    # env = PlayWrapper(env)
    for wrapper in task_config.env_wrappers:
        env = wrapper(env, args_cli=args_cli)

    # 4. Runner Initialization
    # PlayWrapper가 (obs, extras) 형태로 변환해주므로 에러 해결됨
    runner = DwaqOnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
    
    runner.load(resume_path, load_optimizer=False)
    runner.alg.actor_critic.eval()
    
    print(f"[INFO]: Successfully loaded custom agent.")

    # 5. Simulation Loop Setup
    obs, extras = env.get_observations()
    
    obs_rms = runner.obs_rms
    
    # History Buffer 초기화
    for i in range(runner.history_len):
        runner.obs_history_buffer[:, i, :] = obs.clone()

    vel_dim = getattr(runner, "est_vel_dim", 3)
    dummy_vel = torch.zeros(env.num_envs, vel_dim, device=runner.device)

    print("[INFO]: Starting inference loop...")
    # print(env.scene["robot"].data.joint_names)
    
    # 6. Main Loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # (A) Normalize Observation
            obs_norm = obs.clone()
            if obs_rms is not None:
                obs_norm = (obs_norm - obs_rms.mean) / torch.sqrt(obs_rms.var + 1e-6)
            
            # (B) Normalize History
            if obs_rms is not None:
                history_norm = (runner.obs_history_buffer - obs_rms.mean) / torch.sqrt(obs_rms.var + 1e-6)
            else:
                history_norm = runner.obs_history_buffer

            obs_history_flat = history_norm.view(env.num_envs, -1)


            # (D) Actor Input Construction
            actor_obs_input = obs_norm

            # (E) Action
            actions = runner.alg.actor_critic.act_inference(actor_obs_input, obs_history_flat)
            
            # (F) Step
            next_obs, rewards, dones, infos = env.step(actions)
            
            # (G) Update History
            runner.obs_history_buffer[:, :-1, :] = runner.obs_history_buffer[:, 1:, :].clone()
            runner.obs_history_buffer[:, -1, :] = next_obs.clone()

            obs_norm_next = next_obs.clone()
            if obs_rms is not None:
                obs_norm_next = (obs_norm_next - obs_rms.mean) / torch.sqrt(obs_rms.var + 1e-6)

            obs = next_obs

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()