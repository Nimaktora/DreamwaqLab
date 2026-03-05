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

"""Script to train RL agent with RSL-RL."""
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import pickle  # <--- [추가]
import yaml    # <--- [추가]
import argparse
import logging
import os
import sys
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)


## ~ 환경 설정, 아이작 심 실행
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# my arguments
parser.add_argument(
    "--num_steps_per_env",
    type=int,
    default=24,
    help="Number of steps per environment for each training iteration.",
)
parser.add_argument("--num_iters", type=int, default=1500, help="Number of training iterations.")
parser.add_argument(
    "--policy_name",
    type=str,
    default=None,
    help="Name of the policy class.",
)
parser.add_argument(
    "--teacher_path",
    type=str,
    default=None,
    help="Path for teacher model to use.",
)
parser.add_argument(
    "--logging",
    type=str,
    default="WARNING",
    help="Logging level shall be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
)
parser.add_argument(
    "--wandb_continue_run",
    type=str,
    default=None,
    help="Wandb run name to continue logging.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
original_argv = sys.argv[:]  # Make a copy of the original sys.argv
sys.argv = [sys.argv[0]] + hydra_args

# Set up logger
# check if logging level is valid
if args_cli.logging.upper() not in logging._nameToLevel.keys():  # type: ignore
    raise ValueError(f"[ERROR]: Invalid logging level. Allowed values are: {logging._nameToLevel.keys()}.")  # type: ignore
logging.basicConfig(level=args_cli.logging.upper(), format="%(message)s", force=True)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import traceback
from datetime import datetime

import carb
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils.hydra import hydra_task_config

from exts.dreamwaq.tasks import registry
from exts.dreamwaq.utils import (
    get_checkpoint_path,
    dump_hydra_config,
    remove_empty_dicts,
    seed_everything,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Get the task configuration from the registry based on the task name
task_config = registry[args_cli.task]
if not task_config:
    raise ValueError(
        f"Task '{args_cli.task}' is not found in the registry.\n"
        f"Available tasks are: {', '.join(registry.keys())}\n"
        "Please check the task name and try again."
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: task_config.agent_cfg):  # type: ignore
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # seed everything for reproducibility
    seed_everything(args_cli.seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    # note: due to missing device in env_cfg after migration to hydra (see
    # isaaclab_tasks.utils.parse_cfg.parse_env_cfg), we set explicityly it here
    env_cfg.sim.device = agent_cfg.device = args_cli.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = args_cli.log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO]: Logging experiment in directory: {log_dir}")
    if agent_cfg.logger == "wandb" and not args_cli.log_project_name:
        agent_cfg.wandb_project = agent_cfg.experiment_name
    if agent_cfg.logger == "wandb":
        agent_cfg.note = args_cli.note
        agent_cfg.group = args_cli.group
        agent_cfg.job_type = args_cli.job_type
        agent_cfg.wandb_continue_run = args_cli.wandb_continue_run if args_cli.wandb_continue_run else None
    sys.argv = original_argv  # Restore the original sys.argv before wandb initialization
    print(f"[INFO]: Loggings with {agent_cfg.logger} is selected")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO]: Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore
    # Apply environment wrappers from the registry (IN ORDER)
    for wrapper in task_config.env_wrappers:
        env = wrapper(env, args_cli=args_cli)

    # Load algo-soecific configurations
    # Teacher student training
    if args_cli.teacher_path is not None:
        runner_kwargs = {"teacher_path": args_cli.teacher_path}
    else:
        runner_kwargs = {}

    # Instantiate the runner from the registry
    runner_class = task_config.runner
    runner = runner_class(
        env,
        remove_empty_dicts(agent_cfg.to_dict()),
        log_dir=log_dir,
        device=agent_cfg.device,
        **runner_kwargs,
    )

    # write git state to logs
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_hydra_config((hydra_args, env_cfg, agent_cfg), os.path.join(log_dir, "params"))
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)
    
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)

    # [수정] WANDB 활성화 여부 확인
    use_wandb = agent_cfg.logger == "wandb"

    if use_wandb:
        # run_name을 log_dir의 폴더 이름으로 설정 (예: 2025-12-22_12-00-00_run_name)
        run_name = os.path.basename(log_dir)
        
        # Resume 모드일 경우 처리
        run_id = agent_cfg.wandb_continue_run if agent_cfg.wandb_continue_run else wandb.util.generate_id()
        resume_mode = "allow" if agent_cfg.wandb_continue_run else None

        wandb.init(
            project=agent_cfg.wandb_project,
            group=agent_cfg.group,
            job_type=agent_cfg.job_type,
            name=run_name,
            id=run_id,
            resume=resume_mode,
            config={
                "env_cfg": env_cfg.to_dict(),
                "agent_cfg": agent_cfg.to_dict(),
                "cmd_args": vars(args_cli)
            },
            # RSL-RL은 Tensorboard를 백엔드로 사용하므로 이를 WandB와 동기화합니다.
            sync_tensorboard=False,  
            monitor_gym=True,       # gym 비디오 기록 동기화
            save_code=True,         # 코드 저장
            dir=log_dir             # 로컬 메타데이터 저장 위치
        )
        
        # 명시적인 Tensorboard Patching (sync_tensorboard=True로 대체 가능하지만 확실하게 하기 위함)
        # save=False로 설정하여 tfevents 파일 자체를 wandb 서버로 업로드하는 것을 방지(로그만 전송)
        wandb.tensorboard.patch(save=False, tensorboard_x=True)

    # run training
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        # 학습이 끝나거나 에러가 발생해도 wandb 종료 보장
        if use_wandb:
            wandb.finish()

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main execution
        main()  # type: ignore
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        logging.error(f"An error occurred: {err}")
        logging.error(traceback.format_exc())
        os._exit(1)
    finally:
        # close sim app
        simulation_app.close()
