# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from exts.dreamwaq.learning.modules.ac_dwaq import ActorCriticDwaq
from exts.dreamwaq.learning.modules.rms import RunningMeanStd
from exts.dreamwaq.learning.modules.cenet import CENet 
from exts.dreamwaq.learning.algorithms.dwaq_ppo import PPOWAQ
from exts.dreamwaq.envs.wrappers.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper


class DwaqOnPolicyRunner:
    """On-policy runner for DWAQ training and evaluation."""

    def __init__(self, env: RslRlVecEnvWrapper, train_cfg, log_dir=None, device="cpu", **kwargs):
        torch.set_printoptions(threshold=float('inf'))
        if not isinstance(train_cfg, dict):
            train_cfg = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else vars(train_cfg)

        self.cfg = train_cfg
        self.alg_cfg = dict(train_cfg.get("algorithm", {}))
        self.policy_cfg = dict(train_cfg.get("policy", {}))
        self.vae_cfg = dict(train_cfg.get("vae", {}))
        self.device = device
        self.env = env

        if "class_name" in self.alg_cfg:
            self.alg_cfg.pop("class_name")
        if "class_name" in self.policy_cfg:
            self.policy_cfg.pop("class_name")
        if "class_name" in self.vae_cfg:
            self.vae_cfg.pop("class_name")

        obs_dict, extras = self.env.get_observations()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        self.proprio_dim = obs.shape[1]
        self.latent_dim = 19
        self.est_vel_dim = 3
        self.history_len = 5

        privileged_obs = self._get_privileged_obs(extras)
        self.true_vel_dim = 3
        self.disturb_force_dim = 3
        self.height_map_scan_dim = max(privileged_obs.shape[1] - self.true_vel_dim - self.disturb_force_dim, 0)

        self.num_actor_obs = self.proprio_dim
        self.num_critic_obs = privileged_obs.shape[1] + self.num_actor_obs
        self.num_actions = self.env.num_actions
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.actor_critic = ActorCriticDwaq(
            self.num_actor_obs,
            self.num_critic_obs,
            self.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        if not hasattr(self.actor_critic, "evaluate"):
            self.actor_critic.evaluate = self.actor_critic.critic_evaluate
        if not hasattr(self.actor_critic, "reset"):
            self.actor_critic.reset = lambda dones: None
        if not hasattr(self.actor_critic, "nan_detected"):
            self.actor_critic.nan_detected = False
        self.alg = PPOWAQ(self.actor_critic, device=self.device, **self.alg_cfg)
        self.cenet = CENet(
            device=self.device
        ).to(self.device)
        self.cenet_optimizer = torch.optim.Adam(self.cenet.parameters(), lr=1e-3)

        if self.vae_cfg:
            for key, value in self.vae_cfg.items():
                if hasattr(self.cenet, key):
                    setattr(self.cenet, key, value)

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.num_actor_obs+self.latent_dim],
            [self.num_critic_obs],
            [self.env.num_actions],
        )
        self.cenet.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.history_len * self.proprio_dim],
            [self.true_vel_dim],
            [self.proprio_dim],
        )

        self.obs_rms = RunningMeanStd(shape=[obs.shape[1]], device=self.device) if self.cfg.get("obs_rms", True) else None
        self.privileged_obs_rms = (
            RunningMeanStd(shape=[privileged_obs.shape[1]], device=self.device)
            if self.cfg.get("privileged_obs_rms", True)
            else None
        )
        self.true_vel_rms = (
            RunningMeanStd(shape=[self.est_vel_dim], device=self.device)
            if self.cfg.get("true_vel_rms", True)
            else None
        )

        self.obs_history_buffer = torch.zeros(
            self.env.num_envs,
            self.history_len,
            self.proprio_dim,
            dtype=torch.float,
            device=self.device,
        )
        self.episodic_sum_buffer = torch.zeros(self.env.num_envs, device=self.device)

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()

    def get_history_flat(self):
        return self.obs_history_buffer.reshape(self.env.num_envs, -1)

    def _get_privileged_obs(self, extras):
        if "observations" in extras and "critic" in extras["observations"]:
            # print("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
            # print(extras["observations"]["critic"])
            # 974848
            priv = extras["observations"]["critic"]
            # print(priv[:, self.proprio_dim:].numel)
            return priv[:, self.proprio_dim:]
        # else:
        #     print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        return torch.zeros(self.env.num_envs, self.true_vel_dim + self.disturb_force_dim, device=self.device)

    def _compute_log(self, done_ids, infos):
        if len(done_ids) == 0:
            return

        total_per_env = self.episodic_sum_buffer[done_ids]
        if "episode" not in infos:
            infos["episode"] = {}
        infos["episode"]["reward"] = torch.mean(total_per_env).item()

        if "log" in infos:
            for k, v in infos["log"].items():
                if not isinstance(v, (float, int, torch.Tensor)):
                    continue

                if k.startswith("Episode_Reward/"):
                    clean_k = k.replace("Episode_Reward/", "")
                    new_key = f"rew_{clean_k}" if not clean_k.startswith("rew_") else clean_k
                elif k.startswith("Episode_Termination/"):
                    clean_k = k.replace("Episode_Termination/", "")
                    new_key = f"term_{clean_k}" if not clean_k.startswith("term_") else clean_k
                else:
                    new_key = k

                val = v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
                infos["episode"][new_key] = val

        self.episodic_sum_buffer[done_ids] = 0.0

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        self.alg.actor_critic.train()
        self.cenet.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
# #!#$!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#
        obs_dict, extras = self.env.get_observations()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        privileged_obs = self._get_privileged_obs(extras) # 238-45
        critic_obs_input = torch.cat((obs, privileged_obs), dim=-1)

        if self.obs_rms is not None:
            self.obs_rms.update(obs.detach())
        if self.privileged_obs_rms is not None:
            self.privileged_obs_rms.update(privileged_obs.detach())

        true_vel = privileged_obs[:, :self.est_vel_dim]
        if self.true_vel_rms is not None:
            self.true_vel_rms.update(true_vel.detach())
        # ------
        obs_norm = obs.clone()
        if self.obs_rms is not None:
            obs_norm = (obs_norm - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-6)

        priv_obs_norm = privileged_obs.clone()
        if self.privileged_obs_rms is not None:
            priv_obs_norm = (priv_obs_norm - self.privileged_obs_rms.mean) / torch.sqrt(self.privileged_obs_rms.var + 1e-6)

        true_vel_norm = true_vel.clone()
        if self.true_vel_rms is not None:
            true_vel_norm = (true_vel_norm - self.true_vel_rms.mean) / torch.sqrt(self.true_vel_rms.var + 1e-6)
        # ------

        for i in range(self.history_len):
            self.obs_history_buffer[:, i, :] = obs.clone()

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if self.obs_rms is not None:
                        history_norm = (self.obs_history_buffer - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-6)
                        obs_history_flat = history_norm.view(self.env.num_envs, -1)
                    else:
                        obs_history_flat = self.obs_history_buffer.view(self.env.num_envs, -1)

                    
                    est_next_obs, est_vel, mu, logvar, z = self.cenet.before_action(obs_history_flat, true_vel_norm)
                    vel_input = est_vel
                    actor_obs_input = torch.cat((obs_norm, vel_input, z), dim=-1)
                    critic_obs_input = torch.cat((obs_norm, priv_obs_norm), dim=-1)
                    actions = self.alg.act(actor_obs_input, critic_obs_input)

                    next_obs_dict, rewards, dones, infos = self.env.step(actions)
                    # rewards = rewards.to(self.device)
                    # dones = dones.to(self.device)

                    self.episodic_sum_buffer += rewards
                    cur_episode_length += 1

                    next_obs = next_obs_dict["policy"] if isinstance(next_obs_dict, dict) else next_obs_dict
                    next_extras = infos
                    next_privileged_obs = self._get_privileged_obs(next_extras)
                    next_true_vel = next_privileged_obs[:, :self.est_vel_dim]
                    done_ids = dones.nonzero(as_tuple=False).flatten()

                    if len(done_ids) > 0:
                        rewbuffer.extend(self.episodic_sum_buffer[done_ids].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[done_ids].cpu().numpy().tolist())
                        self._compute_log(done_ids, infos)
                        cur_episode_length[done_ids] = 0

                    if self.obs_rms is not None:
                        self.obs_rms.update(next_obs.detach())
                    if self.privileged_obs_rms is not None:
                        self.privileged_obs_rms.update(next_privileged_obs.detach())
                    if self.true_vel_rms is not None:
                        self.true_vel_rms.update(next_true_vel.detach())

                    obs_norm_next = next_obs.clone()
                    if self.obs_rms is not None: 
                        obs_norm_next = (obs_norm_next - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-6)

                    self.cenet.after_action(obs_norm_next)
                    self.alg.process_env_step(rewards, dones, infos)

                    self.obs_history_buffer[:, :-1, :] = self.obs_history_buffer[:, 1:, :].clone()
                    self.obs_history_buffer[:, -1, :] = next_obs.clone()

                    if len(done_ids) > 0:
                        self.obs_history_buffer[done_ids, :, :] = next_obs[done_ids].unsqueeze(1)

                    obs = next_obs
                    obs_norm = obs_norm_next
                    privileged_obs = next_privileged_obs
                    priv_obs_norm = privileged_obs.clone()
                    true_vel = next_true_vel
                    if self.privileged_obs_rms is not None:
                        priv_obs_norm = (priv_obs_norm - self.privileged_obs_rms.mean) / torch.sqrt(self.privileged_obs_rms.var + 1e-6)
                    true_vel = next_true_vel
                    true_vel_norm = true_vel.clone()
                    if self.true_vel_rms is not None:
                        true_vel_norm = (true_vel_norm - self.true_vel_rms.mean) / torch.sqrt(self.true_vel_rms.var + 1e-6)
                    extras = infos

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

                collection_time = time.time() - start

                critic_obs_input = torch.cat((obs_norm, priv_obs_norm), dim=-1)
                self.alg.compute_returns(critic_obs_input)

            mean_loss, mean_surrogate_loss = self.alg.update()
            # if mean_loss is None:
            #     mean_loss = {
            #         "value_function": 0.0,
            #         "surrogate": 0.0,
            #         "entropy_loss": 0.0,
            #     }
            mean_cenet_loss, vel_loss, recon_loss, kl_loss = self.cenet.update()
            learn_time = time.time() - start - collection_time

            if self.log_dir is not None:
                self.log(
                    {
                        "it": it,
                        "num_learning_iterations": num_learning_iterations,
                        "ep_infos": ep_infos,
                        "rewbuffer": rewbuffer,
                        "lenbuffer": lenbuffer,
                        "collection_time": collection_time,
                        "learn_time": learn_time,
                        "mean_value_loss": mean_loss,
                        "mean_surrogate_loss": mean_surrogate_loss,
                        # "mean_entropy_loss": mean_loss["entropy_loss"],
                        "mean_cenet_loss": mean_cenet_loss,
                        "vel_loss": vel_loss,
                        "recon_loss": recon_loss,
                        "kl_loss": kl_loss,
                    }
                )

            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            keys = set()
            for info in locs["ep_infos"]:
                keys.update(info.keys())

            for key in sorted(keys):
                if key == "reward":
                    continue

                infotensor = torch.tensor([], device=self.device)
                for info in locs["ep_infos"]:
                    if key not in info:
                        continue
                    val = info[key]
                    if not isinstance(val, (float, int, torch.Tensor)):
                        continue
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor([val], device=self.device)
                    if len(val.shape) == 0:
                        val = val.unsqueeze(0)
                    infotensor = torch.cat((infotensor, val))

                if len(infotensor) > 0:
                    value = torch.mean(infotensor)
                    self.writer.add_scalar(f"Episode/{key}", value, locs["it"])
                    ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1e-6))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        # self.writer.add_scalar("Loss/entropy", locs["mean_entropy_loss"], locs["it"])
        self.writer.add_scalar("Loss/cenet", locs["mean_cenet_loss"], locs["it"])
        self.writer.add_scalar("Loss/cenet_vel_est", locs["vel_loss"], locs["it"])
        self.writer.add_scalar("Loss/cenet_reconstruction", locs["recon_loss"], locs["it"])
        self.writer.add_scalar("Loss/cenet_kl", locs["kl_loss"], locs["it"])
        # self.writer.add_scalar("Loss/learning_rate", self.alg.lr, locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        str_fmt = (
            f" \033[1m Learning iteration {locs['it']}/"
            f"{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        )

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"{'#' * width}\n"
                f"{str_fmt.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s "
                f"(collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
                f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
                # f"{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"
                f"{'-' * width}\n"
                f"{'CENet loss:':>{pad}} {locs['mean_cenet_loss']:.4f}\n"
                f"{'CENet velocity estimation error:':>{pad}} {locs['vel_loss']:.4f}\n"
                f"{'CENet reconstruction loss:':>{pad}} {locs['recon_loss']:.4f}\n"
                f"{'CENet KL divergence loss:':>{pad}} {locs['kl_loss']:.4f}\n"
                f"{'-' * width}\n"
                f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        else:
            log_string = (
                f"{'#' * width}\n"
                f"{str_fmt.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s "
                f"(collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
                f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
                # f"{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"
                f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
            )

        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        rms_data = {}
        if self.obs_rms is not None:
            rms_data["obs_rms"] = {
                "mean": self.obs_rms.mean,
                "var": self.obs_rms.var,
                "count": self.obs_rms.count,
            }
        if self.privileged_obs_rms is not None:
            rms_data["privileged_obs_rms"] = {
                "mean": self.privileged_obs_rms.mean,
                "var": self.privileged_obs_rms.var,
                "count": self.privileged_obs_rms.count,
            }
        if self.true_vel_rms is not None:
            rms_data["true_vel_rms"] = {
                "mean": self.true_vel_rms.mean,
                "var": self.true_vel_rms.var,
                "count": self.true_vel_rms.count,
            }

        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "cenet_state_dict": self.cenet.state_dict(),
                'cenet_optimizer_state_dict': self.cenet.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
                "rms": rms_data,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])

        if "cenet_state_dict" in loaded_dict:
            self.cenet.load_state_dict(loaded_dict["cenet_state_dict"])

        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.cenet.optimizer.load_state_dict(loaded_dict['cenet_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict["iter"]

        if "rms" in loaded_dict:
            rms_data = loaded_dict["rms"]
            if "obs_rms" in rms_data and self.obs_rms is not None:
                self.obs_rms.mean.data.copy_(rms_data["obs_rms"]["mean"])
                self.obs_rms.var.data.copy_(rms_data["obs_rms"]["var"])
                self.obs_rms.count = rms_data["obs_rms"]["count"]

            if "privileged_obs_rms" in rms_data and self.privileged_obs_rms is not None:
                self.privileged_obs_rms.mean.data.copy_(rms_data["privileged_obs_rms"]["mean"])
                self.privileged_obs_rms.var.data.copy_(rms_data["privileged_obs_rms"]["var"])
                self.privileged_obs_rms.count = rms_data["privileged_obs_rms"]["count"]

            if "true_vel_rms" in rms_data and self.true_vel_rms is not None:
                self.true_vel_rms.mean.data.copy_(rms_data["true_vel_rms"]["mean"])
                self.true_vel_rms.var.data.copy_(rms_data["true_vel_rms"]["var"])
                self.true_vel_rms.count = rms_data["true_vel_rms"]["count"]


        # from exts.dreamwaq.tasks.envs.unitree import UNITREE_GO1_CFG
        # from isaaclab.managers import SceneEntityCfg
        # from isaaclab.envs import ManagerBasedRLEnv
        # from isaaclab.assets import Articulation
        # from typing import TYPE_CHECKING
        # if TYPE_CHECKING:
        #     from isaaclab.envs import ManagerBasedRLEnv
        # import exts.dreamwaq.envs.mdp.rewards as mdp
        # # def _get_robot(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> Articulation:
        # #     return env.scene[asset_cfg.name]



        # asset_cfg = SceneEntityCfg("robot", joint_names=UNITREE_GO1_CFG.joint_sdk_names)
        # env = ManagerBasedRLEnv(asset_cfg)
        # # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        # robot = mdp._get_robot(env, asset_cfg)

        # print("requested joint names:", UNITREE_GO1_CFG.joint_sdk_names)
        # print("resolved joint ids:", asset_cfg.joint_ids)
        # print("resolved joint names:", [robot.data.joint_names[i] for i in asset_cfg.joint_ids])
        # print(self.env.scene["robot"].data.joint_names)
        # print(self.obs_rms.mean.data)
        # print(self.obs_rms.var.data)


        # print(rms_data["obs_rms"]["mean"], rms_data["obs_rms"]["var"])
        # print(rms_data["privileged_obs_rms"]["mean"], rms_data["privileged_obs_rms"]["var"]) 
        # print(rms_data["true_vel_rms"]["mean"], rms_data["true_vel_rms"]["var"])
# 491524096.0001
# 491524096.0001
# 491524096.0001
