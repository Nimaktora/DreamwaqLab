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

import numpy as np
import statistics
import time, os
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv
from exts.dreamwaq.learning.modules.ac_dwaq import ActorCriticDwaq, CENet, CenetRolloutStorage
from exts.dreamwaq.learning.modules.rms import RunningMeanStd
from exts.dreamwaq.learning.algorithms.dwaq_ppo import PPOWAQ
from exts.dreamwaq.envs.wrappers.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper # for type hint

class DwaqOnPolicyRunner:
    """
    On-policy runner for training and evaluation
    """
    def __init__(self, 
                 env: RslRlVecEnvWrapper,
                 train_cfg, 
                 log_dir=None, 
                 device='cpu',
                 **kwargs):

        # === 1. Config Handling ===
        if not isinstance(train_cfg, dict):
            train_cfg = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else vars(train_cfg)

        self.cfg = train_cfg
        self.alg_cfg = train_cfg.get("algorithm", {})
        self.policy_cfg = train_cfg.get("policy", {})
        self.vae_cfg = train_cfg.get("vae", {})
        self.device = device
        self.env = env
        
        # -Handle Hydra config garbage
        if "class_name" in self.alg_cfg: self.alg_cfg.pop("class_name")

        # === 2. Extract Dimensions & Setup ===
        obs_dict, extras = self.env.get_observations()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        # IMU + command + jointstates + prevaction
        self.proprio_dim = 45
        self.latent_dim = 19
        self.est_vel_dim = 3
        self.z_dim = 16
        self.history_len = 5

        # observation, body vel, disturbance force, height map scan
        self.true_vel_dim = 3
        self.disturb_force_dim = 3
        self.height_map_scan_dim = 187

        self.num_actor_obs = self.proprio_dim + self.latent_dim
        self.num_critic_obs = self.num_actor_obs + self.true_vel_dim + self.disturb_force_dim + self.height_map_scan_dim

        self.num_actions = self.env.num_actions
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.obs_hist_shape = self.history_len, self.proprio_dim

        # == 3. Initialize Modules ==
        self.actor_critic = ActorCriticDwaq(
            self.num_actor_obs, self.num_critic_obs, self.num_actions, **self.policy_cfg
        ).to(self.device)
        self.alg = PPOWAQ(self.actor_critic, device=self.device, **self.alg_cfg)
        # self.cenet = CENet(**self.vae_cfg)
        self.cenet = self.actor_critic.cenet
        for key, value in self.vae_cfg.items():
            setattr(self.cenet, key, value)

        # == 4. Storage Initialization ==
        # 1) init_storage
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.num_actor_obs],
            [self.num_critic_obs],
            [self.env.num_actions],
        )
        self.cenet.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.history_len * self.proprio_dim],
            [self.true_vel_dim],
            [self.proprio_dim] # true_onext_shape
        )
        

        # == 5. RMS Initialization ==
        self.rms_dict = {}
        self.obs_rms = None
        self.privileged_obs_rms = None
        self.true_vel_rms = None

        if self.cfg.get("obs_rms", True):
            self.obs_rms = RunningMeanStd(shape=[obs.shape[1]], device=self.device)
        
        if self.cfg.get("privileged_obs_rms", True):
            if "observations" in extras and "critic" in extras["observations"]:
                priv_shape = [extras["observations"]["critic"].shape[1]]
            else:
                priv_shape = [self.proprio_dim + self.true_vel_dim + self.disturb_force_dim + self.height_map_scan_dim]
            self.privileged_obs_rms = RunningMeanStd(shape=priv_shape, device=self.device)
 
        if self.cfg.get("true_vel_rms", True):
            self.true_vel_rms = RunningMeanStd(shape=[self.est_vel_dim], device=self.device)

        # 6. Buffers & Variables
        self.obs_history_buffer = torch.zeros(
            self.env.num_envs, self.history_len, self.proprio_dim, 
            dtype=torch.float, device=self.device
        )
        
        # 7. Reward Accumulator
        self.episodic_sum_buffer = torch.zeros(self.env.num_envs, device=self.device)

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        self.env.reset()

    def get_history_flat(self):
        return self.obs_history_buffer.view(self.env.num_envs, -1)
    
    # it doesn't contain policy observation
    def _get_privileged_obs(self, extras):
        if "observations" in extras and "critic" in extras["observations"]:
            return extras["observations"]["critic"]
        return torch.zeros(self.env.num_envs, 
                           self.true_vel_dim + self.disturb_force_dim + self.height_map_scan_dim, 
                           device=self.device)

    def _compute_log(self, done_ids, infos):
        """
        Computes boot_prob based on reward variance and retrieves individual reward terms
        from 'log' (Isaac Lab standard) to inject into 'episode' for logging.

        """
        if len(done_ids) == 0:
            return

        # 1. Compute Boot Prob (Manual Logic)
        total_per_env = self.episodic_sum_buffer[done_ids]
        std_reward = torch.std(total_per_env)
        mean_reward = torch.mean(total_per_env)
        

        # 2. Inject BootProb & Manual Reward Sum
        if "episode" not in infos:
            infos["episode"] = {}
        
        # 'reward' key is standard for logging total return
        infos["episode"]["reward"] = torch.mean(total_per_env).item()

        # 3. Retrieve Individual Reward Terms from Isaac Lab's 'log'
        # Isaac Lab populates 'log' in infos when episodes finish.
        if "log" in infos:
            for k, v in infos["log"].items():
                # Only process scalar/tensor values
                if isinstance(v, (float, int, torch.Tensor)):

                    if k.startswith("Episode_Reward/"):
                        clean_k = k.replace("Episode_Reward/", "")
                        new_key = f"rew_{clean_k}" if not clean_k.startswith("rew_") else clean_k
        
                    elif k.startswith("Episode_Termination/"):
                        clean_k = k.replace("Episode_Termination/", "")
                        new_key = f"term_{clean_k}" if not clean_k.startswith("term_") else clean_k
                    
                    # If it's a tensor, take mean
                    if isinstance(v, torch.Tensor):
                         val = v.item()
                    else:
                         val = v
                    infos["episode"][new_key] = val

        # 4. Reset internal buffer for next episode
        self.episodic_sum_buffer[done_ids] = 0.0

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
                )
        
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # --- Initial Step ---
        obs_dict, extras = self.env.get_observations()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        privileged_obs = self._get_privileged_obs(extras)
        
        if self.obs_rms is not None:
            self.obs_rms.update(obs.detach())
        if self.privileged_obs_rms is not None:
            self.privileged_obs_rms.update(privileged_obs.detach())
            
        true_vel = privileged_obs[:, 0:self.est_vel_dim]
        if self.true_vel_rms is not None:
            self.true_vel_rms.update(true_vel.detach())

        obs_norm = obs.clone()
        if self.obs_rms is not None:
            obs_norm = (obs_norm - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-6)
        
        priv_obs_norm = privileged_obs.clone()
        if self.privileged_obs_rms is not None:
            priv_obs_norm = (priv_obs_norm - self.privileged_obs_rms.mean) / torch.sqrt(self.privileged_obs_rms.var + 1e-6)

                
        for i in range(self.history_len):
            self.obs_history_buffer[:, i, :] = obs.clone()

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            # --- Rollout Loop ---
            prev_critic_obs_input = torch.zeros(self.env.num_envs, self.num_critic_obs, device=self.device)
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # critic_obs_input = torch.cat((obs_norm, priv_obs_norm), dim=-1)
                    actor_obs_input = obs_norm
                    critic_obs_input = priv_obs_norm
                    
                    if self.obs_rms is not None:
                         history_norm = (self.obs_history_buffer - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-6)
                         obs_history_flat = history_norm.view(self.env.num_envs, -1)
                    else:
                         obs_history_flat = self.obs_history_buffer.view(self.env.num_envs, -1)
                    true_vel_norm = true_vel.clone()
                    if self.true_vel_rms is not None:
                        true_vel_norm = (true_vel_norm - self.true_vel_rms.mean) / torch.sqrt(self.true_vel_rms.var + 1e-6)

                    self.cenet.before_action(obs_history_flat, true_vel_norm)
                    # actions = self.alg.act(actor_obs_input, critic_obs_input, obs_history=obs_history_flat)
                    actions = self.alg.act(actor_obs_input, critic_obs_input, obs_history=obs_history_flat)

                    next_obs_dict, rewards, dones, infos = self.env.step(actions)
                    
                    # Accumulate for BootProb logic
                    self.episodic_sum_buffer += rewards
                    cur_episode_length += 1

                    next_obs = next_obs_dict["policy"] if isinstance(next_obs_dict, dict) else next_obs_dict
                    next_extras = infos
                    next_privileged_obs = self._get_privileged_obs(next_extras)
                    next_true_vel = next_privileged_obs[:, : self.est_vel_dim]

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

                    priv_obs_norm_next = next_privileged_obs.clone()
                    if self.privileged_obs_rms is not None:
                        priv_obs_norm_next.update(next_privileged_obs.detach())

                    self.cenet.after_action(obs_norm_next)
                    self.alg.process_env_step(obs_norm_next, rewards, dones, infos)

                    self.obs_history_buffer[:, :-1, :] = self.obs_history_buffer[:, 1:, :].clone()
                    self.obs_history_buffer[:, -1, :] = next_obs.clone()
                    
                    if len(done_ids) > 0:
                        self.obs_history_buffer[done_ids, :, :] = next_obs[done_ids].unsqueeze(1)

                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    obs = next_obs
                    obs_norm = obs_norm_next
                    privileged_obs = next_privileged_obs
                    priv_obs_norm = priv_obs_norm_next
                    true_vel = next_true_vel
                    true_vel_norm = true_vel.clone()
                    if self.true_vel_rms is not None:
                        true_vel_norm = (true_vel_norm - self.true_vel_rms.mean) / torch.sqrt(self.true_vel_rms.var + 1e-6)
                    extras = next_extras

                    if self.log_dir is not None:
                        # Prioritize 'episode' populated by our logic, then 'log'
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                    prev_critic_obs_input = critic_obs_input
                stop = time.time()
                collection_time = stop - start
                
                # critic_obs_input = torch.cat((obs_norm, priv_obs_norm), dim=-1)
                self.alg.compute_returns(priv_obs_norm)

            # Update Phase
            mean_loss = self.alg.update()
            mean_value_loss = mean_loss["value_function"], 
            mean_surrogate_loss = mean_loss["surrogate"],
            mean_entropy_loss = mean_loss["entropy_loss"],
            mean_cenet_loss, vel_loss, recon_loss, kl_loss = self.cenet.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                locs = locals()
                # Ensure all vars are passed to log
                self.log(locs)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        if locs['ep_infos']:
            # Get all unique keys across all info dictionaries in the buffer
            keys = set()
            for info in locs['ep_infos']:
                keys.update(info.keys())
            
            # Sort keys to ensure consistent order, prioritize 'rew_' keys
            sorted_keys = sorted(list(keys))
            
            for key in sorted_keys:
                # Skip if it's the generic 'reward' key (already printed separately) or 'boot_prob' (print last)
                if key == 'reward' or key == 'boot_prob':
                    continue

                infotensor = torch.tensor([], device=self.device)
                for info in locs['ep_infos']:
                    if key not in info: continue
                    val = info[key]
                    if not isinstance(val, (float, int, torch.Tensor)): continue
                    
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor([val], device=self.device)
                    if len(val.shape) == 0:
                        val = val.unsqueeze(0)
                    infotensor = torch.cat((infotensor, val))
                
                if len(infotensor) > 0:
                    value = torch.mean(infotensor)
                    self.writer.add_scalar(f'Episode/{key}', value, locs['it'])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
            
            # Add boot_prob at the end of the list
            if 'boot_prob' in keys:
                 infotensor = torch.tensor([], device=self.device)
                 for info in locs['ep_infos']:
                     if 'boot_prob' in info:
                         val = info['boot_prob']
                         if not isinstance(val, torch.Tensor): val = torch.tensor([val], device=self.device)
                         infotensor = torch.cat((infotensor, val))
                 if len(infotensor) > 0:
                    value = torch.mean(infotensor)
                    self.writer.add_scalar(f'Episode/boot_prob', value, locs['it'])
                    ep_string += f"""{f'Mean episode boot_prob:':>{pad}} {value:.4f}\n"""


        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))


        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/cenet', locs['mean_autoenc_loss'], locs['it'])
        self.writer.add_scalar('Loss/cenet_vel_est', locs['vel_loss'], locs['it'])
        self.writer.add_scalar('Loss/cenet_reconstruction', locs['recon_loss'], locs['it'])
        self.writer.add_scalar('Loss/cenet_kl', locs['kl_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])

        # Terminal Output Construction
        str_fmt = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_fmt.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                          f"""{'-' * width}\n"""
                          f"""{'CENet loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                          f"""{'CENet velocity estimation error:':>{pad}} {locs['vel_loss']:.4f}\n"""
                          f"""{'CENet reconstruction loss:':>{pad}} {locs['recon_loss']:.4f}\n"""
                          f"""{'CENet KL divergence loss:':>{pad}} {locs['kl_loss']:.4f}\n"""
                          f"""{'-' * width}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
             log_string = (f"""{'#' * width}\n"""
                          f"""{str_fmt.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        rms_data = {}
        if self.obs_rms is not None:
            rms_data["obs_rms"] = {
                'mean': self.obs_rms.mean,
                'var': self.obs_rms.var,
                'count': self.obs_rms.count
            }
        if self.privileged_obs_rms is not None:
            rms_data["privileged_obs_rms"] = {
                'mean': self.privileged_obs_rms.mean,
                'var': self.privileged_obs_rms.var,
                'count': self.privileged_obs_rms.count
            }
        if self.true_vel_rms is not None:
            rms_data["true_vel_rms"] = {
                'mean': self.true_vel_rms.mean,
                'var': self.true_vel_rms.var,
                'count': self.true_vel_rms.count
            }

        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer_ac.state_dict(),
            'cenet_encoder_state_dict' : self.alg.actor_critic.cenet.encoder.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            'rms': rms_data
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        
        if load_optimizer:
           self.alg.optimizer_ac.load_state_dict(loaded_dict["optimizer_state_dict"])
 
        self.current_learning_iteration = loaded_dict['iter']
        
        if 'rms' in loaded_dict:
            rms_data = loaded_dict['rms']
            if 'obs_rms' in rms_data and self.obs_rms is not None:
                self.obs_rms.mean.data.copy_(rms_data['obs_rms']['mean'])
                self.obs_rms.var.data.copy_(rms_data['obs_rms']['var'])
                self.obs_rms.count = rms_data['obs_rms']['count']
            
            if 'privileged_obs_rms' in rms_data and self.privileged_obs_rms is not None:
                self.privileged_obs_rms.mean.data.copy_(rms_data['privileged_obs_rms']['mean'])
                self.privileged_obs_rms.var.data.copy_(rms_data['privileged_obs_rms']['var'])
                self.privileged_obs_rms.count = rms_data['privileged_obs_rms']['count']
                
            if 'true_vel_rms' in rms_data and self.true_vel_rms is not None:
                self.true_vel_rms.mean.data.copy_(rms_data['true_vel_rms']['mean'])
                self.true_vel_rms.var.data.copy_(rms_data['true_vel_rms']['var'])
                self.true_vel_rms.count = rms_data['true_vel_rms']['count']