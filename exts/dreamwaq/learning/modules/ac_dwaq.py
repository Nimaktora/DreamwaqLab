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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.modules import rnn


# === Actor Critic Network ===
# Related to: dwaq_ppo.py / dwaq_rollout.py / rsl_rl_cfg.py
# 
class ActorCriticDwaq(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs, # only observation, not include latant vector
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        super().__init__()
        # 1. kwargs log
        if kwargs:
            print("ActorCriticDwaq.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        
        # 2. Activation function set
        act_name = activation
        activation = get_activation(act_name)

        # 3. Network setting and dimension check
        self.cenet = CENet()
        num_actor_input = num_actor_obs + 19
        num_critic_input = num_critic_obs

        # 4. Actor network
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_input, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # 5. Critic network
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_input, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        # 6. PPO Network logging 
        print('{::^60}'.format(' PPO Structure '))
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    # Note: actor-critic network doesn't learn CENet
    def _build_actor_input(self, obs_current, obs_history):
        with torch.no_grad():
            code, _, _, _ = self.cenet.forward(obs_history)
        code = code.detach() 
        # For debugging
        # code = torch.zeros(obs_current.shape[0], 19, device='cuda:0')
        return torch.cat([code, obs_current], dim=-1)
    
    # distribution
    def update_distribution(self, observations, obs_history=None):
        if obs_history is None:
            raise RuntimeError("obs_history is required to build 45+19 actor input.")
        actor_input = self._build_actor_input(observations, obs_history)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, self.std) # Normalization

    # actor network running - stochastic action
    def act(self, observations, obs_history=None, **kwargs):
        self.update_distribution(observations, obs_history=obs_history)
        return self.distribution.sample()
    
    # actor network inference - deterministic action (relatively)
    def act_inference(self, observations, obs_history):
        actor_in = self._build_actor_input(observations, obs_history)
        # print(actor_in)
        return self.actor(actor_in)
    
    # PPO log
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    # Value function calculate
    def critic_evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    

# === Custom network : CENet === 
class CenetRolloutStorage:
    # Transition Class, it contains outer info from CENet
    # 225 dim obs history, True vel, Next obs 
    class Transition:
        def __init__(self):
            self.observation_histories = None
            self.true_velocities = None
            self.true_next_observations = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_history_shape,
        true_vel_shape,
        true_onext_shape,
        device=torch.device('cuda'),
    ):

        self.device = device

        self.obs_history_shape = obs_history_shape
        self.true_vel_shape = true_vel_shape
        self.true_onext_shape = true_onext_shape

        self.observation_histories = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *obs_history_shape,
            device=self.device,
            requires_grad=False,
        )
        self.true_velocities = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *true_vel_shape,
            device=self.device,
            requires_grad=False,
        )
        self.true_next_observations = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *true_onext_shape,
            device=self.device,
            requires_grad=False,
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

    def add_transitions_before_action(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.true_velocities[self.step].copy_(transition.true_velocities)

    def add_transitions_after_action(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.true_next_observations[self.step].copy_(transition.true_next_observations)

        # Increment the step
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        
        # Random index
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observation_histories = self.observation_histories.flatten(0, 1)
        true_velocities = self.true_velocities.flatten(0, 1)
        true_next_observations = self.true_next_observations.flatten(0, 1)

        for epoch in range(num_epochs):
            for batch_idx in range(num_mini_batches):
                start = batch_idx * mini_batch_size
                end = (batch_idx + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_history_batch = observation_histories[batch_idx]
                true_vel_batch = true_velocities[batch_idx]
                true_onext_batch = true_next_observations[batch_idx]

                # Yield a mini-batch of data
                yield obs_history_batch, true_vel_batch, true_onext_batch
# === CENet network info ===
# 45 x 5 = 225 dim actor obseravtion history  input 
# -> encoder -> 3 dim linear velocity & 19 dim latent vector 
# -> decoder -> 45 dim actor observation output
class CENet(nn.Module):
    def __init__(self,
                num_learning_epochs=1,
                num_mini_batches=1,
                input_dim=225, 
                latent_dim=19,
                output_dim=45,
                beta=1,
                beta_limit=4, # TODO: Check this 
                learning_rate=0.001,
                min_learning_rate=0.001,
                patience=100,
                factor=0.8,
                device=torch.device('cuda'),
                ):
        
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 3 + (latent_dim-3) *2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, output_dim)
        )

        # Logging
        print("{::^60}".format(" CENet Structure "))
        print(f"Encoder MLP: {self.encoder}")
        print(f"Decoder MLP: {self.decoder}")

        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.beta = beta
        self.beta_limit = beta_limit
        self.current_epoch = 0
        self.storage = None  # initialized later
        self.learning_rate = learning_rate
        self.min_lr = min_learning_rate
        self.patience = patience
        self.factor = factor
        self.device = device

        # Rollout Storage
        self.transition = CenetRolloutStorage.Transition()

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="min",
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )
    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        obs_history_shape,
        true_vel_shape,
        true_onext_shape,
    ):

        self.storage = CenetRolloutStorage(
            num_envs,
            num_transitions_per_env,
            obs_history_shape,
            true_vel_shape,
            true_onext_shape,
            device=self.device,
        )

    # Reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        # return mu

    # input dim -> THIS FUNCTION -> latent vector / CENet output / reparamterized values
    def forward(self, obs_history):
        dist = self.encoder(obs_history)
        est_vel = dist[:, :3]
        dist_half = (dist.shape[1] - 3) // 2
        mu = dist[:, 3:3 + dist_half]
        logvar = dist[:, 3 + dist_half:]
        z = self.reparameterize(mu, logvar)
        latent_19 = torch.cat([est_vel, z], dim=-1)
        decode = self.decoder(latent_19)
        return latent_19, decode, mu, logvar
    
    # obs hist, true lin vel 
    # TooIp ready
    def before_action(self, obs_history, true_vel):
        self.transition.observation_histories = obs_history
        self.transition.true_velocities = true_vel
        latent_19, est_next_obs, mu, logvar = self.forward(obs_history)

        est_vel = latent_19[:,0:3]
        z = latent_19[:,3:19]
        
        self.storage.add_transitions_before_action(self.transition)
        return est_next_obs, est_vel, mu, logvar, z
    
    # Jinjja observation compare ready
    def after_action(self, next_obs):
        self.transition.true_next_observations = next_obs
        # Record
        self.storage.add_transitions_after_action(self.transition)
        self.transition.clear()

    # uodate (core part)
    def update(self):
        mean_total_loss = 0
        mean_vel_loss = 0
        mean_recon_loss = 0
        mean_kl_loss = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for obs_history_batch, true_vel_batch, true_onext_batch in generator:

            # model's forward process should be here
            latent_19, est_next_obs, mu, logvar = self.forward(obs_history_batch)
            est_vel = latent_19[:,0:3]
            z = latent_19[:,3:19]
        
            est_onext_batch = est_next_obs
            est_vel_batch = est_vel
            mu_batch = mu
            logvar_batch = logvar
            context_vec_batch = z

            # loss calculation
            mse_loss = nn.MSELoss()
            vel_loss = mse_loss(est_vel_batch, true_vel_batch)
            recon_loss = mse_loss(est_onext_batch, true_onext_batch)

            klds = -0.5 * (1 + logvar_batch - mu_batch.pow(2) - logvar_batch.exp())
            kl_loss = klds.sum(1).mean(0, True) * self.beta
            # kl_loss = (-0.5 * torch.mean(1 + logvar_batch - mu_batch.pow(2) - logvar_batch.exp())) * self.beta

            total_loss = vel_loss + recon_loss + kl_loss

            self.optimizer.zero_grad()

            # Gradient step
            total_loss.backward()

            self.optimizer.step()

            mean_total_loss += total_loss.item()
            mean_vel_loss += vel_loss.item()
            mean_recon_loss += recon_loss.item()
            mean_kl_loss += kl_loss.item()

        self.scheduler.step(total_loss)

        self.current_epoch += 1

        # ----------------- GRAD CHECK ------------------
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.sum())
        #     else:
        #         print(name, param.grad)

        num_updates = self.num_learning_epochs * self.num_mini_batches

        mean_vae_loss = mean_total_loss / num_updates
        mean_vel_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_kl_loss /= num_updates

        self.storage.clear()

        # Update beta
        # self.beta = min(self.beta * 1.01, self.beta_limit)
        return mean_vae_loss, mean_vel_loss, mean_recon_loss, mean_kl_loss     

# Activation function  
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
