# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the rsl_rl repository:
# https://github.com/leggedrobotics/rsl_rl
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications by Amr Mousa (2025).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from exts.dreamwaq.learning.modules import ActorCriticDwaq
from exts.dreamwaq.learning.storage import DwaqRolloutStorage
from exts.dreamwaq.utils import AdaBelief, AdamNorm


class PPOWAQ:
    actor_critic: ActorCriticDwaq

    def __init__(
        self,
        actor_critic=ActorCriticDwaq,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        lr_max=1.0e-3,
        lr_min=1.0e-5,
        adam_betas=(0.9, 0.999),
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        optimizer="adam",
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.lr = (lr_max + lr_min) / 2.0  # Start at the average of lr_max and lr_min
        self.max_lr = lr_max
        self.min_lr = lr_min

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        optimizer = "adam" if not optimizer else optimizer.lower()
        if optimizer == "Adam".lower():
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, betas=adam_betas)  # type: ignore
        elif optimizer == "AdamNorm".lower():
            self.optimizer = AdamNorm(self.actor_critic.parameters(), lr=self.lr, betas=adam_betas)
        elif optimizer == "AdaBelief".lower():
            self.optimizer = AdaBelief(self.actor_critic.parameters(), lr=self.lr, betas=adam_betas)
        else:
            raise ValueError(f"[ERROR]: Unsupported optimizer: {optimizer}")
        print(f"[INFO]: Using optimizer: {self.optimizer}")
        self.transition = DwaqRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.kl_estimate = 0.0

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.storage = DwaqRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def init_schedules(self, tot_iter):
        if self.schedule == "cosine":
            self.lr_schedule.init(tot_iter)

    def update_schedules(self, it):
        """Update learning rate and other schedules based on the iteration count."""
        if self.schedule == "cosine":
            self.lr = self.lr_schedule(it)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

        # KL-based adaptive learning rate
        elif self.schedule == "adaptive":
            assert self.desired_kl is not None, "[ERROR]: desired_kl must be set for adaptive scheduling"
            with torch.inference_mode():
                for param_group in self.optimizer.param_groups:
                    kl_mean = self.kl_estimate  # Store KL estimate during training
                    if kl_mean > self.desired_kl * 2.0:
                        self.lr = max(self.min_lr, self.lr / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.lr = min(self.max_lr, self.lr * 1.5)
                    param_group["lr"] = self.lr

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, obs_history):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
            # TODO: check it!
            actions, next_hidden_state_a = self.actor_critic.act(obs, hidden_states=self.transition.hidden_states[0])
            self.transition.actions = actions.detach()  # type: ignore
            values, next_hidden_state_c = self.actor_critic.evaluate(
                critic_obs, hidden_states=self.transition.hidden_states[1]
            )
            self.transition.values = values.detach()
            self.actor_critic.set_hidden_states(next_hidden_state_a, next_hidden_state_c)
            self.transition.next_hidden_states = (next_hidden_state_a, next_hidden_state_c)  # type: ignore
        else:
            self.transition.actions = self.actor_critic.act(obs, obs_history=obs_history).detach() # type: ignore
            self.transition.values = self.actor_critic.critic_evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.obs_history = obs_history

        return self.transition.actions

    def process_env_step(self, next_obs, rewards, dones, infos):
        self.transition.next_observations = next_obs.clone()
        if "critic" in infos["observations"]:
            self.transition.next_critic_observations = infos["observations"]["critic"].clone()
        else:
            self.transition.next_critic_observations = next_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)

            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * time_outs,
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        if self.actor_critic.is_recurrent:
            hidden_states_c = self.actor_critic.get_hidden_states()[1]
            last_values, _ = self.actor_critic.evaluate(last_critic_obs, hidden_states=hidden_states_c)
        else:
            last_values = self.actor_critic.evaluate(last_critic_obs)
        last_values = last_values.detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _compute_auxiliary_loss(self, batch: dict) -> dict:
        """Compute any auxiliary loss. Override this in subclasses if needed."""
        return {}

    def _compute_surrogate_loss(self, actions_log_prob, old_actions_log_prob, advantages):
        ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob))
        advantages = torch.squeeze(advantages).clone()
        surrogate = advantages * ratio
        surrogate_clipped = advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        return torch.clamp(-torch.min(surrogate, surrogate_clipped).mean(), -1e2, 1e2)

    def _compute_value_loss(self, value, target_values, returns):
        if self.use_clipped_value_loss:
            value_clipped = target_values + (value - target_values).clamp(-self.clip_param, self.clip_param)
            value_losses = (value - returns).pow(2)
            value_losses_clipped = (value_clipped - returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns - value).pow(2).mean()
        return self.value_loss_coef * value_loss
    
    def _update_post_backprop(self):
        pass

    def update(self, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_aux_loss = {}
        mean_entropy_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        num_updates = self.num_learning_epochs * self.num_mini_batches

        for batch in generator:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                next_critic_obs_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                obs_history_batch,
                hid_states_batch,
                masks_batch,
            ) = batch

            self.original_batch_size = returns_batch.shape[0]
            self.actor_critic.act(obs_batch,obs_history=obs_history_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL estimate for adaptive scheduling
            if self.schedule == "adaptive":
                with torch.inference_mode():
                    # Ensure numerical stability using torch.clamp
                    safe_sigma_ratio = torch.clamp(sigma_batch / old_sigma_batch, min=1e-5, max=1e5)
                    kl = torch.sum(
                        torch.log(safe_sigma_ratio)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    self.kl_estimate = kl.mean().item()

            # Update schedules before computing loss
            self.update_schedules(it)

            # Compute losses
            surrogate_loss = self._compute_surrogate_loss(
            actions_log_prob_batch,
            old_actions_log_prob_batch,
            advantages_batch,
            )
            value_loss = self._compute_value_loss(
                value_batch,
                target_values_batch,
                returns_batch,
            )
            entropy_loss = -self.entropy_coef * entropy_batch.mean()

            aux_loss_dict = {}
            aux_loss = sum(aux_loss_dict.values()) if len(aux_loss_dict) > 0 else 0.0

            loss = surrogate_loss + value_loss + entropy_loss + aux_loss

            # Gradient step
            if torch.isnan(loss).any() or self.actor_critic.nan_detected:
                print("[WARNING]: NaN detected in loss. Skipping update.")
                self.optimizer.zero_grad()
                self.storage.clear()
                return None
            else:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self._update_post_backprop()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_loss.item()

            mean_loss = {
                "value_function": mean_value_loss / num_updates,
                "surrogate": mean_surrogate_loss / num_updates,
                "entropy_loss": mean_entropy_loss / num_updates,
            }
        self.storage.clear()

        return mean_loss