import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.modules import rnn


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

    def mini_batch_generator(self, num_mini_batches, num_epochs):
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
                device=torch.device('cpu'),
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
        std = torch.exp(0.5 * logvar).requires_grad_(True)
        eps = torch.randn_like(std)
        return mu + eps*std
        # return mu

    # input dim -> THIS FUNCTION -> latent vector / CENet output / reparamterized values
    def forward(self, obs_history):
        dist = self.encoder(obs_history)
        est_vel = dist[:, :3]
        dist_half = (dist.shape[1] - 3) // 2
        mu = dist[:, 3:3 + dist_half].requires_grad_(True)
        logvar = dist[:, 3 + dist_half:].requires_grad_(True)
        z = self.reparameterize(mu, logvar).requires_grad_(True)
        latent_19 = torch.cat([est_vel, z], dim=-1)
        
        decode = self.decoder(latent_19)
        # latent_19 = torch.zeros(dist.shape[0],19,device="cuda:0") # debugging
        return latent_19, decode, mu, logvar
    
    def forward_inference(self, obs_history):
        dist = self.encoder(obs_history)
        est_vel = dist[:, :3]
        dist_half = (dist.shape[1] - 3) // 2
        mu = dist[:, 3:3 + dist_half].requires_grad_(True)
        logvar = dist[:, 3 + dist_half:].requires_grad_(True)
        z = self.reparameterize(mu, logvar).requires_grad_(True)
        latent_19 = torch.cat([est_vel, mu], dim=-1)
        decode = self.decoder(latent_19)
        # latent_19 = torch.zeros(dist.shape[0],19,device="cuda:0") # debugging
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
        mean_vae_loss = 0

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
            mse_loss = nn.MSELoss(reduction='none')
            vel_loss = mse_loss(est_vel_batch, true_vel_batch).sum(dim=1).mean()
            recon_loss = mse_loss(est_onext_batch, true_onext_batch).sum(dim=1).mean()
            # mse_loss = nn.MSELoss()
            # vel_loss = mse_loss(est_vel_batch, true_vel_batch)
            # recon_loss = mse_loss(est_onext_batch, true_onext_batch)

            klds = -0.5 * (1 + logvar_batch - mu_batch.pow(2) - logvar_batch.exp())
            kl_loss = klds.sum(1).mean() * self.beta
            # kl_loss = klds.sum(1).mean(0, True) * self.beta
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

        self.scheduler.step(mean_vae_loss)
        self.current_epoch += 1

        self.storage.clear()

        # Update beta
        # self.beta = min(self.beta * 1.01, self.beta_limit)
        return mean_vae_loss, mean_vel_loss, mean_recon_loss, mean_kl_loss     