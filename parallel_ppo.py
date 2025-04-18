import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Tuple, Dict, Optional, Callable
import gymnasium as gym
import pathlib
# import gymnasium_robotics
import gym_emg

def make_env(env_fn, seed=0):
    def _init():
        env = env_fn()
        env.reset(seed=seed + np.random.randint(0, 1000))  # Different seed for each env
        return env
    return _init

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        # nn.init.xavier_uniform_(module.weight, gain=0.01)
        torch.nn.init.orthogonal_(module.weight, gain=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable

        # Critic
        self.value = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.apply(initialize_weights)

    def forward(self, x):
        """Returns action, log probability of action, and value"""
        x = self.shared(x)
        return self.mean(x), self.log_std.exp(), self.value(x)

    def evaluate_actions(self, states, actions):
        """Returns log probability of actions, entropy, and value"""
        mean, std, value = self.forward(states)
        dist = Normal(mean, std)
        logprobs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return logprobs, entropy, value

class RolloutBuffer:
    def __init__(self, buffer_size: int, num_envs: int, obs_shape: Tuple, action_dim: int):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.pos = 0
        self.full = False

        # Initialize buffers
        self.observations = np.zeros((buffer_size, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs, action_dim), dtype=np.float32)
        self.logprobs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.advantages = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_envs), dtype=np.float32)

    def add(self, obs, action, logprob, reward, done, value):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.logprobs[self.pos] = logprob.flatten()
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get(self, batch_size: Optional[int] = None):
        if not self.full:
            raise ValueError("Buffer is not full yet")

        # Flatten the buffer
        obs = self.observations.reshape(-1, *self.observations.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        old_logprobs = self.logprobs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)

        if batch_size is not None:
            indices = np.random.permutation(len(obs))
            for i in range(0, len(obs), batch_size):
                batch_indices = indices[i:i+batch_size]
                yield (
                    obs[batch_indices],
                    actions[batch_indices],
                    old_logprobs[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices]
                )
        else:
            yield (obs, actions, old_logprobs, advantages, returns)

    def compute_returns_and_advantage(self, last_values: np.ndarray, gamma: float, gae_lambda: float):
        # Convert to numpy
        last_values = last_values.copy()
        last_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

class PPO:
    def __init__(self, env_fn: Callable, policy_class: nn.Module, num_envs=8, n_steps=2048, epochs=10, batch_size=256,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, ent_coef=0.0, vf_coef=0.5,
                 max_grad_norm=0.5, learning_rate=3e-4, device=None):

        self.env = SubprocVecEnv([make_env(env_fn, seed=i) for i in range(num_envs)])
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_mean_reward = -np.inf

        # Get observation and action spaces
        obs_shape = self.env.observation_space.shape
        action_dim = self.env.action_space.shape[0]

        # Initialize policy and optimizer
        self.policy = policy_class(obs_shape[0], action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.rollout_buffer = RolloutBuffer(n_steps, num_envs, obs_shape, action_dim)

        self.current_obs = self.env.reset()

    def collect_rollouts(self) -> bool:
        self.rollout_buffer.pos = 0
        self.rollout_buffer.full = False

        for step in range(self.n_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self.current_obs).float().to(self.device)

                # Get actions and values
                actions, values, logprobs = self.policy(obs_tensor)
                actions = actions.cpu().numpy()
                values = values.cpu().numpy().flatten()
                logprobs = logprobs.cpu().numpy()

            # Execute in environment
            actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            next_obs, rewards, dones, _ = self.env.step(actions)

            for i, done in enumerate(dones):
                if done:
                    next_obs[i] = self.env.reset()[i]  # Reset the specific environment

            if np.mean(rewards) > self.best_mean_reward:
                self.best_mean_reward = np.mean(rewards)
                torch.save(self.policy.state_dict(), "best_model_ppo.pth")

            self.rollout_buffer.add(
                self.current_obs,
                actions,
                logprobs,
                rewards,
                dones,
                values
            )

            self.current_obs = next_obs

        # Compute returns and advantages
        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(self.current_obs).float().to(self.device)
            _, _, last_values = self.policy(last_obs_tensor)
            last_values = last_values.cpu().numpy().flatten()

        self.rollout_buffer.compute_returns_and_advantage(last_values, self.gamma, self.gae_lambda)

        return True

    def train(self) -> Dict[str, float]:
        # Update policy for n_epochs
        approx_kl = 0.0
        clip_fraction = 0.0
        loss_pi = 0.0
        loss_vf = 0.0
        loss_ent = 0.0

        for _ in range(self.epochs):
            for batch_data in self.rollout_buffer.get(self.batch_size):
                obs, actions, old_logprobs, advantages, returns = batch_data

                # Convert to tensor
                obs_tensor = torch.as_tensor(obs).float().to(self.device)
                actions_tensor = torch.as_tensor(actions).float().to(self.device)
                old_logprobs_tensor = torch.as_tensor(old_logprobs).float().to(self.device)
                advantages_tensor = torch.as_tensor(advantages).float().to(self.device)
                returns_tensor = torch.as_tensor(returns).float().to(self.device)

                # Normalize advantages
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-5)

                # Get current policy outputs
                logprobs, entropy, values = self.policy.evaluate_actions(obs_tensor, actions_tensor)
                values = values.flatten()

                # Policy loss
                ratio = torch.exp(logprobs - old_logprobs_tensor)
                policy_loss_1 = advantages_tensor * ratio
                policy_loss_2 = advantages_tensor * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if self.clip_range_vf is None:
                    value_loss = 0.5 * ((returns_tensor - values) ** 2).mean()
                else:
                    values_clipped = self.rollout_buffer.values + torch.clamp(
                        values - self.rollout_buffer.values,
                        -self.clip_range_vf,
                        self.clip_range_vf
                    )
                    value_loss_1 = (values - returns_tensor) ** 2
                    value_loss_2 = (values_clipped - returns_tensor) ** 2
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

                # Entropy loss
                entropy_loss = -torch.mean(entropy)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Logging
                approx_kl = (old_logprobs_tensor - logprobs).mean().item()
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                loss_pi += policy_loss.item()
                loss_vf += value_loss.item()
                loss_ent += entropy_loss.item()

        # Average over epochs
        n_updates = self.epochs * (self.n_steps * self.num_envs // self.batch_size)
        loss_pi /= n_updates
        loss_vf /= n_updates
        loss_ent /= n_updates

        return {
            "loss/policy": loss_pi,
            "loss/value": loss_vf,
            "loss/entropy": loss_ent,
            "policy/approx_kl": approx_kl,
            "policy/clip_fraction": clip_fraction
        }

    def learn(self, total_timesteps: int):
        assert total_timesteps % (self.n_steps * self.num_envs) == 0, "Total timesteps must be divisible by n_steps * num_envs"
        num_timesteps = 0

        pbar = tqdm(total=total_timesteps, desc="Training", position=0, leave=True)
        while num_timesteps < total_timesteps:
            # Collect rollouts
            self.collect_rollouts()

            self.train()

            # Update timestep counter
            num_timesteps += self.n_steps * self.num_envs

            pbar.update(self.n_steps * self.num_envs)

            # Mean cumulative reward per episode

            # pbar.set_postfix(mean_reward=mean_reward)

        return self


if __name__ == "__main__":
    name = 'gym_emg/SingleHand-v0'
    # name = 'gym_emg/TwoHands-v0'
    datapath = f"{pathlib.Path('~').expanduser()}/code/EMG-RL/data"

    def create_env():
        # gym.register_envs(gymnasium_robotics)
        # env = gym.make("MountainCarContinuous-v0", max_episode_steps=1024)
        # env = gym.make(name, datapath=datapath, n_substeps=5, subsampling=1, subject=2, exercise=3)
        env = gym.make(name)
        return env
    
    n_envs = 1
    n_steps = 1000
    epochs = 10
    total_timesteps = n_envs * n_steps * 1000 #n_envs * n_steps * epochs

    ppo = PPO(
        env_fn=create_env,
        policy_class=ActorCritic,
        num_envs=n_envs,
        n_steps=n_steps,
        epochs=epochs,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=3e-4,
        ent_coef=0.008,
        vf_coef=0.2,
        max_grad_norm=0.5,
        clip_range_vf=None,
        device='cpu',
    )

    ppo.learn(total_timesteps=total_timesteps)
