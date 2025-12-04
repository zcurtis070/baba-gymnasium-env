
# Code modified from assignments; failed because not using convolutional neural network on

from abc import ABC, abstractmethod
from enum import Enum
import gymnasium as gym
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from torch.optim import Adam

# from lib import plotting


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_sample, act_dim, hidden_sizes):
        super().__init__()

        obs_dim = int(np.prod(obs_sample.shape))
        sizes = [obs_dim] + hidden_sizes

        self.layers = nn.ModuleList()

        # Shared layers
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

        # Actor head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], act_dim))
        # Critic head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, obs):
        if obs.ndim == len(obs.shape):
            obs = obs.unsqueeze(0)

        x = torch.flatten(obs, start_dim=1)

        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[i](x))

        # Actor head
        probs = F.softmax(self.layers[-2](x), dim=-1)
        # Critic head
        value = self.layers[-1](x)

        return torch.squeeze(probs, 0 if probs.shape[0] == 1 else -1), \
            torch.squeeze(value, 0 if value.shape[0] == 1 else -1)


class PPO(ABC):
    def __init__(self, env, eval_env):
        self.env = env
        self.eval_env = eval_env
        self.layers = [128, 128]
        self.alpha = 1
        self.gamma = 1
        self.epsilon = 1
        self.mem_size = 100
        self.batch_size = 30
        self.steps = 100

        # Create actor-critic network
        state, _ = env.reset()
        self.actor_critic = ActorCriticNetwork(state, env.action_space.n, hidden_sizes=[128, 128])

        self.policy = self.create_greedy_policy()

        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.alpha)

        # Replay buffer
        self.replay_memory = deque(maxlen=self.mem_size)

    def step(self, action):
        """
        Take one step in the environment while keeping track of statistical information
        Param:
            action:
        Return:
            next_state: The next state
            reward: Immediate reward
            done: Is next_state terminal
            info: Gym transition information
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # reward += self.calc_reward(next_state)

        # Update statistics
        # self.statistics[Statistics.Rewards.value] += reward
        # self.statistics[Statistics.Steps.value] += 1
        # self.total_steps += 1

        return next_state, reward, terminated or truncated, info

    def create_greedy_policy(self):
        """
        Creates a greedy policy.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            return torch.argmax(self.actor_critic(state)[0]).detach().numpy()

        return policy_fn

    def select_action(self, state):
        """
        Selects an action given state.

        Returns:
            The selected action (as an int)
            The probability of the selected action (as a tensor)
            The critic's value estimate (as a tensor)
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        probs, value = self.actor_critic(state)

        probs_np = probs.detach().numpy()
        action = np.random.choice(len(probs_np), p=probs_np)

        return action, probs[action], value

    def memorize(self, state, action, reward, done):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        probs, value = self.actor_critic(state_t.unsqueeze(0))
        probs = probs.squeeze(0)
        value = value.squeeze(0)

        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action))

        self.replay_memory.append(
            (state_t, action, log_prob.detach(), value.detach(), reward, done)
        )

    def compute_returns(self, rewards, dones):
        returns = []
        G = 0
        gamma = self.gamma

        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + gamma * G * (1 - d)
            returns.insert(0, G)

        returns = torch.stack(returns)

        return returns

    def compute_gae(self, rewards, dones, values_old):
        advantages = []
        gae = 0
        next_value = 0
        gamma = self.gamma
        lam = 0.5

        for t in reversed(range(len(rewards))):
            if t < len(values_old) - 1:
                next_value = values_old[t + 1]
            else:
                next_value = 0

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values_old[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae

            advantages.insert(0, gae)

        advantages = torch.stack(advantages)

        return advantages

    def replay(self):
        if len(self.replay_memory) < self.batch_size:
            return

        ppo_epochs = 30

        # Extract transitions
        batch = list(self.replay_memory)  # Do NOT random.sample: PPO is on-policy
        states, actions, log_probs_old, values_old, rewards, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        log_probs_old = torch.stack(log_probs_old)
        values_old = torch.stack(values_old).squeeze(-1)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 1. Compute returns + advantages
        advantages = self.compute_gae(rewards, dones, values_old)
        returns = advantages + values_old

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. PPO update loop
        for _ in range(ppo_epochs):
            # Compute new log-probs and values
            probs, values = self.actor_critic(states)
            dist = torch.distributions.Categorical(probs)
            log_probs_new = dist.log_prob(actions)

            ratio = torch.exp(log_probs_new - log_probs_old)

            # Clipped surrogate
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = F.mse_loss(values.squeeze(-1), returns)

            # Combine
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear replay buffer (because PPO is on-policy)
        self.replay_memory.clear()

    def update_actor_critic(self, advantage, prob, value):
        """
        Performs actor critic update.

        args:
            advantage: Advantage of the chosen action (tensor).
            prob: Probability associated with the chosen action (tensor).
            value: Critic's state value estimate (tensor).
        """
        # Compute loss
        actor_loss = self.actor_loss(advantage.detach(), prob).mean()
        critic_loss = self.critic_loss(advantage.detach(), value).mean()

        loss = actor_loss + critic_loss

        # Update actor critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_episode(self):
        """
        Run a single episode of the PPO algorithm.
        """

        state, _ = self.env.reset()
        gamma = self.gamma
        for _ in range(self.steps):
            # Choose action
            action, prob, value = self.select_action(state)

            # Take action
            next_state, reward, done, _ = self.step(action)

            # next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
            # next_prob, next_value = self.actor_critic(next_state_tensor)

            reward = torch.as_tensor(reward, dtype=torch.float32)
            # advantage = reward + gamma * next_value.detach() * (1 - done) - value

            # Store action taken
            self.memorize(state, action, reward, done)
            state = next_state

            # Check if done
            if done:
                break

        # Experience replay
        self.replay()

    def actor_loss(self, advantage, prob):
        """
        The policy gradient loss function.

        args:
            advantage: Advantage of the chosen action.
            prob: Probability associated with the chosen action.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

        # Reused from REINFORCE
        loss = -1.0 * torch.log(prob) * advantage
        return loss

    def critic_loss(self, advantage, value):
        """
        The integral of the critic gradient

        args:
            advantage: Advantage of the chosen action.
            value: Critic's state value estimate.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        loss = -1.0 * advantage * value
        return loss

    def __str__(self):
        return "PPO"

    def plot(self, stats, smoothing_window=20, final=False):
        pass
        # plotting.plot_episode_stats(stats, smoothing_window, final=final)
