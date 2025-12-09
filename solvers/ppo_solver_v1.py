
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

        # --- Determine shape of observation ---
        self.obs_shape = obs_sample.shape
        C, H, W = self.obs_shape

        # --- Minimal CNN feature extractor ---
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # halves H,W
        )

        # --- Compute size after convs ---
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy)
            conv_flat_dim = int(np.prod(conv_out.shape[1:]))

        # --- Shared MLP after CNN ---
        mlp_layers = []
        prev = conv_flat_dim
        for hs in hidden_sizes:
            mlp_layers.append(nn.Linear(prev, hs))
            prev = hs
        self.mlp = nn.Sequential(*mlp_layers)

        # --- Actor head ---
        self.actor_head = nn.Linear(hidden_sizes[-1], act_dim)

        # --- Critic head ---
        self.critic_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs):
        # ---- 1. Ensure batch dimension ----
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        # ---- 2. Convert from HWC → CHW (image → PyTorch format) ----
        # If the environment outputs (H, W, C), fix it here:
        if obs.shape[-1] <= 4 and self.obs_shape[0] <= 4:
            # This checks if the image's channel dimension is last
            # Usually Baba-Is-Auto images are HWC when coming from Gym
            obs = obs.permute(0, 3, 1, 2)

        # ---- 3. Convolutional Feature Extractor ----
        x = self.conv(obs)

        # ---- 4. Flatten Feature Map ----
        x = torch.flatten(x, start_dim=1)

        # ---- 5. Shared MLP ----
        x = F.relu(self.mlp(x))

        # ---- 6. Actor & Critic Heads ----
        probs = F.softmax(self.actor_head(x), dim=-1)
        value = self.critic_head(x)

        return probs.squeeze(0), value.squeeze(0)


class PPO(ABC):
    def __init__(self, env, eval_env, alpha=0.0003, gamma=0.9, epsilon=0.5, batch_size=64, lam=0.95, epochs=5):
        self.env = env
        self.eval_env = eval_env
        self.layers = [128, 128]
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # reward decay
        self.epsilon = epsilon  # clip size
        self.lam = lam
        self.mem_size = 100
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.steps = 200

        # Create actor-critic network
        state, _ = env.reset()
        self.actor_critic = ActorCriticNetwork(state, env.action_space.n, hidden_sizes=[128, 128])

        self.policy = self.create_greedy_policy()

        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.alpha)

        # Replay buffer
        self.replay_memory = deque(maxlen=self.mem_size)

        # Store best return info
        self.best_return = -float("inf")
        self.best_path = None  # will store (states, actions, rewards)

        self.best_block_return = -float("inf")
        self.best_block_path = None

        self.block_returns = []

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
        gamma = self.gamma
        lam = self.lam

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

        ppo_epochs = self.num_epochs

        # ppo_epochs = int(self.batch_size / 4) + 1

        # Extract transitions
        batch = list(self.replay_memory)
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
            actor_loss = self.actor_loss(ratio, advantages)

            # Critic loss
            critic_loss = self.critic_loss(values, returns)

            # Combine
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear replay buffer (because PPO is on-policy)
        self.replay_memory.clear()

    def train_episode(self):
        """
        Run a single episode of the PPO algorithm.
        """

        state, _ = self.env.reset()

        # ---- Added: path tracking ----
        episode_states = []
        episode_actions = []
        episode_rewards = []

        total_return = 0.0

        for _ in range(self.steps):
            # Choose action
            action, prob, value = self.select_action(state)

            # Take action
            next_state, reward, done, _ = self.step(action)

            episode_actions.append(action)
            episode_rewards.append(reward)
            total_return += reward

            # Store action taken
            self.memorize(state, action, reward, done)
            state = next_state

            # Check if done
            if done:
                break

        # Save best path found so far
        if total_return > self.best_return:
            self.best_return = total_return
            self.best_path = {
                "states": episode_states,
                "actions": episode_actions,
                "rewards": episode_rewards,
                "return": total_return,
            }
        # Save best path found in block
        if total_return > self.best_block_return:
            self.best_block_return = total_return
            self.best_block_path = {
                "states": episode_states,
                "actions": episode_actions,
                "rewards": episode_rewards,
                "return": total_return,
            }

        self.block_returns.append(total_return)

    def train_block(self):
        # Store best return info
        self.best_block_return = -float("inf")
        self.best_block_path = None  # will store (states, actions, rewards)

        self.block_returns = []

        for i in range(self.batch_size):
            self.train_episode()
        self.replay()

        # print(self.block_returns)

    def actor_loss(self, ratio, advantages):
        """
        The policy gradient loss function.

        args:
            ratio: ratio between old and new log_probs.
            advantage: Advantage of the chosen action.

        Returns:
            The unreduced loss (as a tensor).
        """
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        return actor_loss

    def critic_loss(self, values, returns):
        """
        The integral of the critic gradient
        """

        critic_loss = F.mse_loss(values.squeeze(-1), returns)
        return critic_loss

    def __str__(self):
        return "PPO"

    def plot(self, stats, smoothing_window=20, final=False):
        pass
        # plotting.plot_episode_stats(stats, smoothing_window, final=final)

    def display_best_path(self, delay=0.1):
        if self.best_path is None:
            print("No best path recorded yet.")
            return

        print(f"Displaying best episode (return = {self.best_path['return']})")

        env = self.eval_env  # use a separate env so training isn't disturbed
        state, _ = env.reset()

        for t, action in enumerate(self.best_path["actions"]):
            env.render()  # show the frame
            time.sleep(delay)

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            if done:
                env.render()
                print("Episode finished.")
                return

    def display_path(self, path, delay=0.1):

        print(f"Displaying episode with return = {path['return']}")

        env = self.eval_env  # use a separate env so training isn't disturbed
        state, _ = env.reset()

        for t, action in enumerate(self.best_path["actions"]):
            env.render()  # show the frame
            time.sleep(delay)

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            if done:
                env.render()
                print("Episode finished.")
                return

    def record_path(self, path, map_path):
        ###
        # temporary env, use rgb_array as render mode
        tmp_env = gym.make("BabaIsYou-v1", world_path=map_path, render_mode="rgb_array")

        # wrap the env in the record video
        vid_env = gym.wrappers.RecordVideo(env=tmp_env, video_folder="../videos/",
                                           name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)

        # env reset for a fresh start
        print(f"Displaying episode with return = {path['return']}")
        state, _ = vid_env.reset()

        # Start the recorder
        vid_env.start_video_recorder()

        for t, action in enumerate(path["actions"]):
            vid_env.render()  # show the frame

            next_state, reward, term, trunc, _ = vid_env.step(action)
            done = term or trunc

            if done:
                vid_env.render()
                print("Episode finished.")

        # Don't forget to close the video recorder before the env
        vid_env.close_video_recorder()

        # Close the environment
        vid_env.close()

