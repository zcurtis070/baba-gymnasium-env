
# PPO implementation mostly generated from ChatGPT after having trouble with previous PPO implementation.

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import my_baba_env
import pygame
import time

# --- Hyperparameters ---
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPS = 0.2
PPO_EPOCHS = 4
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Actor-Critic Network ---
class CNNActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        C, H, W = input_shape
        # Convolutional layers
        self.conv1 = nn.Conv2d(C, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1

        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(H)))
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(W)))
        linear_input_size = conv_h * conv_w * 128

        # Fully connected layers
        self.fc = nn.Linear(linear_input_size, 256)

        # Actor and Critic heads
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        # x: [batch, C, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)


# --- PPO Update Function ---
def ppo_update(policy, optimizer, states, actions, old_log_probs, returns, advantages, clip_eps=CLIP_EPS):
    probs, values = policy(states)
    dist = Categorical(probs)
    log_probs = dist.log_prob(actions)
    ratio = torch.exp(log_probs - old_log_probs)

    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    actor_loss = -torch.min(surrogate1, surrogate2).mean()
    critic_loss = F.mse_loss(values.squeeze(-1), returns)

    loss = actor_loss + 0.5 * critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --- Main Training Loop ---
def train(max_episodes=1000):
    local_path_m = "/Users/maxmoody/Documents/GitHub/baba-gymnasium-env/" \
                   "babelib/Resources/Maps/med_box.txt"

    local_path_z = "/home/zcurtis070/baba-gymnasium-env" \
                   "/babelib/Resources/Maps/baba_is_you.txt"

    env = gym.make("BabaIsYou-v1", world_path=local_path_m)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    policy = CNNActorCritic(obs_shape[::-1], n_actions).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False

        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        while not done:
            # Convert to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            probs, value = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # Store transition
            states.append(state_tensor)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = next_state

        # Convert to tensors
        states = torch.cat(states)
        actions = torch.stack(actions).to(DEVICE)
        log_probs = torch.stack(log_probs).to(DEVICE).detach()
        values = torch.cat(values).squeeze(-1).to(DEVICE).detach()

        # Compute returns and advantages
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        advantages = returns - values.detach()

        # PPO epochs
        for _ in range(PPO_EPOCHS):
            for i in range(0, len(states), BATCH_SIZE):
                s_batch = states[i:i + BATCH_SIZE]
                a_batch = actions[i:i + BATCH_SIZE]
                lp_batch = log_probs[i:i + BATCH_SIZE]
                r_batch = returns[i:i + BATCH_SIZE]
                adv_batch = advantages[i:i + BATCH_SIZE]

                ppo_update(policy, optimizer, s_batch, a_batch, lp_batch, r_batch, adv_batch)

        print(f"Episode {episode} Reward: {sum(rewards)}")


# --- Run Training ---
if __name__ == "__main__":
    train()
