
# Script to run ppo_solver_v0-1

import gymnasium as gym
import my_baba_env
import pygame
import time

import ppo_solver_v0_1

local_path = "/Users/maxmoody/Documents/GitHub/baba-gymnasium-env/"\
             "babelib/Resources/Maps/volcano.txt"

env = gym.make("BabaIsYou-v1", world_path=local_path)
eval_env = gym.make("BabaIsYou-v1", render_mode="human", world_path=local_path)
obs, info = env.reset()

print("Window open. Close the window or press CTRL+C to exit.")

running = True
while running:
    # Let pygame process window events so it doesn't freeze or auto-close
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    episodes = 1000
    solver = ppo_solver.PPO(env, eval_env)

    for i_episode in range(episodes):
        solver.train_episode()

    # Take a random action so we can see the game update
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    # Slow down display so the window stays visible
    # time.sleep(0.2)

env.close()
