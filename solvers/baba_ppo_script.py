
# Script to run ppo_solver_v1

import gymnasium as gym
import my_baba_env
import pygame
import time
import statistics

import ppo_solver_v1

local_path_biy = "/Users/maxmoody/Documents/GitHub/baba-gymnasium-env/"\
                 "babelib/Resources/Maps/baba_is_you.txt"

local_path_oor = "/Users/maxmoody/Documents/GitHub/baba-gymnasium-env/"\
                 "babelib/Resources/Maps/out_of_reach.txt"

local_path_mm = "/Users/maxmoody/Documents/GitHub/baba-gymnasium-env/"\
                 "babelib/Resources/Maps/med_maze.txt"

local_path_mb = "/Users/maxmoody/Documents/GitHub/baba-gymnasium-env/"\
                 "babelib/Resources/Maps/med_box.txt"

local_path_rb = "/Users/maxmoody/Documents/GitHub/baba-gymnasium-env/"\
                 "babelib/Resources/Maps/rule_break.txt"

local_path = local_path_mb

env = gym.make("BabaIsYou-v1", world_path=local_path)
eval_env = gym.make("BabaIsYou-v1", render_mode="human", world_path=local_path)
obs, info = env.reset()

print("Window open. Close the window or press CTRL+C to exit.")

episode_blocks = 32
solver = ppo_solver_v1.PPO(env, eval_env, alpha=0.0003, gamma=0.99, epsilon=0.2, lam=0.95, batch_size=64, epochs=5)

mean_reward = []

running = True
while running:
    # Let pygame process window events, so it doesn't freeze or auto-close
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    start = time.perf_counter()

    for i_block in range(episode_blocks):
        print("Block " + str(i_block + 1) + " running")
        solver.train_block()

        mean_reward.append(statistics.fmean(solver.block_returns))

    print("Mean rewards for each block: ")
    print(mean_reward)

    end = time.perf_counter()

    print("total time = " + str(end - start))

    if solver.best_return <= -100.0:
        solver.display_best_path(delay=0.05)
    else:
        solver.display_best_path(delay=0.2)

    running = False

# Take a random action, so we can see the game update
# action = env.action_space.sample()
# obs, reward, terminated, truncated, info = env.step(action)

# if terminated or truncated:
# obs, info = env.reset()

# Slow down display so the window stays visible
# time.sleep(0.2)

env.close()
