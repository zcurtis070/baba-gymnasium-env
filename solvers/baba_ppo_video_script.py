
# Script to run ppo_solver_v1

import gymnasium as gym
import my_baba_env
import pygame
import time
import statistics

import ppo_solver_v1

map_dict = {"biy": "../babelib/Resources/Maps/baba_is_you.txt", "oor": "../babelib/Resources/Maps/out_of_reach.txt",
            "vc": "../babelib/Resources/Maps/volcano.txt", "mm": "../more_maps/med_maze.txt",
            "mb": "../more_maps/med_box.txt", "rb": "../more_maps/rule_break.txt"}

map_path = "../babelib/Resources/Maps/baba_is_you.txt"

# Take user inputs for parameters
episode_blocks = 32
episode_blocks_s = input("Enter desired number of episode blocks to run (default 32) or enter to skip: ")
if episode_blocks_s.strip():
    episode_blocks = int(episode_blocks_s)

alpha = 0.0003
alpha_s = input("Enter learning rate/alpha (default 0.0003) or enter to skip: ")
if alpha_s.strip():
    alpha = float(alpha_s)

gamma = 0.95
gamma_s = input("Enter reward decay (default 0.95) or enter to skip: ")
if gamma_s.strip():
    gamma = float(gamma_s)

epsilon = 0.25
epsilon_s = input("Enter clip size (default 0.25) or enter to skip: ")
if epsilon_s.strip():
    epsilon = float(epsilon_s)

map_list = ["biy", "mm", "mb", "rb", "oor", "vc"]
map_string = input("Enter map choice: \"biy\" for baba is you, \"mm\" for med_maze, \"mb\" for med_box, "
                   "\"rb\" for rule break, \"oor\" for out of reach, or \"vc\" for volcano: ")
while map_string not in map_list:
    print("Invalid map.")
    map_string = input("Enter map choice: \"biy\" for baba is you, \"mm\" for med_maze, \"mb\" for med_box, "
                       "\"rb\" for rule break, \"oor\" for out of reach, or \"vc\" for volcano: ")

map_path = map_dict[map_string]

env = gym.make("BabaIsYou-v1", world_path=map_path)
eval_env = gym.make("BabaIsYou-v1", render_mode="human", world_path=map_path)
obs, info = env.reset()

print("Window open. Close the window or press CTRL+C to exit.")

solver = ppo_solver_v1.PPO(env, eval_env, alpha=alpha, gamma=gamma, epsilon=epsilon, lam=0.95, batch_size=64, epochs=5)

mean_reward = []

running = True
while running:
    # Let pygame process window events, so it doesn't freeze or auto-close
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    start = time.perf_counter()

    paths = []

    for i_block in range(episode_blocks):
        print("Block " + str(i_block + 1) + " running")
        solver.train_block()

        # Add mean reward for training block
        mean_reward.append(statistics.fmean(solver.block_returns))

        # Taking best runs from early in training, middle of training, end of training
        if i_block in [0, int(episode_blocks/2), (episode_blocks - 1)]:
            paths.append(solver.best_block_path)

    print("Mean rewards for each block: ")
    print(mean_reward)

    end = time.perf_counter()

    print("total time = " + str(end - start))

    # Save videos of best runs

    ###
    # temporary env, use rgb_array as render mode
    tmp_env = gym.make("BabaIsYou-v1", world_path=map_path, render_mode="rgb_array")
    tmp_env.reset()
    tmp_env.render()

    # wrap the env in the record video
    vid_env = gym.wrappers.RecordVideo(
        env=tmp_env,
        video_folder="../videos/",
        name_prefix="test-video",
        episode_trigger=lambda ep: True  # or lambda ep: ep % 2 == 0
    )

    # frame = vid_env.render()
    # print("Frame type:", type(frame))
    # print("Frame shape:", getattr(frame, 'shape', None))

    path = paths[0]
    for path in paths:
        print(f"Recording episode with return = {path['return']}")
        state, _ = vid_env.reset()  # start recording

        for t, action in enumerate(path["actions"]):

            next_state, reward, term, trunc, _ = vid_env.step(action)
            done = term or trunc

            if done:
                vid_env.render()
                print("Episode finished.")

    print(f"Recording episode with return = {solver.best_path['return']}")
    state, _ = vid_env.reset()  # start recording

    for t, action in enumerate(solver.best_path["actions"]):

        next_state, reward, term, trunc, _ = vid_env.step(action)
        done = term or trunc

        if done:
            vid_env.render()
            print("Episode finished.")

    # Close the environment
    vid_env.close()

    running = False

env.close()
