

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

local_paths = [local_path_biy, local_path_mm, local_path_mb, local_path_rb]

local_path = local_path_mb

alpha_vals = [0.0003, 0.0001]

gamma_vals = [0.9, 0.99]

epsilon_vals = [0.2, 0.3]

for path in local_paths:
    print(path)

    start_time = time.perf_counter()

    for a in alpha_vals:
        print(a)
        for g in gamma_vals:
            print(g)
            for e in epsilon_vals:
                print(e)
                for r in range(3):
                    with open('data/output.txt', 'a') as file:
                        # Append content to the file
                        file.write("Testing alpha=" + str(a) + ", gamma=" + str(g) + ", epsilon=" + str(e) + " on " + path + ", run " + str(r+1))
                        file.write('\n')

                    env = gym.make("BabaIsYou-v1", world_path=local_path)
                    eval_env = gym.make("BabaIsYou-v1", world_path=local_path)
                    obs, info = env.reset()

                    episode_blocks = 32
                    solver = ppo_solver_v1.PPO(env, eval_env, alpha=0.0003, gamma=0.99, epsilon=0.2, lam=0.95, batch_size=64, epochs=5)

                    mean_reward = []

                    running = True
                    while running:
                        # Let pygame process window events, so it doesn't freeze or auto-close
                        '''for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False'''

                        for i_block in range(episode_blocks):
                            # print("Block " + str(i_block + 1) + " running")
                            solver.train_block()

                            '''
                            with open('data/output.txt', 'a') as file:
                                # Append content to the file
                                file.write(str(solver.block_returns))
                                file.write('\n')
                                '''

                            mean_reward.append(statistics.fmean(solver.block_returns))

                        # print("Mean rewards for each block: ")
                        # print(mean_reward)

                        with open('data/output.txt', 'a') as file:
                            # Append content to the file
                            file.write("Mean rewards for each block: ")
                            file.write(str(mean_reward))
                            file.write('\n\n')

                        running = False
                    env.close()

    end_time = time.perf_counter()

    with open('data/output.txt', 'a') as file:
        # Append content to the file
        file.write("Time taken on" + path + ": ")
        file.write(str(end_time - start_time))
        file.write("\n\n\n")
