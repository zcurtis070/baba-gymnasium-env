import gymnasium as gym
import my_baba_env
import pygame
import time

import ppo_solver_v0_1

local_path = "/Users/maxmoody/Documents/GitHub/baba-gymnasium-env/"\
             "babelib/Resources/Maps/volcano.txt"

env = gym.make("BabaIsYou-v1", world_path=local_path)
obs = env.reset()
print(obs)
print(env.observation_space, env.action_space)
print(env.observation_space.shape, env.action_space.shape)
