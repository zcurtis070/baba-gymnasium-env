import gymnasium as gym
import my_baba_env
import pygame
import time

env = gym.make("BabaIsYou-v1", render_mode="human",world_path="/home/zcurtis070/baba-is-auto/Resources/Maps/volcano.txt")
obs, info = env.reset()

print("Window open. Close the window or press CTRL+C to exit.")

running = True
while running:
    # Let pygame process window events so it doesn't freeze or auto-close
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Take a random action so we can see the game update
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    # Slow down display so the window stays visible
    time.sleep(0.2)

env.close()
