import gymnasium as gym
import numpy as np
import pygame
import my_baba_env   # your baba-is-auto gym wrapper

# ------------------------------------------------------------------- #
# 1. DEMO COLLECTION WRAPPER
# ------------------------------------------------------------------- #

class DemoCollector(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.trajectory = []   # single episode
        self.demos = []        # list of episodes

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.trajectory = []   # clear for new episode
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.trajectory.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": terminated or truncated,
            "info": info
        })

        if terminated or truncated:
            # save completed episode
            self.demos.append(self.trajectory)

        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------- #
# 2. KEYBOARD INPUT
# ------------------------------------------------------------------- #

KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
}

def get_human_action():
    """Blocks until the user presses a valid keyboard key."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in KEY_TO_ACTION:
                return KEY_TO_ACTION[event.key]


# ------------------------------------------------------------------- #
# 3. PLAY ENVIRONMENT AND RECORD DEMOS
# ------------------------------------------------------------------- #

env = gym.make(
    "BabaIsYou-v1",
    render_mode="human",
    world_path="/home/zcurtis070/baba-gymnasium-env/babelib/Resources/Maps/volcano.txt"
)

env = DemoCollector(env)

obs, info = env.reset()

print("🎮 Recording demonstration. Use Arrow Keys to play. Close the window to stop.")

while True:
    action = get_human_action()
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    if terminated or truncated:
        print("Episode finished!")
        break


# ------------------------------------------------------------------- #
# 4. SAVE DEMONSTRATIONS
# ------------------------------------------------------------------- #

np.savez_compressed("baba_demos.npz", demos=env.demos)
print("✅ Saved demos to baba_demos.npz")
