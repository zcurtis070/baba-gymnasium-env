import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyBaba
import my_baba_env.wrappers.rendering as rendering
import os
class BabaIsYouEnv(gym.Env):
    """
    Gymnasium environment wrapper for Baba Is Auto (pyBaba).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        world_path=os.path.expanduser("~/baba-is-auto/Resources/Maps/baba_is_you.txt"),
        render_mode=None,
        max_steps=200,
    ):
        super().__init__()

        self.world_path = world_path
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Load game
        self.game = pyBaba.Game(self.world_path)

        # Renderer for human / rgb
        self.renderer = None
        if render_mode in ("human", "rgb_array"):
            self.renderer = rendering.Renderer(self.game)

        # Action space (UP, DOWN, LEFT, RIGHT)
        self.action_map = {
            0: pyBaba.Direction.UP,
            1: pyBaba.Direction.DOWN,
            2: pyBaba.Direction.LEFT,
            3: pyBaba.Direction.RIGHT,
        }
        self.action_space = spaces.Discrete(4)

        # ----------------------------
        # OBSERVATION SPACE SETUP
        # ----------------------------

        # Map dimensions
        self.H = self.game.GetMap().GetHeight()
        self.W = self.game.GetMap().GetWidth()

        if self.H == 0 or self.W == 0:
            raise ValueError(
                f"ERROR: Map failed to load. world_path={self.world_path} "
                f"returned width={self.W}, height={self.H}.\n"
                "Double-check the map path."
            )

        # Get flattened state from pyBaba
        flat = np.array(
            pyBaba.Preprocess.StateToTensor(self.game), dtype=np.float32
        )

        flat_size = flat.size

        # Compute number of channels
        self.C = flat_size // (self.H * self.W)

        if (self.C * self.H * self.W) != flat_size:
            raise ValueError(
                f"State tensor size mismatch: flat={flat_size}, "
                f"expected divisible by {self.H*self.W}"
            )

        # Example observation reshaped to (C, H, W)
        sample_obs = flat.reshape(self.C, self.H, self.W)

        # Define observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.C, self.H, self.W),
            dtype=np.float32,
        )

        self.steps = 0


    # ---------------------
    # RESET (Gymnasium-style)
    # ---------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.game.Reset()
        self.steps = 0

        obs = self._get_obs()
        info = {}

        if self.render_mode in ("human", "rgb_array"):
            self.render()

        return obs, info

    # ---------------------
    # STEP (Gymnasium-style)
    # ---------------------
    def step(self, action):
        # Apply action
        self.game.MovePlayer(self.action_map[action])
        self.steps += 1

        # Game outcome
        play_state = self.game.GetPlayState()

        terminated = False
        truncated = False

        if play_state == pyBaba.PlayState.WON:
            reward = 200.0
            terminated = True

        elif play_state == pyBaba.PlayState.LOST:
            reward = -100.0
            terminated = True

        else:
            reward = -0.5

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {}

        if self.render_mode in ("human", "rgb_array"):
            self.render()

        return obs, reward, terminated, truncated, info

    # ---------------------
    # OBSERVATION
    # ---------------------
    def _get_obs(self):
        flat = np.array(pyBaba.Preprocess.StateToTensor(self.game), dtype=np.float32)
        return flat.reshape(self.C, self.H, self.W)

    # ---------------------
    # RENDERING
    # ---------------------
    def render(self):
        if self.render_mode == "human":
            self.renderer.render(self.game.GetMap(), "human")

        elif self.render_mode == "rgb_array":
            # return RGB frame
            return self.renderer.render(self.game.GetMap(), "rgb_array")

    # ---------------------
    # CLEANUP
    # ---------------------
    def close(self):
        if self.renderer:
            self.renderer.quit_game()
