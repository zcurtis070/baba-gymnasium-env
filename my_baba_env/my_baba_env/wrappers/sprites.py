import pyBaba
import pygame
import os

# Base directory of this file (my_baba_env/my_baba_env/wrappers)
BASE_DIR = os.path.dirname(__file__)

# Full paths to sprite folders
ICON_DIR = os.path.join(BASE_DIR, "sprites", "icon")
TEXT_DIR = os.path.join(BASE_DIR, "sprites", "text")

BLOCK_SIZE = 48


def load_sprite(folder, name):
    """
    Helper function to load and scale a sprite.
    """
    path = os.path.join(folder, f"{name}.gif")
    return pygame.transform.scale(
        pygame.image.load(path),
        (BLOCK_SIZE, BLOCK_SIZE)
    )


class SpriteLoader:
    def __init__(self):
        # Icon objects (Baba, Flag, Rock, Tile, Wall)
        self.icon_images = {
            pyBaba.ObjectType.ICON_BABA: "BABA",
            pyBaba.ObjectType.ICON_FLAG: "FLAG",
            pyBaba.ObjectType.ICON_WALL: "WALL",
            pyBaba.ObjectType.ICON_ROCK: "ROCK",
            pyBaba.ObjectType.ICON_TILE: "TILE",
            pyBaba.ObjectType.ICON_LAVA: "LAVA",
            pyBaba.ObjectType.ICON_WATER: "WATER",
            pyBaba.ObjectType.ICON_GRASS: "GRASS", 
        }

        for key, name in self.icon_images.items():
            self.icon_images[key] = load_sprite(ICON_DIR, name)

        # Text tiles (BABA, IS, YOU, FLAG, WIN, WALL, STOP, ROCK, PUSH)
        self.text_images = {
            pyBaba.ObjectType.BABA: "BABA",
            pyBaba.ObjectType.IS: "IS",
            pyBaba.ObjectType.YOU: "YOU",
            pyBaba.ObjectType.FLAG: "FLAG",
            pyBaba.ObjectType.WIN: "WIN",
            pyBaba.ObjectType.WALL: "WALL",
            pyBaba.ObjectType.STOP: "STOP",
            pyBaba.ObjectType.ROCK: "ROCK",
            pyBaba.ObjectType.PUSH: "PUSH",
            pyBaba.ObjectType.LAVA: "LAVA",
            pyBaba.ObjectType.HOT: "HOT",
            pyBaba.ObjectType.MELT: "MELT", 
            pyBaba.ObjectType.WATER: "WATER",
            pyBaba.ObjectType.SINK: "SINK",
        }

        for key, name in self.text_images.items():
            self.text_images[key] = load_sprite(TEXT_DIR, name)
