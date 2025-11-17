import pygame
import pyBaba
from my_baba_env.wrappers import sprites


BLOCK_SIZE = 48
COLOR_BACKGROUND = pygame.Color(0, 0, 0)


class Renderer:
    def __init__(self, game, enable_render=True):
        pygame.init()
        pygame.display.set_caption("OpenAI Gym - baba-babaisyou-v0")

        self.game = game
        self.game_over = False
        self.enable_render = enable_render

        if self.enable_render:
            self.screen_size = (
                game.GetMap().GetWidth() * BLOCK_SIZE,
                game.GetMap().GetHeight() * BLOCK_SIZE,
            )
            self.screen = pygame.display.set_mode(
                (self.screen_size[0], self.screen_size[1]),
                pygame.DOUBLEBUF,
            )

            self.sprite_loader = sprites.SpriteLoader()
            self.draw(game.GetMap())

    def draw_obj(self, map, x_pos, y_pos):
        objects = map.At(x_pos, y_pos)
        types = list(objects.GetTypes())

        # 1) Draw background tile first (if present)
        if pyBaba.ObjectType.ICON_TILE in types:
            tile_image = self.sprite_loader.icon_images[pyBaba.ObjectType.ICON_TILE]
            tile_rect = tile_image.get_rect()
            tile_rect.topleft = (x_pos * BLOCK_SIZE, y_pos * BLOCK_SIZE)
            self.screen.blit(tile_image, tile_rect)

        # 2) Draw everything else on top
        for obj_type in types:
            # skip tile & empty here (already drawn / nothing to draw)
            if obj_type in (pyBaba.ObjectType.ICON_TILE, pyBaba.ObjectType.ICON_EMPTY):
                continue

            if pyBaba.IsTextType(obj_type):
                obj_image = self.sprite_loader.text_images[obj_type]
            else:
                obj_image = self.sprite_loader.icon_images[obj_type]

            obj_rect = obj_image.get_rect()
            obj_rect.topleft = (x_pos * BLOCK_SIZE, y_pos * BLOCK_SIZE)
            self.screen.blit(obj_image, obj_rect)


    def draw(self, map):
        # CLEAR ENTIRE SCREEN BEFORE DRAWING
        self.screen.fill(COLOR_BACKGROUND)

        for y_pos in range(map.GetHeight()):
            for x_pos in range(map.GetWidth()):
                self.draw_obj(map, x_pos, y_pos)

    def render(self, map, mode="human"):
        try:
            if not self.game_over:

                # Draw ONE complete frame
                self.draw(map)

                if mode == "human":
                    pygame.display.flip()

            self.process_event()

        except Exception as e:
            self.game_over = True
            self.quit_game()
            raise e

    def process_event(self):
        if not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_game()

    def quit_game(self):
        self.game_over = True
        if self.enable_render:
            pygame.display.quit()
        pygame.quit()
