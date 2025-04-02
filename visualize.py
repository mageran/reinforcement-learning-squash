import pygame
import numpy as np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (233, 43, 23)
BLUE = (0, 0, 255)

WIDTH, HEIGHT = 400, 600
BALL_RADIUS = 15
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
PADDLE_Y = HEIGHT - PADDLE_HEIGHT



class SquashVisualizer:

    def __init__(self, squash_env):
        self.env = squash_env
        pygame.init()
        self.font = pygame.font.SysFont(None, 36)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.message = None
        pygame.display.set_caption("Squash Game")
    
    def render(self, delay=0.1):
        ball_position = self.env.ball_position * np.array([WIDTH, HEIGHT])
        paddle_position = self.env.paddle_position * WIDTH - (PADDLE_WIDTH // 2)
        self.screen.fill(BLACK)
        pygame.draw.circle(self.screen, WHITE, ball_position.astype(int), BALL_RADIUS)
        if isinstance(self.env.ball_target_x, (int, float)):
            pos = np.array([self.env.ball_target_x, 1]) * np.array([WIDTH, HEIGHT])
            pygame.draw.circle(self.screen, RED, pos.astype(int), BALL_RADIUS)
        pygame.draw.rect(self.screen, BLUE, (int(paddle_position), PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT))
        if self.message is not None:
            text = self.font.render(self.message, True, WHITE)
            text_rect = text.get_rect()
            text_rect.topleft = (0, 0)
            self.screen.blit(text, text_rect)

        pygame.display.flip()
        pygame.time.delay(delay)
    
    def set_message(self, message):
        #text = self.font.render(message, True, WHITE)
        self.message = message

    def clear_message(self):
        self.message = None
