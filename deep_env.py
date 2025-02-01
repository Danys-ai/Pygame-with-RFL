import pygame
import sys
import numpy as np
import random
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import gymnasium as gym

# Constants
WIDTH, HEIGHT = 1000, 1000  # Screen size
GRID_SIZE = 10  # Number of grid cells
CELL_SIZE = WIDTH // GRID_SIZE  # Size of each cell
MOVES_LIMIT = 20  # Maximum moves

# Colors
BLACK = (30, 30, 30)
WHITE = (255, 255, 255)

class DungeonEscapeEnv(Env):
    def __init__(self):
        super(DungeonEscapeEnv, self).__init__()
        self.action_space = Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = Box(low=0, high=GRID_SIZE - 1, shape=(2,), dtype=np.int32)
        
        # Initialize seed properly
        self.np_random = None
        self.seed()

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Dungeon Escape")
        self.font = pygame.font.Font(None, 36)
        
        # Load images
        self.player_img = pygame.image.load("Player.png")
        self.obstacle_img = pygame.image.load("feuer.jpg")
        self.reward_img = pygame.image.load("dung gold.jpeg")
        self.exit_img = pygame.image.load("exit new.png")
        
        # Scale images
        self.player_img = pygame.transform.scale(self.player_img, (CELL_SIZE, CELL_SIZE))
        self.obstacle_img = pygame.transform.scale(self.obstacle_img, (CELL_SIZE, CELL_SIZE))
        self.reward_img = pygame.transform.scale(self.reward_img, (CELL_SIZE, CELL_SIZE))
        self.exit_img = pygame.transform.scale(self.exit_img, (CELL_SIZE, CELL_SIZE))
        
        # Game elements
        self.player_pos = [0, 0]
        self.exit_pos = [GRID_SIZE - 1, GRID_SIZE - 1]
        self.obstacles = [(4, 5), (5, 5), (6, 6)]
        self.rewards = [(3, 4), (7, 7), (8, 2)]
        
        self.moves_used = 0
        self.points = 0
        self.running = True
        self.render_required = True  # Track when to update screen

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        # Reset game state
        self.player_pos = [0, 0]
        self.moves_used = 0
        self.points = 0
        self.running = True
        self.render_required = True
        
        return np.array(self.player_pos, dtype=np.int32), {}

    def step(self, action):
        if action == 0 and self.player_pos[1] > 0:  # Up
            self.player_pos[1] -= 1
        elif action == 1 and self.player_pos[1] < GRID_SIZE - 1:  # Down
            self.player_pos[1] += 1
        elif action == 2 and self.player_pos[0] > 0:  # Left
            self.player_pos[0] -= 1
        elif action == 3 and self.player_pos[0] < GRID_SIZE - 1:  # Right
            self.player_pos[0] += 1
        
        self.moves_used += 1
        self.render_required = True
        
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        if tuple(self.player_pos) in self.rewards:
            reward += 10
            self.points += 10
            self.rewards.remove(tuple(self.player_pos))
        
        if tuple(self.player_pos) in self.obstacles:
            terminated = True  # Hitting an obstacle ends the episode
            reward = -100
            info = {"message": "Game Over! You hit an obstacle."}
        
        if tuple(self.player_pos) == tuple(self.exit_pos):
            terminated = True  # Reaching the exit ends the episode
            reward += 50
            self.points += 50
            info = {"message": f"Congratulations! You escaped with {self.points} points."}
        
        if self.moves_used >= MOVES_LIMIT:
            truncated = True  # Running out of moves truncates the episode
            reward = -50
            info = {"message": "Out of moves! Game Over."}
        
        return np.array(self.player_pos, dtype=np.int32), float(reward), terminated, truncated, info

    def render(self, mode='human'):
        if self.render_required:
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_elements()
            self.draw_info()
            pygame.display.flip()
            self.render_required = False  # Prevent unnecessary redraws

    def close(self):
        pygame.quit()
        sys.exit()

    def draw_grid(self):
        for x in range(0, WIDTH, CELL_SIZE):
            for y in range(0, HEIGHT, CELL_SIZE):
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, WHITE, rect, 1)

    def draw_elements(self):
        self.screen.blit(self.player_img, (self.player_pos[0] * CELL_SIZE, self.player_pos[1] * CELL_SIZE))
        for obs in self.obstacles:
            self.screen.blit(self.obstacle_img, (obs[0] * CELL_SIZE, obs[1] * CELL_SIZE))
        for reward in self.rewards:
            self.screen.blit(self.reward_img, (reward[0] * CELL_SIZE, reward[1] * CELL_SIZE))
        self.screen.blit(self.exit_img, (self.exit_pos[0] * CELL_SIZE, self.exit_pos[1] * CELL_SIZE))

    def draw_info(self):
        info_text = f"Moves Left: {MOVES_LIMIT - self.moves_used}  Points: {self.points}"
        text_surface = self.font.render(info_text, True, WHITE)
        self.screen.blit(text_surface, (10, 10))

