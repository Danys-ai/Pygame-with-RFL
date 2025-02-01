import gymnasium as gym
from gymnasium import spaces
#from gym import spaces
import numpy as np
import random

class TreasureHuntEnv(gym.Env):
    def __init__(self, grid_size=10, max_steps=100):
        super(TreasureHuntEnv, self).__init__()
        
        # Grid Configuration
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Action space: 4 discrete actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: Grid flattened into a single vector
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(grid_size, grid_size), dtype=np.int32
        )
        
        # Rewards
        self.reward_treasure = 10
        self.reward_trap = -5
        self.reward_exit = 50
        self.step_penalty = -1

        # Initialize the environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        # Reset step counter
        self.steps = 0
        
        # Create a new grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid[:, :] = 0  # Empty spaces
        
        # Place treasures (value = 1)
        for _ in range(10):
            x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            self.grid[x, y] = 1
        
        # Place traps (value = 2)
        for _ in range(10):
            x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if self.grid[x, y] == 0:  # Ensure no overlap
                self.grid[x, y] = 2

        # Place exit (value = 3)
        self.grid[self.grid_size - 1, self.grid_size - 1] = 3
        
        # Player's starting position
        self.player_pos = [0, 0]
        
        # Observation: Initial grid state
        observation = self._get_observation()
        return observation, {}
    
    def step(self, action):
        self.steps += 1
        
        # Move player based on the action
        if action == 0 and self.player_pos[0] > 0:  # Up
            self.player_pos[0] -= 1
        elif action == 1 and self.player_pos[0] < self.grid_size - 1:  # Down
            self.player_pos[0] += 1
        elif action == 2 and self.player_pos[1] > 0:  # Left
            self.player_pos[1] -= 1
        elif action == 3 and self.player_pos[1] < self.grid_size - 1:  # Right
            self.player_pos[1] += 1
        
        # Calculate reward
        current_cell = self.grid[self.player_pos[0], self.player_pos[1]]
        reward = self.step_penalty  # Default step penalty
        
        if current_cell == 1:  # Treasure
            reward += self.reward_treasure
            self.grid[self.player_pos[0], self.player_pos[1]] = 0  # Remove treasure
        elif current_cell == 2:  # Trap
            reward += self.reward_trap
        elif current_cell == 3:  # Exit
            reward += self.reward_exit
            done = True
            return self._get_observation(), reward, done, False, {}
        
        # Check termination
        done = False
        if self.steps >= self.max_steps:  # Step limit reached
            done = True
        
        return self._get_observation(), reward, done, False, {}
    
    def render(self):
        # Simple text-based rendering
        print("\nGrid:")
        for row in range(self.grid_size):
            line = ""
            for col in range(self.grid_size):
                if self.player_pos == [row, col]:
                    line += "P "  # Player's position
                elif self.grid[row, col] == 1:
                    line += "T "  # Treasure
                elif self.grid[row, col] == 2:
                    line += "X "  # Trap
                elif self.grid[row, col] == 3:
                    line += "E "  # Exit
                else:
                    line += "0 "  # Empty space
            print(line)
    
    def _get_observation(self):
        # Flattened grid representation with player position
        obs = np.copy(self.grid)
       # obs[self.player_pos[0], self.player_pos[1]] = 9  # Mark player position
        return obs

