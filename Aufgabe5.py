import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import pygame
import random
import matplotlib.pyplot as plt


class TreasureHuntEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, grid_size=10, max_steps=100, rooms=3):
        super(TreasureHuntEnv, self).__init__()

        # Grid Configuration
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.rooms = rooms

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

        # Pygame setup
        self.cell_size = 50  # Size of each grid cell
        self.window_size = self.grid_size * self.cell_size
        self.window = None
        self.clock = None

        # Colors
        self.colors = {
            "empty": (200, 200, 200),
            "player": (0, 0, 255),
            "treasure": (255, 215, 0),
            "trap": (255, 0, 0),
            "exit": (0, 255, 0)
        }

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        # Reset step counter
        self.steps = 0

        # Create a new grid
        self.grid = self.generate_dungeon(self.grid_size, treasures=10, traps=10, rooms=self.rooms)
        self.player_pos = [0, 0]  # Player's starting position

        return self._get_observation(), {}

    def generate_dungeon(self, grid_size, treasures, traps, rooms):
        room_size = grid_size // rooms
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)

        for r in range(rooms):
            for c in range(rooms):
                room_x = r * room_size
                room_y = c * room_size

                for _ in range(treasures // (rooms * rooms)):
                    x, y = random.randint(0, room_size - 1), random.randint(0, room_size - 1)
                    grid[room_x + x, room_y + y] = 1

                for _ in range(traps // (rooms * rooms)):
                    x, y = random.randint(0, room_size - 1), random.randint(0, room_size - 1)
                    if grid[room_x + x, room_y + y] == 0:
                        grid[room_x + x, room_y + y] = 2

        grid[-1, -1] = 3  # Exit
        return grid

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
            return self._get_observation(), reward, True, False, {}

        # Check termination
        done = self.steps >= self.max_steps
        return self._get_observation(), reward, done, False, {}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))  # Black background

        # Draw the grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell_value = self.grid[row, col]
                color = self.colors["empty"]  # Default color

                if cell_value == 1:
                    color = self.colors["treasure"]
                elif cell_value == 2:
                    color = self.colors["trap"]
                elif cell_value == 3:
                    color = self.colors["exit"]

                pygame.draw.rect(
                    self.window,
                    color,
                    pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                )

                pygame.draw.rect(
                    self.window,
                    (0, 0, 0),  # Black gridlines
                    pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size),
                    1
                )

        # Draw the player
        pygame.draw.rect(
            self.window,
            self.colors["player"],
            pygame.Rect(
                self.player_pos[1] * self.cell_size,
                self.player_pos[0] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()

    def _get_observation(self):
        return np.copy(self.grid)


# Train and evaluate models

def train_and_evaluate():
    env = TreasureHuntEnv(grid_size=10, max_steps=100, rooms=3)

    # Train with DQN
    dqn_model = DQN("MlpPolicy", env, verbose=1)
    dqn_model.learn(total_timesteps=20000)
    print("Evaluating DQN...")
    mean_reward_dqn, _ = evaluate_policy(dqn_model, env, n_eval_episodes=5)
    print(f"DQN Mean Reward: {mean_reward_dqn}")

    # Train with PPO
    ppo_model = PPO("MlpPolicy", env, verbose=1)
    ppo_model.learn(total_timesteps=20000)
    print("Evaluating PPO...")
    mean_reward_ppo, _ = evaluate_policy(ppo_model, env, n_eval_episodes=5)
    print(f"PPO Mean Reward: {mean_reward_ppo}")

    # Train with A2C
    a2c_model = A2C("MlpPolicy", env, verbose=1)
    a2c_model.learn(total_timesteps=20000)
    print("Evaluating A2C...")
    mean_reward_a2c, _ = evaluate_policy(a2c_model, env, n_eval_episodes=5)
    print(f"A2C Mean Reward: {mean_reward_a2c}")

    # Plot results
    algorithms = ["DQN", "PPO", "A2C"]
    mean_rewards = [mean_reward_dqn, mean_reward_ppo, mean_reward_a2c]

    plt.bar(algorithms, mean_rewards, color=["blue", "orange", "green"])
    plt.title("Comparison of Algorithm Performance")
    plt.xlabel("Algorithm")
    plt.ylabel("Mean Reward")
    plt.show()

    print("Visualizing with the trained agent...")
    visualize_agent(env, dqn_model)

    #print("Visualizing with the trained PPO agent...")
    #visualize_agent(env, ppo_model)

    #print("Visualizing with the trained A2C agent...")
    #visualize_agent(env, a2c_model)

    env.close()


def visualize_agent(env, model):
    observation, _ = env.reset()
    done = False
    env.render()
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _, _ = env.step(action)
        env.render()


# Run the training and evaluation
if __name__ == "__main__":
    train_and_evaluate()
