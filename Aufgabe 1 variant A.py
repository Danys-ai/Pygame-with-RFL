import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600  # Screen size
GRID_SIZE = 10            # Number of grid cells
CELL_SIZE = WIDTH // GRID_SIZE  # Size of each cell
MOVES_LIMIT = 20          # Maximum moves

# Colors
BLACK = (30, 30, 30)
WHITE = (255, 255, 255)

# Load images
player_img = pygame.image.load("Player.png")
obstacle_img = pygame.image.load("feuer.jpg")
reward_img = pygame.image.load("dung gold.jpeg")
exit_img = pygame.image.load("exit new.png")

# Scale images
player_img = pygame.transform.scale(player_img, (CELL_SIZE, CELL_SIZE))
obstacle_img = pygame.transform.scale(obstacle_img, (CELL_SIZE, CELL_SIZE))
reward_img = pygame.transform.scale(reward_img, (CELL_SIZE, CELL_SIZE))
exit_img = pygame.transform.scale(exit_img, (CELL_SIZE, CELL_SIZE))

# Game elements
player_pos = [0, 0]
exit_pos = [GRID_SIZE - 1, GRID_SIZE - 1]
obstacles = [(4, 5), (5, 5), (6, 6)]
rewards = [(3, 4), (7, 7), (8, 2)]

moves_used = 0
points = 0
running = True

# Setup screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dungeon Escape")
font = pygame.font.Font(None, 36)

def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        for y in range(0, HEIGHT, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WHITE, rect, 1)

def draw_elements():
    screen.blit(player_img, (player_pos[0] * CELL_SIZE, player_pos[1] * CELL_SIZE))
    for obs in obstacles:
        screen.blit(obstacle_img, (obs[0] * CELL_SIZE, obs[1] * CELL_SIZE))
    for reward in rewards:
        screen.blit(reward_img, (reward[0] * CELL_SIZE, reward[1] * CELL_SIZE))
    screen.blit(exit_img, (exit_pos[0] * CELL_SIZE, exit_pos[1] * CELL_SIZE))

def draw_info():
    info_text = f"Moves Left: {MOVES_LIMIT - moves_used}  Points: {points}"
    text_surface = font.render(info_text, True, WHITE)
    screen.blit(text_surface, (10, 10))

def game_over(message):
    screen.fill(BLACK)
    game_over_text = font.render(message, True, WHITE)
    screen.blit(game_over_text, (WIDTH//3, HEIGHT//3))
    pygame.display.flip()
    pygame.time.delay(3000)
    pygame.quit()
    sys.exit()

# Game loop
while running:
    screen.fill(BLACK)
    draw_grid()
    draw_elements()
    draw_info()
    
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and player_pos[1] > 0:
                player_pos[1] -= 1
            elif event.key == pygame.K_DOWN and player_pos[1] < GRID_SIZE - 1:
                player_pos[1] += 1
            elif event.key == pygame.K_LEFT and player_pos[0] > 0:
                player_pos[0] -= 1
            elif event.key == pygame.K_RIGHT and player_pos[0] < GRID_SIZE - 1:
                player_pos[0] += 1
            
            moves_used += 1
            
            if tuple(player_pos) in rewards:
                points += 10
                rewards.remove(tuple(player_pos))
            
            if tuple(player_pos) in obstacles:
                game_over("Game Over! You hit an obstacle.")
            
            if tuple(player_pos) == tuple(exit_pos):
                points += 50  # Ultimate reward for finding the exit
                game_over(f"Congratulations! You escaped with {points} points.")
            
            if moves_used >= MOVES_LIMIT:
                game_over("Out of moves! Game Over.")
    
pygame.quit()