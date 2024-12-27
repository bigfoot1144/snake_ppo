import pygame
import numpy as np
from gymnasium import Env, spaces
import random

COLLISION_REWARD = -1
SCORE_REWARD = 1
WINNER_REWARD = 10

class SnakeEnv(Env):
    def __init__(self, seed=None, render_mode="human"):
        super(SnakeEnv, self).__init__()

        self.grid_size = 32

        self.observation_space = spaces.Box(low=0, high=3, shape=[self.grid_size * self.grid_size], dtype=np.int16)

        self.action_space = spaces.Discrete(5)  # up, down, left, right, nothing

        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen_width = 640
            self.screen_height = 640
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.snake_head_x = self.grid_size // 2
        self.snake_head_y = self.grid_size // 2
        self.snake_direction = 0

    def reset(self, seed=None):
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)
        self.snake_head_y = self.grid_size // 2
        self.snake_head_x = self.grid_size // 2
        self.state[self.snake_head_y, self.snake_head_x] = 1
        self.snake_path = np.empty([self.grid_size * self.grid_size, 2], dtype=np.int32) # grid_size * grid_size, yx 
        self.snake_path.fill(-1)
        self.snake_index = 0
        self.snake_length = 1
        self.snake_path[0][0] = self.snake_head_y
        self.snake_path[0][1] = self.snake_head_x

        self.episodic_reward = 0
        self.episodic_length = 0

        #snake
        zero_indices = np.argwhere(self.state == 0)
        random_index = random.choice(zero_indices)
        self.apple_y = random_index[1]
        self.apple_x = random_index[0]
        self.state[random_index[0]][random_index[1]] = 3
        return self.state.flatten(), {}

    def step(self, action):
        reward = 0
        terminated = 0
        truncated = 0
        info = {}
        apple_found = False

        if action == 0:  
            self.snake_direction = 0
        elif action == 1:  
            self.snake_direction = 1
        elif action == 2:  
            self.snake_direction = 2
        elif action == 3:  
            self.snake_direction = 3
        elif action == 4:  
            pass

        if self.snake_direction == 0:  
            self.snake_head_y -= 1
        elif self.snake_direction == 1:  
            self.snake_head_y += 1
        elif self.snake_direction == 2:  
            self.snake_head_x -= 1
        elif self.snake_direction == 3:  
            self.snake_head_x += 1

        if (self.snake_head_x < 0 or self.snake_head_x > (self.grid_size-1) or
            self.snake_head_y < 0 or self.snake_head_y > (self.grid_size-1) or
            (self.state[self.snake_head_y][self.snake_head_x] != 0 and self.state[self.snake_head_y][self.snake_head_x] != 3)):

            terminated = 1
            reward = COLLISION_REWARD
        else:
            if self.state[self.snake_head_y][self.snake_head_x] == 3:
                apple_found = True
                self.snake_length += 1
                
            self.snake_index += 1 
            self.snake_path[((self.snake_index) % (self.grid_size * self.grid_size))][1] = self.snake_head_y
            self.snake_path[((self.snake_index) % (self.grid_size * self.grid_size))][0] = self.snake_head_x
            
            self.state[self.snake_head_y][self.snake_head_x] = 2
            
            if(not apple_found):
                del_y = self.snake_path[((self.snake_index - self.snake_length) % (self.grid_size * self.grid_size))][1]
                del_x = self.snake_path[((self.snake_index - self.snake_length) % (self.grid_size * self.grid_size))][0]
                self.state[del_y][del_x] = 0
                self.snake_path[((self.snake_index + 1) % (self.grid_size * self.grid_size)), 1] = -1
                self.snake_path[((self.snake_index + 1) % (self.grid_size * self.grid_size)), 0] = -1
                if self.snake_length > 1:
                    next_box_y = self.snake_path[((self.snake_index - 1) % (self.grid_size * self.grid_size))][1]
                    next_box_x = self.snake_path[((self.snake_index - 1) % (self.grid_size * self.grid_size))][0]
                    self.state[next_box_y][next_box_x] = 1
            else:
                zero_indices = np.argwhere(self.state == 0)
                if len(zero_indices) == 0:
                    print("WINNER")
                    terminated = 1
                    reward = WINNER_REWARD
                else:
                    random_index = random.choice(zero_indices)
                    self.state[random_index[0]][random_index[1]] = 3
                    self.apple_y = random_index[1]
                    self.apple_x = random_index[0]

        self.episodic_reward += reward
        self.episodic_length += 1
        if terminated:
            info["final_info"] = (self.episodic_reward, self.episodic_length)
        return self.state.flatten(), reward, terminated, truncated, info

    def close(self):
        pygame.quit()

    def render(self):
        if self.render_mode == "human":
            self.screen.fill((255, 255, 255))
            self.draw_state()
            pygame.display.flip()
            self.clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    quit()

    def draw_state(self):
        if self.render_mode == "human":
            cell_size = self.screen_width // self.grid_size
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    pygame.draw.rect(self.screen, (0, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size), 1)

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.state[i, j] > 0:
                        pygame.draw.rect(self.screen, (0, 255, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
            
            pygame.draw.rect(self.screen, (255, 0, 0), (self.apple_y * cell_size, self.apple_x * cell_size, cell_size, cell_size))

    def get_action(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 0
        elif keys[pygame.K_DOWN]:
            return 1
        elif keys[pygame.K_LEFT]:
            return 2
        elif keys[pygame.K_RIGHT]:
            return 3
        else:
            return 4

if __name__ == "__main__":
    env = SnakeEnv(render_mode="human")
    env.reset()

    while True:
        action = env.get_action()
        state, _, done, _, _ = env.step(action)
        env.render()

        if done:
            env.reset()
            env.snake_head_x = env.grid_size // 2
            env.snake_head_y = env.grid_size // 2
