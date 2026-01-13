import gymnasium
from gymnasium import spaces
from pettingzoo.utils import AECEnv, agent_selector
import numpy as np
import pygame

class XInARowEnv(AECEnv):
    def __init__(self, height, width, win_con, p1, p2, render_mode=None):
        """
        Creates an x-in-a-row board game
        
        :param grid_size: size of grid
        :param win_con: number of spaces in a row required to win
        :param p1: symbol representing p1
        :param p2: symbol representing p2
        """
        self.height = height
        self.width = width
        self.win_con = win_con
        
        self.max_steps = height*width
        self.current_step = 0

        self.possible_agents = [p1, p2]
        self.agents = self.possible_agents[:]

        # Set the board
        self.board = [[None for _j in range(width)] for _i in range(height)]

        # Observation space
        self.observation_spaces = {
            agent:spaces.MultiBinary([2, height, width]) # 2 channels: one for your pieces, one for opponent pieces
            for agent in self.possible_agents
        }

        # Action space
        self.action_spaces = {
            agent: spaces.Discrete(height*width) # Each cell is a possible action. Illegal moves will be masked in the NN
            for agent in self.possible_agents
        }

        # Agent selector
        self._agent_selector = agent_selector.agent_selector(self.agents)

        # Rendering
        self.window = None
        self.clock = None

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.agents = self.possible_agents[:]
        
        self.board = [[None for _j in range(self.width)] for _i in range(self.height)]

        self.rewards = {agent:0 for agent in self.agents}
        self.terminations = {agent:False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent:{"action_mask":[1 for cell in range(self.height * self.width)]} for agent in self.agents}

        self._agent_selector = agent_selector.agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self.observe(self.agent_selection), self.infos
    
    def step(self, action):
        # self._clear_rewards()
        agent = self.agent_selection
        self.current_step += 1

        # Place piece
        row = action//self.height
        col = action%self.height

        if self.board[row][col] == None: # Should implement some sort of failsafe in case it tries to put a piece on an actual spot
            self.board[row][col] = agent
            # Update information: mask cell for agents as illegal move
            for a in self.agents:
                self.infos[a]["action_mask"][action] = 0
        
        # Check victory/termination and assign reward
        if self.is_victory(agent, row, col):
            # Simple reward system: -1 for loss, +1 for win, 0 for draw
            for a in self.agents:
                if a == agent:
                    self.rewards[a] = 1
                else:
                    self.rewards[a] = -1

                self.terminations[a] = True
        # Check truncation (board completely full)
        if self.current_step >= self.max_steps:
            for a in self.agents:
                self.truncations[a] = True
        # self._accumulate_rewards()

        # Immediately end episode if terminated/truncated
        if all(self.terminations.values()) or all(self.truncations.values()):
            self.agents = []
            return
        self.agent_selection = self._agent_selector.next()
        while self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self.agent_selection = self._agent_selector.next()

    def is_victory(self, agent, row, col):
        # Checks whether the agent that just played a piece in (row, col) has won
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)] # Vertical, horizontal, and both diagonals
        for direction in directions:
            num_in_a_row = 1
            # Check on both sides of the recently played cell 
            current_row = row + direction[0]
            current_col = col + direction[1]
            while 0 <= current_row < self.height and 0 <= current_col < self.width and self.board[current_row][current_col] == agent:
                num_in_a_row += 1
                current_row += direction[0]
                current_col += direction[1]
                if num_in_a_row == self.win_con:
                    return True

            current_row = row - direction[0]
            current_col = col - direction[1]
            while 0 <= current_row < self.height and 0 <= current_col < self.width and self.board[current_row][current_col] == agent:
                num_in_a_row += 1
                current_row -= direction[0]
                current_col -= direction[1]
                if num_in_a_row == self.win_con:
                    return True
        return False
    
    def observe(self, agent):
        obs = np.zeros((2, self.height, self.width))
        for row in range(self.height):
            for col in range(self.width):
                if self.board[row][col] == agent:
                    obs[0, row, col] = 1
                elif self.board[row][col] in self.agents:
                    obs[1, row, col] = 1
        return obs
    
    def render(self):
        if self.render_mode is None:
            return

        cell_size = 80
        margin = 50
        width = self.width * cell_size
        height = self.height * cell_size

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((width, height))
            pygame.display.set_caption("X In A Row")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((width, height))
        canvas.fill((255, 255, 255))

        # Draw grid lines
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                pygame.draw.rect(canvas, (0, 0, 0), rect, width=3)

                val = self.board[r][c]
                if val is None:
                    continue

                center = (c * cell_size + cell_size // 2, r * cell_size + cell_size // 2)

                # Draw player token text
                font = pygame.font.SysFont("arial", 48, bold=True)

                text_surface = font.render(str(val), True, (0, 0, 0))
                text_rect = text_surface.get_rect(
                    center=(c * cell_size + cell_size // 2, r * cell_size + cell_size // 2)
                )

                canvas.blit(text_surface, text_rect)

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()

        if self.render_mode == "human":
            self.clock.tick(10)

        # Support rgb_array render mode
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None