import gymnasium
from gymnasium import spaces
from pettingzoo.utils import AECEnv, agent_selector
import numpy as np
import pygame
import time

class XInARowEnv(AECEnv):
    metadata = {"is_parallelizable":True}
    
    def __init__(self, height, width, win_con, p1, p2, render_mode=None, render_delay = 0.5):
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
            agent:spaces.Dict({
                "observation":spaces.MultiBinary([2, height, width]), # 2 channels: one for your pieces, one for opponent pieces
                "action_mask":spaces.MultiBinary(height*width),
            })
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
        self.window_size = 800
        self.bg_color = (240, 240, 240)
        self.grid_color = (0, 0, 0)
        self.token_color = (0, 0, 0)
        self.window = None
        self.clock = None
        self.render_mode = render_mode
        self.render_delay = render_delay

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.agents = self.possible_agents[:]
        
        self.board = [[None for _j in range(self.width)] for _i in range(self.height)]

        self.cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent:0 for agent in self.agents}
        self.terminations = {agent:False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {}

        self._agent_selector = agent_selector.agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self.observe(self.agent_selection), self.infos
    
    def step(self, action):
        self._clear_rewards()
        agent = self.agent_selection
        self.current_step += 1

        # Place piece
        row = action//self.width
        col = action%self.width

        if self.board[row][col] == None: # Legal move: proceed as normal
            self.board[row][col] = agent
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
            elif self.current_step >= self.max_steps:
                for a in self.agents:
                    self.rewards[a] = 0 # Small reward for draw maybe?
                    self.truncations[a] = True
        else: # Failsafe: heavy penalty for illegal move
            self.rewards[agent] = -2
            for a in self.agents:
                self.terminations[a] = True
        self._accumulate_rewards()

        # Immediately end episode if terminated/truncated
        if all(self.terminations.values()) or all(self.truncations.values()):
            self.agents = []
            return
        self.agent_selection = self._agent_selector.next()
        while self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self.agent_selection = self._agent_selector.next()

    def is_victory(self, agent, row, col):
        if self.board[row][col] != agent:
            return False
        
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
    
    def _clear_rewards(self):
        for agent in self.agents:
            self.rewards[agent] = 0.0

    def _accumulate_rewards(self):
        for agent in self.agents:
            self.cumulative_rewards[agent] += self.rewards[agent]

    def observe(self, agent):
        obs = np.zeros((2, self.height, self.width))
        mask = np.ones(self.height*self.width)
        for row in range(self.height):
            for col in range(self.width):
                if self.board[row][col] == agent:
                    obs[0, row, col] = 1
                    mask[row*self.width + col] = 0
                elif self.board[row][col] in self.agents:
                    obs[1, row, col] = 1
                    mask[row*self.width + col] = 0
        return {"observation": obs, "action_mask": mask}
    
    def render(self):
        if self.render_mode is None:
            return

        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("X In A Row")
            self.clock = pygame.time.Clock()
        
        self.screen.fill(self.bg_color)

        rows = self.height
        cols = self.width
        board_cells = max(rows, cols)
        cell_size = self.window_size//board_cells

        # Compute padding to center board
        total_board_width = cols * cell_size
        total_board_height = rows * cell_size

        pad_x = (self.window_size - total_board_width) // 2
        pad_y = (self.window_size - total_board_height) // 2

        # Draw grid
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(
                    pad_x + c * cell_size,
                    pad_y + r * cell_size,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(self.screen, self.grid_color, rect, 3)

                val = self.board[r][c]
                if val is None:
                    continue

                # Draw token text
                font_size = int(cell_size * 0.6)
                font = pygame.font.SysFont("arial", font_size, bold=True)

                text_surface = font.render(str(val), True, self.token_color)
                text_rect = text_surface.get_rect(center=rect.center)

                self.screen.blit(text_surface, text_rect)

        pygame.display.flip()

        if self.render_delay > 0:
            time.sleep(self.render_delay)
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]