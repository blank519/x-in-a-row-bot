import gymnasium
from gymnasium import spaces
from pettingzoo.utils import AECEnv, agent_selector
import numpy as np
import pygame
import time
import game_utils as utils

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
        self.board = utils.new_board(self.height, self.width)

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
        
        self.board = utils.new_board(self.height, self.width)

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
            if utils.check_winner(self.board, agent, row, col, self.win_con):
                # Simple reward system: -1 for loss, +1 for win, 0 for draw
                for a in self.agents:
                    if a == agent:
                        self.rewards[a] = 1.0
                    else:
                        self.rewards[a] = -1.0

                    self.terminations[a] = True
            # Check truncation/draw (board completely full)
            elif utils.is_draw(self.current_step, self.max_steps):
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
    
    def _clear_rewards(self):
        for agent in self.agents:
            self.rewards[agent] = 0.0

    def _accumulate_rewards(self):
        for agent in self.agents:
            self.cumulative_rewards[agent] += self.rewards[agent]

    def observe(self, agent):
        other_agent = self.agents[0] if agent == self.agents[1] else self.agents[1]
        obs = utils.build_observation(self.board, agent, other_agent, self.height, self.width)
        mask = utils.action_mask(self.board, self.height, self.width)
        return {"observation": obs, "action_mask": mask}
    
    def render(self):
        if self.render_mode is None:
            return

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "rgb_array":
                self.screen = pygame.Surface((self.window_size, self.window_size))
            else:
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

        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))
            return frame

        pygame.display.flip()

        if self.render_mode == "human" and self.render_delay > 0:
            time.sleep(self.render_delay)
        
    def close(self):
        if hasattr(self, "screen"):
            try:
                pygame.display.quit()
            finally:
                pygame.quit()
            delattr(self, "screen")

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]