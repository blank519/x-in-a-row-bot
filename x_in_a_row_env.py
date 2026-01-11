import gymnasium
from gymnasium import spaces
from pettingzoo.utils import AECEnv

class XInARowEnv(AECEnv):
    def __init__(self, grid_size, win_con, p1, p2):
        """
        Creates an x-in-a-row board game
        
        :param grid_size: size of grid
        :param win_con: number of spaces in a row required to win
        :param p1: symbol representing p1
        :param p2: symbol representing p2
        """
        self.grid_size = grid_size
        self.win_con = win_con
        self.possible_agents = {p1, p2}