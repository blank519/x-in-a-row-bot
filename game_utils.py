import numpy as np

"""
Contains the shared game logic used by the web app and the environment. 
This includes setting the board, checking for win/draw conditions, creating observation and action spaces, 
and action masking.
"""


def new_board(height, width) -> list[list[str | None]]:
    return [[None for _ in range(width)] for _ in range(height)]


def flatten_index(row: int, col: int, width: int) -> int:
    return row * width + col

def check_winner(board: list[list[str | None]], agent: str, row: int, col: int, win_con: int) -> bool:
    """Specifically checks if the last played move at (row, col) resulted in a winner. 
       Works for any board size and any win condition length."""
    if board[row][col] != agent:
        return False
    
    height = len(board)
    width = len(board[0])
    # Checks whether the agent that just played a piece in (row, col) has won
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)] # Vertical, horizontal, and both diagonals
    for direction in directions:
        num_in_a_row = 1
        # Check on both sides of the recently played cell 
        current_row = row + direction[0]
        current_col = col + direction[1]
        while 0 <= current_row < height and 0 <= current_col < width and board[current_row][current_col] == agent:
            num_in_a_row += 1
            current_row += direction[0]
            current_col += direction[1]
            if num_in_a_row == win_con:
                return True

        current_row = row - direction[0]
        current_col = col - direction[1]
        while 0 <= current_row < height and 0 <= current_col < width and board[current_row][current_col] == agent:
            num_in_a_row += 1
            current_row -= direction[0]
            current_col -= direction[1]
            if num_in_a_row == win_con:
                return True
    return False

def is_draw(num_turns: int, max_turns: int) -> bool:
    return num_turns >= max_turns


def build_observation(board: list[list[str | None]], learner_symbol: str, opponent_symbol: str, height: int, width: int) -> np.ndarray:
    obs = np.zeros((2, height, width), dtype=np.int8)
    for r in range(height):
        for c in range(width):
            cell = board[r][c]
            if cell == learner_symbol:
                obs[0, r, c] = 1
            elif cell == opponent_symbol:
                obs[1, r, c] = 1
    return obs


def action_mask(board: list[list[str | None]], height: int, width: int) -> np.ndarray:
    mask = np.ones(height * width, dtype=np.int8)
    for r in range(height):
        for c in range(width):
            if board[r][c] is not None:
                mask[flatten_index(r, c, width)] = 0
    return mask