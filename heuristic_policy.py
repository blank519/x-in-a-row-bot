import numpy as np


def _check_win(pieces: np.ndarray, win_con: int) -> bool:
    height, width = pieces.shape
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))

    for r in range(height):
        for c in range(width):
            if pieces[r, c] == 0:
                continue
            for dr, dc in directions:
                count = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < height and 0 <= cc < width and pieces[rr, cc] == 1:
                    count += 1
                    if count >= win_con:
                        return True
                    rr += dr
                    cc += dc
    return False


class WinBlockRandomPolicy:
    def __init__(self, height: int, width: int, win_con: int):
        self.height = int(height)
        self.width = int(width)
        self.win_con = int(win_con)

    def __call__(self, obs: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator) -> int:
        mask = np.asarray(action_mask, dtype=np.int8)
        legal_actions = np.flatnonzero(mask.astype(bool)).astype(np.int64)
        if legal_actions.size == 0:
            return 0

        obs = np.asarray(obs, dtype=np.int8)
        my_pieces = obs[0]
        opp_pieces = obs[1]

        #Win immediately if possible
        for a in legal_actions:
            r = int(a) // self.width
            c = int(a) % self.width
            test = my_pieces.copy()
            test[r, c] = 1
            if _check_win(test, self.win_con):
                return int(a)

        #Block opponent's immediate win
        for a in legal_actions:
            r = int(a) // self.width
            c = int(a) % self.width
            test = opp_pieces.copy()
            test[r, c] = 1
            if _check_win(test, self.win_con):
                return int(a)

        #Otherwise random legal move
        return int(rng.choice(legal_actions))
