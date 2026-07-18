import numpy as np


def _check_win(pieces: np.ndarray, win_con: int, r: int, c: int) -> bool:
    height, width = pieces.shape
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))

    for dr, dc in directions:
        count = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < height and 0 <= cc < width and pieces[rr, cc] == 1:
            count += 1
            if count >= win_con:
                return True
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < height and 0 <= cc < width and pieces[rr, cc] == 1:
            count += 1
            if count >= win_con:
                return True
            rr -= dr
            cc -= dc
    return False


class XInARowHeuristicPolicy:
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
            r = a // self.width
            c = a % self.width
            test = my_pieces.copy()
            test[r, c] = 1
            if _check_win(test, self.win_con, r, c):
                #print("Win detected")
                return int(a)

        #Block opponent's immediate win
        for a in legal_actions:
            r = a // self.width
            c = a % self.width
            test = opp_pieces.copy()
            test[r, c] = 1
            if _check_win(test, self.win_con, r, c):
                #print("Block detected")
                return int(a)

        #Otherwise random legal move
        #print("Random move")
        return int(rng.choice(legal_actions))

class GomokuOffensiveHeuristicPolicy(XInARowHeuristicPolicy):
    def __init__(self):
        super().__init__(height=15, width=15, win_con=5)

    def _offensive_score(self, pieces: np.ndarray, r: int, c: int) -> float:
        # Score a hypothetical move at (r, c) by how strongly it extends our own
        # runs toward five, rewarding longer runs and open (unblocked) ends.
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        total = 0.0

        for dr, dc in directions:
            count = 1
            open_ends = 0

            rr, cc = r + dr, c + dc
            while 0 <= rr < self.height and 0 <= cc < self.width and pieces[rr, cc] == 1:
                count += 1
                rr += dr
                cc += dc
            if 0 <= rr < self.height and 0 <= cc < self.width and pieces[rr, cc] == 0:
                open_ends += 1

            rr, cc = r - dr, c - dc
            while 0 <= rr < self.height and 0 <= cc < self.width and pieces[rr, cc] == 1:
                count += 1
                rr -= dr
                cc -= dc
            if 0 <= rr < self.height and 0 <= cc < self.width and pieces[rr, cc] == 0:
                open_ends += 1

            # Fully blocked runs that cannot reach five contribute nothing.
            if count < self.win_con and open_ends == 0:
                continue

            # Reward longer runs strongly, and open runs more than half-open ones.
            total += (count ** 2) * (open_ends + 1)

        return total

    def __call__(self, obs: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator) -> int:
        # Offense-focused: win when possible, otherwise keep building our own
        # lines toward five. Immediate opponent wins are still blocked so the
        # policy survives long enough to keep attacking.
        mask = np.asarray(action_mask, dtype=np.int8)
        legal_actions = np.flatnonzero(mask.astype(bool)).astype(np.int64)
        if legal_actions.size == 0:
            return 0

        obs = np.asarray(obs, dtype=np.int8)
        my_pieces = obs[0]
        opp_pieces = obs[1]

        #Win immediately if possible
        for a in legal_actions:
            r = a // self.width
            c = a % self.width
            test = my_pieces.copy()
            test[r, c] = 1
            if _check_win(test, self.win_con, r, c):
                #print("Win detected")
                return int(a)

        #Block opponent's immediate win
        for a in legal_actions:
            r = a // self.width
            c = a % self.width
            test = opp_pieces.copy()
            test[r, c] = 1
            if _check_win(test, self.win_con, r, c):
                #print("Block detected")
                return int(a)

        #Offensive heuristic: pick the move that best extends our own runs
        best_score = -1.0
        best_actions: list = []
        for a in legal_actions:
            r = a // self.width
            c = a % self.width
            test = my_pieces.copy()
            test[r, c] = 1
            score = self._offensive_score(test, r, c)
            if score > best_score:
                best_score = score
                best_actions = [int(a)]
            elif score == best_score:
                best_actions.append(int(a))

        if best_actions:
            #print("Offensive move detected")
            return int(rng.choice(best_actions))

        #Otherwise random legal move
        #print("Random move")
        return int(rng.choice(legal_actions))


class GomokuDefensiveHeuristicPolicy(XInARowHeuristicPolicy):
    def __init__(self):
        super().__init__(height=15, width=15, win_con=5)

    def __call__(self, obs: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator) -> int:
        # Same as XInARowHeuristic but with a new heuristic to block 4-in-a-row opportunities from opponent
        mask = np.asarray(action_mask, dtype=np.int8)
        legal_actions = np.flatnonzero(mask.astype(bool)).astype(np.int64)
        if legal_actions.size == 0:
            return 0

        obs = np.asarray(obs, dtype=np.int8)
        my_pieces = obs[0]
        opp_pieces = obs[1]

        #Win immediately if possible
        for a in legal_actions:
            r = a // self.width
            c = a % self.width
            test = my_pieces.copy()
            test[r, c] = 1
            if _check_win(test, self.win_con, r, c):
                #print("Win detected")
                return int(a)

        #Block opponent's immediate win
        for a in legal_actions:
            r = a // self.width
            c = a % self.width
            test = opp_pieces.copy()
            test[r, c] = 1
            if _check_win(test, self.win_con, r, c):
                #print("Block detected")
                return int(a)

        #New heuristic with priority over random move: block 4-in-a-row opportunities from opponent
        #on at least 1 side to prevent unblockable win
        for a in legal_actions:
            r = a // self.width
            c = a % self.width
            test = opp_pieces.copy()
            test[r, c] = 1
            directions = ((1, 0), (0, 1), (1, 1), (1, -1))
            for dr, dc in directions:
                # Assumes that count will be less than 5, otherwise second heuristic would have triggered
                count = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < self.height and 0 <= cc < self.width and test[rr, cc] == 1:
                    count += 1
                    rr += dr
                    cc += dc
                unblocked = (0 <= rr < self.height and 0 <= cc < self.width and my_pieces[rr, cc] == 0)

                rr, cc = r - dr, c - dc
                while 0 <= rr < self.height and 0 <= cc < self.width and test[rr, cc] == 1:
                    count += 1
                    rr -= dr
                    cc -= dc
                # Check if both ends are unblocked
                unblocked &= (0 <= rr < self.height and 0 <= cc < self.width and my_pieces[rr, cc] == 0)

                if count >= 4 and unblocked:
                    #print("Block 4-in-a-row detected")
                    return int(a)

        #Otherwise random legal move
        #print("Random move")
        return int(rng.choice(legal_actions))