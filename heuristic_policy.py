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
    def __init__(self, height: int, width: int, win_con: int, mistake_rate: float = 0.0):
        self.height = int(height)
        self.width = int(width)
        self.win_con = int(win_con)
        # Probability of ignoring the heuristic on a given turn and playing a
        # uniformly random legal move instead. Intended to be annealed toward 0
        # over training so early opponents are beatable. Wiring the schedule
        # into the training loop is deliberately left to the caller.
        self.mistake_rate = float(mistake_rate)

    def set_mistake_rate(self, mistake_rate: float) -> None:
        self.mistake_rate = float(mistake_rate)

    def _maybe_random_mistake(self, legal_actions: np.ndarray, rng: np.random.Generator):
        # Returns a random legal action with probability ``mistake_rate``,
        # otherwise None (meaning: proceed with the heuristic).
        if self.mistake_rate > 0.0 and float(rng.random()) < self.mistake_rate:
            return int(rng.choice(legal_actions))
        return None

    def __call__(self, obs: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator) -> int:
        mask = np.asarray(action_mask, dtype=np.int8)
        legal_actions = np.flatnonzero(mask.astype(bool)).astype(np.int64)
        if legal_actions.size == 0:
            return 0

        mistake = self._maybe_random_mistake(legal_actions, rng)
        if mistake is not None:
            return mistake

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
    def __init__(self, mistake_rate: float = 0.0):
        super().__init__(height=15, width=15, win_con=5, mistake_rate=mistake_rate)

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

        mistake = self._maybe_random_mistake(legal_actions, rng)
        if mistake is not None:
            return mistake

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
    def __init__(self, mistake_rate: float = 0.0):
        super().__init__(height=15, width=15, win_con=5, mistake_rate=mistake_rate)

    def __call__(self, obs: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator) -> int:
        # Same as XInARowHeuristic but with a new heuristic to block 4-in-a-row opportunities from opponent
        mask = np.asarray(action_mask, dtype=np.int8)
        legal_actions = np.flatnonzero(mask.astype(bool)).astype(np.int64)
        if legal_actions.size == 0:
            return 0

        mistake = self._maybe_random_mistake(legal_actions, rng)
        if mistake is not None:
            return mistake

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

class GomokuCombinedHeuristicPolicy(GomokuOffensiveHeuristicPolicy):
    def __init__(self, mistake_rate: float = 0):
        super().__init__(mistake_rate)

    def __call__(self, obs: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator) -> int:
        """Prioritize defensive heuristic, then offensive heuristic, with a chance of a random move = mistake_rate"""
        mask = np.asarray(action_mask, dtype=np.int8)
        legal_actions = np.flatnonzero(mask.astype(bool)).astype(np.int64)
        if legal_actions.size == 0:
            return 0

        # Priority 1: Check for random mistake 
        mistake = self._maybe_random_mistake(legal_actions, rng)
        if mistake is not None:
            return mistake

        obs = np.asarray(obs, dtype=np.int8)
        my_pieces = obs[0]
        opp_pieces = obs[1]

        # Priority 2: Win immediately if possible
        best_actions = []
        for a in legal_actions:
            r = a // self.width
            c = a % self.width
            test = my_pieces.copy()
            test[r, c] = 1
            if _check_win(test, self.win_con, r, c):
                #print("Win detected")
                best_actions.append(int(a))
        # If there are winning moves, choose one randomly
        if best_actions:
            return int(rng.choice(best_actions))

        # Priority 3: Block opponent's immediate win
        for a in legal_actions:
            r = a // self.width
            c = a % self.width
            test = opp_pieces.copy()
            test[r, c] = 1
            if _check_win(test, self.win_con, r, c):
                #print("Block detected")
                best_actions.append(int(a))
        # If there are blocking moves, choose one randomly
        if best_actions:
            return int(rng.choice(best_actions))

        # Priority 4: Block formation of open-4 chains from opponent (Defensive Heuristic)
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
                    best_actions.append(int(a))
        # If there are blocking moves, choose one randomly
        if best_actions:
            return int(rng.choice(best_actions))

        # Priority 5: Extend the longest chain in a row (Offensive heuristic)
        #Offensive heuristic: pick the move that best extends our own runs
        best_score = -1.0
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
        
        #If all else fails, choose a random legal move
        return int(rng.choice(legal_actions))