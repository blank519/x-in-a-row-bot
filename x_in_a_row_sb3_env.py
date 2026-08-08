import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy.lib.stride_tricks import sliding_window_view

from x_in_a_row_env import XInARowEnv
import game_utils


def _build_line_index_groups(height: int, width: int, win_con: int) -> list:
    """Precompute flat-index arrays for every row, column, and diagonal that is
    at least ``win_con`` cells long. Used for windowed threat scoring."""
    lines: list = []

    for r in range(height):
        if width >= win_con:
            lines.append(np.array([r * width + c for c in range(width)], dtype=np.int64))

    for c in range(width):
        if height >= win_con:
            lines.append(np.array([r * width + c for r in range(height)], dtype=np.int64))

    # Main diagonals going down and right, grouped by consistent k = c - r.
    for k in range(-(height - 1), width):
        cells = [r * width + (r + k) for r in range(height) if 0 <= r + k < width]
        # 0 <= r+k < width checks if the column r+k is within bounds
        if len(cells) >= win_con:
            lines.append(np.array(cells, dtype=np.int64))

    # Anti-diagonals going down and left, grouped by consistent s = r + c.
    for s in range(0, height + width - 1):
        cells = [r * width + (s - r) for r in range(height) if 0 <= s - r < width]
        if len(cells) >= win_con:
            lines.append(np.array(cells, dtype=np.int64))

    return lines


def _build_threat_weights(win_con: int) -> np.ndarray:
    """Weight assigned to a length-``win_con`` window by how many of one player's
    stones it contains (and none of the opponent's). Singletons are ignored so
    only genuine multi-stone threats contribute."""
    weights = np.zeros(win_con + 1, dtype=np.float32)
    for k in range(2, win_con + 1):
        weights[k] = (k * k) / float(win_con * win_con) # Exponentially greater weight for longer threats
    return weights


def _board_threat_potential(
    own_flat: np.ndarray,
    opp_flat: np.ndarray,
    lines: list,
    win_con: int,
    weights: np.ndarray,
    defense_weight: float = 1.0,
) -> float:
    """Calculate raw (uncoefficiented) potential = own threat mass minus opponent 
    threat mass, summed over every win_con-length window. Windows containing both
    players' stones contribute nothing. ``defense_weight`` > 1 scales up the
    opponent-threat term so that blocking (and not allowing threats) is rewarded
    more strongly than building one's own."""
    total = 0.0
    for idx in lines: # each "idx" is an array of indices representing a line
        own_line = own_flat[idx]
        opp_line = opp_flat[idx]
        own_counts = sliding_window_view(own_line, win_con).sum(axis=1).astype(np.int64)
        opp_counts = sliding_window_view(opp_line, win_con).sum(axis=1).astype(np.int64)

        own_only = opp_counts == 0 # count number of windows where opponent has 0 stones
        opp_only = own_counts == 0 # count number of windows where own has 0 stones
        total += float(weights[own_counts[own_only]].sum())
        total -= defense_weight * float(weights[opp_counts[opp_only]].sum())
    return total


def _opponent_threat_mass(
    own_flat: np.ndarray,
    opp_flat: np.ndarray,
    lines: list,
    win_con: int,
    weights: np.ndarray,
) -> float:
    """Total weighted threat mass of the OPPONENT alone: sum over win_con-length
    windows that contain only opponent stones. A learner stone placed inside such
    a window makes it mixed (no longer counted), so the drop in this quantity
    caused by the learner's move measures how much threat that move blocked.
    """
    total = 0.0
    for idx in lines:
        own_line = own_flat[idx]
        opp_line = opp_flat[idx]
        own_counts = sliding_window_view(own_line, win_con).sum(axis=1).astype(np.int64)
        opp_counts = sliding_window_view(opp_line, win_con).sum(axis=1).astype(np.int64)
        opp_only = own_counts == 0
        total += float(weights[opp_counts[opp_only]].sum())
    return total


class SingleAgentSelfPlayEnv(gym.Env):
    def __init__(self, height, width, win_con, p1_symbol = "X", p2_symbol = "O", render_mode = None, 
                 opponent_policy = "random", randomize_learner = False,
                 reward_shaping_coef = 0.0, reward_shaping_gamma = 0.99,
                 reward_shaping_defense_weight = 1.0,
                 block_reward_coef = 0.0,
                 defensive_opening_prob = 0.0,
                 defensive_opening_neighbor_radius = 2):
        super().__init__()
        self.height = height
        self.width = width
        self.win_con = win_con

        self._p1_symbol = p1_symbol
        self._p2_symbol = p2_symbol
        self.randomize_learner = randomize_learner
        # Probability that reset() installs a designed defensive puzzle instead of
        # a normal empty-board start. See _install_defensive_opening.
        self.defensive_opening_prob = float(defensive_opening_prob)
        # Parity/neutral learner stones in a puzzle are placed within this
        # Chebyshev radius of the threat line so the position stays consistent
        # with the local-move masking used during training.
        self._defensive_opening_radius = int(defensive_opening_neighbor_radius)

        self.learner_symbol = p1_symbol
        self.opponent_symbol = p2_symbol
        self.render_mode = render_mode
        self._opponent = opponent_policy

        self._env = XInARowEnv(
            height=height,
            width=width,
            win_con=win_con,
            p1=p1_symbol,
            p2=p2_symbol,
            render_mode=render_mode,
        )

        self.action_space = spaces.Discrete(height * width)
        self.observation_space = spaces.MultiBinary((2, height, width))

        self._last_action_mask: np.ndarray | None = None

        # Potential-based reward shaping (Ng et al., 1999): shaped reward is
        # gamma * Phi(s') - Phi(s), which does not change the optimal policy but
        # gives a dense signal that rewards building threats and blocking the
        # opponent's. Disabled when reward_shaping_coef == 0.
        self._reward_shaping_coef = float(reward_shaping_coef)
        self._reward_shaping_gamma = float(reward_shaping_gamma)
        # >1 emphasizes blocking the opponent over building one's own threats.
        self._reward_shaping_defense_weight = float(reward_shaping_defense_weight)
        # Immediate (non-potential) reward for reducing the opponent's threat mass
        # with the learner's own move. Unlike potential-based shaping this DOES
        # change the optimal policy, so it directly incentivizes blocking.
        # Disabled when block_reward_coef == 0.
        self._block_reward_coef = float(block_reward_coef)
        self._shaping_lines = _build_line_index_groups(self.height, self.width, self.win_con)
        self._shaping_weights = _build_threat_weights(self.win_con)
        self._prev_potential = 0.0

    def _current_potential(self, obs: np.ndarray) -> float:
        if not self._reward_shaping_coef:
            return 0.0
        own_flat = np.asarray(obs[0], dtype=np.float32).reshape(-1)
        opp_flat = np.asarray(obs[1], dtype=np.float32).reshape(-1)
        raw = _board_threat_potential(
            own_flat, opp_flat, self._shaping_lines, self.win_con, self._shaping_weights,
            defense_weight=self._reward_shaping_defense_weight,
        )
        return self._reward_shaping_coef * raw

    def set_opponent(self, opponent):
        self._opponent = opponent

    def _legal_actions_from_mask(self, mask: np.ndarray) -> np.ndarray:
        legal = np.flatnonzero(mask.astype(bool))
        if legal.size == 0:
            # Should not happen unless env is terminal; return a dummy.
            return np.array([0], dtype=np.int64)
        return legal.astype(np.int64)

    def _observe_for_learner(self) -> np.ndarray:
        obs_dict = self._env.observe(self.learner_symbol)
        self._last_action_mask = np.asarray(obs_dict["action_mask"], dtype=np.int8)
        return np.asarray(obs_dict["observation"], dtype=np.int8)

    def action_masks(self) -> np.ndarray:
        if self._last_action_mask is None:
            _ = self._observe_for_learner()
        return np.asarray(self._last_action_mask, dtype=np.int8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_learner:
            if bool(self.np_random.integers(0, 2)):
                self.learner_symbol = self._p1_symbol
                self.opponent_symbol = self._p2_symbol
            else:
                self.learner_symbol = self._p2_symbol
                self.opponent_symbol = self._p1_symbol

        self._last_action_mask = None
        _obs_dict, _info = self._env.reset(seed=seed, options=options)

        # Optionally overwrite the empty board with a designed defensive puzzle so
        # the learner is forced to block an immediate opponent threat. Sparse
        # self-play never reaches these states, so the learner otherwise never
        # learns to defend (it just races to build its own line and loses).
        installed = False
        if self.defensive_opening_prob > 0.0 and self.np_random.random() < self.defensive_opening_prob:
            installed = self._install_defensive_opening()

        # Ensure we start on learner turn. When a puzzle is installed the turn is
        # already handed to the learner, so skip the opponent pre-move.
        if not installed and self._env.agent_selection != self.learner_symbol:
            self._play_opponent_until_learner_turn()
            if self.render_mode == "human":
                self._env.render()

        obs = self._observe_for_learner()
        self._prev_potential = self._current_potential(obs)
        info = {}
        return obs, info

    # ------------------------------------------------------------------
    # Defensive-opening curriculum
    # ------------------------------------------------------------------
    @staticmethod
    def _start_bounds(span, dim, margin):
        """Inclusive [lo, hi] range for a line's start index so that both the
        start and start+span stay within [margin, dim-1-margin]."""
        lo_pos = margin
        hi_pos = dim - 1 - margin
        return max(lo_pos, lo_pos - span), min(hi_pos, hi_pos - span)

    def _random_line_cells(self, length, margin=0):
        """Return `length` consecutive in-bounds cells along a random one of the
        four line directions, with every cell kept `margin` away from the board
        edge. Returns a list of (row, col) or None if none fits."""
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        dr, dc = directions[int(self.np_random.integers(0, len(directions)))]
        span_r = dr * (length - 1)
        span_c = dc * (length - 1)
        r_lo, r_hi = self._start_bounds(span_r, self.height, margin)
        c_lo, c_hi = self._start_bounds(span_c, self.width, margin)
        if r_lo > r_hi or c_lo > c_hi:
            return None
        sr = int(self.np_random.integers(r_lo, r_hi + 1))
        sc = int(self.np_random.integers(c_lo, c_hi + 1))
        cells = [(sr + i * dr, sc + i * dc) for i in range(length)]
        # Randomly flip so the open (block) end varies between the two ends.
        if bool(self.np_random.integers(0, 2)):
            cells.reverse()
        return cells

    def _cells_near(self, line_cells, radius, forbidden):
        """Empty, non-forbidden board cells within Chebyshev `radius` of any cell
        in `line_cells`. Used to place parity stones close to the threat so the
        puzzle resembles a realistic locally-masked position."""
        cells = set()
        for (r, c) in line_cells:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < self.height and 0 <= cc < self.width):
                        continue
                    if (rr, cc) in forbidden:
                        continue
                    if self._env.board[rr][cc] is not None:
                        continue
                    cells.add((rr, cc))
        return list(cells)

    def _set_turn_to_learner(self):
        """Hand the move to the learner. The underlying env keeps the invariant
        that the agent_selector's last-returned agent equals agent_selection, so
        for the second player we advance the selector exactly once."""
        first_player = self._env.possible_agents[0]
        if self.learner_symbol == first_player:
            self._env.agent_selection = first_player
        else:
            self._env.agent_selection = self._env._agent_selector.next()

    def _install_defensive_opening(self):
        """Overwrite the freshly-reset board with a designed position in which the
        opponent (attacker) has an immediate line threat and it is the learner's
        (defender's) turn to block. Returns True if a position was installed.

        Stone counts are chosen so it is legally the learner's turn under
        X-moves-first alternation. Extra "neutral" defender stones (needed only
        for parity) are placed far from the threat where they cannot help either
        side."""
        h, w, wc = self.height, self.width, self.win_con
        if h < wc + 1 or w < wc + 1:
            return False

        attacker = self.opponent_symbol
        defender = self.learner_symbol
        first_player = self._env.possible_agents[0]
        margin = 3

        for _attempt in range(8):
            scenario = int(self.np_random.integers(0, 2))
            if scenario == 0:
                # "Block the four": attacker has win_con-1 in a row with one end
                # already blocked by a defender stone; the open end is the unique
                # winning square, so the learner must play it or lose next move.
                cells = self._random_line_cells(wc + 1, margin)
                if cells is None:
                    continue
                att_cells = cells[1:wc]        # win_con-1 attacker stones
                def_cells = [cells[0]]         # blocked end
                must_stay_empty = [cells[wc]]  # open end (the correct block)
            else:
                # "Block the open three": attacker has win_con-2 in a row with
                # both ends open; not blocking concedes an unstoppable open four.
                cells = self._random_line_cells(wc, margin)
                if cells is None:
                    continue
                att_cells = cells[1:wc - 1]    # win_con-2 attacker stones
                def_cells = []
                dr = cells[1][0] - cells[0][0]
                dc = cells[1][1] - cells[0][1]
                # Keep both open ends AND the cells just beyond them empty so the
                # attacker's open-four threat is preserved (and so parity stones
                # can't accidentally neutralise the puzzle).
                must_stay_empty = [
                    cells[0], cells[wc - 1],
                    (cells[0][0] - dr, cells[0][1] - dc),
                    (cells[wc - 1][0] + dr, cells[wc - 1][1] + dc),
                ]

            n_att = len(att_cells)
            # Make it the learner's turn: equal counts if the learner moves first,
            # otherwise the attacker (first player) has exactly one more stone.
            target_def = n_att if defender == first_player else n_att - 1
            extra_def = target_def - len(def_cells)
            if extra_def < 0:
                continue

            # Commit the attacker line and the pre-placed blocked-end stone first
            # so neighbour/emptiness checks see them.
            for (r, c) in att_cells:
                self._env.board[r][c] = attacker
            for (r, c) in def_cells:
                self._env.board[r][c] = defender

            # Place the remaining parity ("neutral") defender stones near the
            # threat line rather than scattered across the board, so the position
            # matches what the learner sees under local-move masking. Skip any
            # cell that would hand the defender its own winning line.
            forbidden = set(att_cells) | set(def_cells) | set(must_stay_empty)
            candidates = self._cells_near(att_cells, self._defensive_opening_radius, forbidden)
            placed_neutrals = []
            for idx in self.np_random.permutation(len(candidates)):
                if len(placed_neutrals) >= extra_def:
                    break
                r, c = candidates[int(idx)]
                if self._env.board[r][c] is not None:
                    continue
                self._env.board[r][c] = defender
                if game_utils.check_winner(self._env.board, defender, r, c, wc):
                    self._env.board[r][c] = None
                    continue
                placed_neutrals.append((r, c))

            if len(placed_neutrals) < extra_def:
                # Not enough legal room near the line; roll back and retry.
                for (r, c) in placed_neutrals + def_cells + att_cells:
                    self._env.board[r][c] = None
                continue

            self._env.current_step = n_att + len(def_cells) + len(placed_neutrals)
            self._set_turn_to_learner()
            return True

        return False

    def _opponent_action_mask(self, mask, obs):
        """Hook for subclasses to restrict the opponent's legal moves (e.g. local
        masking so the opponent plays under the same locality constraint as the
        learner). Base implementation returns the mask unchanged."""
        return mask

    def _play_opponent_until_learner_turn(self):
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._env.render()
        while self._env.agents and self._env.agent_selection == self.opponent_symbol:
            opp_obs = self._env.observe(self.opponent_symbol)
            mask = np.asarray(opp_obs["action_mask"], dtype=np.int8)
            mask = np.asarray(self._opponent_action_mask(mask, opp_obs["observation"]), dtype=np.int8)
            legal = self._legal_actions_from_mask(mask)

            if self._opponent == "random" or self._opponent is None:
                opp_action = int(self.np_random.choice(legal))
            elif callable(self._opponent):
                opp_action = int(self._opponent(opp_obs["observation"], mask, self.np_random))
            elif hasattr(self._opponent, "predict"):
                opp_action, _state = self._opponent.predict(
                    opp_obs["observation"],
                    action_masks=mask,
                    deterministic=True,
                )
                opp_action = int(opp_action)
            else:
                opp_action = int(self.np_random.choice(legal))

            if mask[int(opp_action)] == 0:
                print("Illegal move detected")
                opp_action = int(self.np_random.choice(legal))

            self._env.step(opp_action)

    def step(self, action: int):
        if not self._env.agents:
            obs = np.zeros(self.observation_space.shape, dtype=np.int8)
            info = {}
            return obs, 0.0, True, True, info

        # If not our turn, advance with opponent moves.
        if self._env.agent_selection != self.learner_symbol:
            self._play_opponent_until_learner_turn()
            if not self._env.agents:
                obs = np.zeros(self.observation_space.shape, dtype=np.int8)
                info = {}
                return obs, 0.0, True, True, info

        # Clip invalid actions: MaskablePPO should prevent this, but keep a hard guard.
        mask = self.action_masks()
        if mask[int(action)] == 0:
            legal = self._legal_actions_from_mask(mask)
            action = int(self.np_random.choice(legal))

        # Opponent threat mass on the board immediately before the learner's move,
        # used to compute the immediate block reward below.
        opp_mass_before = 0.0
        if self._block_reward_coef:
            obs_before = self._observe_for_learner()
            opp_mass_before = _opponent_threat_mass(
                obs_before[0].astype(np.float32).reshape(-1),
                obs_before[1].astype(np.float32).reshape(-1),
                self._shaping_lines, self.win_con, self._shaping_weights,
            )

        self._env.step(int(action))

        # If game ended after learner move.
        terminated = not self._env.agents
        truncated = False
        reward = float(self._env.rewards.get(self.learner_symbol, 0.0))

        # Immediate block reward: how much the learner's own move reduced the
        # opponent's threat mass (measured before the opponent replies). A win
        # already terminates with +1, so skip it in that case.
        block_reward = 0.0
        if self._block_reward_coef and not terminated:
            obs_after = self._observe_for_learner()
            opp_mass_after = _opponent_threat_mass(
                obs_after[0].astype(np.float32).reshape(-1),
                obs_after[1].astype(np.float32).reshape(-1),
                self._shaping_lines, self.win_con, self._shaping_weights,
            )
            block_reward = self._block_reward_coef * max(0.0, opp_mass_before - opp_mass_after)
            reward += block_reward

        if not terminated:
            self._play_opponent_until_learner_turn()
            terminated = not self._env.agents

            if terminated:
                reward = float(self._env.rewards.get(self.learner_symbol, 0.0)) + block_reward

        if terminated:
            truncated = bool(self._env.truncations.get(self.learner_symbol, False))
            obs = np.zeros(self.observation_space.shape, dtype=np.int8)
            new_potential = 0.0  # Absorbing state has zero potential by convention.
        else:
            obs = self._observe_for_learner()
            new_potential = self._current_potential(obs)

        if self._reward_shaping_coef:
            reward += self._reward_shaping_gamma * new_potential - self._prev_potential
            self._prev_potential = new_potential

        info = {"block_reward": block_reward}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


def apply_local_move_mask(
    action_mask: np.ndarray,
    obs: np.ndarray,
    height: int,
    width: int,
    radius,
    min_stones_before_mask: int = 1,
) -> np.ndarray:
    """Restrict legal moves to cells within Chebyshev radius of existing stones.

    When there are fewer than ``min_stones_before_mask`` stones on the board, the
    move is instead restricted to cells within ``radius`` of the board center so
    the agent opens near the middle. Falls back to the original mask if the
    filtered mask would be empty.
    """
    base_mask = np.asarray(action_mask, dtype=np.int8)
    if radius is None or radius < 0:
        return base_mask

    obs_arr = np.asarray(obs)
    if obs_arr.ndim != 3 or obs_arr.shape[0] < 2:
        return base_mask

    occupied = (obs_arr[0] + obs_arr[1]) > 0
    if int(occupied.sum()) < int(min_stones_before_mask):
        center_r = height // 2
        center_c = width // 2
        center_candidate = np.zeros((height, width), dtype=bool)
        center_candidate[
            max(0, center_r - radius):min(height, center_r + radius + 1),
            max(0, center_c - radius):min(width, center_c + radius + 1),
        ] = True
        center_masked = base_mask.astype(bool) & center_candidate.reshape(-1)
        if center_masked.any():
            return center_masked.astype(np.int8)
        return base_mask

    candidate = np.zeros((height, width), dtype=bool)
    for r, c in np.argwhere(occupied):
        candidate[
            max(0, int(r) - radius):min(height, int(r) + radius + 1),
            max(0, int(c) - radius):min(width, int(c) + radius + 1),
        ] = True

    masked = base_mask.astype(bool) & candidate.reshape(-1)
    if masked.any():
        return masked.astype(np.int8)
    return base_mask


class CurriculumMaskedSelfPlayEnv(SingleAgentSelfPlayEnv):
    """Single-agent self-play env with optional learner locality masking."""

    def __init__(self, *args, learner_local_mask_radius=None,
                 opponent_local_mask_radius=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._learner_local_mask_radius = learner_local_mask_radius
        self._opponent_local_mask_radius = opponent_local_mask_radius

    def set_learner_local_mask_radius(self, radius):
        self._learner_local_mask_radius = None if radius is None else int(radius)

    def set_opponent_local_mask_radius(self, radius):
        self._opponent_local_mask_radius = None if radius is None else int(radius)

    def action_masks(self) -> np.ndarray:
        base_mask = super().action_masks()
        return apply_local_move_mask(
            action_mask=base_mask,
            obs=self._env.observe(self.learner_symbol)["observation"],
            height=self.height,
            width=self.width,
            radius=self._learner_local_mask_radius,
        )

    def _opponent_action_mask(self, mask, obs):
        return apply_local_move_mask(
            action_mask=mask,
            obs=obs,
            height=self.height,
            width=self.width,
            radius=self._opponent_local_mask_radius,
        )
