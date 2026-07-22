import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy.lib.stride_tricks import sliding_window_view

from x_in_a_row_env import XInARowEnv


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
) -> float:
    """Calculate raw (uncoefficiented) potential = own threat mass minus opponent 
    threat mass, summed over every win_con-length window. Windows containing both
    players' stones contribute nothing."""
    total = 0.0
    for idx in lines: # each "idx" is an array of indices representing a line
        own_line = own_flat[idx]
        opp_line = opp_flat[idx]
        own_counts = sliding_window_view(own_line, win_con).sum(axis=1).astype(np.int64)
        opp_counts = sliding_window_view(opp_line, win_con).sum(axis=1).astype(np.int64)

        own_only = opp_counts == 0 # count number of windows where opponent has 0 stones
        opp_only = own_counts == 0 # count number of windows where own has 0 stones
        total += float(weights[own_counts[own_only]].sum())
        total -= float(weights[opp_counts[opp_only]].sum())
    return total


class SingleAgentSelfPlayEnv(gym.Env):
    def __init__(self, height, width, win_con, p1_symbol = "X", p2_symbol = "O", render_mode = None, 
                 opponent_policy = "random", randomize_learner = False,
                 reward_shaping_coef = 0.0, reward_shaping_gamma = 0.99):
        super().__init__()
        self.height = height
        self.width = width
        self.win_con = win_con

        self._p1_symbol = p1_symbol
        self._p2_symbol = p2_symbol
        self.randomize_learner = randomize_learner

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
        self._shaping_lines = _build_line_index_groups(self.height, self.width, self.win_con)
        self._shaping_weights = _build_threat_weights(self.win_con)
        self._prev_potential = 0.0

    def _current_potential(self, obs: np.ndarray) -> float:
        if not self._reward_shaping_coef:
            return 0.0
        own_flat = np.asarray(obs[0], dtype=np.float32).reshape(-1)
        opp_flat = np.asarray(obs[1], dtype=np.float32).reshape(-1)
        raw = _board_threat_potential(
            own_flat, opp_flat, self._shaping_lines, self.win_con, self._shaping_weights
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

        # Ensure we start on learner turn.
        if self._env.agent_selection != self.learner_symbol:
            self._play_opponent_until_learner_turn()
            if self.render_mode == "human":
                self._env.render()

        obs = self._observe_for_learner()
        self._prev_potential = self._current_potential(obs)
        info = {}
        return obs, info

    def _play_opponent_until_learner_turn(self):
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._env.render()
        while self._env.agents and self._env.agent_selection == self.opponent_symbol:
            opp_obs = self._env.observe(self.opponent_symbol)
            mask = np.asarray(opp_obs["action_mask"], dtype=np.int8)
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

        self._env.step(int(action))

        # If game ended after learner move.
        terminated = not self._env.agents
        truncated = False
        reward = float(self._env.rewards.get(self.learner_symbol, 0.0))

        if not terminated:
            self._play_opponent_until_learner_turn()
            terminated = not self._env.agents

            if terminated:
                reward = float(self._env.rewards.get(self.learner_symbol, 0.0))

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

        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
