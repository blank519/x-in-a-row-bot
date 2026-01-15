import numpy as np
import gymnasium as gym
from gymnasium import spaces

from x_in_a_row_env import XInARowEnv


class SingleAgentSelfPlayEnv(gym.Env):
    def __init__(
        self,
        height,
        width,
        win_con,
        p1_symbol = "X",
        p2_symbol = "O",
        render_mode = None,
        opponent_policy = "random",
        randomize_learner = False,
    ):
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

        obs = self._observe_for_learner()
        info = {}
        return obs, info

    def _play_opponent_until_learner_turn(self):
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
            truncated = bool(self._env.truncations.get(self.learner_symbol, False))
            obs = np.zeros(self.observation_space.shape, dtype=np.int8)
        else:
            obs = self._observe_for_learner()

        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
