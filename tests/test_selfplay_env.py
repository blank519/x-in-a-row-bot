"""The self-play environment should run a full episode to termination no matter
what kind of opponent is plugged in: the built-in random policy, a scripted
heuristic (a callable), or a trained-agent-style object exposing ``.predict``.

These tests drive the *learner* with uniformly random legal moves and only vary
the *opponent* type, exercising every branch of
``SingleAgentSelfPlayEnv._play_opponent_until_learner_turn``.
"""

import numpy as np
import pytest

from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv, CurriculumMaskedSelfPlayEnv
from heuristic_policy import XInARowHeuristicPolicy

HEIGHT, WIDTH, WIN_CON = 6, 6, 4


class _StubAgent:
    """Minimal stand-in for a trained SB3 model: exposes the ``.predict``
    interface used for snapshot/self-play opponents, returning a legal move
    without importing torch."""

    def predict(self, observation, action_masks=None, deterministic=True):
        mask = np.asarray(action_masks, dtype=np.int8)
        legal = np.flatnonzero(mask.astype(bool))
        return int(legal[0]), None


def _make_opponent(kind):
    if kind == "random":
        return "random"
    if kind == "heuristic":
        return XInARowHeuristicPolicy(HEIGHT, WIDTH, WIN_CON)
    if kind == "agent":
        return _StubAgent()
    raise ValueError(kind)


def _play_full_episode(env, seed):
    """Play the learner with random legal moves until the episode ends. Returns
    the number of learner steps taken. Asserts invariants along the way."""
    obs, info = env.reset(seed=seed)
    assert env.observation_space.contains(obs), "reset obs outside observation space"

    rng = np.random.default_rng(seed)
    max_steps = env.height * env.width + 5
    for step_i in range(max_steps):
        mask = np.asarray(env.action_masks(), dtype=np.int8)
        assert mask.shape == (env.height * env.width,)
        legal = np.flatnonzero(mask.astype(bool))
        assert legal.size > 0, "no legal moves on a non-terminal turn"

        action = int(rng.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)

        assert np.isfinite(reward), "reward must be finite"
        assert "block_reward" in info
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            return step_i + 1

    pytest.fail("episode did not terminate within height*width+5 steps")


@pytest.mark.parametrize("opponent_kind", ["random", "heuristic", "agent"])
@pytest.mark.parametrize("randomize_learner", [False, True])
def test_single_agent_env_runs_with_any_opponent(opponent_kind, randomize_learner):
    env = SingleAgentSelfPlayEnv(
        height=HEIGHT,
        width=WIDTH,
        win_con=WIN_CON,
        opponent_policy=_make_opponent(opponent_kind),
        randomize_learner=randomize_learner,
    )
    # A handful of episodes with different seeds to cover both start parities and
    # different opponent move sequences.
    for seed in range(4):
        steps = _play_full_episode(env, seed=seed)
        assert 1 <= steps <= HEIGHT * WIDTH
    env.close()


@pytest.mark.parametrize("opponent_kind", ["random", "heuristic", "agent"])
def test_curriculum_masked_env_runs_with_any_opponent(opponent_kind):
    # Same, but through the subclass actually used in training, with locality
    # masking active for both learner and opponent.
    env = CurriculumMaskedSelfPlayEnv(
        height=HEIGHT,
        width=WIDTH,
        win_con=WIN_CON,
        opponent_policy=_make_opponent(opponent_kind),
        randomize_learner=True,
        learner_local_mask_radius=2,
        opponent_local_mask_radius=2,
    )
    for seed in range(4):
        steps = _play_full_episode(env, seed=seed)
        assert 1 <= steps <= HEIGHT * WIDTH
    env.close()


def test_local_mask_restricts_to_neighbourhood():
    # With stones on the board, the learner's legal moves should be a subset of
    # the fully-legal moves and confined to the neighbourhood of occupied cells.
    env = CurriculumMaskedSelfPlayEnv(
        height=HEIGHT,
        width=WIDTH,
        win_con=WIN_CON,
        opponent_policy="random",
        randomize_learner=False,
        learner_local_mask_radius=1,
    )
    env.reset(seed=0)
    # Place a couple of stones so masking has an anchor.
    env._env.board[2][2] = env.learner_symbol
    env._env.board[3][3] = env.opponent_symbol
    env._last_action_mask = None

    masked = np.asarray(env.action_masks(), dtype=np.int8).reshape(HEIGHT, WIDTH)
    occupied = np.zeros((HEIGHT, WIDTH), dtype=bool)
    occupied[2, 2] = occupied[3, 3] = True
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if masked[r, c]:
                near = any(
                    abs(r - orr) <= 1 and abs(c - occ) <= 1
                    for orr, occ in np.argwhere(occupied)
                )
                assert near, f"cell ({r},{c}) allowed but not within radius 1"
    env.close()
