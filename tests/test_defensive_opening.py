"""The "defensive opening" curriculum (``defensive_opening_prob``) occasionally
replaces the empty-board reset with a designed puzzle: the opponent already has
a near-winning line and it is the learner's turn to block or lose.

These tests verify (a) the probability gate is wired up correctly and (b) the
installed position has the promised structure -- the opponent holds 3 or 4 in a
row and every learner stone sits close to that threat line.
"""

import numpy as np
import pytest

from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv

# Board large enough for the curriculum's hardcoded edge margin (3) to fit a
# length win_con(+1) line for both scenarios.
HEIGHT, WIDTH, WIN_CON = 13, 13, 5
RADIUS = 2


def _make_env(prob):
    return SingleAgentSelfPlayEnv(
        height=HEIGHT,
        width=WIDTH,
        win_con=WIN_CON,
        opponent_policy="random",
        randomize_learner=False,  # learner is always X / first player
        defensive_opening_prob=prob,
        defensive_opening_neighbor_radius=RADIUS,
    )


def _max_run(pieces):
    """Length of the longest straight run of 1s in a 2D binary array."""
    h, w = pieces.shape
    best = 0
    for r in range(h):
        for c in range(w):
            if pieces[r, c] != 1:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and pieces[pr, pc] == 1:
                    continue  # not the start of this run; count it once from its head
                n, rr, cc = 0, r, c
                while 0 <= rr < h and 0 <= cc < w and pieces[rr, cc] == 1:
                    n += 1
                    rr += dr
                    cc += dc
                best = max(best, n)
    return best


def _installed(obs):
    return int(obs[1].sum()) > 0


def test_prob_zero_never_installs_puzzle():
    env = _make_env(0.0)
    for seed in range(40):
        obs, _ = env.reset(seed=seed)
        # Learner is first player, so with no puzzle the board is completely empty.
        assert obs[0].sum() == 0 and obs[1].sum() == 0
    env.close()


def test_prob_one_always_installs_valid_puzzle():
    env = _make_env(1.0)
    scenarios_seen = set()
    for seed in range(40):
        obs, _ = env.reset(seed=seed)
        learner, opponent = obs[0], obs[1]

        assert _installed(obs), f"seed {seed}: no defensive puzzle installed"

        # (1) Opponent (attacker) holds exactly 3 or 4 in a row, and all its
        #     stones belong to that single line.
        opp_run = _max_run(opponent)
        assert opp_run in (WIN_CON - 2, WIN_CON - 1), f"seed {seed}: opp run {opp_run}"
        assert int(opponent.sum()) == opp_run, "opponent stones should form one line"
        scenarios_seen.add(opp_run)

        # (2) It is legally the learner's turn. Learner is the first player here,
        #     so equal stone counts <=> learner to move.
        assert env._env.agent_selection == env.learner_symbol
        assert int(learner.sum()) == int(opponent.sum())

        # (3) Every learner stone is within Chebyshev RADIUS of some attacker
        #     stone (i.e. the defender's pieces are all "nearby" the threat).
        attacker_cells = np.argwhere(opponent == 1)
        for lr, lc in np.argwhere(learner == 1):
            dist = np.max(np.abs(attacker_cells - np.array([lr, lc])), axis=1).min()
            assert dist <= RADIUS, f"seed {seed}: learner stone ({lr},{lc}) too far"

    # Over 40 seeds we expect to hit both the "block the four" (run 4) and
    # "block the open three" (run 3) scenarios.
    assert scenarios_seen == {WIN_CON - 2, WIN_CON - 1}, scenarios_seen
    env.close()


def test_install_frequency_tracks_probability():
    prob = 0.5
    env = _make_env(prob)
    n = 400
    installed = sum(_installed(env.reset(seed=seed)[0]) for seed in range(n))
    frac = installed / n
    # Generous tolerance: this checks the gate is actually probabilistic, not the
    # exact rate.
    assert 0.4 <= frac <= 0.6, f"install fraction {frac} not near {prob}"
    env.close()
