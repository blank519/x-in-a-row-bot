import os

import numpy as np

from heuristic_policy import XInARowHeuristicPolicy
from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv


class HeuristicEvaluator:
    def __init__(
        self,
        height,
        width,
        win_con,
        heuristic,
        n_games_per_side = 50,
        best_model_path = "best_vs_heuristic",
        deterministic = True,
        seed = None,
        verbose = 1,
    ):
        self.height = height
        self.width = width
        self.win_con = win_con
        self.n_games_per_side = n_games_per_side
        self.best_model_path = best_model_path
        self.deterministic = deterministic
        self.seed = seed
        self.verbose = verbose

        self.heuristic = heuristic

        self.eval_num = 0
        self.best_eval_num = 0

        self.best_key = None
        self.best_metrics = None

        self.p1_env = self._make_env()
        self.p2_env = self._make_env()

        self.p1_env.learner_symbol = "X"
        self.p1_env.opponent_symbol = "O"
        self.p2_env.learner_symbol = "O"
        self.p2_env.opponent_symbol = "X"

        self.p1_env.set_opponent(self.heuristic)
        self.p2_env.set_opponent(self.heuristic)

    def _make_env(self):
        return SingleAgentSelfPlayEnv(
            height=self.height,
            width=self.width,
            win_con=self.win_con,
            p1_symbol="X",
            p2_symbol="O",
            render_mode=None,
            opponent_policy=self.heuristic,
            randomize_learner=False,
        )

    def _play_one(self, model, learner_symbol, rng):
        if learner_symbol == "X":
            env = self.p1_env
        else:
            env = self.p2_env

        obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = False
        truncated = False

        last_reward = 0.0
        while not (terminated or truncated):
            action_masks = env.action_masks()
            action, _state = model.predict(obs, action_masks=action_masks, deterministic=self.deterministic)
            obs, reward, terminated, truncated, _info = env.step(int(action))
            last_reward = float(reward)

        return last_reward

    def evaluate(self, model):
        self.eval_num += 1
        if self.seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(int(self.seed))

        results = {
            "X": {"win": 0, "draw": 0, "loss": 0},
            "O": {"win": 0, "draw": 0, "loss": 0},
        }

        def outcome_from_reward(r: float):
            if r > 0.5:
                return "win"
            if r < -0.5:
                return "loss"
            return "draw"

        for _ in range(self.n_games_per_side):
            r = self._play_one(model, learner_symbol="X", rng=rng)
            results["X"][outcome_from_reward(r)] += 1

        for _ in range(self.n_games_per_side):
            r = self._play_one(model, learner_symbol="O", rng=rng)
            results["O"][outcome_from_reward(r)] += 1

        wins = results["X"]["win"] + results["O"]["win"]
        draws = results["X"]["draw"] + results["O"]["draw"]
        losses = results["X"]["loss"] + results["O"]["loss"]

        x_win_rate = results["X"]["win"] / self.n_games_per_side
        x_draw_rate = results["X"]["draw"] / self.n_games_per_side
        x_loss_rate = results["X"]["loss"] / self.n_games_per_side

        o_win_rate = results["O"]["win"] / self.n_games_per_side
        o_draw_rate = results["O"]["draw"] / self.n_games_per_side
        o_loss_rate = results["O"]["loss"] / self.n_games_per_side

        worst_loss_rate = max(x_loss_rate, o_loss_rate)
        worst_win_rate = min(x_win_rate, o_win_rate)
        worst_draw_rate = min(x_draw_rate, o_draw_rate)

        total_games = 2 * self.n_games_per_side
        metrics = {
            "win_rate": wins / total_games,
            "draw_rate": draws / total_games,
            "loss_rate": losses / total_games,
            "x_win_rate": x_win_rate,
            "x_draw_rate": x_draw_rate,
            "x_loss_rate": x_loss_rate,
            "o_win_rate": o_win_rate,
            "o_draw_rate": o_draw_rate,
            "o_loss_rate": o_loss_rate,
            "worst_loss_rate": worst_loss_rate,
            "worst_win_rate": worst_win_rate,
            "worst_draw_rate": worst_draw_rate,
        }

        # Selection priority:
        # 1. Minimize the worst-side loss rate (avoid models that are good as X but blunder as O, or vice versa)
        # 2. Maximize the worst-side win rate
        # 3. Maximize the worst-side draw rate
        # Tie-breakers:
        # 4. Minimize overall loss rate
        # 5. Maximize overall win rate
        key = (
            -metrics["worst_loss_rate"],
            metrics["worst_win_rate"],
            metrics["worst_draw_rate"],
            -metrics["loss_rate"],
            metrics["win_rate"],
        )
        return key, metrics

    def maybe_save(self, model, num_timesteps: int):
        key, metrics = self.evaluate(model)

        improved = self.best_key is None or key > self.best_key
        if improved:
            self.best_eval_num = self.eval_num
            self.best_key = key
            self.best_metrics = metrics

            os.makedirs(os.path.dirname(self.best_model_path) or ".", exist_ok=True)
            model.save(self.best_model_path)

        if self.verbose > 0:
            tag = "BEST" if improved else "keep"
            print(
                f"[VsHeuristicEval] {tag} eval number {self.best_eval_num} @ {num_timesteps} steps | "
                f"(X: wr={metrics['x_win_rate']:.3f}, dr={metrics['x_draw_rate']:.3f}, lr={metrics['x_loss_rate']:.3f})"
                f"(O: wr={metrics['o_win_rate']:.3f}, dr={metrics['o_draw_rate']:.3f}, lr={metrics['o_loss_rate']:.3f})"
            )

        return improved, metrics

    def close(self):
        self.p1_env.close()
        self.p2_env.close()