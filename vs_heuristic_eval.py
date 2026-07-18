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
        heuristic = None,
        heuristics = None,
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

        # Accept either a single heuristic or a list; the model is evaluated
        # against every heuristic and selected on the worst-case outcome.
        if heuristics is not None:
            self.heuristics = list(heuristics)
        elif heuristic is not None:
            self.heuristics = [heuristic]
        else:
            raise ValueError("Provide either `heuristic` or `heuristics`.")
        if len(self.heuristics) == 0:
            raise ValueError("`heuristics` must contain at least one policy.")

        self.heuristic_names = self._build_heuristic_names(self.heuristics)

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

        self.p1_env.set_opponent(self.heuristics[0])
        self.p2_env.set_opponent(self.heuristics[0])

    @staticmethod
    def _build_heuristic_names(heuristics):
        names = []
        seen = {}
        for h in heuristics:
            base = type(h).__name__
            if base in seen:
                seen[base] += 1
                names.append(f"{base}_{seen[base]}")
            else:
                seen[base] = 0
                names.append(base)
        return names

    def _make_env(self):
        return SingleAgentSelfPlayEnv(
            height=self.height,
            width=self.width,
            win_con=self.win_con,
            p1_symbol="X",
            p2_symbol="O",
            render_mode=None,
            opponent_policy=self.heuristics[0],
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

        def outcome_from_reward(r: float):
            if r > 0.5:
                return "win"
            if r < -0.5:
                return "loss"
            return "draw"

        metrics = {}
        # (win_rate, draw_rate, loss_rate) for every (heuristic, side) pairing.
        side_rates = []
        agg = {"win": 0, "draw": 0, "loss": 0}

        for h, name in zip(self.heuristics, self.heuristic_names):
            self.p1_env.set_opponent(h)
            self.p2_env.set_opponent(h)

            results = {
                "X": {"win": 0, "draw": 0, "loss": 0},
                "O": {"win": 0, "draw": 0, "loss": 0},
            }

            for _ in range(self.n_games_per_side):
                r = self._play_one(model, learner_symbol="X", rng=rng)
                results["X"][outcome_from_reward(r)] += 1

            for _ in range(self.n_games_per_side):
                r = self._play_one(model, learner_symbol="O", rng=rng)
                results["O"][outcome_from_reward(r)] += 1

            for side in ("X", "O"):
                win_rate = results[side]["win"] / self.n_games_per_side
                draw_rate = results[side]["draw"] / self.n_games_per_side
                loss_rate = results[side]["loss"] / self.n_games_per_side
                metrics[f"{name}/{side.lower()}_win_rate"] = win_rate
                metrics[f"{name}/{side.lower()}_draw_rate"] = draw_rate
                metrics[f"{name}/{side.lower()}_loss_rate"] = loss_rate
                side_rates.append((win_rate, draw_rate, loss_rate))

                for outcome in ("win", "draw", "loss"):
                    agg[outcome] += results[side][outcome]

        total_games = 2 * self.n_games_per_side * len(self.heuristics)

        # Worst-case across every (heuristic, side) pairing so a saved model must
        # be robust against all opponents, not just the easiest one.
        worst_loss_rate = max(lr for _, _, lr in side_rates)
        worst_win_rate = min(wr for wr, _, _ in side_rates)
        worst_draw_rate = min(dr for _, dr, _ in side_rates)

        metrics["win_rate"] = agg["win"] / total_games
        metrics["draw_rate"] = agg["draw"] / total_games
        metrics["loss_rate"] = agg["loss"] / total_games
        metrics["worst_loss_rate"] = worst_loss_rate
        metrics["worst_win_rate"] = worst_win_rate
        metrics["worst_draw_rate"] = worst_draw_rate

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
                f"overall (wr={metrics['win_rate']:.3f}, dr={metrics['draw_rate']:.3f}, lr={metrics['loss_rate']:.3f}) "
                f"worst-case (wr={metrics['worst_win_rate']:.3f}, dr={metrics['worst_draw_rate']:.3f}, lr={metrics['worst_loss_rate']:.3f})"
            )

        return improved, metrics

    def close(self):
        self.p1_env.close()
        self.p2_env.close()