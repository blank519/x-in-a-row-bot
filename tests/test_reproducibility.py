"""Acceptance tests for deterministic training and reproducibility provenance."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import random
from pathlib import Path

import gymnasium as gym
import mlflow
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from mlflow.tracking import MlflowClient
from stable_baselines3.common.vec_env import DummyVecEnv

import reproducibility
from finetune_ppo_persistent_pool import load_snapshot_pool
from reproducibility import (
    configure_reproducibility,
    policy_state_sha256,
    sha256_file,
)
from self_play_gomoku import BoardCnnExtractor as GomokuExtractor
from self_play_gomoku import SelfPlaySnapshotCallback
from train_ppo_tic_tac_toe import BoardCnnExtractor as TicTacToeExtractor
from train_ppo_tic_tac_toe import SelfPlaySnapshotCallback as TicTacToeCallback
from vs_heuristic_eval import HeuristicEvaluator
from x_in_a_row_sb3_env import CurriculumMaskedSelfPlayEnv


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METADATA_PARAMS = {
    "seed",
    "eval_seed",
    "device",
    "deterministic_algorithms",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "allow_tf32",
    "cublas_workspace_config",
    "python_version",
    "numpy_version",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "stable_baselines3_version",
    "sb3_contrib_version",
    "gymnasium_version",
    "git_commit",
    "git_dirty",
    "entrypoint",
    "entrypoint_sha256",
    "input_artifact_count",
    "reproducibility_manifest_sha256",
}
REQUIRED_SOURCE_NAMES = [
    "train_ppo_gomoku.py",
    ".python-version",
    "requirements.txt",
    "reproducibility.py",
    "game_utils.py",
    "heuristic_policy.py",
    "self_play_gomoku.py",
    "vs_heuristic_eval.py",
    "x_in_a_row_env.py",
    "x_in_a_row_sb3_env.py",
]


class _NoSampleSpace(spaces.Box):
    """Observation space whose sample method proves extractors are pure."""

    def sample(self, mask=None):  # pragma: no cover - must never be reached
        raise AssertionError("extractor construction must not sample the space")


def _assert_equal_state_dicts(left: th.nn.Module, right: th.nn.Module) -> None:
    assert left.state_dict().keys() == right.state_dict().keys()
    for key, value in left.state_dict().items():
        assert th.equal(value, right.state_dict()[key]), key


def test_configure_reproducibility_seeds_all_rngs_and_enables_strict_mode(monkeypatch):
    """The shared setup must replay CPU RNG draws and reject silent backend drift."""
    configure_reproducibility(123)
    first = (random.random(), np.random.random(), th.rand(4))
    configure_reproducibility(123)
    second = (random.random(), np.random.random(), th.rand(4))

    assert first[:2] == second[:2]
    assert th.equal(first[2], second[2])
    assert th.are_deterministic_algorithms_enabled()
    assert th.backends.cudnn.deterministic is True
    assert th.backends.cudnn.benchmark is False
    assert th.backends.cuda.matmul.allow_tf32 is False
    assert th.backends.cudnn.allow_tf32 is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    configure_reproducibility(124)
    assert not th.equal(first[2], th.rand(4))
    for invalid in (-1, True, 1.5, "1", None):
        with pytest.raises(ValueError):
            configure_reproducibility(invalid)  # type: ignore[arg-type]

    monkeypatch.setattr(th.cuda, "is_initialized", lambda: True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="already initialized"):
        configure_reproducibility(1)


def test_feature_extractors_do_not_sample_and_reseed_exactly():
    """Shape discovery must neither consume a space RNG nor mutate BatchNorm state."""
    gomoku_space = _NoSampleSpace(0, 1, shape=(2, 15, 15), dtype=np.float32)
    ttt_space = _NoSampleSpace(0, 1, shape=(2, 3, 3), dtype=np.float32)

    configure_reproducibility(7)
    gomoku_a = GomokuExtractor(gomoku_space)
    configure_reproducibility(7)
    gomoku_b = GomokuExtractor(gomoku_space)
    _assert_equal_state_dicts(gomoku_a, gomoku_b)

    for module in gomoku_a.cnn.modules():
        if isinstance(module, th.nn.BatchNorm2d):
            assert th.count_nonzero(module.running_mean) == 0
            assert th.equal(module.running_var, th.ones_like(module.running_var))

    configure_reproducibility(9)
    ttt_a = TicTacToeExtractor(ttt_space)
    configure_reproducibility(9)
    ttt_b = TicTacToeExtractor(ttt_space)
    _assert_equal_state_dicts(ttt_a, ttt_b)


def test_policy_fingerprint_is_semantic_and_seed_sensitive():
    """Equal policy tensors hash equally; a changed seed or tensor changes the digest."""
    class Model:
        def __init__(self, seed: int):
            th.manual_seed(seed)
            self.policy = th.nn.Sequential(th.nn.Linear(3, 4), th.nn.Linear(4, 2))

    same_a, same_b, changed = Model(11), Model(11), Model(12)
    digest = policy_state_sha256(same_a)
    assert digest == policy_state_sha256(same_b)
    assert digest != policy_state_sha256(changed)
    assert len(digest) == 64 and digest == digest.lower()

    with th.no_grad():
        next(same_a.policy.parameters()).add_(1)
    assert digest != policy_state_sha256(same_a)


def test_vector_env_seed_sequence_and_curriculum_rollout_are_deterministic():
    """Vec-env seed+i ownership and stochastic curriculum/mask behavior must replay."""
    vec = DummyVecEnv(
        [
            lambda: CurriculumMaskedSelfPlayEnv(
                height=9,
                width=9,
                win_con=5,
                randomize_learner=True,
                defensive_opening_prob=1.0,
            )
            for _ in range(3)
        ]
    )
    assert vec.seed(41) == [41, 42, 43]

    def rollout(seed: int):
        env = CurriculumMaskedSelfPlayEnv(
            height=9,
            width=9,
            win_con=5,
            randomize_learner=True,
            defensive_opening_prob=1.0,
        )
        obs, _ = env.reset(seed=seed)
        trace = [(obs.tobytes(), env.action_masks().tobytes(), env.learner_symbol)]
        for _ in range(6):
            legal = np.flatnonzero(env.action_masks())
            obs, reward, terminated, truncated, _ = env.step(int(legal[0]))
            next_mask = b"" if terminated or truncated else env.action_masks().tobytes()
            trace.append(
                (
                    obs.tobytes(),
                    next_mask,
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                )
            )
            if terminated or truncated:
                break
        return trace

    assert rollout(91) == rollout(91)
    assert rollout(91) != rollout(92)


def test_callbacks_forward_the_run_seed_to_evaluation():
    """Both independent callback stacks expose and store the separately logged eval seed."""
    for callback_type in (SelfPlaySnapshotCallback, TicTacToeCallback):
        signature = inspect.signature(callback_type.__init__)
        names = list(signature.parameters)
        assert names.index("eval_seed") == names.index("eval_games_per_side") + 1
        assert signature.parameters["eval_seed"].default == 0

    class FirstLegalModel:
        def predict(self, _obs, action_masks, deterministic=True):
            assert deterministic
            return int(np.flatnonzero(action_masks)[0]), None

    # Two fresh evaluators with the same seed must produce the same ordered metrics.
    from heuristic_policy import XInARowHeuristicPolicy

    kwargs = dict(
        height=3,
        width=3,
        win_con=3,
        heuristic=XInARowHeuristicPolicy(3, 3, 3),
        n_games_per_side=3,
        seed=73,
        verbose=0,
    )
    _, metrics_a = HeuristicEvaluator(**kwargs).evaluate(FirstLegalModel())
    _, metrics_b = HeuristicEvaluator(**kwargs).evaluate(FirstLegalModel())
    assert list(metrics_a.items()) == list(metrics_b.items())


def test_mlflow_manifest_contains_exact_sources_ordered_inputs_and_hashes(tmp_path):
    """A run must retain complete source/input bytes and a verifiable deterministic manifest."""
    configure_reproducibility(42)
    inputs = [tmp_path / "same.zip", tmp_path / "other" / "same.zip"]
    inputs[0].write_bytes(b"first-input")
    inputs[1].parent.mkdir()
    inputs[1].write_bytes(b"second-input")

    tracking_uri = (tmp_path / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment("repro-acceptance")
    with mlflow.start_run() as active:
        reproducibility.log_reproducibility_metadata(
            seed=42,
            eval_seed=99,
            device="cpu",
            entrypoint_path=str(ROOT / "train_ppo_gomoku.py"),
            input_artifacts=inputs,
        )
        run_id = active.info.run_id

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert REQUIRED_METADATA_PARAMS <= set(run.data.params)
    assert run.data.params["input_artifact_count"] == "2"
    assert run.data.params["entrypoint_sha256"] == sha256_file(ROOT / "train_ppo_gomoku.py")

    download_dir = tmp_path / "download"
    download_dir.mkdir()
    manifest_path = Path(
        client.download_artifacts(run_id, "reproducibility/manifest.json", str(download_dir))
    )
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert hashlib.sha256(manifest_bytes).hexdigest() == run.data.params[
        "reproducibility_manifest_sha256"
    ]
    assert [Path(item["path"]).name for item in manifest["source"]] == REQUIRED_SOURCE_NAMES
    assert [item["path"] for item in manifest["inputs"]] == [str(path.resolve()) for path in inputs]

    for index, (item, original) in enumerate(zip(manifest["inputs"], inputs)):
        assert item["sha256"] == sha256_file(original)
        assert item["size_bytes"] == original.stat().st_size
        artifact = (
            f"reproducibility/inputs/{index:04d}-{item['sha256'][:12]}/{original.name}"
        )
        copy_dir = tmp_path / f"copy-{index}"
        copy_dir.mkdir()
        copied = Path(client.download_artifacts(run_id, artifact, str(copy_dir)))
        assert copied.read_bytes() == original.read_bytes()

    source_artifacts = {
        item.path for item in client.list_artifacts(run_id, "reproducibility/source")
    }
    assert source_artifacts == {
        f"reproducibility/source/{name}" for name in REQUIRED_SOURCE_NAMES
    }
    top_level = {
        item.path for item in client.list_artifacts(run_id, "reproducibility")
    }
    assert {
        "reproducibility/manifest.json",
        "reproducibility/pip_freeze.txt",
        "reproducibility/git_diff.patch",
    } <= top_level


def test_missing_input_fails_before_metadata_is_logged(tmp_path):
    """A nonexistent continuation input must reject the run without a partial metadata record."""
    tracking_uri = (tmp_path / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("missing-input")
    with mlflow.start_run() as active:
        with pytest.raises(FileNotFoundError):
            reproducibility.log_reproducibility_metadata(
                seed=1,
                eval_seed=2,
                device="cpu",
                entrypoint_path=str(ROOT / "train_ppo_gomoku.py"),
                input_artifacts=(tmp_path / "absent.zip",),
            )
        run_id = active.info.run_id
    assert MlflowClient(tracking_uri=tracking_uri).get_run(run_id).data.params == {}


def test_persistent_pool_selection_is_numeric_ordered_and_device_bound(tmp_path, monkeypatch):
    """Only numeric last-k immutable snapshots may be loaded, exactly once and in manifest order."""
    for name in (
        "opponent_snapshot_10.zip",
        "opponent_snapshot_2.zip",
        "opponent_snapshot_7.zip",
        "opponent_snapshot_bad.zip",
        "unrelated.zip",
    ):
        (tmp_path / name).write_bytes(name.encode())

    calls = []

    def fake_load(path, device):
        calls.append((path, device))
        return Path(path).name

    monkeypatch.setattr("finetune_ppo_persistent_pool.MaskablePPO.load", fake_load)
    models, max_index, paths = load_snapshot_pool(str(tmp_path), 2, "cpu")
    expected = [
        str(tmp_path / "opponent_snapshot_7.zip"),
        str(tmp_path / "opponent_snapshot_10.zip"),
    ]
    assert paths == expected
    assert models == [Path(path).name for path in expected]
    assert calls == [(path, "cpu") for path in expected]
    assert max_index == 10


def test_provenance_is_the_first_operation_inside_each_training_run():
    """Invalid provenance must abort before any entry point creates a partial MLflow record."""
    entrypoints = (
        "train_ppo_gomoku.py",
        "train_ppo_tic_tac_toe.py",
        "finetune_ppo.py",
        "finetune_ppo_persistent_pool.py",
    )
    for filename in entrypoints:
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"), filename=filename)
        run_with = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "start_run"
                for item in node.items
            )
        )
        first = run_with.body[0]
        assert (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Call)
            and getattr(first.value.func, "id", None) == "log_reproducibility_metadata"
        ), f"{filename} logs run data before validating required reproducibility provenance"
