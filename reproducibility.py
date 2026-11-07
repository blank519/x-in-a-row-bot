"""Strict seeding, MLflow provenance, and policy fingerprinting helpers.

This module is the single owner of the deterministic-run configuration shared
by the training and fine-tuning entry points:

- ``configure_reproducibility`` seeds every RNG stack and enables strict
  deterministic Torch algorithms before any CUDA/context use.
- ``log_reproducibility_metadata`` records the exact source, runtime, git
  state, and immutable continuation inputs of a run into the active MLflow run.
- ``policy_state_sha256`` produces a deterministic fingerprint of a learned
  policy's tensors (not its ``.zip`` container, whose timestamps are not a
  semantic result).
- ``sha256_file`` streams a file through SHA-256.

Importing this module does not execute any of the above; entry points call the
functions explicitly.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile

from collections.abc import Sequence

import numpy as np
import torch as th

import mlflow

# Shared library/config sources recorded with every run's provenance. The
# entry-point file itself is logged first, followed by this list in order.
_SHARED_SOURCE_FILENAMES = (
    ".python-version",
    "requirements.txt",
    "reproducibility.py",
    "game_utils.py",
    "heuristic_policy.py",
    "self_play_gomoku.py",
    "vs_heuristic_eval.py",
    "x_in_a_row_env.py",
    "x_in_a_row_sb3_env.py",
)

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

_UNAVAILABLE = "none"


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the lowercase SHA-256 hex digest of the file's bytes."""
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_size(path: str | os.PathLike[str]) -> int:
    return os.path.getsize(os.fspath(path))


def _manifest_entry(path: str) -> dict:
    return {
        "path": path,
        "sha256": sha256_file(path),
        "size_bytes": _file_size(path),
    }


def _run_git(args: list[str], cwd: str) -> bytes:
    """Run a git command in ``cwd`` and return stdout; fail loudly on error."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed with exit code {result.returncode}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def configure_reproducibility(seed: int) -> None:
    """Configure every RNG and Torch backend for strict determinism.

    Must be called before environment creation or any CUDA query. Raises
    ``ValueError`` for a non-integer or negative seed and ``RuntimeError`` when
    CUDA is already initialized with a conflicting cuBLAS workspace config.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    current_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if th.cuda.is_initialized() and current_config != _CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUDA is already initialized with "
            f"CUBLAS_WORKSPACE_CONFIG={current_config!r}; deterministic "
            f"execution requires {_CUBLAS_WORKSPACE_CONFIG!r} to be set "
            "before CUDA initialization"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    th.use_deterministic_algorithms(True, warn_only=False)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.backends.cuda.matmul.allow_tf32 = False
    th.backends.cudnn.allow_tf32 = False


def log_reproducibility_metadata(
    *,
    seed: int,
    eval_seed: int,
    device: str,
    entrypoint_path: str,
    input_artifacts: Sequence[str | os.PathLike[str]] = (),
) -> None:
    """Log the full reproducibility provenance of the active MLflow run.

    Requires an active MLflow run and existing entry-point/input files; any
    missing file or failed provenance command raises before a partial record
    is emitted.
    """
    if mlflow.active_run() is None:
        raise RuntimeError(
            "log_reproducibility_metadata requires an active MLflow run; call "
            "mlflow.start_run(...) first"
        )

    entrypoint_path = os.path.abspath(os.fspath(entrypoint_path))
    if not os.path.isfile(entrypoint_path):
        raise FileNotFoundError(f"entry-point file not found: {entrypoint_path}")

    input_paths = [os.path.abspath(os.fspath(p)) for p in input_artifacts]
    for path in input_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"input artifact not found: {path}")

    repo_root = os.path.dirname(os.path.abspath(__file__))

    # Ordered source list: the entry point first, then the shared files.
    source_paths = [entrypoint_path]
    for name in _SHARED_SOURCE_FILENAMES:
        path = os.path.join(repo_root, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"shared source file not found: {path}")
        source_paths.append(path)

    git_commit = _run_git(["rev-parse", "HEAD"], repo_root).decode("utf-8").strip()
    git_status = _run_git(["status", "--porcelain"], repo_root).decode("utf-8")
    git_diff = _run_git(["diff", "--binary", "HEAD"], repo_root)
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        check=False,
    )
    if pip_freeze.returncode != 0:
        raise RuntimeError(
            f"pip freeze failed with exit code {pip_freeze.returncode}: "
            f"{pip_freeze.stderr.decode('utf-8', errors='replace').strip()}"
        )

    cuda_version = th.version.cuda if th.cuda.is_available() else None
    cudnn_version = None
    if th.backends.cudnn.is_available():
        cudnn_version = th.backends.cudnn.version()

    manifest = {
        "source": [_manifest_entry(p) for p in source_paths],
        "inputs": [_manifest_entry(p) for p in input_paths],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    mlflow.log_params(
        {
            "seed": seed,
            "eval_seed": eval_seed,
            "device": device,
            "deterministic_algorithms": th.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": th.backends.cudnn.deterministic,
            "cudnn_benchmark": th.backends.cudnn.benchmark,
            "allow_tf32": (
                th.backends.cuda.matmul.allow_tf32 and th.backends.cudnn.allow_tf32
            ),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": th.__version__,
            "cuda_version": cuda_version if cuda_version else _UNAVAILABLE,
            "cudnn_version": cudnn_version if cudnn_version else _UNAVAILABLE,
            "stable_baselines3_version": importlib.metadata.version("stable-baselines3"),
            "sb3_contrib_version": importlib.metadata.version("sb3-contrib"),
            "gymnasium_version": importlib.metadata.version("gymnasium"),
            "git_commit": git_commit,
            "git_dirty": bool(git_status.strip()),
            "entrypoint": entrypoint_path,
            "entrypoint_sha256": sha256_file(entrypoint_path),
            "input_artifact_count": len(input_paths),
            "reproducibility_manifest_sha256": manifest_sha256,
        }
    )

    tmp_dir = tempfile.mkdtemp(prefix="repro_provenance_")
    try:
        with open(os.path.join(tmp_dir, "manifest.json"), "wb") as handle:
            handle.write(manifest_bytes)
        mlflow.log_artifact(
            os.path.join(tmp_dir, "manifest.json"),
            artifact_path="reproducibility",
        )

        with open(os.path.join(tmp_dir, "pip_freeze.txt"), "wb") as handle:
            handle.write(pip_freeze.stdout)
        mlflow.log_artifact(
            os.path.join(tmp_dir, "pip_freeze.txt"),
            artifact_path="reproducibility",
        )

        with open(os.path.join(tmp_dir, "git_diff.patch"), "wb") as handle:
            handle.write(git_diff)
        mlflow.log_artifact(
            os.path.join(tmp_dir, "git_diff.patch"),
            artifact_path="reproducibility",
        )

        source_dir = os.path.join(tmp_dir, "source")
        os.makedirs(source_dir)
        for path in source_paths:
            shutil.copyfile(path, os.path.join(source_dir, os.path.basename(path)))
        mlflow.log_artifacts(source_dir, artifact_path="reproducibility/source")

        for index, path in enumerate(input_paths):
            input_sha256 = sha256_file(path)
            input_dir = os.path.join(tmp_dir, "inputs", f"{index:04d}-{input_sha256[:12]}")
            os.makedirs(input_dir)
            shutil.copyfile(path, os.path.join(input_dir, os.path.basename(path)))
            mlflow.log_artifacts(
                input_dir,
                artifact_path=f"reproducibility/inputs/{index:04d}-{input_sha256[:12]}",
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def policy_state_sha256(model: object) -> str:
    """Return a deterministic SHA-256 fingerprint of ``model.policy`` tensors.

    Iterates ``model.policy.state_dict()`` keys in sorted order and hashes each
    key, dtype, shape, and contiguous CPU tensor bytes. The ``.zip`` container
    is never hashed, since its timestamps are not a semantic policy result.
    """
    state_dict = model.policy.state_dict()

    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        cpu_tensor = tensor.detach().to("cpu").contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(cpu_tensor.dtype).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(cpu_tensor.numpy().tobytes())
    return digest.hexdigest()