"""Black-box acceptance tests for the project-local launch_run Pi tool.

The production path is exercised only with isolated, harmless Python fixtures.
No Gomoku trainer or repository MLflow run is started by this module.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / ".pi" / "extensions" / "launch_run.ts"
PI_ROOT = Path("/mnt/c/Users/wange/AppData/Local/pi-node/current")
LOADER = PI_ROOT / "node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js"


def _node_executable() -> str:
    executable = shutil.which("node")
    if executable:
        return executable
    bundled = PI_ROOT / "node.exe"
    if bundled.exists():
        return str(bundled)
    pytest.skip("Pi's Node runtime is unavailable")


def _node_path(path: Path) -> str:
    path = path.resolve()
    if os.name != "nt" and shutil.which("wslpath"):
        return subprocess.run(
            ["wslpath", "-w", str(path)], check=True, text=True, capture_output=True
        ).stdout.strip()
    return str(path)


def _run_node(script: str, payload: dict, *, timeout: int = 90) -> dict:
    completed = subprocess.run(
        [_node_executable(), "--input-type=module", "-e", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stderr
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise AssertionError(
            f"No JSON response. stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        ) from error


def _registration(*, auto_discover: bool) -> dict:
    """Load through Pi's real loader and return the registered public definition."""
    script = r"""
const { pathToFileURL } = await import('node:url');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const p = JSON.parse(input);
try {
  const loader = await import(pathToFileURL(p.loader).href);
  const loaded = p.autoDiscover
    ? await loader.discoverAndLoadExtensions([], p.cwd, p.agentDir)
    : await loader.loadExtensions([p.extension], p.cwd);
  if (loaded.errors.length) throw new Error(JSON.stringify(loaded.errors));
  const names = loaded.extensions.flatMap((extension) => [...extension.tools.keys()]);
  const registrations = loaded.extensions
    .map((extension) => extension.tools.get('launch_run'))
    .filter(Boolean);
  if (registrations.length !== 1) throw new Error(`registrations=${registrations.length}`);
  const tool = registrations[0].definition;
  console.log(JSON.stringify({
    ok: true,
    names,
    description: tool.description,
    promptSnippet: tool.promptSnippet,
    promptGuidelines: tool.promptGuidelines,
    parameters: tool.parameters,
  }));
} catch (error) {
  console.log(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }));
}
"""
    empty_agent_dir = ROOT / "artifacts" / "launch_run_tool" / "tmp" / "empty-agent"
    empty_agent_dir.mkdir(parents=True, exist_ok=True)
    return _run_node(
        script,
        {
            "loader": _node_path(LOADER),
            "extension": _node_path(EXTENSION),
            "cwd": _node_path(ROOT),
            "agentDir": _node_path(empty_agent_dir),
            "autoDiscover": auto_discover,
        },
    )


def _call_tool(
    repository: Path,
    arguments: dict,
    *,
    abort_after_started: Path | None = None,
    timeout: int = 90,
) -> dict:
    """Invoke the actual registered definition; the Node loader exits afterward."""
    script = r"""
const { access } = await import('node:fs/promises');
const { pathToFileURL } = await import('node:url');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const p = JSON.parse(input);
try {
  const loader = await import(pathToFileURL(p.loader).href);
  const loaded = await loader.loadExtensions([p.extension], p.cwd);
  if (loaded.errors.length) throw new Error(JSON.stringify(loaded.errors));
  const registrations = loaded.extensions
    .map((extension) => extension.tools.get('launch_run'))
    .filter(Boolean);
  if (registrations.length !== 1) throw new Error(`registrations=${registrations.length}`);
  const tool = registrations[0].definition;
  const controller = p.abortAfterStarted ? new AbortController() : undefined;
  const updates = [];
  const pending = tool.execute(
    'launch-run-acceptance',
    p.arguments,
    controller?.signal,
    (update) => updates.push(update.content?.[0]?.text ?? ''),
    { cwd: p.cwd },
  );
  if (p.abortAfterStarted) {
    const deadline = Date.now() + 30000;
    while (true) {
      try { await access(p.abortAfterStarted); break; }
      catch {
        if (Date.now() >= deadline) throw new Error('fixture never reached its started flag');
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
    }
    controller.abort();
  }
  try {
    const result = await pending;
    console.log(JSON.stringify({ ok: true, result, updates }));
  } catch (error) {
    console.log(JSON.stringify({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
      errorName: error instanceof Error ? error.name : null,
      updates,
    }));
  }
} catch (error) {
  console.log(JSON.stringify({
    ok: false,
    harnessError: error instanceof Error ? error.message : String(error),
  }));
}
"""
    return _run_node(
        script,
        {
            "loader": _node_path(LOADER),
            "extension": _node_path(EXTENSION),
            "cwd": _node_path(repository),
            "arguments": arguments,
            "abortAfterStarted": (
                _node_path(abort_after_started) if abort_after_started else None
            ),
        },
        timeout=timeout,
    )


def _fixture_repository(tmp_path: Path) -> Path:
    """Build the minimum WSL repository contract without copying the real venv."""
    if os.name == "nt":
        pytest.skip("These process-lifecycle acceptance tests run from the project's WSL venv")
    repository = tmp_path / "launch fixture repository"
    python_link = repository / ".venv" / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    # Use a tiny executable shim rather than a symlink. Python resolves a symlink
    # as this empty fixture venv and consequently cannot import MLflow; exec keeps
    # the detached PID while running the project's real, dependency-complete venv.
    python_link.write_text(
        f"#!/bin/sh\nexec {ROOT / '.venv' / 'bin' / 'python'} \"$@\"\n",
        encoding="utf-8",
    )
    python_link.chmod(0o755)
    return repository


def _write_fixture(repository: Path, name: str, body: str) -> Path:
    script = repository / name
    script.write_text(body, encoding="utf-8")
    return script


def _proc_environment(pid: int) -> dict[str, str] | None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    environment: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            environment[key.decode(errors="replace")] = value.decode(errors="replace")
    return environment


def _fixture_processes(run_name: str) -> list[int]:
    encoded = base64.b64encode(run_name.encode()).decode()
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        environment = _proc_environment(int(entry.name))
        if environment and environment.get("PI_LAUNCH_RUN_NAME_B64") == encoded:
            result.append(int(entry.name))
    return sorted(result)


def _pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_until(predicate, *, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _stop_group(pgid: int) -> None:
    """Best-effort fixture cleanup that never targets an unverified production PID."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if not _group_alive(pgid):
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        if _wait_until(lambda: not _group_alive(pgid), timeout=3):
            return
    assert not _group_alive(pgid), f"fixture PGID {pgid} survived cleanup"


@pytest.fixture(autouse=True)
def _no_fixture_leaks():
    """Each test must remove every harmless process identity it creates."""
    yield
    leftovers: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        environment = _proc_environment(int(entry.name))
        if environment and environment.get("PI_LAUNCH_RUN_MARKER", "").startswith(
            "pi-launch-run-v1:"
        ):
            cmdline = Path(f"/proc/{entry.name}/cmdline").read_bytes().replace(b"\0", b" ")
            if b"launch fixture repository" in cmdline:
                leftovers.append(int(entry.name))
    for pid in leftovers:
        try:
            _stop_group(os.getpgid(pid))
        except ProcessLookupError:
            pass
    assert not [pid for pid in leftovers if _pid_alive(pid)]


def test_real_loader_registration_and_low_power_schema():
    # Real auto-discovery must expose one obvious two-argument launch path and
    # keep every safety-tuning field optional with conservative defaults.
    assert LOADER.exists(), "Pi extension loader is unavailable"
    response = _registration(auto_discover=True)
    assert response["ok"], response
    assert response["names"].count("launch_run") == 1
    schema = response["parameters"]
    assert set(schema["properties"]) == {
        "scriptPath",
        "runName",
        "requiredFreeVramMiB",
        "startupTimeoutSeconds",
    }
    assert set(schema["required"]) == {"scriptPath", "runName"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["requiredFreeVramMiB"]["minimum"] == 1
    assert schema["properties"]["startupTimeoutSeconds"]["maximum"] == 3600
    guidance = " ".join(response["promptGuidelines"])
    assert "launch_run instead" in guidance
    assert "exact" in guidance and "runName" in guidance
    assert "duplicate" in response["description"].lower()
    assert "VRAM" in response["description"]
    assert response["promptSnippet"]


def test_two_argument_launch_returns_pid_log_progress_and_survives_loader_exit(tmp_path: Path):
    # A low-power-shaped call supplies only @scriptPath and runName. The returned
    # session leader must remain alive after the invoking Node loader has exited.
    repository = _fixture_repository(tmp_path)
    script = _write_fixture(
        repository,
        "survivor.py",
        """import signal, time
stop = False
def finish(*_args):
    global stop
    stop = True
signal.signal(signal.SIGINT, finish)
signal.signal(signal.SIGTERM, finish)
print('| iterations | 1 |', flush=True)
while not stop:
    time.sleep(0.05)
""",
    )
    run_name = "acceptance-survival"
    response = _call_tool(repository, {"scriptPath": f"@{script.name}", "runName": run_name})
    assert response["ok"], response
    result = response["result"]
    details = result["details"]
    pid = details["pid"]
    try:
        assert details["outcome"] == "started"
        assert pid == details["pgid"] and pid > 1
        assert str(pid) in result["content"][0]["text"]
        assert details["startupEvidence"]["type"] == "log_progress"
        assert "iterations" in details["startupEvidence"]["value"]
        assert details["gpu"]["freeMiB"] >= details["requiredFreeVramMiB"] == 4096
        assert details["launchArgv"][1] == "-u"
        assert "nohup setsid" in details["launchCommand"]
        log_path = repository / "logs" / f"{run_name}.log"
        record_path = repository / "logs" / f"{run_name}.pid.json"
        assert log_path.read_text(encoding="utf-8").find("iterations") >= 0
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["pid"] == pid and record["startToken"] == details["startToken"]
        # _call_tool returned only after Node exited, so this delay demonstrates
        # that neither the loader nor its terminal owns the detached WSL group.
        time.sleep(0.4)
        assert _pid_alive(pid) and _group_alive(pid)
        assert _fixture_processes(run_name) == [pid]
    finally:
        _stop_group(pid)


def test_low_vram_fails_before_spawn_with_observed_reason(tmp_path: Path):
    # An impossible requirement must report observed and required MiB and must
    # not execute even the first line of the fixture.
    repository = _fixture_repository(tmp_path)
    started = repository / "should-not-start"
    script = _write_fixture(
        repository,
        "vram.py",
        f"from pathlib import Path\nPath({str(started)!r}).write_text('spawned')\n",
    )
    run_name = "acceptance-vram"
    response = _call_tool(
        repository,
        {
            "scriptPath": script.name,
            "runName": run_name,
            "requiredFreeVramMiB": 999_999,
        },
    )
    assert not response["ok"], response
    assert "Insufficient free VRAM" in response["error"]
    assert "required 999999 MiB" in response["error"]
    assert "observed GPU" in response["error"]
    assert "No Python process was spawned" in response["error"]
    assert not started.exists()
    assert not (repository / "logs" / f"{run_name}.pid.json").exists()
    assert _fixture_processes(run_name) == []


def test_crash_timeout_and_cancellation_clean_only_launched_process(tmp_path: Path):
    # Early exit, missing progression, and post-spawn AbortSignal cancellation
    # must all throw with log diagnostics and leave no PID record or process.
    repository = _fixture_repository(tmp_path)
    crash = _write_fixture(
        repository,
        "crash.py",
        "print('fixture fatal crash', flush=True)\nraise SystemExit(7)\n",
    )
    crashed_name = "acceptance-crash"
    crashed = _call_tool(
        repository,
        {"scriptPath": crash.name, "runName": crashed_name, "startupTimeoutSeconds": 3},
    )
    assert not crashed["ok"], crashed
    assert "fixture fatal crash" in crashed["error"]
    assert "Log:" in crashed["error"] and "Log tail:" in crashed["error"]
    assert _fixture_processes(crashed_name) == []
    assert not (repository / "logs" / f"{crashed_name}.pid.json").exists()

    quiet_body = """import signal, time
from pathlib import Path
Path('START_FILE').write_text('started')
stop = False
def finish(*_args):
    global stop
    stop = True
signal.signal(signal.SIGINT, finish)
signal.signal(signal.SIGTERM, finish)
print('generic banner only', flush=True)
while not stop:
    time.sleep(0.05)
"""
    timeout_flag = repository / "timeout-started"
    timeout_script = _write_fixture(
        repository,
        "timeout.py",
        quiet_body.replace("START_FILE", timeout_flag.name),
    )
    timeout_name = "acceptance-timeout"
    timed_out = _call_tool(
        repository,
        {
            "scriptPath": timeout_script.name,
            "runName": timeout_name,
            "startupTimeoutSeconds": 1,
        },
    )
    assert timeout_flag.exists()
    assert not timed_out["ok"], timed_out
    assert "Startup timed out" in timed_out["error"]
    assert "cleanup signals=SIGINT" in timed_out["error"]
    assert _fixture_processes(timeout_name) == []
    assert not (repository / "logs" / f"{timeout_name}.pid.json").exists()

    cancel_flag = repository / "cancel-started"
    cancel_script = _write_fixture(
        repository,
        "cancel.py",
        quiet_body.replace("START_FILE", cancel_flag.name),
    )
    cancel_name = "acceptance-cancel"
    cancelled = _call_tool(
        repository,
        {
            "scriptPath": cancel_script.name,
            "runName": cancel_name,
            "startupTimeoutSeconds": 30,
        },
        abort_after_started=cancel_flag,
    )
    assert cancel_flag.exists()
    assert not cancelled["ok"], cancelled
    assert cancelled["errorName"] == "AbortError"
    assert "cancelled" in cancelled["error"].lower()
    assert "cleanup signals=SIGINT" in cancelled["error"]
    assert _fixture_processes(cancel_name) == []
    assert not (repository / "logs" / f"{cancel_name}.pid.json").exists()


def test_duplicate_replacement_proves_old_group_dead_and_one_replacement(tmp_path: Path):
    # Two real calls with a case-insensitively equal identity must serialize,
    # stop the first verified group, and leave exactly one replacement process.
    repository = _fixture_repository(tmp_path)
    script = _write_fixture(
        repository,
        "duplicate.py",
        """import signal, time
stop = False
def finish(*_args):
    global stop
    stop = True
signal.signal(signal.SIGINT, finish)
signal.signal(signal.SIGTERM, finish)
print('total_timesteps: 1', flush=True)
while not stop:
    time.sleep(0.05)
""",
    )
    first_name = "Acceptance-Duplicate"
    second_name = "acceptance-duplicate"
    first = _call_tool(repository, {"scriptPath": script.name, "runName": first_name})
    assert first["ok"], first
    old_pid = first["result"]["details"]["pid"]
    replacement_pid: int | None = None
    try:
        second = _call_tool(repository, {"scriptPath": script.name, "runName": second_name})
        assert second["ok"], second
        details = second["result"]["details"]
        replacement_pid = details["pid"]
        assert replacement_pid != old_pid
        assert len(details["duplicates"]) == 1
        duplicate = details["duplicates"][0]
        assert duplicate["pid"] == old_pid
        assert duplicate["signals"][0] == "SIGINT"
        assert duplicate["leaderIdentityGone"] is True
        assert duplicate["processGroupGone"] is True
        assert duplicate["markerGone"] is True
        assert _wait_until(lambda: not _pid_alive(old_pid))
        assert not _group_alive(old_pid)
        assert _pid_alive(replacement_pid) and _group_alive(replacement_pid)
        # The marker embeds a lower-cased run identity, so one exact fixture
        # process under either name is the only acceptable postcondition.
        assert sorted(set(_fixture_processes(first_name) + _fixture_processes(second_name))) == [
            replacement_pid
        ]
    finally:
        if replacement_pid is not None:
            _stop_group(replacement_pid)
        elif _group_alive(old_pid):
            _stop_group(old_pid)


def test_injection_validation_and_stale_pid_record_do_not_signal_unrelated_process(tmp_path: Path):
    # Shell metacharacters in a valid root filename must remain argv data; unsafe
    # names/traversal must fail, and a stale PID record must not kill a live PID.
    repository = _fixture_repository(tmp_path)
    injected = repository / "injected_marker.py"
    script = _write_fixture(
        repository,
        "fixture;touch injected_marker.py",
        """import signal, time
stop = False
def finish(*_args):
    global stop
    stop = True
signal.signal(signal.SIGINT, finish)
signal.signal(signal.SIGTERM, finish)
print('| total_timesteps | 2 |', flush=True)
while not stop:
    time.sleep(0.05)
""",
    )
    sentinel = subprocess.Popen(["sleep", "60"], start_new_session=True)
    run_name = "acceptance-stale-record"
    logs = repository / "logs"
    logs.mkdir()
    stale_record = {
        "version": 1,
        "pid": sentinel.pid,
        "pgid": sentinel.pid,
        "startToken": "definitely-not-the-live-start-token",
        "marker": "pi-launch-run-v1:forged:stale",
    }
    (logs / f"{run_name}.pid.json").write_text(json.dumps(stale_record), encoding="utf-8")
    launched_pid: int | None = None
    try:
        unsafe_name = _call_tool(
            repository,
            {"scriptPath": script.name, "runName": "unsafe;touch-pwned"},
        )
        assert not unsafe_name["ok"] and "filesystem-safe" in unsafe_name["error"]
        traversal = _call_tool(
            repository,
            {"scriptPath": "../escape.py", "runName": "safe-name"},
        )
        assert not traversal["ok"] and "root-level" in traversal["error"]

        launched = _call_tool(
            repository,
            {"scriptPath": f"@{script.name}", "runName": run_name},
        )
        assert launched["ok"], launched
        launched_pid = launched["result"]["details"]["pid"]
        assert sentinel.poll() is None, "stale/PID-reused metadata killed an unrelated process"
        assert not injected.exists(), "scriptPath metacharacters were interpreted by a shell"
        returned_script = launched["result"]["details"]["scriptPath"].replace("\\", "/")
        assert returned_script.rsplit("/", 1)[-1] == script.name
        assert _fixture_processes(run_name) == [launched_pid]
    finally:
        if launched_pid is not None:
            _stop_group(launched_pid)
        if sentinel.poll() is None:
            os.killpg(sentinel.pid, signal.SIGKILL)
        sentinel.wait(timeout=5)


def test_failed_mlflow_fixture_is_reconciled_without_active_orphan(tmp_path: Path):
    # A fixture-created MLflow run intentionally remains RUNNING when its owner
    # exits on timeout. launch_run must reconcile it through MlflowClient and
    # leave neither an active run nor a marker-bearing process.
    repository = _fixture_repository(tmp_path)
    script = _write_fixture(
        repository,
        "mlflow_orphan.py",
        """import base64, os, signal, time
from pathlib import Path
from mlflow.tracking import MlflowClient
run_name = base64.b64decode(os.environ['PI_LAUNCH_RUN_NAME_B64']).decode()
root = Path.cwd() / 'mlruns'
client = MlflowClient(tracking_uri=root.resolve().as_uri())
experiment = client.get_experiment_by_name('launch-run-fixture')
if experiment is None:
    experiment_id = client.create_experiment('launch-run-fixture')
else:
    experiment_id = experiment.experiment_id
run = client.create_run(experiment_id, tags={'mlflow.runName': run_name})
Path('mlflow-created').write_text(run.info.run_id)
def finish(*_args):
    raise SystemExit(0)
signal.signal(signal.SIGINT, finish)
signal.signal(signal.SIGTERM, finish)
print('MLflow run created, but this is not progression', flush=True)
while True:
    time.sleep(0.05)
""",
    )
    run_name = "acceptance-mlflow-cleanup"
    response = _call_tool(
        repository,
        {
            "scriptPath": script.name,
            "runName": run_name,
            "startupTimeoutSeconds": 8,
        },
        timeout=120,
    )
    assert not response["ok"], response
    assert "Startup timed out" in response["error"]
    assert "reconciled MLflow run" in response["error"]
    run_id = (repository / "mlflow-created").read_text(encoding="utf-8")
    meta = next((repository / "mlruns").glob(f"*/{run_id}/meta.yaml")).read_text(
        encoding="utf-8"
    )
    status_line = next(line for line in meta.splitlines() if line.startswith("status:"))
    status = status_line.split(":", 1)[1].strip().upper()
    assert status not in {"1", "2", "RUNNING", "SCHEDULED"}
    assert _fixture_processes(run_name) == []
    assert not (repository / "logs" / f"{run_name}.pid.json").exists()
