"""Black-box acceptance tests for the project-local health_sweep Pi tool.

The production path is exercised only with isolated, harmless sleep processes and
OS-temporary repository trees. No trainer or real repository MLflow run is used.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterator

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / ".pi" / "extensions" / "health_sweep.ts"
PI_ROOT = Path("/mnt/c/Users/wange/AppData/Local/pi-node/current")
LOADER = PI_ROOT / "node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js"
TIMEOUT_MS = 600_000
POLL_MS = 5_000


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


def _loader() -> Path:
    if not LOADER.exists():
        pytest.skip("Pi's real extension loader is unavailable")
    return LOADER


def _run_node(script: str, payload: dict, *, timeout: int = 30) -> dict:
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


def _registration(tmp_path: Path, *, auto_discover: bool) -> dict:
    """Load through Pi's real loader and expose the registered public contract."""
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
    .map((extension) => extension.tools.get('health_sweep'))
    .filter(Boolean);
  if (registrations.length !== 1) throw new Error(`registrations=${registrations.length}`);
  const tool = registrations[0].definition;
  console.log(JSON.stringify({
    ok: true, names, name: tool.name, label: tool.label,
    description: tool.description, promptSnippet: tool.promptSnippet,
    promptGuidelines: tool.promptGuidelines, parameters: tool.parameters,
  }));
} catch (error) {
  console.log(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }));
}
"""
    agent_dir = tmp_path / "empty-agent"
    agent_dir.mkdir(parents=True)
    return _run_node(
        script,
        {
            "loader": _node_path(_loader()),
            "extension": _node_path(EXTENSION),
            "cwd": _node_path(ROOT),
            "agentDir": _node_path(agent_dir),
            "autoDiscover": auto_discover,
        },
        timeout=90,
    )


def _call_registered(
    repository: Path,
    arguments: dict,
    *,
    abort_after_ms: int | None = None,
    timeout: int = 30,
) -> dict:
    """Invoke the actual tool definition returned by Pi's loader."""
    script = r"""
const { pathToFileURL } = await import('node:url');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const p = JSON.parse(input);
try {
  const loader = await import(pathToFileURL(p.loader).href);
  const loaded = await loader.loadExtensions([p.extension], p.cwd);
  if (loaded.errors.length) throw new Error(JSON.stringify(loaded.errors));
  const registrations = loaded.extensions
    .map((extension) => extension.tools.get('health_sweep'))
    .filter(Boolean);
  if (registrations.length !== 1) throw new Error(`registrations=${registrations.length}`);
  const tool = registrations[0].definition;
  const controller = p.abortAfterMs === null ? undefined : new AbortController();
  const updates = [];
  if (controller) setTimeout(() => controller.abort(), p.abortAfterMs);
  const started = Date.now();
  try {
    const result = await tool.execute(
      'health-sweep-acceptance', p.arguments, controller?.signal,
      (update) => updates.push(update.content?.[0]?.text ?? ''), { cwd: p.cwd },
    );
    console.log(JSON.stringify({ ok: true, elapsedMs: Date.now() - started, result, updates }));
  } catch (error) {
    console.log(JSON.stringify({
      ok: false, elapsedMs: Date.now() - started,
      error: error instanceof Error ? error.message : String(error),
      errorName: error instanceof Error ? error.name : null, updates,
    }));
  }
} catch (error) {
  console.log(JSON.stringify({ ok: false, harnessError: error instanceof Error ? error.message : String(error) }));
}
"""
    return _run_node(
        script,
        {
            "loader": _node_path(_loader()),
            "extension": _node_path(EXTENSION),
            "cwd": _node_path(repository),
            "arguments": arguments,
            "abortAfterMs": abort_after_ms,
        },
        timeout=timeout,
    )


def _direct_module(tmp_path: Path, body: str, payload: dict | None = None) -> dict:
    """Import exported seams after resolving TypeBox from Pi's installed package."""
    script = r"""
const { createRequire } = await import('node:module');
const { readFile, writeFile } = await import('node:fs/promises');
const { pathToFileURL } = await import('node:url');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const p = JSON.parse(input);
const require = createRequire(p.loader);
const typebox = require.resolve('typebox');
let source = await readFile(p.extension, 'utf8');
source = source.replace('from "typebox"', `from ${JSON.stringify(pathToFileURL(typebox).href)}`);
await writeFile(p.module, source, 'utf8');
const health = await import(`${pathToFileURL(p.module).href}?v=${Date.now()}`);
try {
  const run = new Function('health', 'p', `return (async () => { ${p.body}\n })()`);
  const value = await run(health, p.payload);
  console.log(JSON.stringify({ ok: true, value }));
} catch (error) {
  console.log(JSON.stringify({
    ok: false, error: error instanceof Error ? error.message : String(error),
    errorName: error instanceof Error ? error.name : null,
  }));
}
"""
    module_path = tmp_path / "health-sweep-import.ts"
    return _run_node(
        script,
        {
            "loader": _node_path(_loader()),
            "extension": _node_path(EXTENSION),
            "module": _node_path(module_path),
            "body": body,
            "payload": payload or {},
        },
        timeout=90,
    )


def _proc_identity(pid: int) -> tuple[str, int, int]:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = text[text.rfind(")") + 2 :].split()
    return fields[19], int(fields[2]), int(fields[3])


def _write_mlflow_run(repository: Path, run_name: str, run_id: str, started_ms: int) -> None:
    run = repository / "mlruns" / "1" / run_id
    (run / "tags").mkdir(parents=True)
    (run / "meta.yaml").write_text(
        f"run_id: {run_id}\nrun_name: {run_name}\nstatus: 1\nstart_time: {started_ms}\n",
        encoding="utf-8",
    )
    (run / "tags" / "mlflow.runName").write_text(run_name, encoding="utf-8")


def _write_launch_evidence(
    repository: Path,
    *,
    run_name: str,
    run_id: str,
    pid: int,
    marker: str,
    start_token: str,
    pgid: int,
    launched_ms: int,
    include_progress: bool = True,
) -> None:
    logs = repository / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    launched_at = datetime.fromtimestamp(launched_ms / 1000, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    run_path = repository / "mlruns" / "1" / run_id
    (logs / f"{run_name}.pid.json").write_text(
        json.dumps(
            {
                "version": 1,
                "runName": run_name,
                "pid": pid,
                "pgid": pgid,
                "marker": marker,
                "startToken": start_token,
                "launchedAtMs": launched_ms,
                "launchedAt": launched_at,
                "mlflowRunId": run_id,
                # launch_run is a Windows-hosted Pi tool and records this namespace.
                "mlflowRunPath": _node_path(run_path),
            }
        ),
        encoding="utf-8",
    )
    progress = "| total_timesteps | 8 |\n" if include_progress else ""
    (logs / f"{run_name}.log").write_text(
        f"[launch_run {launched_at}] script=fixture.py run={run_name} gpu=0\n"
        "[Train] device=cpu total_timesteps=999999\n"
        f"{progress}",
        encoding="utf-8",
    )
    _write_mlflow_run(repository, run_name, run_id, launched_ms)


@contextmanager
def _sleep_launch(tmp_path: Path, run_name: str = "healthy-run") -> Iterator[dict]:
    """Create a harmless current-launch-shaped process and remove it afterward."""
    if os.name == "nt":
        pytest.skip("Real /proc fixture runs inside the project WSL virtualenv")
    marker = f"health-sweep-{os.getpid()}-{time.time_ns()}"
    process = subprocess.Popen(
        ["setsid", "env", f"PI_LAUNCH_RUN_MARKER={marker}", "sleep", "120"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while True:
            if Path(f"/proc/{process.pid}/stat").exists():
                start_token, pgid, sid = _proc_identity(process.pid)
                if pgid == process.pid and sid == process.pid:
                    break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        repository = tmp_path / "fixture-repository"
        launched_ms = int(time.time() * 1000)
        run_id = "abcdef0123456789"
        _write_launch_evidence(
            repository,
            run_name=run_name,
            run_id=run_id,
            pid=process.pid,
            marker=marker,
            start_token=start_token,
            pgid=pgid,
            launched_ms=launched_ms,
        )
        yield {
            "repository": repository,
            "runName": run_name,
            "runId": run_id,
            "pid": process.pid,
            "startToken": start_token,
            "marker": marker,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_real_loader_registration_and_strict_low_power_contract(tmp_path: Path) -> None:
    """The real loader must discover one obvious, strict, low-power-friendly tool."""
    explicit = _registration(tmp_path / "explicit", auto_discover=False)
    discovered = _registration(tmp_path / "discovered", auto_discover=True)
    assert explicit["ok"] and discovered["ok"], (explicit, discovered)
    assert explicit["name"] == "health_sweep"
    assert discovered["names"].count("health_sweep") == 1
    public_text = json.dumps(
        {
            "label": explicit["label"],
            "description": explicit["description"],
            "snippet": explicit["promptSnippet"],
            "guidelines": explicit["promptGuidelines"],
        }
    ).lower()
    assert "immediately" in public_text and "launch_run" in public_text
    assert "ad hoc" in public_text and "[]" in public_text
    schema = explicit["parameters"]
    assert schema["required"] == ["runs"] and schema["additionalProperties"] is False
    runs = schema["properties"]["runs"]
    assert runs["minItems"] == 1 and runs["maxItems"] == 16
    item = runs["items"]
    assert item["additionalProperties"] is False
    assert {tuple(branch["required"]) for branch in item["anyOf"]} == {
        ("runName",), ("runId",), ("pid",)
    }
    assert item["properties"]["pid"]["minimum"] == 2


def test_registered_tool_accepts_name_id_pid_and_consistent_combinations(tmp_path: Path) -> None:
    """All selector modes resolve through real files/processes and healthy output is literal []."""
    with _sleep_launch(tmp_path) as fixture:
        response = _call_registered(
            fixture["repository"],
            {
                "runs": [
                    {"runName": fixture["runName"]},
                    {"runId": fixture["runId"][:8]},
                    {"pid": fixture["pid"]},
                    {
                        "runName": fixture["runName"].upper(),
                        "runId": fixture["runId"].upper(),
                        "pid": fixture["pid"],
                    },
                ]
            },
        )
    assert response["ok"], response
    assert response["result"]["content"][0]["text"] == "[]"
    assert response["result"]["details"]["unhealthyRuns"] == []
    assert len(response["result"]["details"]["observations"]) == 4
    assert all(item["healthy"] for item in response["result"]["details"]["observations"])


def test_mixed_batch_order_dead_process_and_pid_identity_mismatch(tmp_path: Path) -> None:
    """Mixed results include only bad inputs, preserve order, and fail dead/reused PIDs closed."""
    with _sleep_launch(tmp_path) as fixture:
        repository = fixture["repository"]
        launched_ms = int(time.time() * 1000)
        _write_launch_evidence(
            repository,
            run_name="dead-run",
            run_id="deadbeef00000000",
            pid=2_000_000_000,
            marker="dead-marker",
            start_token="1",
            pgid=2_000_000_000,
            launched_ms=launched_ms,
        )
        _write_launch_evidence(
            repository,
            run_name="reused-pid-run",
            run_id="feedface00000000",
            pid=fixture["pid"],
            marker=fixture["marker"],
            start_token=str(int(fixture["startToken"]) + 1),
            pgid=fixture["pid"],
            launched_ms=launched_ms,
        )
        response = _call_registered(
            repository,
            {
                "runs": [
                    {"runName": fixture["runName"]},
                    {"runName": "reused-pid-run"},
                    {"runName": "dead-run"},
                ]
            },
        )
    assert response["ok"], response
    unhealthy = response["result"]["details"]["unhealthyRuns"]
    assert [item["inputIndex"] for item in unhealthy] == [1, 2]
    assert json.loads(response["result"]["content"][0]["text"]) == unhealthy
    assert {reason["code"] for reason in unhealthy[0]["reasons"]} >= {
        "process_identity_mismatch"
    }
    assert {reason["code"] for reason in unhealthy[1]["reasons"]} >= {"process_dead"}


def test_zombie_is_not_alive_and_registered_polling_is_cancellable(tmp_path: Path) -> None:
    """The production /proc adapter rejects zombies and AbortSignal stops a real pending poll."""
    if os.name == "nt":
        pytest.skip("Zombie and /proc acceptance runs inside WSL")
    parent = subprocess.Popen(
        [
            shutil.which("python3") or "python3",
            "-c",
            "import os,time; p=os.fork(); "
            "(os._exit(0) if p == 0 else (print(p, flush=True), time.sleep(120)))",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    sleeper = subprocess.Popen(["sleep", "120"])
    try:
        zombie_pid = int(parent.stdout.readline().strip())
        deadline = time.monotonic() + 5
        while True:
            _, _, state = _proc_identity(zombie_pid)
            if Path(f"/proc/{zombie_pid}/stat").read_text().split(") ", 1)[1][0] == "Z":
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        observed = _direct_module(
            tmp_path,
            "const value = await health.inspectProductionProcess(p.pid); return value;",
            {"pid": zombie_pid},
        )
        assert observed["ok"], observed
        assert observed["value"]["state"] == "Z" and observed["value"]["alive"] is False

        response = _call_registered(
            tmp_path / "pending-repository",
            {"runs": [{"pid": sleeper.pid}]},
            abort_after_ms=100,
        )
        assert not response["ok"] and response["errorName"] == "AbortError"
        assert response["elapsedMs"] < 2_000
    finally:
        sleeper.terminate()
        parent.terminate()
        sleeper.wait(timeout=5)
        parent.wait(timeout=5)


def test_parsers_and_current_launch_segment_reject_stale_progress(tmp_path: Path) -> None:
    """Configured timesteps and stale prior-launch tables cannot masquerade as current progress."""
    repository = tmp_path / "log-repository"
    logs = repository / "logs"
    logs.mkdir(parents=True)
    run_name = "segment-run"
    old_ms = 1_700_000_000_000
    current_ms = old_ms + 10_000
    old_at = datetime.fromtimestamp(old_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
    current_at = datetime.fromtimestamp(current_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
    (logs / f"{run_name}.log").write_text(
        f"[launch_run {old_at}] script=x.py run={run_name} gpu=0\n"
        "[Train] device=cpu total_timesteps=1000\n| iterations | 1 |\n"
        f"[launch_run {current_at}] script=x.py run={run_name} gpu=0\n"
        "[Train] device=cpu total_timesteps=1000\n",
        encoding="utf-8",
    )
    body = r"""
const record = {
  kind: 'current', path: 'record', fileRunName: p.runName, runName: p.runName,
  pid: 42, pgid: 42, marker: 'm', startToken: 's',
  launchedAtMs: p.currentMs, launchedAt: p.currentAt,
  mlflowRunId: null, mlflowRunPath: null,
};
const log = await health.inspectProductionLog(p.repository, p.runName, record);
return {
  constants: [health.HEALTH_SWEEP_TIMEOUT_MS, health.HEALTH_SWEEP_POLL_INTERVAL_MS],
  configuredOnly: health.detectPpoProgress('[Train] device=cpu total_timesteps=999'),
  zero: health.detectPpoProgress('| total_timesteps | 0 |'),
  positive: health.detectPpoProgress('| iterations | 2 |'),
  banner: health.detectTrainingBanner('[Train] board=15 device=cuda'),
  fatal: health.detectFatalLogError('Traceback (most recent call last):'),
  log,
};
"""
    response = _direct_module(
        tmp_path,
        body,
        {
            "repository": _node_path(repository),
            "runName": run_name,
            "currentMs": current_ms,
            "currentAt": current_at,
        },
    )
    assert response["ok"], response
    value = response["value"]
    assert value["constants"] == [TIMEOUT_MS, POLL_MS]
    assert value["configuredOnly"] is None and value["zero"] is None
    assert value["positive"] and value["banner"] and value["fatal"]
    assert value["log"]["association"] == "current_launch"
    assert value["log"]["banner"] and value["log"]["progress"] is None


def test_virtual_time_shared_deadline_final_poll_and_delayed_evidence(tmp_path: Path) -> None:
    """One batch deadline handles delayed success and performs the mandatory exact-deadline read."""
    body = r"""
function fixtureIndex(now, includeFinal) {
  const records = ['delayed', 'final', 'never'].map((name, i) => ({
    kind: 'current', path: `/logs/${name}.pid.json`, fileRunName: name, runName: name,
    pid: 100 + i, pgid: 100 + i, marker: `m${i}`, startToken: `s${i}`,
    launchedAtMs: 0, launchedAt: new Date(0).toISOString(),
    mlflowRunId: `${name}-id`, mlflowRunPath: `/mlruns/1/${name}-id`,
  }));
  const available = records.filter((r) =>
    r.runName === 'never' ? false : r.runName === 'delayed' ? now >= 10000 : includeFinal && now >= 600000
  );
  return {
    logsPath: '/logs', mlrunsPath: '/mlruns', launchRecords: records, recordErrors: [],
    mlflowRuns: available.map((r) => ({
      experimentId: '1', runId: r.mlflowRunId, runName: r.runName, status: '1',
      startTimeMs: 0, runPath: r.mlflowRunPath, metaPath: `${r.mlflowRunPath}/meta.yaml`,
    })),
  };
}
function dependencies(includeFinal) {
  let now = 0;
  const sleeps = [];
  const builtAt = [];
  return {
    state: { sleeps, builtAt, get now() { return now; } },
    deps: {
      clock: { now: () => now, sleep: async (ms, signal) => { sleeps.push(ms); now += ms; } },
      buildIndex: async () => { builtAt.push(now); return fixtureIndex(now, includeFinal); },
      inspectProcess: async (pid) => ({
        pid, exists: true, alive: true, state: 'S', pgid: pid, sid: pid,
        startToken: `s${pid - 100}`, marker: `m${pid - 100}`,
      }),
      inspectLog: async (_cwd, name) => {
        const ready = name === 'delayed' ? now >= 10000 : name === 'final' ? includeFinal && now >= 600000 : false;
        return {
          path: `/logs/${name}.log`, exists: ready, association: 'current_launch',
          banner: ready ? '[Train] device=cpu' : null,
          progress: ready ? '| iterations | 1 |' : null, fatal: null,
          boundary: ready ? '[launch_run ...]' : null, bytesRead: 1, truncated: false,
        };
      },
    },
  };
}
const delayedHarness = dependencies(false);
const updates = [];
const delayed = await health.healthSweep(
  { runs: [{ runName: 'delayed' }] }, '/repo', undefined,
  (update) => updates.push(update.content[0].text), delayedHarness.deps
);
const deadlineHarness = dependencies(true);
const deadline = await health.healthSweep(
  { runs: [{ runName: 'final' }, { runName: 'never' }] }, '/repo', undefined, undefined,
  deadlineHarness.deps
);
return {
  delayed, delayedState: delayedHarness.state, updates,
  deadline, deadlineState: deadlineHarness.state,
};
"""
    response = _direct_module(tmp_path, body)
    assert response["ok"], response
    value = response["value"]
    assert value["delayed"]["content"][0]["text"] == "[]"
    assert value["delayed"]["details"]["waitedMs"] == 10_000
    assert sum(value["delayedState"]["sleeps"]) == 10_000
    assert value["updates"] and all("pending" in update.lower() for update in value["updates"])

    deadline = value["deadline"]
    assert deadline["details"]["waitedMs"] == TIMEOUT_MS
    assert deadline["details"]["timedOut"] is True
    assert value["deadlineState"]["builtAt"][-1] == TIMEOUT_MS
    assert sum(value["deadlineState"]["sleeps"]) == TIMEOUT_MS
    unhealthy = deadline["details"]["unhealthyRuns"]
    assert [item["inputIndex"] for item in unhealthy] == [1]
    assert deadline["details"]["observations"][0]["healthy"] is True
    assert {reason["code"] for reason in unhealthy[0]["reasons"]} >= {
        "log_missing", "mlflow_run_missing"
    }


def test_mlflow_missing_ambiguity_fatal_log_and_combined_identity_conflict(tmp_path: Path) -> None:
    """MLflow/fatal failures are actionable, and disagreeing combined selectors must fail closed."""
    body = r"""
function record(name, id, pid) {
  return {
    kind: 'current', path: `/logs/${name}.pid.json`, fileRunName: name, runName: name,
    pid, pgid: pid, marker: `m${pid}`, startToken: `s${pid}`,
    launchedAtMs: 0, launchedAt: new Date(0).toISOString(), mlflowRunId: id,
    mlflowRunPath: `/mlruns/1/${id}`,
  };
}
function mlflow(name, id) {
  return { experimentId: '1', runId: id, runName: name, status: '1', startTimeMs: 0,
    runPath: `/mlruns/1/${id}`, metaPath: `/mlruns/1/${id}/meta.yaml` };
}
async function run(mode, input) {
  let now = 0;
  const baseRecord = record('right-run', 'actual-id', 101);
  const runs = mode === 'missing' ? []
    : mode === 'ambiguous' || mode === 'ambiguous-combined'
      ? [mlflow('right-run', 'abc-one'), mlflow('right-run', 'abc-two')]
    : mode === 'final-id-conflict'
      ? [mlflow('right-run', 'selected-id')]
    : [mlflow('right-run', 'actual-id')];
  const deps = {
    clock: { now: () => now, sleep: async (ms) => { now += ms; } },
    buildIndex: async () => ({ logsPath: '/logs', mlrunsPath: '/mlruns',
      launchRecords: [baseRecord], recordErrors: [], mlflowRuns: runs }),
    inspectProcess: async (pid) => ({ pid, exists: true, alive: true, state: 'S',
      pgid: pid, sid: pid, startToken: `s${pid}`, marker: `m${pid}` }),
    inspectLog: async () => ({ path: '/logs/right-run.log', exists: true,
      association: 'current_launch', banner: '[Train] device=cpu',
      progress: '| iterations | 1 |', fatal: mode === 'fatal' ? 'Traceback (most recent call last):' : null,
      boundary: '[launch_run ...]', bytesRead: 10, truncated: false }),
  };
  return health.healthSweep({ runs: [input] }, '/repo', undefined, undefined, deps);
}
const missing = await run('missing', { runName: 'right-run' });
const ambiguous = await run('ambiguous', { runId: 'abc' });
const fatal = await run('fatal', { runName: 'right-run' });
// A matching case-insensitive prefix must cross-check both launch metadata and MLflow.
const prefixHealthy = await run('healthy', { runName: 'RIGHT-RUN', runId: 'ACTUAL' });
// Zero explicit matches cannot fall back to the name-selected launch record.
const conflict = await run('healthy', { runName: 'right-run', runId: 'wrong-id' });
// Ambiguous explicit matches cannot fall back to the name-selected launch record either.
const ambiguousCombined = await run(
  'ambiguous-combined', { runName: 'right-run', runId: 'abc' }
);
// A unique explicit prefix still fails when the final MLflow identity and launch record disagree.
const finalIdConflict = await run(
  'final-id-conflict', { runName: 'right-run', runId: 'selected' }
);
return { missing, ambiguous, fatal, prefixHealthy, conflict, ambiguousCombined, finalIdConflict };
"""
    response = _direct_module(tmp_path, body)
    assert response["ok"], response
    results = response["value"]
    assert {r["code"] for r in results["missing"]["details"]["unhealthyRuns"][0]["reasons"]} >= {
        "mlflow_run_missing"
    }
    assert {r["code"] for r in results["ambiguous"]["details"]["unhealthyRuns"][0]["reasons"]} >= {
        "identity_ambiguous"
    }
    fatal = results["fatal"]["details"]["unhealthyRuns"][0]
    assert {r["code"] for r in fatal["reasons"]} >= {"fatal_log_error"}
    assert "Traceback" in next(r["evidence"] for r in fatal["reasons"] if r["code"] == "fatal_log_error")

    # Multiple supplied identifiers are assertions about one identity, not fallbacks.
    assert results["prefixHealthy"]["content"][0]["text"] == "[]"
    conflict = results["conflict"]
    assert conflict["content"][0]["text"] != "[]"
    assert {r["code"] for r in conflict["details"]["unhealthyRuns"][0]["reasons"]} & {
        "identity_mismatch", "mlflow_run_missing"
    }
    ambiguous_combined = results["ambiguousCombined"]
    assert ambiguous_combined["content"][0]["text"] != "[]"
    assert {
        r["code"]
        for r in ambiguous_combined["details"]["unhealthyRuns"][0]["reasons"]
    } >= {"identity_ambiguous"}
    final_id_conflict = results["finalIdConflict"]
    assert final_id_conflict["content"][0]["text"] != "[]"
    assert {
        r["code"]
        for r in final_id_conflict["details"]["unhealthyRuns"][0]["reasons"]
    } >= {"identity_mismatch"}
