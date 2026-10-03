"""Black-box acceptance tests for the estimate_run_completion Pi extension."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / ".pi" / "extensions" / "estimate_run_completion.ts"
PI_ROOT = Path("/mnt/c/Users/wange/AppData/Local/pi-node/current")
TIMEOUT_MS = 1_800_000
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
    loader = PI_ROOT / "node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js"
    if not loader.exists():
        pytest.skip("Pi extension loader is unavailable")
    return loader


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
        raise AssertionError(f"No JSON response. stdout={completed.stdout!r}, stderr={completed.stderr!r}") from error


def _call_many(store: Path, queries: list[dict], *, auto_discover: bool = False) -> dict:
    """Load once through Pi and invoke the actual registered tool for each query."""
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
    .map((extension) => extension.tools.get('estimate_run_completion'))
    .filter(Boolean);
  if (registrations.length !== 1) {
    throw new Error(`estimate_run_completion registrations: ${registrations.length}; tools: ${names.join(',')}`);
  }
  const tool = registrations[0].definition;
  const outcomes = [];
  for (const query of p.queries) {
    try {
      const result = await tool.execute(
        'acceptance', { mlrunsPath: p.store, ...query }, undefined, undefined, { cwd: p.cwd }
      );
      outcomes.push({ ok: true, result });
    } catch (error) {
      outcomes.push({ ok: false, error: error instanceof Error ? error.message : String(error) });
    }
  }
  console.log(JSON.stringify({
    ok: true,
    names,
    description: tool.description,
    parameters: tool.parameters,
    outcomes,
  }));
} catch (error) {
  console.log(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }));
}
"""
    agent_dir = store.parent / "empty-pi-agent"
    agent_dir.mkdir(exist_ok=True)
    response = _run_node(
        script,
        {
            "loader": _node_path(_loader()),
            "extension": _node_path(EXTENSION),
            "cwd": _node_path(ROOT),
            "store": _node_path(store),
            "queries": queries,
            "autoDiscover": auto_discover,
            "agentDir": _node_path(agent_dir),
        },
    )
    assert response["ok"], response
    return response


def _call_waiting_tool(store: Path, run_name: str, mode: str) -> dict:
    """Exercise real registered polling with real sleep, delayed I/O, or abort."""
    script = r"""
const { mkdir, writeFile } = await import('node:fs/promises');
const { dirname } = await import('node:path');
const { pathToFileURL } = await import('node:url');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const p = JSON.parse(input);
try {
  const loader = await import(pathToFileURL(p.loader).href);
  const loaded = await loader.loadExtensions([p.extension], p.cwd);
  if (loaded.errors.length) throw new Error(JSON.stringify(loaded.errors));
  const registrations = loaded.extensions
    .map((extension) => extension.tools.get('estimate_run_completion'))
    .filter(Boolean);
  if (registrations.length !== 1) throw new Error(`registrations: ${registrations.length}`);
  const tool = registrations[0].definition;
  const controller = new AbortController();
  const started = Date.now();
  let settled = false;
  const pending = tool.execute(
    'waiting-acceptance',
    { mlrunsPath: p.store, runName: p.runName },
    controller.signal,
    undefined,
    { cwd: p.cwd },
  );
  pending.then(() => { settled = true; }, () => { settled = true; });
  await new Promise((resolve) => setTimeout(resolve, 150));
  const pendingAfterDelay = !settled;
  if (p.mode === 'delayed-metric') {
    await mkdir(dirname(p.metricPath), { recursive: true });
    await writeFile(p.metricPath, p.metricText, 'utf8');
  } else if (p.mode === 'cancel') {
    controller.abort();
  }
  try {
    const result = await pending;
    console.log(JSON.stringify({ ok: true, pendingAfterDelay, elapsedMs: Date.now() - started, result }));
  } catch (error) {
    console.log(JSON.stringify({
      ok: false,
      pendingAfterDelay,
      elapsedMs: Date.now() - started,
      error: error instanceof Error ? error.message : String(error),
      errorName: error instanceof Error ? error.name : null,
    }));
  }
} catch (error) {
  console.log(JSON.stringify({ ok: false, harnessError: error instanceof Error ? error.message : String(error) }));
}
"""
    run = next(path for path in store.glob("*/*") if path.is_dir())
    metric_path = run / "metrics" / "train" / "progress"
    return _run_node(
        script,
        {
            "loader": _node_path(_loader()),
            "extension": _node_path(EXTENSION),
            "cwd": _node_path(ROOT),
            "store": _node_path(store),
            "runName": run_name,
            "mode": mode,
            "metricPath": _node_path(metric_path),
            # Two progress observations make the post-wait rate unambiguous.
            "metricText": "1000000 0.1 10\n1001000 0.2 20\n",
        },
        timeout=12,
    )


def _write_run(
    store: Path,
    experiment: str,
    directory_id: str,
    *,
    run_id: str | None = None,
    name: str | None = None,
    meta_name: str | None = None,
    status: str | None = "RUNNING",
    start: int | str | None = 900_000,
    params: dict[str, str] | None = None,
    metrics: dict[str, str] | None = None,
) -> Path:
    run = store / experiment / directory_id
    run.mkdir(parents=True)
    fields = []
    if run_id is not None:
        fields.append(f"run_id: {run_id}")
    if meta_name is not None:
        fields.append(f"run_name: '{meta_name}'")
    if status is not None:
        fields.append(f"status: {status}")
    if start is not None:
        fields.append(f"start_time: {start}")
    (run / "meta.yaml").write_text("\n".join(fields) + "\n", encoding="utf-8")
    if name is not None:
        (run / "tags").mkdir()
        (run / "tags/mlflow.runName").write_text(name, encoding="utf-8")
    for key, value in (params or {}).items():
        target = run / "params" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
    for key, value in (metrics or {}).items():
        target = run / "metrics" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
    return run


def _new_store(tmp_path: Path) -> Path:
    store = tmp_path / "mlruns"
    experiment = store / "7"
    experiment.mkdir(parents=True)
    (experiment / "meta.yaml").write_text("name: eta-acceptance\n", encoding="utf-8")
    return store


def _result(response: dict, index: int) -> dict:
    outcome = response["outcomes"][index]
    assert outcome["ok"], outcome
    return outcome["result"]


def test_real_pi_registration_schema_exact_name_and_unique_id(tmp_path: Path):
    # Auto-discovery must register this tool exactly once, expose no timeout knob,
    # and resolve both tag names and metadata fallback names case-insensitively.
    store = _new_store(tmp_path)
    _write_run(
        store,
        "7",
        "tag-dir",
        run_id="ABCDEF111111",
        name="Tagged Exact Name",
        meta_name="ignored metadata name",
        status="3",
    )
    _write_run(
        store,
        "7",
        "meta-dir",
        run_id="FEDCBA222222",
        meta_name="Metadata Fallback Name",
        status="FINISHED",
    )
    response = _call_many(
        store,
        [
            {"runName": "tAgGeD eXaCt NaMe"},
            {"runId": "fedcba2"},
            {"runName": "METADATA FALLBACK NAME"},
        ],
        auto_discover=True,
    )
    assert response["names"].count("estimate_run_completion") == 1
    assert set(response["parameters"]["properties"]) == {"mlrunsPath", "runName", "runId"}
    assert "30 minutes" in response["description"]
    assert "non-configurable" in response["description"]
    assert [
        _result(response, index)["details"]["runId"] for index in range(3)
    ] == ["ABCDEF111111", "FEDCBA222222", "FEDCBA222222"]
    for index in range(3):
        details = _result(response, index)["details"]
        assert details["experimentId"] == "7"
        assert details["experimentName"] == "eta-acceptance"
        assert details["mlrunsPath"] == _node_path(store)
        assert details["runPath"]


def test_lookup_validation_not_found_and_ambiguity_are_actionable(tmp_path: Path):
    # Exact duplicate names, non-unique prefixes, absent selectors, and invalid
    # selector combinations must fail rather than selecting an arbitrary run.
    store = _new_store(tmp_path)
    _write_run(store, "7", "a", run_id="shared-prefix-aaa", name="Duplicate")
    _write_run(store, "7", "b", run_id="shared-prefix-bbb", name="duplicate")
    response = _call_many(
        store,
        [
            {"runName": "DUPLICATE"},
            {"runId": "SHARED-PREFIX"},
            {"runName": "absent"},
            {},
            {"runName": "Duplicate", "runId": "shared-prefix-aaa"},
        ],
    )
    errors = [item["error"] for item in response["outcomes"]]
    assert "Ambiguous MLflow run name" in errors[0]
    assert "Ambiguous MLflow run-ID prefix" in errors[1]
    assert "No MLflow run" in errors[2]
    assert "exactly one" in errors[3]
    assert "exactly one" in errors[4]
    assert "shared-prefix-aaa" in errors[0] and "shared-prefix-bbb" in errors[0]


def test_deterministic_metric_arithmetic_deduplicates_steps_and_ignores_bad_rows(tmp_path: Path):
    # Any metric history may contribute progress, but duplicate records at one
    # step count once (latest timestamp wins) and malformed/non-finite rows do not.
    store = _new_store(tmp_path)
    _write_run(
        store,
        "7",
        "eta",
        run_id="eta-history-001",
        name="Deterministic ETA",
        params={"total_timesteps": "100"},
        metrics={
            "eval/a": (
                "malformed\n"
                "1000000 0.1 10\n"
                "1002500 Infinity 25\n"
                "NaN 0.2 30\n"
                "1003000 0.3 50 extra\n"
                "1003000 0.4 50\n"
            ),
            "train/b": "1002000 9.0 30\n1004000 8.0 50\n",
        },
    )
    result = _result(_call_many(store, [{"runId": "ETA-HISTORY"}]), 0)
    details = result["details"]
    assert details["outcome"] == "estimated"
    assert details["currentTimesteps"] == 50
    assert details["observationTimestampMs"] == 1_004_000
    assert details["observedStartTimesteps"] == 10
    assert details["observedTimestepSpan"] == 40
    assert details["observedTimeSpanMs"] == 4_000
    assert details["millisecondsPerTimestep"] == 100
    assert details["estimatedCompletionTimestampMs"] == 1_009_000
    assert details["remainingMilliseconds"] == 0
    assert details["remainingSeconds"] == 0
    assert details["overdue"] is True
    assert details["usedStartTimeFallback"] is False
    assert "overdue" in result["content"][0]["text"]


def test_single_observation_uses_documented_start_time_fallback(tmp_path: Path):
    # One positive-step sample remains estimable from run start, and the result
    # labels that lower-confidence timing basis and all arithmetic operands.
    store = _new_store(tmp_path)
    _write_run(
        store,
        "7",
        "single",
        run_id="single-observation",
        name="Single Observation",
        start=1_000_000,
        params={"num_timesteps": "100"},
        metrics={"nested/progress": "1005000 1.0 25\n"},
    )
    details = _result(_call_many(store, [{"runName": "single observation"}]), 0)["details"]
    assert details["outcome"] == "estimated"
    assert details["totalTimesteps"] == 100
    assert details["totalTimestepsParameter"] == "num_timesteps"
    assert details["currentTimesteps"] == 25
    assert details["usedStartTimeFallback"] is True
    assert details["observedStartTimesteps"] == 0
    assert details["observedStartTimestampMs"] == 1_000_000
    assert details["observedTimestepSpan"] == 25
    assert details["observedTimeSpanMs"] == 5_000
    assert details["millisecondsPerTimestep"] == 200
    assert details["estimatedCompletionTimestampMs"] == 1_020_000


def test_completed_and_failed_status_semantics(tmp_path: Path):
    # Numeric/text terminal statuses and progress reaching the plan have distinct
    # completed versus unknown-failure outcomes and matching human-readable text.
    store = _new_store(tmp_path)
    _write_run(store, "7", "finished", run_id="finished-id", name="Finished", status="3", params=None)
    _write_run(
        store,
        "7",
        "reached",
        run_id="reached-id",
        name="Reached Plan",
        status="RUNNING",
        params={"total_timesteps": "50"},
        metrics={"m": "1000000 1 50\n"},
    )
    _write_run(store, "7", "failed", run_id="failed-id", name="Failed", status="FAILED")
    _write_run(store, "7", "killed", run_id="killed-id", name="Killed", status="5")
    response = _call_many(
        store,
        [{"runId": "finished"}, {"runId": "reached"}, {"runId": "failed"}, {"runId": "killed"}],
    )
    finished = _result(response, 0)
    reached = _result(response, 1)
    failed = _result(response, 2)
    killed = _result(response, 3)
    assert finished["details"]["outcome"] == "completed"
    assert finished["details"]["completionBasis"] == "finished_status"
    assert reached["details"]["outcome"] == "completed"
    assert reached["details"]["completionBasis"] == "planned_timesteps_reached"
    assert finished["details"]["remainingMilliseconds"] == 0
    assert reached["details"]["remainingMilliseconds"] == 0
    assert "Remaining: 0s" in finished["content"][0]["text"]
    for terminal in (failed, killed):
        assert terminal["details"]["outcome"] == "unknown"
        assert terminal["details"]["reason"] == "terminal_failure"
        assert terminal["details"]["remainingMilliseconds"] is None
        assert "terminal_failure" in terminal["content"][0]["text"]


def test_missing_malformed_conflicting_totals_and_unusable_metrics(tmp_path: Path):
    # Invalid plans return immediately with separate reason codes; valid metric
    # records that cannot form a positive interval are not mistaken for no metrics.
    store = _new_store(tmp_path)
    common_metric = {"m": "1000000 1 10\n1001000 2 20\n"}
    _write_run(store, "7", "missing", run_id="total-missing", name="Missing", params=None, metrics=common_metric)
    _write_run(store, "7", "malformed", run_id="total-malformed", name="Malformed", params={"total_timesteps": "NaN"}, metrics=common_metric)
    _write_run(store, "7", "zero", run_id="total-zero", name="Zero", params={"total_timesteps": "0"}, metrics=common_metric)
    _write_run(
        store,
        "7",
        "conflict",
        run_id="total-conflict",
        name="Conflict",
        params={"total_timesteps": "100", "num_timesteps": "200"},
        metrics=common_metric,
    )
    _write_run(
        store,
        "7",
        "unusable",
        run_id="metric-unusable",
        name="Unusable",
        params={"total_timesteps": "100"},
        metrics={"m": "1000000 1 0\nnot a record\n1001000 NaN 10\n"},
    )
    response = _call_many(
        store,
        [
            {"runId": "total-missing"},
            {"runId": "total-malformed"},
            {"runId": "total-zero"},
            {"runId": "total-conflict"},
            {"runId": "metric-unusable"},
        ],
    )
    reasons = [_result(response, index)["details"]["reason"] for index in range(5)]
    assert reasons == [
        "missing_total_timesteps",
        "malformed_total_timesteps",
        "malformed_total_timesteps",
        "conflicting_total_timesteps",
        "unusable_metric_timing",
    ]
    for index in range(5):
        details = _result(response, index)["details"]
        assert details["outcome"] == "unknown"
        assert details["wait"]["pollCount"] == 0
        assert details["wait"]["timedOut"] is False


def test_registered_call_really_waits_then_unblocks_when_metric_appears(tmp_path: Path):
    # This is a real loader/registered-tool call with no initial metrics. A file
    # appears after 150 ms; the five-second production poll proves the call was
    # pending, re-read the same run, and immediately estimated on that poll.
    store = _new_store(tmp_path)
    _write_run(
        store,
        "7",
        "delayed",
        run_id="delayed-id",
        name="Delayed Metric",
        start=999_000,
        params={"total_timesteps": "100"},
        metrics=None,
    )
    response = _call_waiting_tool(store, "Delayed Metric", "delayed-metric")
    assert response["ok"], response
    assert response["pendingAfterDelay"] is True
    assert response["elapsedMs"] >= POLL_MS - 250
    details = response["result"]["details"]
    assert details["outcome"] == "estimated"
    assert details["currentTimesteps"] == 20
    assert details["wait"]["pollCount"] == 1
    assert details["wait"]["metricsObservedAfterWait"] is True
    assert details["wait"]["timedOut"] is False


def test_registered_wait_honors_abort_signal_without_waiting_for_next_poll(tmp_path: Path):
    # Cancellation is tested while the real registered call is asleep, not just
    # with a pre-cancelled signal, so an abort must interrupt the timer promptly.
    store = _new_store(tmp_path)
    _write_run(
        store,
        "7",
        "cancel",
        run_id="cancel-id",
        name="Cancel Wait",
        params={"total_timesteps": "100"},
        metrics=None,
    )
    response = _call_waiting_tool(store, "Cancel Wait", "cancel")
    assert response["ok"] is False
    assert response["pendingAfterDelay"] is True
    assert response["errorName"] == "AbortError"
    assert "cancelled" in response["error"].lower()
    assert response["elapsedMs"] < 2_000


def _write_clock_probe(probe: Path) -> None:
    """Create an evaluator-only extension that calls only the exported clock seam."""
    relative_extension = os.path.relpath(EXTENSION, probe.parent).replace(os.sep, "/")
    if not relative_extension.startswith("."):
        relative_extension = "./" + relative_extension
    probe.write_text(
        f'''import {{ mkdir, writeFile }} from "node:fs/promises";
import {{ dirname }} from "node:path";
import type {{ ExtensionAPI }} from "@earendil-works/pi-coding-agent";
import {{ Type }} from "typebox";
import {{
  estimateRunCompletion,
  METRIC_POLL_INTERVAL_MS,
  METRIC_WAIT_TIMEOUT_MS,
}} from {json.dumps(relative_extension)};

export default function probe(pi: ExtensionAPI): void {{
  pi.registerTool({{
    name: "estimate_clock_probe",
    label: "Estimate clock probe",
    description: "Evaluator-only exported-seam probe",
    parameters: Type.Object({{
      store: Type.String(),
      runName: Type.String(),
      start: Type.Number(),
      metricPath: Type.String(),
      writeAtDeadline: Type.Boolean(),
      metricText: Type.String(),
    }}),
    async execute(_id, params, signal, _update, ctx) {{
      let now = params.start;
      let written = false;
      const sleeps: number[] = [];
      const clock = {{
        now: () => now,
        sleep: async (milliseconds: number) => {{
          sleeps.push(milliseconds);
          now += milliseconds;
          if (params.writeAtDeadline && !written && now >= params.start + METRIC_WAIT_TIMEOUT_MS) {{
            await mkdir(dirname(params.metricPath), {{ recursive: true }});
            await writeFile(params.metricPath, params.metricText, "utf8");
            written = true;
          }}
        }},
      }};
      const result = await estimateRunCompletion(
        {{ mlrunsPath: params.store, runName: params.runName }}, ctx.cwd, signal, clock
      );
      return {{
        ...result,
        details: {{
          ...result.details,
          probe: {{ sleeps, sleepTotal: sleeps.reduce((a, b) => a + b, 0), written,
                    timeout: METRIC_WAIT_TIMEOUT_MS, poll: METRIC_POLL_INTERVAL_MS }},
        }},
      }};
    }},
  }});
}}
''',
        encoding="utf-8",
    )


def _call_clock_probe(probe: Path, store: Path, run_name: str, *, write_at_deadline: bool) -> dict:
    script = r"""
const { pathToFileURL } = await import('node:url');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const p = JSON.parse(input);
try {
  const loader = await import(pathToFileURL(p.loader).href);
  const loaded = await loader.loadExtensions([p.probe], p.cwd);
  if (loaded.errors.length) throw new Error(JSON.stringify(loaded.errors));
  const registrations = loaded.extensions
    .map((extension) => extension.tools.get('estimate_clock_probe'))
    .filter(Boolean);
  if (registrations.length !== 1) throw new Error(`probe registrations: ${registrations.length}`);
  const result = await registrations[0].definition.execute(
    'clock-probe', p.params, undefined, undefined, { cwd: p.cwd }
  );
  console.log(JSON.stringify({ ok: true, result }));
} catch (error) {
  console.log(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }));
}
"""
    run = next(path for path in store.glob("*/*") if path.is_dir())
    response = _run_node(
        script,
        {
            "loader": _node_path(_loader()),
            "probe": _node_path(probe),
            "cwd": _node_path(ROOT),
            "params": {
                "store": _node_path(store),
                "runName": run_name,
                "start": 1_000_000,
                "metricPath": _node_path(run / "metrics" / "deadline"),
                "writeAtDeadline": write_at_deadline,
                "metricText": "1000000 0.1 10\n2800000 0.2 20\n",
            },
        },
    )
    assert response["ok"], response
    return response["result"]


def test_fixed_timeout_and_final_deadline_read_via_exported_clock_seam(tmp_path: Path):
    # The evaluator-only probe imports the production core and clock constants;
    # it does not copy logic or expose a timeout. Virtual sleeps prove the exact
    # 30-minute boundary quickly, including the mandatory read after final sleep.
    probe = ROOT / "artifacts" / "estimate_run_completion_tool" / "tmp" / "clock_probe.ts"
    probe.parent.mkdir(parents=True, exist_ok=True)
    _write_clock_probe(probe)
    try:
        timeout_store = _new_store(tmp_path / "timeout")
        _write_run(
            timeout_store,
            "7",
            "malformed-only",
            run_id="malformed-metrics",
            name="Malformed Metrics",
            params={"total_timesteps": "100"},
            metrics={"bad": "partial row\n1 NaN 10\n2 0.5 Infinity\n"},
        )
        timed_out = _call_clock_probe(probe, timeout_store, "Malformed Metrics", write_at_deadline=False)
        timeout_details = timed_out["details"]
        timeout_probe = timeout_details["probe"]
        assert timeout_details["outcome"] == "unknown"
        assert timeout_details["reason"] == "metric_timeout"
        assert timeout_details["wait"]["timedOut"] is True
        assert timeout_details["wait"]["waitedMs"] == TIMEOUT_MS
        assert timeout_details["wait"]["deadlineMs"] - timeout_details["wait"]["startedAtMs"] == TIMEOUT_MS
        assert timeout_details["wait"]["pollCount"] == TIMEOUT_MS // POLL_MS
        assert timeout_probe["timeout"] == TIMEOUT_MS
        assert timeout_probe["poll"] == POLL_MS
        assert timeout_probe["sleepTotal"] == TIMEOUT_MS
        assert set(timeout_probe["sleeps"]) == {POLL_MS}
        assert "no valid metrics" in timed_out["content"][0]["text"]

        boundary_store = _new_store(tmp_path / "boundary")
        _write_run(
            boundary_store,
            "7",
            "boundary",
            run_id="deadline-boundary",
            name="Deadline Boundary",
            start=900_000,
            params={"total_timesteps": "100"},
            metrics=None,
        )
        boundary = _call_clock_probe(probe, boundary_store, "Deadline Boundary", write_at_deadline=True)
        boundary_details = boundary["details"]
        assert boundary_details["outcome"] == "estimated"
        assert boundary_details["currentTimesteps"] == 20
        assert boundary_details["wait"]["waitedMs"] == TIMEOUT_MS
        assert boundary_details["wait"]["pollCount"] == TIMEOUT_MS // POLL_MS
        assert boundary_details["wait"]["metricsObservedAfterWait"] is True
        assert boundary_details["wait"]["timedOut"] is False
        assert boundary_details["probe"]["written"] is True
        assert boundary_details["probe"]["sleepTotal"] == TIMEOUT_MS
    finally:
        shutil.rmtree(probe.parent, ignore_errors=True)


def test_public_tool_has_no_timeout_override_and_uses_filesystem_core():
    # A source-level regression guard complements the black-box schema check:
    # production binds its system clock/fixed core and never imports MLflow.
    source = EXTENSION.read_text(encoding="utf-8")
    schema_source = source.split("export const estimateRunCompletionSchema", 1)[1].split(
        "export type EstimateRunCompletionInput", 1
    )[0]
    assert "timeout:" not in schema_source.lower()
    assert "METRIC_WAIT_TIMEOUT_MS = 1_800_000" in source
    assert "METRIC_POLL_INTERVAL_MS = 5_000" in source
    assert "estimateRunCompletion(params, ctx.cwd, signal, SYSTEM_ESTIMATE_CLOCK)" in source
    assert 'from "node:fs/promises"' in source
    import_lines = "\n".join(line.lower() for line in source.splitlines() if "import" in line)
    assert 'from "mlflow"' not in import_lines
    assert "mlflowclient" not in source.lower()
