"""Black-box acceptance tests for the registered extract_run_evidence Pi tool."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / ".pi" / "extensions" / "extract_run_evidence.ts"
PI_ROOT = Path("/mnt/c/Users/wange/AppData/Local/pi-node/current")


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


def _call_tool(
    store: Path,
    calls: list[dict],
    *,
    auto_discover: bool = False,
    timeout: int = 45,
) -> dict:
    """Load the real extension and execute one or more calls through its registration."""
    loader = PI_ROOT / "node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js"
    if not loader.exists():
        pytest.skip("Pi extension loader is unavailable")
    script = r"""
const { createHash } = await import('node:crypto');
const { readFile, rm } = await import('node:fs/promises');
const { dirname, basename } = await import('node:path');
const { pathToFileURL } = await import('node:url');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const p = JSON.parse(input);
const loaderModule = await import(pathToFileURL(p.loader).href);
const loaded = p.autoDiscover
  ? await loaderModule.discoverAndLoadExtensions([], p.cwd, p.agentDir)
  : await loaderModule.loadExtensions([p.extension], p.cwd);
if (loaded.errors.length) throw new Error(JSON.stringify(loaded.errors));
const names = loaded.extensions.flatMap((extension) => [...extension.tools.keys()]);
const registrations = loaded.extensions
  .map((extension) => extension.tools.get('extract_run_evidence'))
  .filter(Boolean);
if (registrations.length !== 1) {
  throw new Error(`extract_run_evidence registrations: ${registrations.length}; tools: ${names.join(',')}`);
}
const tool = registrations[0].definition;
const outcomes = [];
for (const call of p.calls) {
  const controller = call.preAbort || call.abortDelayMs !== undefined
    ? new AbortController()
    : undefined;
  if (call.preAbort) controller.abort();
  if (call.abortDelayMs !== undefined) {
    setTimeout(() => controller.abort(), call.abortDelayMs);
  }
  try {
    const result = await tool.execute(
      'acceptance',
      { mlrunsPath: p.store, ...call.query },
      controller?.signal,
      undefined,
      { cwd: p.cwd },
    );
    const text = result.content[0].text;
    const canonical = JSON.stringify(result.details.evidence, null, 2) + '\n';
    let report = null;
    if (result.details.fullReportPath) {
      const reportText = await readFile(result.details.fullReportPath, 'utf8');
      report = {
        basename: basename(result.details.fullReportPath),
        matchesCanonical: reportText === canonical,
        sha256: createHash('sha256').update(reportText).digest('hex'),
        bytes: Buffer.byteLength(reportText),
        lines: reportText.split('\n').length - 1,
      };
      await rm(dirname(result.details.fullReportPath), { recursive: true, force: true });
    }
    outcomes.push({
      ok: true,
      result,
      textMatchesCanonical: text === canonical,
      textEndsNewline: text.endsWith('\n'),
      textBytes: Buffer.byteLength(text),
      textLines: text.split('\n').length - 1,
      canonicalSha256: createHash('sha256').update(canonical).digest('hex'),
      report,
    });
  } catch (error) {
    outcomes.push({
      ok: false,
      name: error instanceof Error ? error.name : null,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}
console.log(JSON.stringify({
  names,
  registrationCount: registrations.length,
  description: tool.description,
  promptSnippet: tool.promptSnippet,
  promptGuidelines: tool.promptGuidelines,
  parameters: tool.parameters,
  outcomes,
}));
"""
    agent_dir = store.parent / "empty-pi-agent"
    agent_dir.mkdir(exist_ok=True)
    payload = {
        "loader": _node_path(loader),
        "extension": _node_path(EXTENSION),
        "cwd": _node_path(ROOT),
        "store": _node_path(store),
        "agentDir": _node_path(agent_dir),
        "autoDiscover": auto_discover,
        "calls": calls,
    }
    completed = subprocess.run(
        [_node_executable(), "--input-type=module", "-e", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _write_run(
    store: Path,
    experiment: str,
    directory_id: str,
    *,
    run_id: str | None = None,
    tag_name: str | None = None,
    meta_name: str | None = None,
    start: int | str | None = 1704067200000,
    end: int | str | None = 1704070800000,
    status: str | None = "FINISHED",
    params: dict[str, str] | None = None,
    metrics: dict[str, str] | None = None,
    artifacts: dict[str, bytes] | None = None,
) -> Path:
    run = store / experiment / directory_id
    run.mkdir(parents=True)
    fields = [f"experiment_id: {experiment}"]
    if run_id is not None:
        fields.append(f"run_id: {run_id}")
    if meta_name is not None:
        fields.append(f"run_name: '{meta_name}'")
    if start is not None:
        fields.append(f"start_time: {start}")
    if end is not None:
        fields.append(f"end_time: {end}")
    if status is not None:
        fields.append(f"status: {status}")
    (run / "meta.yaml").write_text("\n".join(fields) + "\n", encoding="utf-8")
    if tag_name is not None:
        (run / "tags").mkdir()
        (run / "tags/mlflow.runName").write_text(tag_name, encoding="utf-8")
    for key, value in (params or {}).items():
        target = run / "params" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
    for key, value in (metrics or {}).items():
        target = run / "metrics" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
    for key, value in (artifacts or {}).items():
        target = run / "artifacts" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    return run


@pytest.fixture
def evidence_store(tmp_path: Path) -> Path:
    """Create a multi-experiment store with healthy, ambiguous, and sparse runs."""
    store = tmp_path / "mlruns"
    for experiment, name in (("10", "Experiment Ten"), ("20", "Experiment Twenty")):
        directory = store / experiment
        directory.mkdir(parents=True)
        (directory / "meta.yaml").write_text(f"name: '{name}'\n", encoding="utf-8")

    score = (
        "1 0.1 0\n"
        "2 0.2 1\n"
        "3 0.3 2\n"
        "4 0.4 3\n"
        "5 0.5 4\n"
        "6 0.6 5\n"
        "7 0.9 2\n"  # duplicate step: this later append must win
        "malformed\n"
        "8 NaN 6\n"
        "9 0.7 -1\n"
        "10 0.8 1.5\n"
        "11 0.9 7 extra\n"
        "12 Infinity 8\n"
    )
    _write_run(
        store,
        "10",
        "alpha-directory",
        run_id="abc111alpha",
        tag_name="Baseline Alpha",
        meta_name="Metadata Alpha",
        params={"zeta": "last", "nested/group/value": "first"},
        metrics={
            "eval/score": score,
            "eval/a-b": "1 1.0 0\n",
            "eval/ab": "1 2.0 0\n",
        },
        artifacts={"models/model.zip": b"12345", "notes.txt": b"abc"},
    )
    _write_run(
        store,
        "20",
        "candidate-directory",
        run_id="def222candidate",
        tag_name="Candidate Beta",
        meta_name="Metadata Beta",
        start=1704153600000,
        end=None,
        params={"learning_rate": "0.0001"},
        metrics={"eval/score": "1 0.25 0\n2 0.75 2\n"},
        artifacts={"candidate.bin": b"candidate"},
    )
    # These runs create both ID-prefix and exact-name ambiguity across experiments.
    _write_run(store, "10", "prefix-one", run_id="sameprefix-one", tag_name="Duplicate Name")
    _write_run(store, "20", "prefix-two", run_id="sameprefix-two", tag_name="Duplicate Name")
    # Metadata-name fallback and missing optional trees exercise damaged/sparse isolation.
    _write_run(
        store,
        "20",
        "sparse444",
        run_id="sparse444",
        meta_name="Sparse Fallback",
        start="broken",
        end=None,
        status=None,
    )
    return store


def _evidence(outcome: dict) -> dict:
    assert outcome["ok"], outcome
    return outcome["result"]["details"]["evidence"]


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_real_registration_standard_json_multi_run_order_and_determinism(evidence_store: Path):
    """The auto-discovered tool emits canonical JSON and preserves caller labels/order."""
    query = {
        "runs": [
            {
                "label": "candidate",
                "runId": "DEF222",
                "runName": "candidate beta",
                "experimentId": "20",
            },
            {"label": "baseline", "runId": "ABC111", "experimentId": "10"},
        ],
        "metricKeys": ["eval/score", "missing/metric"],
    }
    before = _tree_digest(evidence_store)
    response = _call_tool(evidence_store, [{"query": query}, {"query": query}], auto_discover=True)
    assert response["registrationCount"] == 1
    assert response["names"].count("extract_run_evidence") == 1
    assert "one-off MLflow evidence parsers" in response["description"]
    normalized_description = response["description"].replace(",", "")
    assert "2000" in normalized_description and "50" in normalized_description
    assert response["promptSnippet"] and len(response["promptGuidelines"]) == 3
    schema = response["parameters"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["runs"]["minItems"] == 1
    assert schema["properties"]["runs"]["maxItems"] == 32

    first, second = response["outcomes"]
    assert first["textMatchesCanonical"] and first["textEndsNewline"]
    assert second["textMatchesCanonical"]
    assert first["canonicalSha256"] == second["canonicalSha256"]
    assert first["result"]["details"]["evidence"] == second["result"]["details"]["evidence"]
    evidence = _evidence(first)
    assert evidence["schemaVersion"] == 1
    assert evidence["window"] == {"mode": "recent", "recentCount": 5}
    assert [run["label"] for run in evidence["runs"]] == ["candidate", "baseline"]
    assert [run["runId"] for run in evidence["runs"]] == ["def222candidate", "abc111alpha"]
    assert before == _tree_digest(evidence_store), "the MLflow fixture must remain read-only"


def test_mlflow_metadata_nested_files_and_metric_record_semantics(evidence_store: Path):
    """Nested params/metrics/artifacts and malformed/duplicate samples follow the contract."""
    response = _call_tool(
        evidence_store,
        [{"query": {"runs": [{"label": "base", "runName": "BASELINE ALPHA"}], "metricKeys": ["eval/score"]}}],
    )
    run = _evidence(response["outcomes"][0])["runs"][0]
    assert run["experimentId"] == "10" and run["experimentName"] == "Experiment Ten"
    assert run["directoryName"] == "alpha-directory"
    assert run["tagRunName"] == "Baseline Alpha"
    assert run["metadataRunName"] == "Metadata Alpha"
    assert run["runName"] == "Baseline Alpha"
    assert run["status"] == "FINISHED"
    assert run["startTime"] == "2024-01-01T00:00:00.000Z"
    assert run["endTime"] == "2024-01-01T01:00:00.000Z"
    assert list(run["parameters"]) == ["nested/group/value", "zeta"]
    assert run["parameterCount"] == 2
    assert run["artifacts"] == [
        {"path": "models/model.zip", "bytes": 5},
        {"path": "notes.txt", "bytes": 3},
    ]
    metric = run["metrics"]["eval/score"]
    assert metric["rawValidCount"] == 7
    assert metric["uniqueStepCount"] == 6
    assert metric["invalidLineCount"] == 6
    assert metric["duplicateStepCount"] == 1
    assert metric["firstSample"] == {"timestamp": 1, "value": 0.1, "step": 0}
    assert metric["latestSample"] == {"timestamp": 6, "value": 0.6, "step": 5}
    assert metric["minStep"] == 0 and metric["maxStep"] == 5
    assert metric["window"]["steps"] == [1, 2, 3, 4, 5]
    assert metric["window"]["values"] == [0.2, 0.9, 0.4, 0.5, 0.6]
    assert metric["window"]["presentCount"] == 5
    assert metric["window"]["missingSteps"] == []
    assert metric["window"]["mean"] == pytest.approx(0.52)


def test_aligned_missing_windows_sparse_run_and_empty_projection(evidence_store: Path):
    """Aligned windows null-fill absent steps while sparse selected runs stay represented."""
    response = _call_tool(
        evidence_store,
        [
            {
                "query": {
                    "runs": [
                        {"label": "base", "runId": "abc111alpha"},
                        {"label": "sparse", "runName": "sparse fallback", "experimentId": "20"},
                    ],
                    "metricKeys": ["eval/score", "does/not/exist"],
                    "alignedSteps": [0, 2, 7],
                }
            },
            {
                "query": {
                    "runs": [{"label": "base", "runId": "abc111alpha"}],
                    "metricKeys": [],
                    "recentCount": 2,
                }
            },
        ],
    )
    aligned = _evidence(response["outcomes"][0])
    assert aligned["window"] == {"mode": "aligned", "alignedSteps": [0, 2, 7]}
    base, sparse = aligned["runs"]
    window = base["metrics"]["eval/score"]["window"]
    assert window == {
        "steps": [0, 2, 7],
        "values": [0.1, 0.9, None],
        "presentCount": 2,
        "missingSteps": [7],
        "mean": 0.5,
        "min": 0.1,
        "max": 0.9,
    }
    assert base["missingMetricKeys"] == ["does/not/exist"]
    assert sparse["metrics"] == {}
    assert sparse["missingMetricKeys"] == ["does/not/exist", "eval/score"]
    assert sparse["startTimeMs"] is None and sparse["startTime"] is None
    assert "Malformed start_time in meta.yaml." in sparse["warnings"]
    assert "Missing or unreadable params directory." in sparse["warnings"]
    assert "Missing or unreadable metrics directory." in sparse["warnings"]
    assert "Missing or unreadable artifacts directory." in sparse["warnings"]

    empty = _evidence(response["outcomes"][1])
    assert empty["window"] == {"mode": "recent", "recentCount": 2}
    assert empty["runs"][0]["metricFileCount"] == 3
    assert empty["runs"][0]["metricCount"] == 0
    assert empty["runs"][0]["metrics"] == {}


def test_identity_alias_and_strict_input_errors_are_actionable(evidence_store: Path):
    """Ambiguous/missing/mismatched identities and aliases fail instead of choosing evidence."""
    calls = [
        {"query": {"runs": [{"label": "x", "runId": "absent"}]}},
        {"query": {"runs": [{"label": "x", "runId": "sameprefix"}]}},
        {"query": {"runs": [{"label": "x", "runName": "duplicate name"}]}},
        {
            "query": {
                "runs": [{"label": "x", "runId": "abc111alpha", "runName": "Candidate Beta"}]
            }
        },
        {
            "query": {
                "runs": [{"label": "Same", "runId": "abc111alpha"}, {"label": "same", "runId": "def222candidate"}]
            }
        },
        {
            "query": {
                "runs": [{"label": "x", "runId": "abc111alpha"}],
                "metricKeys": ["evalab"],
            }
        },
        {"query": {"runs": [{"label": "x"}]}},
        {"query": {"runs": [{"label": "x", "runId": "abc111alpha", "extra": True}]}},
        {"query": {"runs": [{"label": "x", "runId": "abc111alpha"}], "extra": True}},
    ]
    outcomes = _call_tool(evidence_store, calls)["outcomes"]
    assert all(not outcome["ok"] for outcome in outcomes)
    expected = [
        "no run matches",
        "ambiguous",
        "ambiguous",
        "different runs",
        "Duplicate label",
        "Ambiguous metric key",
        "at least one",
        "unknown field",
        "unknown field",
    ]
    for outcome, fragment in zip(outcomes, expected, strict=True):
        assert fragment.lower() in outcome["error"].lower(), outcome


def test_window_validation_and_cancellation(evidence_store: Path):
    """Invalid window configurations fail clearly and an abort signal cancels real execution."""
    identity = {"runs": [{"label": "x", "runId": "abc111alpha"}]}
    calls = [
        {"query": {**identity, "alignedSteps": []}},
        {"query": {**identity, "alignedSteps": [0, 0]}},
        {"query": {**identity, "alignedSteps": [1, 0]}},
        {"query": {**identity, "alignedSteps": [-1]}},
        {"query": {**identity, "recentCount": 0}},
        {"query": {**identity, "alignedSteps": [0], "recentCount": 1}},
        {"query": identity, "preAbort": True},
        {"query": identity, "abortDelayMs": 0},
    ]
    outcomes = _call_tool(evidence_store, calls)["outcomes"]
    assert all(not outcome["ok"] for outcome in outcomes)
    assert all("alignedsteps" in outcome["error"].lower() for outcome in outcomes[:4])
    assert "positive integer" in outcomes[4]["error"]
    assert "must not be supplied" in outcomes[5]["error"]
    assert outcomes[6]["name"] == "AbortError" and "cancelled" in outcomes[6]["error"]
    assert outcomes[7]["name"] == "AbortError" and "cancelled" in outcomes[7]["error"]


def test_oversized_output_truncates_and_recovers_complete_report(tmp_path: Path):
    """Oversized evidence respects Pi limits and the temporary report equals canonical JSON."""
    store = tmp_path / "large-mlruns"
    (store / "99").mkdir(parents=True)
    (store / "99/meta.yaml").write_text("name: large\n", encoding="utf-8")
    metrics = {
        f"nested/metric_{index:03d}": "\n".join(
            f"{step} {index + step / 1000:.6f} {step}" for step in range(12)
        )
        + "\n"
        for index in range(80)
    }
    _write_run(store, "99", "huge-run", run_id="huge-run-id", tag_name="Huge Run", metrics=metrics)
    outcome = _call_tool(
        store,
        [{"query": {"runs": [{"label": "huge", "runId": "huge-run-id"}]}}],
        timeout=60,
    )["outcomes"][0]
    assert outcome["ok"]
    details = outcome["result"]["details"]
    assert details["truncated"] is True
    assert details["fullReportPath"].endswith("evidence.json")
    assert outcome["report"]["basename"] == "evidence.json"
    assert outcome["report"]["matchesCanonical"] is True
    assert outcome["report"]["sha256"] == outcome["canonicalSha256"]
    assert outcome["report"]["bytes"] > 50 * 1024 or outcome["report"]["lines"] > 2000
    text = outcome["result"]["content"][0]["text"]
    assert "Evidence truncated" in text and details["fullReportPath"] in text
    assert outcome["textBytes"] <= 50 * 1024
    assert outcome["textLines"] <= 2000
    assert details["evidence"]["runs"][0]["metricCount"] == 80
