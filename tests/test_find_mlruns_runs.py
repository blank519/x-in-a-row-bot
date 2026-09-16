"""Black-box acceptance tests for the project-local find_mlruns_runs Pi tool."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / ".pi" / "extensions" / "find_mlruns_runs.ts"
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


def _call_tool(store: Path, query: dict, *, expect_error: bool = False) -> dict:
    loader = PI_ROOT / "node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js"
    if not loader.exists():
        pytest.skip("Pi extension loader is unavailable")
    script = r"""
const { readFile, rm } = await import('node:fs/promises');
const { dirname } = await import('node:path');
const { pathToFileURL } = await import('node:url');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const p = JSON.parse(input);
try {
  const { loadExtensions } = await import(pathToFileURL(p.loader).href);
  const loaded = await loadExtensions([p.extension], p.cwd);
  if (loaded.errors.length) throw new Error(JSON.stringify(loaded.errors));
  if (loaded.extensions.length !== 1) throw new Error(`loaded ${loaded.extensions.length} extensions`);
  const names = [...loaded.extensions[0].tools.keys()];
  const registration = loaded.extensions[0].tools.get('find_mlruns_runs');
  if (!registration) throw new Error(`registered tools: ${names.join(',')}`);
  const tool = registration.definition;
  const result = await tool.execute('acceptance', p.query, undefined, undefined, { cwd: p.cwd });
  let reportStats = null;
  if (result.details?.fullReportPath) {
    const report = await readFile(result.details.fullReportPath, 'utf8');
    reportStats = {
      length: report.length,
      lines: report.split(/\r?\n/).length,
      hasHugeValue: report.includes('Z'.repeat(1000)),
      hasManyLinesValue: report.includes(['value', 'value', 'value'].join('\n')),
    };
    await rm(dirname(result.details.fullReportPath), { recursive: true, force: true });
  }
  console.log(JSON.stringify({ ok: true, names, result, reportStats }));
} catch (error) {
  console.log(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }));
}
"""
    payload = {
        "loader": _node_path(loader),
        "extension": _node_path(EXTENSION),
        "cwd": _node_path(ROOT),
        "query": {"mlrunsPath": _node_path(store), **query},
    }
    completed = subprocess.run(
        [_node_executable(), "--input-type=module", "-e", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout.strip().splitlines()[-1])
    if expect_error:
        assert not response["ok"], response
    else:
        assert response["ok"], response
    return response


def _write_run(
    store: Path,
    experiment: str,
    directory_id: str,
    *,
    run_id: str | None = None,
    name: str | None = None,
    meta_name: str | None = None,
    start: int | str | None = None,
    end: int | str | None = None,
    status: str | None = "FINISHED",
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
    if start is not None:
        fields.append(f"start_time: {start}")
    if end is not None:
        fields.append(f"end_time: {end}")
    if status is not None:
        fields.append(f"status: {status}")
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


@pytest.fixture
def mlflow_store(tmp_path: Path) -> Path:
    store = tmp_path / "mlruns"
    exp = store / "10"
    exp.mkdir(parents=True)
    (exp / "meta.yaml").write_text("name: ppo-gomoku\n", encoding="utf-8")
    (store / "20").mkdir()
    # Exact Jan 1/2/3 2024 UTC boundaries make inclusive range checks unambiguous.
    _write_run(
        store,
        "10",
        "alpha-dir",
        run_id="alpha111aaa",
        name="Baseline Alpha",
        start=1704067200000,
        end=1704070800000,
        params={"learning_rate": "0.0001", "nested/label": "WarmUp Alpha"},
        metrics={
            "eval/average_win_rate": "1 0.20 0\n2 0.40 1\n",
            "eval/GomokuOffensiveHeuristicPolicy/o_win_rate": "1 0.30 0\n",
            "train/loss": "1 9.0 0\n",
        },
    )
    _write_run(
        store,
        "10",
        "beta-dir",
        run_id="beta222bbb",
        name="Candidate BETA",
        start=1704153600000,
        end=None,
        params={"learning_rate": "0.0002", "nested/label": "WarmUp Beta", "empty": ""},
        metrics={
            "eval/average_win_rate": "1 0.50 0\n",
            "eval/GomokuOffensiveHeuristicPolicy/o_win_rate": "1 0.70 0\n",
        },
    )
    _write_run(
        store,
        "10",
        "gamma-dir",
        run_id="gamma333ccc",
        meta_name="Gamma: fallback name",
        start=1704240000000,
        end=1704243600000,
        params={"learning_rate": "0.0002", "batch_size": "64"},
        metrics={
            "eval/average_win_rate": "1 0.10 0\n2 broken 1\n",
            # Only complete finite timestamp/value/step records are valid. Scanning
            # backward must skip extra-field, non-finite, and partially written tails.
            "eval/GomokuOffensiveHeuristicPolicy/o_win_rate": (
                "3 0.80 2\n"
                "4 0.91 NaN\n"
                "NaN 0.92 3\n"
                "5 Infinity 4\n"
                "6 0.98 5 extra\n"
                "7 0.99\n"
            ),
            "empty_metric": "not a metric\n",
        },
    )
    # Displayable sparse/damaged run: no tags, params, metrics, end time, or valid start.
    _write_run(store, "20", "damaged444", start="broken", status=None)
    # A child directory with no run metadata must not abort or become a run.
    (store / "10" / "half-written").mkdir()
    return store


def _ids(response: dict) -> list[str]:
    return [run["runId"] for run in response["result"]["details"]["runs"]]


def test_real_pi_registration_combined_filters_and_complete_inventories(mlflow_store: Path):
    response = _call_tool(
        mlflow_store,
        {
            "experimentName": "PPO-GOMOKU",
            "runName": "candidate",
            "runId": "BETA22",
            "parameterFilters": [
                {"key": "learning_rate", "operator": "gte", "value": 0.0002},
                {"key": "nested-label", "operator": "contains", "value": "beta"},
            ],
            "metricFilters": [
                {"key": "eval/average_win_rate", "operator": "eq", "value": 0.5},
                {"key": "evalgomokuoffensiveheuristicpolicyowinrate", "operator": "gt", "value": 0.6},
            ],
            "limit": 10,
        },
    )
    assert response["names"] == ["find_mlruns_runs"]
    details = response["result"]["details"]
    assert details["count"] == 1
    run = details["runs"][0]
    assert run["experimentId"] == "10" and run["experimentName"] == "ppo-gomoku"
    assert run["runName"] == "Candidate BETA" and run["status"] == "FINISHED"
    assert run["endTime"] is None
    assert list(run["parameters"]) == ["empty", "learning_rate", "nested/label"]
    assert list(run["metrics"]) == [
        "eval/average_win_rate",
        "eval/GomokuOffensiveHeuristicPolicy/o_win_rate",
    ]
    text = response["result"]["content"][0]["text"]
    assert "Run ID: beta222bbb" in text and "- empty: (empty)" in text
    assert "timestamp=1, step=0" in text


def test_identifiers_latest_and_inclusive_date_ranges(mlflow_store: Path):
    assert _ids(_call_tool(mlflow_store, {"runId": "alpha111aaa"})) == ["alpha111aaa"]
    assert _ids(_call_tool(mlflow_store, {"runName": "ALPHA"})) == ["alpha111aaa"]
    # Date strings and Unix seconds/milliseconds are inclusive at both ends.
    bounded = _call_tool(
        mlflow_store,
        {"startedAfter": "2024-01-02", "startedBefore": 1704240000, "limit": 10},
    )
    assert _ids(bounded) == ["gamma333ccc", "beta222bbb"]
    ended = _call_tool(
        mlflow_store,
        {"endedAfter": 1704070800000, "endedBefore": "2024-01-03T01:00:00Z", "limit": 10},
    )
    assert _ids(ended) == ["gamma333ccc", "alpha111aaa"]
    assert _ids(_call_tool(mlflow_store, {"latest": 2})) == ["gamma333ccc", "beta222bbb"]


def test_arbitrary_metric_rankings_use_latest_valid_samples(mlflow_store: Path):
    average = _call_tool(mlflow_store, {"rankByMetric": "eval/average_win_rate", "limit": 10})
    assert _ids(average) == ["beta222bbb", "alpha111aaa", "gamma333ccc"]
    offensive = _call_tool(
        mlflow_store,
        {
            "rankByMetric": "eval/GomokuOffensiveHeuristicPolicy/o_win_rate",
            "rankOrder": "desc",
            "limit": 10,
        },
    )
    assert _ids(offensive) == ["gamma333ccc", "beta222bbb", "alpha111aaa"]
    sample = offensive["result"]["details"]["runs"][0]["metrics"][
        "eval/GomokuOffensiveHeuristicPolicy/o_win_rate"
    ]
    assert sample == {"value": 0.8, "timestamp": 3, "step": 2}


def test_missing_data_damaged_runs_empty_matches_and_actionable_errors(mlflow_store: Path):
    all_runs = _call_tool(mlflow_store, {"limit": 10})
    assert set(_ids(all_runs)) == {"alpha111aaa", "beta222bbb", "gamma333ccc", "damaged444"}
    damaged = next(run for run in all_runs["result"]["details"]["runs"] if run["runId"] == "damaged444")
    assert damaged["runName"] is None and damaged["endTime"] is None
    assert damaged["parameters"] == {} and damaged["metrics"] == {}
    assert any("Malformed start_time" in warning for warning in damaged["warnings"])
    empty = _call_tool(mlflow_store, {"metricFilters": [{"key": "missing", "value": 1}]})
    assert empty["result"]["details"]["count"] == 0
    assert "Found 0 matching" in empty["result"]["content"][0]["text"]
    for query, message in [
        ({"startedAfter": "definitely-not-a-date"}, "startedAfter"),
        ({"startedAfter": "2024-02-01", "startedBefore": "2024-01-01"}, "must not be later"),
        ({"limit": 0}, "limit"),
    ]:
        error = _call_tool(mlflow_store, query, expect_error=True)
        assert message in error["error"]


def test_ambiguous_alias_is_rejected_but_exact_key_works(mlflow_store: Path):
    run = mlflow_store / "10" / "alpha-dir"
    (run / "params/a-b").write_text("one", encoding="utf-8")
    (run / "params/a_b").write_text("two", encoding="utf-8")
    error = _call_tool(
        mlflow_store,
        {"runId": "alpha", "parameterFilters": [{"key": "ab", "value": "one"}]},
        expect_error=True,
    )
    assert "Ambiguous parameter key" in error["error"]
    assert "a-b" in error["error"] and "a_b" in error["error"]
    exact = _call_tool(
        mlflow_store,
        {"runId": "alpha", "parameterFilters": [{"key": "a-b", "value": "one"}]},
    )
    assert _ids(exact) == ["alpha111aaa"]


def test_output_truncation_keeps_structured_contract_and_full_report(mlflow_store: Path):
    params = mlflow_store / "10" / "alpha-dir" / "params"
    huge = params / "huge"
    huge.write_text("Z" * 60_000, encoding="utf-8")
    response = _call_tool(mlflow_store, {"runId": "alpha"})
    result = response["result"]
    assert result["details"]["truncated"] is True
    assert result["details"]["runs"][0]["parameters"]["huge"] == "Z" * 60_000
    text = result["content"][0]["text"]
    assert len(text.encode("utf-8")) <= 50 * 1024
    assert "Output truncated" in text
    assert "Full report saved to:" in text
    assert response["reportStats"]["length"] > 60_000
    assert response["reportStats"]["hasHugeValue"] is True

    # The annotation itself must not push a line-limited result past Pi's cap.
    huge.unlink()
    (params / "many_lines").write_text("\n".join(["value"] * 2_100), encoding="utf-8")
    line_response = _call_tool(mlflow_store, {"runId": "alpha"})
    line_result = line_response["result"]
    line_text = line_result["content"][0]["text"]
    expected_many_lines = "\n".join(["value"] * 2_100)
    assert line_result["details"]["truncated"] is True
    assert line_result["details"]["runs"][0]["parameters"]["many_lines"] == expected_many_lines
    assert len(line_text.splitlines()) <= 2_000
    assert len(line_text.encode("utf-8")) <= 50 * 1024
    assert "Output truncated" in line_text
    assert "Full report saved to:" in line_text
    assert line_response["reportStats"]["lines"] > 2_100
    assert line_response["reportStats"]["hasManyLinesValue"] is True


def test_extension_uses_filesystem_not_mlflow_api():
    source = EXTENSION.read_text(encoding="utf-8")
    assert "node:fs/promises" in source
    lowered_import_lines = "\n".join(line.lower() for line in source.splitlines() if "import" in line)
    assert 'from "mlflow"' not in lowered_import_lines
    assert "mlflowclient" not in source.lower()
