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


def _call_tool(
    store: Path,
    query: dict,
    *,
    expect_error: bool = False,
    auto_discover: bool = False,
    abort: bool = False,
) -> dict:
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
  const loaderModule = await import(pathToFileURL(p.loader).href);
  const loaded = p.autoDiscover
    ? await loaderModule.discoverAndLoadExtensions([], p.cwd, p.agentDir)
    : await loaderModule.loadExtensions([p.extension], p.cwd);
  if (loaded.errors.length) throw new Error(JSON.stringify(loaded.errors));
  const names = loaded.extensions.flatMap((extension) => [...extension.tools.keys()]);
  const registrations = loaded.extensions
    .map((extension) => extension.tools.get('find_mlruns_runs'))
    .filter(Boolean);
  if (registrations.length !== 1) {
    throw new Error(`find_mlruns_runs registrations: ${registrations.length}; tools: ${names.join(',')}`);
  }
  const tool = registrations[0].definition;
  const controller = p.abort ? new AbortController() : undefined;
  controller?.abort();
  const result = await tool.execute(
    'acceptance', p.query, controller?.signal, undefined, { cwd: p.cwd }
  );
  let reportStats = null;
  if (result.details?.fullReportPath) {
    const report = await readFile(result.details.fullReportPath, 'utf8');
    reportStats = {
      length: report.length,
      lines: report.split(/\r?\n/).length,
      hasHugeValue: report.includes('Z'.repeat(1000)),
      hasManyLinesValue: report.includes(['value', 'value', 'value'].join('\n')),
      hasHistoryTail: report.includes('timestamp=2499, step=2499'),
    };
    await rm(dirname(result.details.fullReportPath), { recursive: true, force: true });
  }
  console.log(JSON.stringify({
    ok: true,
    names,
    description: tool.description,
    parameters: tool.parameters,
    result,
    reportStats,
  }));
} catch (error) {
  console.log(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }));
}
"""
    agent_dir = store.parent / "empty-pi-agent"
    agent_dir.mkdir(exist_ok=True)
    payload = {
        "loader": _node_path(loader),
        "extension": _node_path(EXTENSION),
        "cwd": _node_path(ROOT),
        "agentDir": _node_path(agent_dir),
        "autoDiscover": auto_discover,
        "abort": abort,
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


def _run(response: dict, run_id: str) -> dict:
    return next(run for run in response["result"]["details"]["runs"] if run["runId"] == run_id)


@pytest.fixture
def history_store(tmp_path: Path) -> Path:
    """Build independent histories whose four rank modes produce distinct orders."""
    store = tmp_path / "history-mlruns"
    exp = store / "42"
    exp.mkdir(parents=True)
    (exp / "meta.yaml").write_text("name: history-evaluation\n", encoding="utf-8")
    _write_run(
        store,
        "42",
        "alpha-dir",
        run_id="history-alpha-aaa",
        name="History Candidate Alpha",
        start=1704844800000,
        end=1704848400000,
        params={
            "group": "blue",
            "depth": "1",
            "alpha_only": "yes",
            "optim/learning-rate": "0.01",
        },
        metrics={
            # Malformed/non-finite rows are interspersed so recent windows must
            # count only valid appended records, not physical file lines.
            "eval/score": (
                "10 0.0 0\n"
                "partial row\n"
                "11 0.2 1\n"
                "NaN 99 50\n"
                "12 0.9 2\n"
                "13 Infinity 3\n"
                "14 0.95 4 extra\n"
            ),
            "eval/tie": "1 0.5 0\n",
            "train/aux": "1 10 0\n",
        },
    )
    _write_run(
        store,
        "42",
        "beta-dir",
        run_id="history-beta-bbb",
        name="History Candidate Beta",
        start=1704931200000,
        end=1704934800000,
        params={"group": "blue", "depth": "2", "beta_only": "yes"},
        metrics={
            "eval/score": "20 1.2 0\n21 0.6 1\n22 0.8 2\n",
            "eval/tie": "1 0.5 0\n",
            "train/aux": "1 20 0\n",
        },
    )
    _write_run(
        store,
        "42",
        "gamma-dir",
        run_id="history-gamma-ccc",
        name="History Candidate Gamma",
        start=1705017600000,
        end=1705021200000,
        params={"group": "blue", "depth": "3", "gamma_only": "yes"},
        metrics={
            "eval/score": "30 -1.0 0\n31 1.1 1\n32 0.7 2\n",
            "train/aux": "1 30 0\n",
        },
    )
    # Sparse and malformed-only runs prove schema drift isolation and that
    # missing ranked/predicate metrics do not poison otherwise valid runs.
    _write_run(
        store,
        "42",
        "malformed-dir",
        run_id="history-malformed-ddd",
        name="History Candidate Malformed",
        start=1705104000000,
        params={"group": "blue", "malformed_only": "yes"},
        metrics={"eval/score": "not a sample\n1 NaN 0\n2 0.4\n"},
    )
    _write_run(
        store,
        "42",
        "sparse-dir",
        run_id="history-sparse-eee",
        name="History Candidate Sparse",
        start=1705190400000,
        params=None,
        metrics=None,
    )
    return store


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


def test_real_loader_exposes_history_options_and_opt_in_defaults(history_store: Path):
    # The actual Pi loader must expose exactly one tool whose schema documents
    # every optional control without changing the default latest-only contract.
    response = _call_tool(
        history_store, {"runId": "history-alpha"}, auto_discover=True
    )
    assert response["names"].count("find_mlruns_runs") == 1
    properties = response["parameters"]["properties"]
    assert {
        "metricFilters",
        "rankMode",
        "rankRecentCount",
        "returnMetrics",
        "returnParameters",
        "metricHistory",
        "metricHistoryLimit",
    } <= properties.keys()
    assert "default to the latest valid sample" in response["description"]
    assert "absent returns all" in properties["returnMetrics"]["description"]
    metric_filter_properties = properties["metricFilters"]["items"]["properties"]
    assert {"historyMode", "recentCount"} <= metric_filter_properties.keys()
    assert metric_filter_properties["recentCount"]["minimum"] == 1
    assert properties["rankRecentCount"]["minimum"] == 1
    assert properties["metricHistoryLimit"]["minimum"] == 1

    run = response["result"]["details"]["runs"][0]
    assert run["runId"] == "history-alpha-aaa"
    assert run["runName"] == "History Candidate Alpha"
    assert list(run["parameters"]) == ["alpha_only", "depth", "group", "optim/learning-rate"]
    assert list(run["metrics"]) == ["eval/score", "eval/tie", "train/aux"]
    assert run["metrics"]["eval/score"] == {"value": 0.9, "timestamp": 12, "step": 2}
    assert "metricHistories" not in run
    assert "Metric histories:" not in response["result"]["content"][0]["text"]


def test_full_and_recent_history_filters_are_existential_over_valid_samples(history_store: Path):
    # Whole-history finds old peaks, whereas fixed recent windows use the final
    # N valid records after malformed and non-finite rows have been discarded.
    all_history = _call_tool(
        history_store,
        {
            "metricFilters": [
                {"key": "evalscore", "operator": "gt", "value": 1.0, "historyMode": "all"}
            ],
            "limit": 10,
        },
    )
    assert _ids(all_history) == ["history-gamma-ccc", "history-beta-bbb"]

    recent_larger_than_history = _call_tool(
        history_store,
        {
            "metricFilters": [
                {
                    "key": "eval/score",
                    "operator": "gt",
                    "value": 1.0,
                    "historyMode": "recent",
                    "recentCount": 10,
                }
            ],
            "limit": 10,
        },
    )
    assert _ids(recent_larger_than_history) == ["history-gamma-ccc", "history-beta-bbb"]

    recent_two = _call_tool(
        history_store,
        {
            "metricFilters": [
                {
                    "key": "eval/score",
                    "operator": "gt",
                    "value": 1.0,
                    "historyMode": "recent",
                    "recentCount": 2,
                }
            ],
            "limit": 10,
        },
    )
    assert _ids(recent_two) == ["history-gamma-ccc"]
    recent_one = _call_tool(
        history_store,
        {
            "metricFilters": [
                {
                    "key": "eval/score",
                    "operator": "gt",
                    "value": 1.0,
                    "historyMode": "recent",
                    "recentCount": 1,
                }
            ]
        },
    )
    assert _ids(recent_one) == []

    # Omitting historyMode remains latest-only, including the existential `ne`
    # interpretation documented for history scopes.
    latest = _call_tool(
        history_store,
        {"metricFilters": [{"key": "eval/score", "operator": "gt", "value": 0.75}], "limit": 10},
    )
    assert _ids(latest) == ["history-beta-bbb", "history-alpha-aaa"]
    not_equal = _call_tool(
        history_store,
        {
            "metricFilters": [
                {"key": "eval/score", "operator": "ne", "value": 0.9, "historyMode": "all"}
            ],
            "runId": "history-alpha",
        },
    )
    assert _ids(not_equal) == ["history-alpha-aaa"]


def test_every_rank_mode_and_order_uses_the_requested_history_score(history_store: Path):
    # Values were chosen so latest/max/min/recent-mean produce four distinct
    # permutations, proving that rankMode changes the score rather than display.
    cases = [
        ({"rankMode": "latest", "rankOrder": "desc"}, ["history-alpha-aaa", "history-beta-bbb", "history-gamma-ccc"]),
        ({"rankMode": "max", "rankOrder": "desc"}, ["history-beta-bbb", "history-gamma-ccc", "history-alpha-aaa"]),
        ({"rankMode": "min", "rankOrder": "asc"}, ["history-gamma-ccc", "history-alpha-aaa", "history-beta-bbb"]),
        (
            {"rankMode": "recent_mean", "rankRecentCount": 2, "rankOrder": "desc"},
            ["history-gamma-ccc", "history-beta-bbb", "history-alpha-aaa"],
        ),
    ]
    for options, expected in cases:
        response = _call_tool(
            history_store,
            {"rankByMetric": "evalscore", "limit": 10, **options},
        )
        assert _ids(response) == expected

    ascending_latest = _call_tool(
        history_store,
        {"rankByMetric": "eval/score", "rankOrder": "asc", "limit": 10},
    )
    assert _ids(ascending_latest) == ["history-gamma-ccc", "history-beta-bbb", "history-alpha-aaa"]
    # Equal scores retain the existing newest-first deterministic tie breaker.
    tied = _call_tool(history_store, {"rankByMetric": "eval/tie", "limit": 10})
    assert _ids(tied) == ["history-beta-bbb", "history-alpha-aaa"]


def test_projections_are_post_selection_and_identity_is_always_human_visible(history_store: Path):
    # Selected aliases project per-run maps; missing projection-only keys warn
    # but never exclude sparse or schema-drifted runs.
    selected = _call_tool(
        history_store,
        {
            "returnMetrics": ["evalscore", "missing_metric"],
            "returnParameters": ["depth", "alpha_only", "optimlearningrate"],
            "limit": 10,
        },
    )
    assert len(_ids(selected)) == 5
    alpha = _run(selected, "history-alpha-aaa")
    beta = _run(selected, "history-beta-bbb")
    sparse = _run(selected, "history-sparse-eee")
    assert alpha["parameters"] == {
        "alpha_only": "yes",
        "depth": "1",
        "optim/learning-rate": "0.01",
    }
    assert list(alpha["metrics"]) == ["eval/score"]
    assert beta["parameters"] == {"depth": "2"}
    assert list(beta["metrics"]) == ["eval/score"]
    assert sparse["parameters"] == {} and sparse["metrics"] == {}
    assert any('Requested metric "missing_metric"' in warning for warning in alpha["warnings"])
    assert any('Requested parameter "alpha_only"' in warning for warning in beta["warnings"])
    text = selected["result"]["content"][0]["text"]
    for run in selected["result"]["details"]["runs"]:
        assert f"Run ID: {run['runId']}" in text
        assert f"Run name: {run['runName']}" in text
    assert "  - eval/score: 0.9 (timestamp=12, step=2)" in text
    assert "  - train/aux:" not in text

    # Empty arrays return no inventory while filters and ranking still use data.
    empty = _call_tool(
        history_store,
        {
            "parameterFilters": [{"key": "group", "value": "blue"}],
            "metricFilters": [{"key": "eval/score", "operator": "gt", "value": 0.75}],
            "rankByMetric": "eval/score",
            "returnMetrics": [],
            "returnParameters": [],
            "metricHistory": "all",
            "limit": 10,
        },
    )
    assert _ids(empty) == ["history-alpha-aaa", "history-beta-bbb"]
    for run in empty["result"]["details"]["runs"]:
        assert run["parameters"] == {}
        assert run["metrics"] == {}
        assert run["metricHistories"] == {}
    empty_text = empty["result"]["content"][0]["text"]
    assert empty_text.count("Parameters:\n  (none)") == 2
    assert empty_text.count("Metrics:\n  (none)") == 2
    assert empty_text.count("Metric histories:\n  (none)") == 2


def test_all_and_recent_returned_histories_preserve_latest_maps_and_text(history_store: Path):
    # Requested histories augment rather than replace the stable latest map, and
    # the human report displays the same valid records in append order.
    all_history = _call_tool(
        history_store,
        {
            "runId": "history-alpha",
            "returnMetrics": ["eval/score"],
            "returnParameters": [],
            "metricHistory": "all",
        },
    )
    run = all_history["result"]["details"]["runs"][0]
    expected = [
        {"value": 0, "timestamp": 10, "step": 0},
        {"value": 0.2, "timestamp": 11, "step": 1},
        {"value": 0.9, "timestamp": 12, "step": 2},
    ]
    assert run["metrics"] == {"eval/score": expected[-1]}
    assert run["metricHistories"] == {"eval/score": expected}
    text = all_history["result"]["content"][0]["text"]
    assert "  - eval/score: 0.9 (timestamp=12, step=2)" in text
    assert "    - 0 (timestamp=10, step=0)" in text
    assert "    - 0.9 (timestamp=12, step=2)" in text
    assert "Infinity" not in text and "NaN" not in text

    recent = _call_tool(
        history_store,
        {
            "runId": "history-alpha",
            "returnMetrics": ["eval/score"],
            "metricHistory": "recent",
            "metricHistoryLimit": 2,
        },
    )
    recent_run = recent["result"]["details"]["runs"][0]
    assert recent_run["metrics"] == {"eval/score": expected[-1]}
    assert recent_run["metricHistories"] == {"eval/score": expected[-2:]}
    recent_all = _call_tool(
        history_store,
        {
            "runId": "history-alpha",
            "returnMetrics": ["eval/score"],
            "metricHistory": "recent",
            "metricHistoryLimit": 10,
        },
    )
    assert recent_all["result"]["details"]["runs"][0]["metricHistories"] == {
        "eval/score": expected
    }


def test_all_criteria_compose_before_preselection_ranking_and_projection(history_store: Path):
    # Name/ID/experiment/inclusive times/parameter/history metric predicates are
    # arbitrary AND criteria. Projection is presentation-only after selection.
    combined = _call_tool(
        history_store,
        {
            "experimentId": "42",
            "experimentName": "HISTORY-EVALUATION",
            "runName": "candidate beta",
            "runId": "HISTORY-B",
            "startedAfter": "2024-01-11T00:00:00Z",
            "startedBefore": 1704931200,
            "endedAfter": 1704934800000,
            "endedBefore": "2024-01-11T01:00:00Z",
            "parameterFilters": [
                {"key": "group", "value": "blue"},
                {"key": "depth", "operator": "gte", "value": 2},
            ],
            "metricFilters": [
                {"key": "evalscore", "operator": "gt", "value": 1, "historyMode": "all"},
                {"key": "eval/score", "operator": "lt", "value": 1, "historyMode": "recent", "recentCount": 1},
            ],
            "returnMetrics": [],
            "returnParameters": [],
            "limit": 10,
        },
    )
    assert _ids(combined) == ["history-beta-bbb"]
    assert _run(combined, "history-beta-bbb")["metrics"] == {}

    # Filtering precedes newest preselection; ranking then cannot resurrect an
    # older high-max run, and multiple matches remain available without `latest`.
    multiple = _call_tool(
        history_store,
        {
            "parameterFilters": [{"key": "group", "value": "blue"}],
            "metricFilters": [
                {"key": "eval/score", "operator": "gt", "value": 1, "historyMode": "all"}
            ],
            "rankByMetric": "eval/score",
            "rankMode": "max",
            "limit": 10,
        },
    )
    assert _ids(multiple) == ["history-beta-bbb", "history-gamma-ccc"]
    preselected = _call_tool(
        history_store,
        {
            "metricFilters": [
                {"key": "eval/score", "operator": "gt", "value": 1, "historyMode": "all"}
            ],
            "latest": 1,
            "rankByMetric": "eval/score",
            "rankMode": "max",
        },
    )
    assert _ids(preselected) == ["history-gamma-ccc"]


def test_sparse_predicate_rank_and_projection_absence_have_distinct_semantics(history_store: Path):
    # Missing predicates fail only that run; missing rank metrics exclude it;
    # missing projections retain it with warnings. Malformed-only files are local.
    predicate = _call_tool(
        history_store,
        {"parameterFilters": [{"key": "depth", "operator": "gte", "value": 1}], "limit": 10},
    )
    assert set(_ids(predicate)) == {
        "history-alpha-aaa",
        "history-beta-bbb",
        "history-gamma-ccc",
    }
    ranked = _call_tool(history_store, {"rankByMetric": "eval/score", "limit": 10})
    assert set(_ids(ranked)) == {
        "history-alpha-aaa",
        "history-beta-bbb",
        "history-gamma-ccc",
    }
    projected = _call_tool(
        history_store,
        {"returnMetrics": ["eval/score"], "returnParameters": ["depth"], "limit": 10},
    )
    assert len(_ids(projected)) == 5
    malformed = _run(projected, "history-malformed-ddd")
    sparse = _run(projected, "history-sparse-eee")
    assert malformed["metrics"] == {} and sparse["metrics"] == {}
    assert any("Metric has no valid samples: eval/score" in warning for warning in malformed["warnings"])
    assert any('Requested metric "eval/score"' in warning for warning in malformed["warnings"])


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ({"metricFilters": [{"key": "eval/score", "value": 1, "historyMode": "recent"}]}, "recentCount"),
        ({"metricFilters": [{"key": "eval/score", "value": 1, "historyMode": "recent", "recentCount": 0}]}, "positive integer"),
        ({"metricFilters": [{"key": "eval/score", "value": 1, "historyMode": "recent", "recentCount": 1.5}]}, "positive integer"),
        ({"metricFilters": [{"key": "eval/score", "value": 1, "historyMode": "latest", "recentCount": 1}]}, "only valid"),
        ({"metricFilters": [{"key": "eval/score", "value": 1, "historyMode": "invalid"}]}, "historyMode"),
        ({"rankMode": "max"}, "requires rankByMetric"),
        ({"rankByMetric": "eval/score", "rankMode": "recent_mean"}, "rankRecentCount"),
        ({"rankByMetric": "eval/score", "rankMode": "recent_mean", "rankRecentCount": -2}, "positive integer"),
        ({"rankByMetric": "eval/score", "rankMode": "latest", "rankRecentCount": 2}, "only valid"),
        ({"rankByMetric": "eval/score", "rankMode": "invalid"}, "rankMode"),
        ({"metricHistory": "recent"}, "metricHistoryLimit"),
        ({"metricHistory": "recent", "metricHistoryLimit": 0}, "positive integer"),
        ({"metricHistory": "all", "metricHistoryLimit": 2}, "only valid"),
        ({"metricHistoryLimit": 2}, "only valid"),
        ({"metricHistory": "invalid"}, "metricHistory"),
    ],
)
def test_invalid_history_options_fail_actionably(history_store: Path, query: dict, message: str):
    # Contradictory modes/counts must fail clearly rather than silently changing scope.
    error = _call_tool(history_store, query, expect_error=True)
    assert message in error["error"]


def test_history_output_truncation_keeps_complete_details_and_full_report(history_store: Path):
    # Explicit full histories may exceed Pi's report cap; the structured result
    # remains complete and the fallback report contains samples beyond the cap.
    metric = history_store / "42" / "alpha-dir" / "metrics" / "eval" / "long_history"
    metric.write_text(
        "".join(f"{index} {index / 1000} {index}\n" for index in range(2500)),
        encoding="utf-8",
    )
    response = _call_tool(
        history_store,
        {
            "runId": "history-alpha",
            "returnMetrics": ["eval/long_history"],
            "metricHistory": "all",
        },
    )
    result = response["result"]
    run = result["details"]["runs"][0]
    assert len(run["metricHistories"]["eval/long_history"]) == 2500
    assert run["metricHistories"]["eval/long_history"][-1] == {
        "value": 2.499,
        "timestamp": 2499,
        "step": 2499,
    }
    assert result["details"]["truncated"] is True
    text = result["content"][0]["text"]
    assert len(text.splitlines()) <= 2000
    assert len(text.encode("utf-8")) <= 50 * 1024
    assert "Output truncated" in text and "Full report saved to:" in text
    assert response["reportStats"]["lines"] > 2500
    assert response["reportStats"]["hasHistoryTail"] is True


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


def test_real_loader_execution_honors_pre_cancelled_search(history_store: Path):
    # Cancellation remains observable through the registered tool execution path.
    error = _call_tool(history_store, {}, expect_error=True, abort=True)
    assert "cancelled" in error["error"].lower()


def test_extension_uses_filesystem_not_mlflow_api():
    source = EXTENSION.read_text(encoding="utf-8")
    assert "node:fs/promises" in source
    lowered_import_lines = "\n".join(line.lower() for line in source.splitlines() if "import" in line)
    assert 'from "mlflow"' not in lowered_import_lines
    assert "mlflowclient" not in source.lower()
