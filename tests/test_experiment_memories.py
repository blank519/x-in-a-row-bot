"""Black-box acceptance tests for the project-local experiment-memory Pi tools."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / ".pi" / "extensions" / "experiment_memories.ts"
PI_ROOT = Path("/mnt/c/Users/wange/AppData/Local/pi-node/current")
SECTIONS = (
    "breakthroughs_and_dead_ends",
    "correlations_and_patterns",
    "champion",
    "promising_directions",
)


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


def _invoke(cwd: Path, calls: list[tuple[str, dict]]) -> dict:
    """Load the production extension once and invoke registered definitions in order."""
    loader = PI_ROOT / "node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js"
    if not loader.exists():
        pytest.skip("Pi extension loader is unavailable")
    script = r"""
const { pathToFileURL } = await import('node:url');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const p = JSON.parse(input);
const { loadExtensions } = await import(pathToFileURL(p.loader).href);
const loaded = await loadExtensions([p.extension], p.cwd);
if (loaded.errors.length) throw new Error(JSON.stringify(loaded.errors));
if (loaded.extensions.length !== 1) throw new Error(`loaded ${loaded.extensions.length} extensions`);
const registrations = loaded.extensions[0].tools;
const names = [...registrations.keys()];
const schemas = Object.fromEntries([...registrations].map(([name, item]) => [name, item.definition.parameters]));
const results = [];
for (let i = 0; i < p.calls.length; i++) {
  const [name, params] = p.calls[i];
  const registration = registrations.get(name);
  if (!registration) {
    results.push({ ok: false, error: `unregistered tool ${name}` });
    continue;
  }
  try {
    const result = await registration.definition.execute(`acceptance-${i}`, params, undefined, undefined, { cwd: p.cwd });
    results.push({ ok: true, result });
  } catch (error) {
    results.push({ ok: false, error: error instanceof Error ? error.message : String(error) });
  }
}
console.log(JSON.stringify({ names, schemas, results }));
"""
    payload = {
        "loader": _node_path(loader),
        "extension": _node_path(EXTENSION),
        "cwd": _node_path(cwd),
        "calls": calls,
    }
    completed = subprocess.run(
        [_node_executable(), "--input-type=module", "-e", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _run(
    name: str,
    *,
    timestamp: str = "2026-09-12T16:43:19.363Z",
    run_id: str | None = "run-001",
    path: str | None = None,
    learning_rate: object = 0.001,
) -> dict:
    entry = {
        "run_name": name,
        "timestamp": timestamp,
        "modified_params": {"learning_rate": learning_rate, "Use_Mask": True},
        "goal_hypothesis": "The changed parameter improves defensive play.",
        "results_insights": [
            {
                "metric": "eval/average_win_rate",
                "outcome": "improved",
                "insight": "The candidate exceeded the baseline.",
            }
        ],
        "reasoning": ["Reward signal", "Opponent mix", "Sampling variance"],
        "tags": ["Warmup", "Defense"],
    }
    if run_id is not None:
        entry["mlruns_run_id"] = run_id
    if path is not None:
        entry["mlruns_path"] = path
    return entry


def _append(experiment: str, run: dict) -> tuple[str, dict]:
    return "write_experiment_memory", {
        "operation": "append_run",
        "experiment": experiment,
        "run": run,
    }


def _query(**filters: object) -> tuple[str, dict]:
    return "query_experiment_memories", {"mode": "runs", **filters}


def test_real_pi_registration_create_append_and_structured_readback(tmp_path: Path):
    # The real Pi loader must expose the strict-v2 timestamp schema and preserve each
    # canonical timestamp through append response, persistence, and structured query.
    first = _run("Baseline Run", run_id=None, path="mlruns/42/arbitrary-existing-run")
    second = _run("Candidate Run", run_id="candidate-002", learning_rate=0.0005)
    first_response = _invoke(
        tmp_path,
        [
            _append("safe-ticket_1", first),
            _query(experiment="SAFE-TICKET_1", run_name="baseline run"),
        ],
    )
    assert first_response["names"] == ["write_experiment_memory", "query_experiment_memories"]
    assert all(first_response["schemas"][name]["type"] == "object" for name in first_response["names"])
    run_schema = first_response["schemas"]["write_experiment_memory"]["properties"]["run"]
    assert "timestamp" in run_schema["required"]
    assert run_schema["properties"]["timestamp"]["pattern"] == (
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
    )
    assert all(item["ok"] for item in first_response["results"]), first_response
    assert first_response["results"][0]["result"]["details"]["entry"] == first
    first_match = first_response["results"][1]["result"]["details"]["matches"]
    assert first_match == [{"experiment": "safe-ticket_1", "entry": first}]
    target = tmp_path / "memories/experiments/safe-ticket_1.json"
    first_text = target.read_text()
    # Capture the pretty-printed run bytes themselves, not just a parsed copy.
    first_run_block = first_text.split('  "runs": [\n', 1)[1].rsplit("\n  ]", 1)[0]

    second_response = _invoke(
        tmp_path,
        [_append("safe-ticket_1", second), _query(experiment="safe-ticket_1")],
    )
    assert all(item["ok"] for item in second_response["results"]), second_response
    final_matches = second_response["results"][1]["result"]["details"]["matches"]
    assert final_matches == [
        {"experiment": "safe-ticket_1", "entry": first},
        {"experiment": "safe-ticket_1", "entry": second},
    ]
    final_text = target.read_text()
    assert f"{first_run_block},\n" in final_text
    document = json.loads(final_text)
    assert document == {"schema_version": 2, "experiment": "safe-ticket_1", "runs": [first, second]}
    assert target.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda run: run.pop("timestamp"), "timestamp"),
        (lambda run: run.update(timestamp=""), "timestamp"),
        (lambda run: run.update(timestamp="2026-09-12"), "timestamp"),
        (lambda run: run.update(timestamp="2026-09-12T18:43:19.363+02:00"), "timestamp"),
        (lambda run: run.update(timestamp="2026-9-12T16:43:19.363Z"), "timestamp"),
        (lambda run: run.update(timestamp="2026-02-30T16:43:19.363Z"), "timestamp"),
        (lambda run: run.update(timestamp="1789231399363"), "timestamp"),
        (lambda run: run.pop("run_name"), "run_name"),
        (lambda run: run.update(run_name="  "), "run_name"),
        (lambda run: run.pop("modified_params"), "modified_params"),
        (lambda run: run.update(modified_params={}), "modified_params"),
        (lambda run: run.pop("goal_hypothesis"), "goal_hypothesis"),
        (lambda run: run.pop("results_insights"), "results_insights"),
        (lambda run: run.update(results_insights=[]), "results_insights"),
        (lambda run: run["results_insights"][0].pop("outcome"), "outcome"),
        (lambda run: run.pop("reasoning"), "reasoning"),
        (lambda run: run.update(reasoning=["one", "two"]), "at least 3"),
        (lambda run: (run.pop("mlruns_run_id"), run.pop("mlruns_path", None)), "evidence locator"),
    ],
    ids=[
        "missing-timestamp",
        "empty-timestamp",
        "date-only-timestamp",
        "offset-timestamp",
        "non-padded-timestamp",
        "invalid-calendar-timestamp",
        "unix-string-timestamp",
        "missing-run-name",
        "empty-run-name",
        "missing-modified-params",
        "empty-modified-params",
        "missing-goal",
        "missing-insights",
        "empty-insights",
        "malformed-insight",
        "missing-reasoning",
        "two-reasons",
        "missing-evidence",
    ],
)
def test_required_run_validation_rejects_without_mutation(tmp_path: Path, mutation, message: str):
    # Every required run-field class, especially strict canonical run-start timestamps,
    # is validated before the canonical file can be changed.
    target = tmp_path / "memories/experiments/validation.json"
    valid = _run("persisted")
    created = _invoke(tmp_path, [_append("validation", valid)])
    assert created["results"][0]["ok"]
    before = target.read_bytes()
    invalid = _run("invalid", run_id="run-bad", path="mlruns/arbitrary")
    mutation(invalid)
    rejected = _invoke(tmp_path, [_append("validation", invalid)])
    result = rejected["results"][0]
    assert not result["ok"] and message.lower() in result["error"].lower()
    assert target.read_bytes() == before


def test_existing_document_validation_and_safe_names_prevent_clobber(tmp_path: Path):
    # Unsafe names cannot escape/collide with the store, and a malformed existing
    # document is reported rather than overwritten or silently repaired.
    calls = [_append(name, _run("unsafe")) for name in ("", "../escape", "a/b", "a\\b", "_TRENDS")]
    rejected = _invoke(tmp_path, calls)
    assert all(not item["ok"] for item in rejected["results"])
    assert not (tmp_path / "escape.json").exists()
    store = tmp_path / "memories/experiments"
    store.mkdir(parents=True, exist_ok=True)
    target = store / "corrupt.json"
    target.write_bytes(b'{"schema_version":2,"experiment":"corrupt","runs":"not-an-array"}\n')
    before = target.read_bytes()
    result = _invoke(tmp_path, [_append("corrupt", _run("new")), _query()])
    assert not result["results"][0]["ok"] and "runs" in result["results"][0]["error"]
    assert not result["results"][1]["ok"] and "malformed experiment memory" in result["results"][1]["error"].lower()
    assert target.read_bytes() == before


def test_schema_version_one_is_rejected_with_migration_guidance(tmp_path: Path):
    # Legacy v1 documents are migration input only; normal reads and writes must reject
    # them actionably and preserve their bytes rather than inventing timestamps.
    store = tmp_path / "memories/experiments"
    store.mkdir(parents=True)
    target = store / "legacy.json"
    legacy_run = _run("legacy")
    legacy_run.pop("timestamp")
    target.write_text(
        json.dumps({"schema_version": 1, "experiment": "legacy", "runs": [legacy_run]}) + "\n"
    )
    before = target.read_bytes()
    response = _invoke(tmp_path, [_query(experiment="legacy"), _append("legacy", _run("new"))])
    assert all(not item["ok"] for item in response["results"])
    assert all("migrate to schema 2" in item["error"].lower() for item in response["results"])
    assert target.read_bytes() == before


def test_all_structured_filters_are_exact_and_and_composed(tmp_path: Path):
    # Experiment/run/parameter/value/metric/outcome/tag filters use case-insensitive
    # exact matching, compose with AND semantics, and no-match queries succeed empty.
    matching = _run("Candidate Run", run_id="candidate-123", learning_rate=0.0005)
    distractor = _run("Other Run", run_id="other-456", learning_rate=0.001)
    distractor["results_insights"][0]["outcome"] = "regressed"
    distractor["tags"] = ["SelfPlay"]
    response = _invoke(
        tmp_path,
        [
            _append("filter-exp", matching),
            _append("other-exp", distractor),
            _query(
                experiment="FILTER-EXP",
                run_name="candidate run",
                modified_param="LEARNING_RATE",
                modified_param_value=0.0005,
                metric="EVAL/AVERAGE_WIN_RATE",
                outcome="IMPROVED",
                tag="DEFENSE",
            ),
            _query(experiment="filter-exp", tag="missing"),
            _query(run_name="Candidate"),
        ],
    )
    assert all(item["ok"] for item in response["results"]), response
    combined = response["results"][2]["result"]["details"]["matches"]
    assert combined == [{"experiment": "filter-exp", "entry": matching}]
    assert response["results"][3]["result"]["details"]["matches"] == []
    assert response["results"][4]["result"]["details"]["matches"] == []
    assert "Found 0 matching" in response["results"][3]["result"]["content"][0]["text"]


def test_trends_all_sections_direct_reads_and_single_section_preservation(tmp_path: Path):
    # The first update creates all four canonical sections; each section can be read
    # directly, and replacing one section preserves the other three byte-semantically.
    initial = {
        "breakthroughs_and_dead_ends": [{"event": "warmup worked"}],
        "correlations_and_patterns": {"block_reward": "positive"},
        "champion": {"run": "candidate-123", "params": {"lr": 0.0005}},
        "promising_directions": ["longer self-play"],
    }
    calls: list[tuple[str, dict]] = []
    for section in SECTIONS:
        calls.append(
            (
                "write_experiment_memory",
                {"operation": "update_trends_section", "section": section, "content": initial[section]},
            )
        )
    for section in SECTIONS:
        calls.append(("query_experiment_memories", {"mode": "trends_section", "section": section}))
    response = _invoke(tmp_path, calls)
    assert all(item["ok"] for item in response["results"]), response
    for index, section in enumerate(SECTIONS, start=4):
        details = response["results"][index]["result"]["details"]
        assert details["section"] == section and details["content"] == initial[section]
        assert set(details) == {"mode", "path", "section", "content"}
    target = tmp_path / "memories/experiments/_TRENDS.json"
    before = json.loads(target.read_text())
    replacement = {"run": "new-champion", "params": {"lr": 0.00025}}
    edited = _invoke(
        tmp_path,
        [
            (
                "write_experiment_memory",
                {"operation": "update_trends_section", "section": "champion", "content": replacement},
            ),
            ("query_experiment_memories", {"mode": "trends_section", "section": "champion"}),
        ],
    )
    assert all(item["ok"] for item in edited["results"]), edited
    after = json.loads(target.read_text())
    expected = deepcopy(before)
    expected["sections"]["champion"] = replacement
    assert after == expected
    assert edited["results"][1]["result"]["details"]["content"] == replacement


def test_recursive_trend_run_references_require_canonical_timestamps(tmp_path: Path):
    # Nested objects identified by run_id or an MLflow run path require canonical
    # timestamps; rejected updates leave the complete prior trends document unchanged.
    valid_reference = {
        "nested": [
            {
                "run_id": "candidate-123",
                "path": "mlruns/42/candidate-123",
                "timestamp": "2026-09-12T16:43:19.363Z",
            }
        ]
    }
    created = _invoke(
        tmp_path,
        [
            (
                "write_experiment_memory",
                {
                    "operation": "update_trends_section",
                    "section": "champion",
                    "content": valid_reference,
                },
            )
        ],
    )
    assert created["results"][0]["ok"], created
    target = tmp_path / "memories/experiments/_TRENDS.json"
    assert json.loads(target.read_text())["schema_version"] == 2
    before = target.read_bytes()

    invalid_contents = [
        {"nested": [{"run_id": "missing-time"}]},
        {"nested": [{"path": "mlruns/42/missing-time"}]},
        {
            "nested": [
                {"run_id": "bad-time", "timestamp": "2026-09-12T18:43:19.363+02:00"}
            ]
        },
    ]
    rejected = _invoke(
        tmp_path,
        [
            (
                "write_experiment_memory",
                {
                    "operation": "update_trends_section",
                    "section": "promising_directions",
                    "content": content,
                },
            )
            for content in invalid_contents
        ],
    )
    assert all(not item["ok"] and "timestamp" in item["error"] for item in rejected["results"])
    assert target.read_bytes() == before


def test_read_only_query_does_not_create_store(tmp_path: Path):
    # A no-match read in a fresh cwd is successful and has no filesystem side effect.
    response = _invoke(tmp_path, [_query(run_name="absent")])
    assert response["results"][0]["ok"]
    assert response["results"][0]["result"]["details"]["matches"] == []
    assert not (tmp_path / "memories").exists()
