"""Black-box acceptance tests for the project-local experiment-memory Pi tools."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
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
EXPERIMENTS = {
    "all-heuristic-curriculum",
    "block-reward-coefficient-sweep",
    "block20-opening-retries",
    "center-mask-shaping-and-lr",
    "combined-heuristic-shaping",
    "defensive-opening-curriculum",
    "extended-heuristic-training",
    "fixed-defensive-curriculum-shaping",
    "full-combined-defense-weight",
    "long-block25-refinements",
    "mistake-rate-block-reward-interaction",
    "potential-reward-shaping",
    "raise_warmup_performance",
    "reproduction-and-forgetting",
    "warmup-duration-control",
}


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


def _append(
    experiment: str, run: dict, conclusion: str = "The candidate is promising but needs replication."
) -> tuple[str, dict]:
    return "write_experiment_memory", {
        "operation": "append_run",
        "experiment": experiment,
        "conclusion": conclusion,
        "run": run,
    }


def _query(**filters: object) -> tuple[str, dict]:
    return "query_experiment_memories", {"mode": "runs", **filters}


def _trend_update(section: str, content: object) -> tuple[str, dict]:
    return "write_experiment_memory", {
        "operation": "update_trends_section",
        "section": section,
        "content": content,
    }


def _collect_key(value: object, key: str) -> list[object]:
    found: list[object] = []
    if isinstance(value, dict):
        for current, child in value.items():
            if current == key:
                found.append(child)
            found.extend(_collect_key(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_key(child, key))
    return found


def test_real_pi_schema_v3_append_conclusion_update_and_structured_readback(tmp_path: Path):
    # The real registration must append schema-v3 runs with a required conclusion,
    # refresh it on append, and query/update it without returning or changing runs.
    first = _run("Baseline Run", run_id=None, path="mlruns/42/arbitrary-existing-run")
    second = _run("Candidate Run", run_id="candidate-002", learning_rate=0.0005)
    response = _invoke(
        tmp_path,
        [
            _append("safe-ticket_1", first, "Initial controlled conclusion."),
            _query(experiment="SAFE-TICKET_1", run_name="baseline run"),
            (
                "query_experiment_memories",
                {"mode": "experiment_conclusion", "experiment": "SAFE-TICKET_1"},
            ),
        ],
    )
    assert response["names"] == ["write_experiment_memory", "query_experiment_memories"]
    write_schema = response["schemas"]["write_experiment_memory"]
    assert write_schema["type"] == "object"
    assert "conclusion" in write_schema["properties"]
    assert "qualification" not in write_schema["properties"]
    run_schema = write_schema["properties"]["run"]
    assert "timestamp" in run_schema["required"]
    assert run_schema["properties"]["timestamp"]["pattern"] == (
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
    )
    assert all(item["ok"] for item in response["results"]), response
    assert response["results"][1]["result"]["details"]["matches"] == [
        {"experiment": "safe-ticket_1", "entry": first}
    ]
    conclusion_details = response["results"][2]["result"]["details"]
    assert conclusion_details["conclusion"] == "Initial controlled conclusion."
    assert "runs" not in conclusion_details

    target = tmp_path / "memories/experiments/safe-ticket_1.json"
    appended = _invoke(
        tmp_path,
        [_append("safe-ticket_1", second, "Refreshed after the candidate append.")],
    )
    assert appended["results"][0]["ok"], appended
    before_update = target.read_text()
    runs_text = before_update.split('  "runs": ', 1)[1]
    updated = _invoke(
        tmp_path,
        [
            (
                "write_experiment_memory",
                {
                    "operation": "update_experiment_conclusion",
                    "experiment": "safe-ticket_1",
                    "conclusion": "Final curated conclusion.",
                },
            ),
            (
                "query_experiment_memories",
                {"mode": "experiment_conclusion", "experiment": "safe-ticket_1"},
            ),
            _query(experiment="safe-ticket_1"),
        ],
    )
    assert all(item["ok"] for item in updated["results"]), updated
    document = json.loads(target.read_text())
    assert document == {
        "schema_version": 3,
        "experiment": "safe-ticket_1",
        "conclusion": "Final curated conclusion.",
        "runs": [first, second],
    }
    assert target.read_text().split('  "runs": ', 1)[1] == runs_text
    assert updated["results"][1]["result"]["details"]["conclusion"] == document["conclusion"]
    assert updated["results"][2]["result"]["details"]["matches"] == [
        {"experiment": "safe-ticket_1", "entry": first},
        {"experiment": "safe-ticket_1", "entry": second},
    ]
    assert target.read_bytes().endswith(b"\n")

    # Missing/blank conclusions fail before mutation for either write operation.
    before = target.read_bytes()
    rejected = _invoke(
        tmp_path,
        [
            (
                "write_experiment_memory",
                {"operation": "append_run", "experiment": "safe-ticket_1", "run": _run("bad")},
            ),
            (
                "write_experiment_memory",
                {
                    "operation": "update_experiment_conclusion",
                    "experiment": "safe-ticket_1",
                    "conclusion": "  ",
                },
            ),
        ],
    )
    assert all(not item["ok"] and "conclusion" in item["error"] for item in rejected["results"])
    assert target.read_bytes() == before


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
    # is validated before the canonical schema-v3 file can be changed.
    target = tmp_path / "memories/experiments/validation.json"
    created = _invoke(tmp_path, [_append("validation", _run("persisted"))])
    assert created["results"][0]["ok"]
    before = target.read_bytes()
    invalid = _run("invalid", run_id="run-bad", path="mlruns/arbitrary")
    mutation(invalid)
    rejected = _invoke(tmp_path, [_append("validation", invalid)])
    result = rejected["results"][0]
    assert not result["ok"] and message.lower() in result["error"].lower()
    assert target.read_bytes() == before


def test_existing_document_validation_safe_names_and_qualification_prevent_clobber(tmp_path: Path):
    # Unsafe names cannot escape the store, and malformed/qualification-bearing
    # experiment documents are rejected instead of overwritten or silently repaired.
    calls = [_append(name, _run("unsafe")) for name in ("", "../escape", "a/b", "a\\b", "_TRENDS")]
    rejected = _invoke(tmp_path, calls)
    assert all(not item["ok"] for item in rejected["results"])
    assert not (tmp_path / "escape.json").exists()
    store = tmp_path / "memories/experiments"
    store.mkdir(parents=True, exist_ok=True)
    target = store / "corrupt.json"
    target.write_bytes(
        b'{"schema_version":3,"experiment":"corrupt","conclusion":"old",'
        b'"qualification":"legacy","runs":[]}\n'
    )
    before = target.read_bytes()
    result = _invoke(tmp_path, [_append("corrupt", _run("new")), _query()])
    assert not result["results"][0]["ok"] and "exactly" in result["results"][0]["error"]
    assert not result["results"][1]["ok"] and "malformed experiment memory" in result["results"][1]["error"].lower()
    assert target.read_bytes() == before


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("kind", ["experiment", "trends"])
def test_legacy_documents_are_controlled_migration_input_only(
    tmp_path: Path, version: int, kind: str
):
    # V1 and v2 experiment/trends documents must be rejected actionably on normal
    # read and write paths, with the legacy bytes preserved rather than upgraded ad hoc.
    store = tmp_path / "memories/experiments"
    store.mkdir(parents=True)
    if kind == "experiment":
        target = store / "legacy.json"
        target.write_text(
            json.dumps({"schema_version": version, "experiment": "legacy", "runs": [_run("legacy")]})
            + "\n"
        )
        calls = [_query(experiment="legacy"), _append("legacy", _run("new"))]
    else:
        target = store / "_TRENDS.json"
        target.write_text(
            json.dumps(
                {
                    "schema_version": version,
                    "sections": {section: [] if section != "champion" else {} for section in SECTIONS},
                }
            )
            + "\n"
        )
        calls = [
            ("query_experiment_memories", {"mode": "trends_section", "section": "champion"}),
            _trend_update("champion", {"name": "replacement"}),
        ]
    before = target.read_bytes()
    response = _invoke(tmp_path, calls)
    assert all(not item["ok"] for item in response["results"])
    assert all(f"schema version {version}" in item["error"].lower() for item in response["results"])
    assert all("migrate to schema 3" in item["error"].lower() for item in response["results"])
    assert target.read_bytes() == before


def test_all_structured_filters_are_exact_and_and_composed(tmp_path: Path):
    # Existing experiment/run/parameter/metric/outcome/tag filters remain exact,
    # case-insensitive and AND-composed after the schema transition.
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
    assert response["results"][2]["result"]["details"]["matches"] == [
        {"experiment": "filter-exp", "entry": matching}
    ]
    assert response["results"][3]["result"]["details"]["matches"] == []
    assert response["results"][4]["result"]["details"]["matches"] == []


def test_compact_pointer_variants_historical_summaries_and_section_preservation(tmp_path: Path):
    # Both exact compact locator variants and evidence-free one-line history are
    # accepted; all four direct queries work and replacing one section preserves three.
    timestamp = "2026-09-12T16:43:19.363Z"
    initial = {
        "breakthroughs_and_dead_ends": {
            "historical_summaries": [
                "old-sweep, 8 runs, Jul 2026 — dead end: heuristic-heavy curricula failed."
            ]
        },
        "correlations_and_patterns": {
            "patterns": [
                {
                    "name": "experiment-level evidence",
                    "evidence": [{"experiment_name": "old-sweep", "timestamp": timestamp}],
                }
            ]
        },
        "champion": {
            "name": "candidate-123",
            "evidence": [{"run_id": "candidate-123", "timestamp": timestamp}],
        },
        "promising_directions": ["longer self-play"],
    }
    calls = [_trend_update(section, initial[section]) for section in SECTIONS]
    calls.extend(
        ("query_experiment_memories", {"mode": "trends_section", "section": section})
        for section in SECTIONS
    )
    response = _invoke(tmp_path, calls)
    assert all(item["ok"] for item in response["results"]), response
    for index, section in enumerate(SECTIONS, start=4):
        details = response["results"][index]["result"]["details"]
        assert details["section"] == section and details["content"] == initial[section]
        assert set(details) == {"mode", "path", "section", "content"}

    target = tmp_path / "memories/experiments/_TRENDS.json"
    before = json.loads(target.read_text())
    replacement = {
        "name": "new-champion",
        "evidence": [{"run_id": "new-run", "timestamp": timestamp}],
    }
    edited = _invoke(
        tmp_path,
        [
            _trend_update("champion", replacement),
            ("query_experiment_memories", {"mode": "trends_section", "section": "champion"}),
        ],
    )
    assert all(item["ok"] for item in edited["results"]), edited
    after = json.loads(target.read_text())
    expected = deepcopy(before)
    expected["sections"]["champion"] = replacement
    assert after == expected
    assert after["schema_version"] == 3
    assert edited["results"][1]["result"]["details"]["content"] == replacement


@pytest.mark.parametrize(
    "content",
    [
        {"evidence": []},
        {"evidence": "not-a-list"},
        {"evidence": [{"run_id": "missing-time"}]},
        {"evidence": [{"timestamp": "2026-09-12T16:43:19.363Z"}]},
        {
            "evidence": [
                {
                    "run_id": "both",
                    "experiment_name": "also-both",
                    "timestamp": "2026-09-12T16:43:19.363Z",
                }
            ]
        },
        {
            "evidence": [
                {
                    "run_id": "extra",
                    "timestamp": "2026-09-12T16:43:19.363Z",
                    "path": "mlruns/42/extra",
                }
            ]
        },
        {
            "evidence": [
                {
                    "run_id": "metric-map",
                    "timestamp": "2026-09-12T16:43:19.363Z",
                    "metrics": {"eval/average_win_rate": 0.5},
                }
            ]
        },
        {"evidence": [{"run_id": "", "timestamp": "2026-09-12T16:43:19.363Z"}]},
        {"evidence": [{"experiment_name": "../unsafe", "timestamp": "2026-09-12T16:43:19.363Z"}]},
        {"evidence": [{"run_id": "offset", "timestamp": "2026-09-12T18:43:19.363+02:00"}]},
        {"evidence": [{"run_id": "invalid-date", "timestamp": "2026-02-30T16:43:19.363Z"}]},
        {
            "evidence": [
                {"run_id": "valid", "timestamp": "2026-09-12T16:43:19.363Z"},
                "prose is not a pointer",
            ]
        },
        {"nested": {"qualification": "legacy reasoning"}},
        {"nested": {"support": []}},
        {"nested": {"terminal_evidence": {}}},
        {"historical_summaries": ["line one\nline two"]},
        {"historical_summaries": [{"summary": "object with attached evidence"}]},
    ],
    ids=[
        "empty-evidence",
        "non-list-evidence",
        "missing-timestamp",
        "missing-locator",
        "both-locators",
        "extra-path",
        "inline-metric-map",
        "empty-run-id",
        "unsafe-experiment-name",
        "noncanonical-offset",
        "invalid-calendar-date",
        "mixed-pointer-prose",
        "qualification",
        "support",
        "terminal-evidence",
        "multiline-history",
        "object-history",
    ],
)
def test_malformed_or_legacy_trend_shapes_reject_atomically(tmp_path: Path, content: object):
    # Every prohibited legacy or malformed compact shape fails before canonical
    # mutation, preserving all existing sections byte-for-byte with no temp residue.
    timestamp = "2026-09-12T16:43:19.363Z"
    created = _invoke(
        tmp_path,
        [
            _trend_update(
                "champion",
                {"evidence": [{"run_id": "good", "timestamp": timestamp}]},
            )
        ],
    )
    assert created["results"][0]["ok"], created
    target = tmp_path / "memories/experiments/_TRENDS.json"
    before = target.read_bytes()
    rejected = _invoke(tmp_path, [_trend_update("promising_directions", content)])
    assert not rejected["results"][0]["ok"], rejected
    assert target.read_bytes() == before
    assert list(target.parent.glob("*.tmp")) == []
    assert list(target.parent.glob(".*.tmp")) == []


def test_checked_in_schema_v3_migration_is_complete_and_evidence_linked():
    # The checked-in migration must retain the exact 61-run corpus, move all
    # conclusions, condense old chronology, and resolve all 50 compact timestamps.
    store = ROOT / "memories/experiments"
    experiment_paths = sorted(path for path in store.glob("*.json") if path.name != "_TRENDS.json")
    assert {path.stem for path in experiment_paths} == EXPERIMENTS
    documents = [(path.stem, json.loads(path.read_text())) for path in experiment_paths]
    assert all(document["schema_version"] == 3 for _, document in documents)
    assert all(set(document) == {"schema_version", "experiment", "conclusion", "runs"} for _, document in documents)
    assert all(document["experiment"] == name and document["conclusion"].strip() for name, document in documents)
    assert sum(len(document["runs"]) for _, document in documents) == 61
    run_corpus = [(name, document["runs"]) for name, document in documents]
    digest = hashlib.sha256(
        json.dumps(run_corpus, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert digest == "1d898786425b7d33fc68c47ce06b5c18e4a29183ab0bfd673c12d5e0eba73b69"

    conclusions = {name: document["conclusion"] for name, document in documents}
    # These phrases pin the four extra correlation/champion qualifications to their
    # mandated per-experiment destinations, in addition to every non-empty conclusion.
    assert "later, longer warmup configuration" in conclusions["raise_warmup_performance"]
    assert "not proven superiority against learned agents or across seeds" in conclusions["raise_warmup_performance"]
    assert "Do not generalize a universal negative effect" in conclusions["potential-reward-shaping"]
    assert "regimen-level correlations" in conclusions["reproduction-and-forgetting"]

    trends = json.loads((store / "_TRENDS.json").read_text())
    assert trends["schema_version"] == 3 and set(trends["sections"]) == set(SECTIONS)
    for forbidden in ("qualification", "support", "terminal_evidence", "path"):
        assert _collect_key(trends, forbidden) == []

    chronology = trends["sections"]["breakthroughs_and_dead_ends"]
    summaries = chronology["historical_summaries"]
    assert len(summaries) == 14
    assert all(
        isinstance(summary, str)
        and summary.strip()
        and "\n" not in summary
        and re.search(r"\b\d+ runs?\b", summary)
        and ("Jul" in summary or "Aug" in summary)
        and ("finding:" in summary or "dead end:" in summary)
        for summary in summaries
    )
    assert len(chronology["current_findings"]) == 1
    assert chronology["current_findings"][0]["group"] == "raise_warmup_performance"
    assert "Sep" in chronology["current_findings"][0]["period"]

    correlations = trends["sections"]["correlations_and_patterns"]
    directions = trends["sections"]["promising_directions"]
    champion = trends["sections"]["champion"]
    assert len(correlations["patterns"]) == 7
    assert len(correlations["global_caveats"]) == 4
    assert len(directions["prioritized"]) == 6
    assert len(directions["evaluation_requirements"]) == 5
    assert {
        "run_name",
        "status",
        "selection_criterion",
        "defining_parameters",
        "evidence",
        "other_strong_comparator",
        "why_not_highest_mean",
    } <= set(champion)

    evidence_lists = _collect_key(trends, "evidence")
    pointers = [pointer for evidence in evidence_lists for pointer in evidence]
    assert len(pointers) == 50
    assert sum(len(_collect_key(item, "evidence")[0]) for item in chronology["current_findings"]) == 4
    assert sum(len(item["evidence"]) for item in correlations["patterns"]) == 25
    assert len(champion["evidence"]) + len(champion["other_strong_comparator"]["evidence"]) + len(champion["why_not_highest_mean"]["evidence"]) == 3
    assert sum(len(item["evidence"]) for item in directions["prioritized"]) == 18
    assert all(set(pointer) in ({"run_id", "timestamp"}, {"experiment_name", "timestamp"}) for pointer in pointers)

    run_map = {
        run["mlruns_run_id"]: run["timestamp"]
        for _, document in documents
        for run in document["runs"]
        if "mlruns_run_id" in run
    }
    mlruns = ROOT / "mlruns/510583218657647424"
    for pointer in pointers:
        if "run_id" not in pointer:
            assert pointer["experiment_name"] in EXPERIMENTS
            continue
        assert run_map[pointer["run_id"]] == pointer["timestamp"]
        meta = mlruns / pointer["run_id"] / "meta.yaml"
        if meta.exists():
            match = re.search(r"^start_time:\s*(\d+)\s*$", meta.read_text(), re.MULTILINE)
            assert match
            raw_timestamp = datetime.fromtimestamp(
                int(match.group(1)) / 1000, tz=timezone.utc
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            assert raw_timestamp == pointer["timestamp"]


def test_real_extension_queries_every_checked_in_conclusion_and_trend_section():
    # The production loader must validate and directly query all migrated documents,
    # proving no checked-in file only looks valid to a separate test-side parser.
    calls: list[tuple[str, dict]] = [
        (
            "query_experiment_memories",
            {"mode": "experiment_conclusion", "experiment": experiment.upper()},
        )
        for experiment in sorted(EXPERIMENTS)
    ]
    calls.extend(
        ("query_experiment_memories", {"mode": "trends_section", "section": section})
        for section in SECTIONS
    )
    response = _invoke(ROOT, calls)
    assert all(item["ok"] for item in response["results"]), response
    assert all(
        response["results"][index]["result"]["details"]["conclusion"].strip()
        for index in range(len(EXPERIMENTS))
    )


def test_read_only_query_does_not_create_store(tmp_path: Path):
    # A no-match read in a fresh cwd is successful and has no filesystem side effect.
    response = _invoke(tmp_path, [_query(run_name="absent")])
    assert response["results"][0]["ok"]
    assert response["results"][0]["result"]["details"]["matches"] == []
    assert not (tmp_path / "memories").exists()
