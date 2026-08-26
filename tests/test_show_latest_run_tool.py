"""Tests for show_latest_run_tool against a synthetic mlruns/ file store."""

import show_latest_run_tool as tool


def _write_run(experiment_dir, run_id, run_name, start_time, end_time=None,
               lifecycle="active", params=None, metrics=None):
    run_dir = experiment_dir / run_id
    exp_meta = experiment_dir / "meta.yaml"
    if not exp_meta.exists():
        exp_meta.write_text(
            "experiment_id: '%s'\nname: test-exp\n" % experiment_dir.name,
            encoding="utf-8",
        )
    (run_dir / "params").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "tags").mkdir(parents=True, exist_ok=True)
    meta = [
        "artifact_uri: file://...",
        "end_time: %s" % (end_time if end_time is not None else 0),
        "experiment_id: '%s'" % experiment_dir.name,
        "lifecycle_stage: %s" % lifecycle,
        "run_id: %s" % run_id,
        "run_name: %s" % run_name,
        "run_uuid: %s" % run_id,
        "start_time: %s" % start_time,
        "status: 3",
        "tags: []",
        "user_id: test",
    ]
    (run_dir / "meta.yaml").write_text("\n".join(meta) + "\n", encoding="utf-8")
    for name, value in (params or {}).items():
        p = run_dir / "params" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(value), encoding="utf-8")
    for name, points in (metrics or {}).items():
        f = run_dir.joinpath("metrics", *name.split("/"))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join("%d %s %d\n" % (ts, v, step) for ts, v, step in points),
                     encoding="utf-8")
    return run_dir


def test_finds_latest_run_by_start_time(tmp_path):
    mlruns = tmp_path / "mlruns"
    models = mlruns / "models"
    models.mkdir(parents=True)
    exp = mlruns / "123"
    exp.mkdir()
    (exp / "meta.yaml").write_text(
        "experiment_id: '123'\nname: ppo-gomoku\n", encoding="utf-8"
    )
    _write_run(exp, "aaa111", "older-run", 1000, params={"lr": "0.001"})
    _write_run(exp, "bbb222", "newer-run", 2000,
               params={"lr": "0.002", "extra_param": "abc"})

    latest = tool.find_latest_run(mlruns)
    assert latest.run_id == "bbb222"
    assert latest.run_name == "newer-run"
    assert latest.experiment_name == "ppo-gomoku"
    assert latest.status_name == "FINISHED"


def test_params_and_metrics_parsed(tmp_path):
    mlruns = tmp_path / "mlruns"
    exp = mlruns / "123"
    exp.mkdir(parents=True)
    _write_run(
        exp,
        "run1",
        "run-name",
        100,
        params={"total_timesteps": "10240000", "p_heuristics": "[0.4]"},
        metrics={
            "train/reward": [(10, 1.0, 0), (20, 2.5, 1), (15, 3.0, 2)],
            "eval/SomeHeuristic/x_win_rate": [(100, 0.75, 0)],
        },
    )
    latest = tool.find_latest_run(mlruns)
    params = latest.params()
    assert params == {"total_timesteps": "10240000", "p_heuristics": "[0.4]"}
    metrics = latest.metrics()
    # Latest value is picked by timestamp (ts=20 -> 2.5), not file order.
    assert metrics["train/reward"] == (2.5, 3)
    assert metrics["eval/SomeHeuristic/x_win_rate"] == (0.75, 1)
    report = tool.format_report(latest)
    assert "run1" in report and "total_timesteps" in report
    assert "train/reward                  = 2.5  (3 points)" in report
    assert "eval/SomeHeuristic/x_win_rate = 0.75" in report


def test_missing_params_or_metrics_dirs(tmp_path):
    mlruns = tmp_path / "mlruns"
    exp = mlruns / "123"
    exp.mkdir(parents=True)
    (exp / "meta.yaml").write_text(
        "experiment_id: '123'\nname: test-exp\n", encoding="utf-8"
    )
    run_dir = exp / "run1"
    run_dir.mkdir()
    (run_dir / "meta.yaml").write_text(
        "run_id: run1\nrun_name: bare\nstart_time: 5\nstatus: 1\n",
        encoding="utf-8",
    )
    latest = tool.find_latest_run(mlruns)
    assert latest.params() == {}
    assert latest.metrics() == {}
    assert "(none logged)" in tool.format_report(latest)


def test_deletes_and_empty_store(tmp_path):
    mlruns = tmp_path / "mlruns"
    exp = mlruns / "123"
    exp.mkdir(parents=True)
    _write_run(exp, "deleted1", "old", 100, lifecycle="deleted")
    _write_run(exp, "active1", "new", 200)
    latest = tool.find_latest_run(mlruns)
    assert latest.run_id == "active1"


def test_only_deleted_runs_still_found(tmp_path):
    mlruns = tmp_path / "mlruns"
    exp = mlruns / "123"
    exp.mkdir(parents=True)
    _write_run(exp, "d1", "a", 100, lifecycle="deleted")
    _write_run(exp, "d2", "b", 200, lifecycle="deleted")
    latest = tool.find_latest_run(mlruns)
    assert latest.run_id == "d2"
