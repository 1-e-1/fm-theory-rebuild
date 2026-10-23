"""Ensure delivery preserves frozen data and actual historical attempt sources."""
import csv
import json
import tempfile
from pathlib import Path

from experiment import reporting


def put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))


def test_review_pack_keeps_prior_failure_source_and_metric_provenance(monkeypatch):
    # Use a dedicated workspace directory; the sandboxed Windows default pytest
    # numbered-temp symlink cannot always be removed during session teardown.
    project_root = reporting.ROOT.resolve()
    parent = (project_root / ".pytest_tmp").resolve()
    assert parent.is_relative_to(project_root)
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reporting_provenance_", dir=parent) as temporary:
        temporary_path = Path(temporary).resolve()
        assert temporary_path.is_relative_to(parent)
        _check_review_pack_preservation(temporary_path, monkeypatch)


def _check_review_pack_preservation(tmp_path, monkeypatch):
    project = tmp_path / "project"
    report = project / "review_pack_phase1_50"
    output = project / "outputs/phase1_iid_robot"
    monkeypatch.setattr(reporting, "ROOT", project)
    monkeypatch.setattr(reporting, "REPORT_ROOT", report)
    monkeypatch.setattr(reporting, "OUTPUT_ROOT", output)
    run_id = "SEQ_IID__FM_STANDARD__seed17"
    previous = output / run_id / "attempt_001"
    current = output / run_id / "attempt_002"
    for attempt, source in ((previous, "source before retry"), (current, "source actually used after retry")):
        put(attempt / "source/runner.py", source)
        put(attempt / "config.json", json.dumps({"source_sha256": source, "execution_source_hashes": {"runner.py": source}}))
        put(attempt / "metric_provenance.json", '{"source": "frozen test split"}')
        put(attempt / "heun_metrics/metric_provenance.json", '{"solver": "Heun"}')
        put(attempt / "training.csv", "step,loss\n1,0.5\n")
    put(previous / "failure.json", '{"error": "original failure", "traceback": "original trace"}')
    put(previous / "training_before_resume.csv", "step,loss\n1,0.9\n")
    # Current project source differs; snapshots must keep each actual attempt.
    put(project / "experiment/runner.py", "current project source")
    put(project / "EXPERIMENT_IMPLEMENTATION_LOG.md", "preserved implementation history")
    for name in ("DATA_SPEC.json", "METRIC_SPEC.json"):
        put(project / "data/seq_iid" / name, "{}")
    for name in ("train.npz", "validation.npz", "test.npz", "stats.npz", "metric_parameters.npz"):
        put(project / "data/seq_iid" / name, b"byte-for-byte frozen test fixture")
    run = {"run_id": run_id, "dataset_id": "SEQ_IID", "model_id": "FM_STANDARD", "seed": 17, "status": "RUNNING", "attempts": []}
    reporting.copy_review_artifacts([{"run": run, "attempt": current}])
    assert (report / "runs" / run_id / "attempt_001/source/runner.py").read_text() == "source before retry"
    assert (report / "runs" / run_id / "attempt_002/source/runner.py").read_text() == "source actually used after retry"
    assert (report / "runs" / run_id / "attempt_001/failure.json").read_text() == (previous / "failure.json").read_text()
    assert (report / "runs" / run_id / "attempt_002/heun_metrics/metric_provenance.json").exists()
    assert (report / "runs" / run_id / "attempt_002/metric_provenance.json").exists()
    assert (report / "curves" / run_id / "attempt_001/training_before_resume.csv").exists()
    assert (report / "data/seq_iid/train.npz").read_bytes() == b"byte-for-byte frozen test fixture"
    assert (report / "EXPERIMENT_IMPLEMENTATION_LOG.md").exists()
    with (report / "tables/attempt_history.csv").open(newline="") as handle:
        attempts = list(csv.DictReader(handle))
    assert len(attempts) == 2
    assert [row["is_current_attempt"] for row in attempts] == ["False", "True"]
    with (report / "tables/attempt_failure_records.csv").open(newline="") as handle:
        failures = list(csv.DictReader(handle))
    assert len(failures) == 1
    assert failures[0]["error"] == "original failure"
    assert failures[0]["traceback"] == "original trace"
