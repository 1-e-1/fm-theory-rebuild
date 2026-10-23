"""Nonformal monitoring must preserve scope, provenance and prior activities."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from phase2_experiment import monitor
from phase2_experiment.scope import ROOT, run_identifier, sha256_file


@pytest.fixture
def workspace(monkeypatch):
    parent = ROOT / ".phase2_test_tmp"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="monitor_", dir=parent) as folder:
        directory = Path(folder)
        monkeypatch.setattr(monitor, "ROOT", directory)
        yield directory


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_pilot(workspace, *, finished=True):
    run_id = run_identifier("SEQ_DELAY", "FM_STANDARD", 17)
    attempt = workspace / "outputs/phase2_pilots" / run_id / "attempt_001"
    attempt.mkdir(parents=True)
    source = attempt / "source/theory_fm_models.py"
    source.parent.mkdir()
    source.write_text("# graphics/metadata test fixture; no training\n", encoding="utf-8")
    hashes = {"theory_fm_models.py": sha256_file(source)}
    spec = workspace / "data/phase2_3datasets/seq_delay/DATA_SPEC.json"
    write_json(spec, {"state_dim": 128})
    config = {"phase": "PHASE2_3DATASETS", "dataset_id": "SEQ_DELAY", "model_id": "FM_STANDARD",
        "seed": 17, "run_id": run_id, "kind": "pilot", "steps": 3, "model_config": {"state_dim": 128},
        "data_spec_path": spec.relative_to(workspace).as_posix(), "data_spec_sha256": sha256_file(spec),
        "source_sha256": hashes["theory_fm_models.py"], "execution_source_hashes": hashes}
    write_json(attempt / "config.json", config)
    with (attempt / "training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "elapsed_seconds", "loss"])
        writer.writerows((step, float(step), 1 / step) for step in range(1, 4 if finished else 2))
    if finished:
        write_json(attempt / "pilot_result.json", {"dataset_id": "SEQ_DELAY", "model_id": "FM_STANDARD", "seed": 17,
            "kind": "pilot_not_formal", "test_accessed": False, "validation_loss": 1., "final_validation_loss": 1., "training_seconds": 3.})
        write_json(attempt / "pilot_validation_diagnostics.json", {"test_accessed": False, "small_sample_nonformal": True})
        write_json(attempt / "training_cost.json", {"updates": 3})
        write_json(attempt / "sampling_diagnostics.json", {"sampling_seconds": 1.})
        write_json(attempt / "source/core_execution_source_hashes.json", hashes)
        np.save(attempt / "pilot_samples.npy", np.zeros((2, 32, 128), dtype=np.float32))
        checkpoint = {"step": 3, "run_identity": {key: config[key] for key in ("dataset_id", "model_id", "seed", "kind", "run_id", "steps")},
            "model_sha256": config["source_sha256"], "data_spec_sha256": config["data_spec_sha256"], "execution_source_hashes": hashes,
            "model": {"weight": torch.zeros(1)}}
        for name in ("checkpoint_final.pt", "checkpoint_best.pt"):
            torch.save(checkpoint, attempt / name)
    return attempt, config


def test_running_pilot_does_not_load_active_checkpoint(workspace, monkeypatch):
    attempt, config = make_pilot(workspace, finished=False)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("active checkpoint must not be read"))
    records, activities, warnings = monitor._scan_pilots({"mode": "pilot", "dataset_id": "SEQ_DELAY", "status": "RUNNING", "pid_alive_at_snapshot": True})
    assert not activities and not warnings
    assert records[0]["status"] == "TRAINING" and records[0]["last_flushed_step"] == 1
    assert records[0]["planned_updates"] == 3
    with (attempt / "training.csv").open("a", encoding="utf-8") as handle:
        handle.write("2,2,")
    assert monitor._training_progress(attempt / "training.csv")["last_flushed_step"] == 1


def test_completed_gate_rejects_incomplete_log_source_and_samples(workspace):
    attempt, config = make_pilot(workspace)
    progress = monitor._training_progress(attempt / "training.csv")
    assert monitor._pilot_completion_issues(attempt, config, progress) == []
    (attempt / "source/theory_fm_models.py").write_text("changed fixture", encoding="utf-8")
    np.save(attempt / "pilot_samples.npy", np.full((2, 32, 128), np.nan, dtype=np.float32))
    progress["steps"] = [1, 3]
    issues = monitor._pilot_completion_issues(attempt, config, progress)
    assert any("complete pilot budget" in issue for issue in issues)
    assert any("source snapshot mismatch" in issue for issue in issues)
    assert any("finite float" in issue for issue in issues)


def test_auxiliary_append_is_idempotent_and_preserves_other_entries(workspace):
    original = {"formal": False, "count_towards_75": False, "custom_field": {"keep": True},
        "activities": [{"activity_id": "preformal_correctness_and_data_audit", "passed_tests": 47, "passed_subtests": 3}]}
    write_json(workspace / "PHASE2_AUXILIARY_RUNS.json", original)
    item = {"activity_id": "new_pilot", "count_towards_75": False}
    assert monitor._register_auxiliary([item])["new_activity_ids"] == ["new_pilot"]
    assert monitor._register_auxiliary([item])["new_activity_ids"] == []
    final = json.loads((workspace / "PHASE2_AUXILIARY_RUNS.json").read_text(encoding="utf-8"))
    assert final["custom_field"] == original["custom_field"]
    assert final["activities"][0] == original["activities"][0]
    assert len(final["activities"]) == 2


def test_snapshot_keeps_formal_denominator_and_registers_real_pilot_evidence(workspace, monkeypatch):
    attempt, config = make_pilot(workspace)
    monkeypatch.setattr(monitor, "load_manifest", lambda: {"runs": [
        {"dataset_id": dataset, "status": "PENDING"} for dataset in monitor.DATASETS for _ in range(25)]})
    write_json(workspace / "phase2_reference/PLOTS_TEST_EVIDENCE.json", {"passed": 3, "training_updates": 0,
        "original_test_npz_opened": False, "test_command": "python -m pytest phase2_tests/test_plots.py -q"})
    result = monitor.snapshot()
    assert result["completed"] == 0 and result["formal_configuration_count"] == 75
    assert result["pilots"]["initial_pilot_groups_completed"] == 1
    assert result["pilots"]["count_towards_75"] is False
    registry = json.loads((workspace / "PHASE2_AUXILIARY_RUNS.json").read_text(encoding="utf-8"))
    pilot = next(item for item in registry["activities"] if item["kind"] == "selected_task_validation_pilot")
    assert pilot["training_updates"] == 3 and pilot["formal_test_accessed"] is False
    assert pilot["count_towards_75"] is False and len(pilot["artifacts"]) >= 10
    assert any(item.get("passed_tests") == 3 for item in registry["activities"])
    progress = (workspace / "PHASE2_PROGRESS.md").read_text(encoding="utf-8")
    assert "**0/75**" in progress and "**1/15**" in progress and "Asia/Shanghai" in progress
    assert not result["training_started_by_monitor"] and not result["test_data_opened_by_monitor"]


def test_saved_five_checkpoint_checks_registered_without_sampling(workspace, monkeypatch):
    attempt = workspace / "outputs/phase2_3datasets/SEQ_DELAY__FM_STANDARD__seed17/attempt_001"
    root = attempt / "checkpoint_checks/checkpoint_diagnostics"
    root.mkdir(parents=True)
    run = {"run_id": "SEQ_DELAY__FM_STANDARD__seed17", "dataset_id": "SEQ_DELAY", "model_id": "FM_STANDARD", "seed": 17, "status": "COMPLETED"}
    monkeypatch.setattr(monitor, "active_attempt_path", lambda row: attempt)
    write_json(attempt / "config.json", {"source_sha256": "0" * 64})
    (attempt / "checkpoint_final.pt").write_bytes(b"read-only metadata fixture")
    diagnostics = {"status": "COMPLETED", "kind": "same_checkpoint_auxiliary_evaluation", "training_updates": 0,
        "formal_configuration_count_added": 0, "dataset_id": "SEQ_DELAY", "model_id": "FM_STANDARD",
        "model_state_sha256": "1" * 64, "seed": 771043, "checks": {}}
    for particles, steps in ((32, 32), (32, 64), (32, 128), (64, 64), (128, 64)):
        key = f"N{particles}_steps{steps}"
        name = f"samples_{key}.npy"
        np.save(root / name, np.zeros((2, particles, 128), dtype=np.float32))
        diagnostics["checks"][key] = {"samples": name, "status": "COMPLETED", "training_updates": 0,
            "cost": {"N": particles, "steps": steps, "seed": 771043}}
    write_json(attempt / "checkpoint_checks.json", diagnostics)
    activities, warnings = monitor._formal_check_activities({"runs": [run]})
    assert not warnings and len(activities) == 1
    activity = activities[0]
    assert activity["saved_checks_count"] == 5 and activity["count_towards_75"] is False
    assert activity["training_updates"] == activity["formal_configurations_added"] == 0
    assert all(check["sampling_seed"] == 771043 and len(check["samples_sha256"]) == 64 for check in activity["checks"])
    (root / "samples_N32_steps32.npy").unlink()
    activities, warnings = monitor._formal_check_activities({"runs": [run]})
    assert not activities and warnings
