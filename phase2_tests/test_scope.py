"""Guard independent scope, old-stage isolation, D128 artifacts and seed statistics."""
import copy
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from phase2_experiment import reporting, scope


@pytest.fixture
def project(monkeypatch):
    actual_root = scope.ROOT.resolve()
    parent = (actual_root / ".phase2_test_tmp").resolve()
    assert parent.is_relative_to(actual_root)
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scope_", dir=parent) as temporary:
        root = Path(temporary).resolve()
        assert root.is_relative_to(parent)
        monkeypatch.setattr(scope, "ROOT", root)
        monkeypatch.setattr(scope, "SCOPE_PATH", root / "PHASE2_SCOPE.json")
        monkeypatch.setattr(scope, "MANIFEST_PATH", root / "PHASE2_MANIFEST.json")
        monkeypatch.setattr(scope, "PROTOCOL_PATH", root / "PHASE2_LOCKED_PROTOCOL.json")
        monkeypatch.setattr(scope, "OUTPUT_ROOT", root / "outputs/phase2_3datasets")
        monkeypatch.setattr(reporting, "ROOT", root)
        monkeypatch.setattr(reporting, "OUTPUT_ROOT", scope.OUTPUT_ROOT)
        monkeypatch.setattr(reporting, "REPORT_ROOT", root / "review_pack_phase2_3datasets")
        yield root


def initialize():
    return scope.initialize_scope(list(scope.DATASETS), authority_note="isolated correctness-test fixture, no research training")


def test_initialize_exact75_preserves_previous_phase_files_and_refuses_reset(project):
    protected = ["manifest.json", "PHASE1_SCOPE.json", "LOCKED_PROTOCOL.json", "PROGRESS.md", "DEFERRED_BY_USER.json"]
    for name in protected:
        (project / name).write_text("protected original " + name, encoding="utf-8")
    originals = {name: (project / name).read_bytes() for name in protected}
    initialize()
    manifest = scope.load_manifest()
    triples = [(r["dataset_id"], r["model_id"], r["seed"]) for r in manifest["runs"]]
    assert tuple(triples) == scope.EXPECTED and len(set(triples)) == 75
    assert all(r["status"] == "PENDING" for r in manifest["runs"])
    assert all(sum(r["dataset_id"] == d for r in manifest["runs"]) == 25 for d in scope.DATASETS)
    assert all((project / name).read_bytes() == original for name, original in originals.items())
    assert "Asia/Shanghai" in (project / "PHASE2_PROGRESS.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="refuses to erase"):
        initialize()
    assert all((project / name).read_bytes() == original for name, original in originals.items())


@pytest.mark.parametrize("triple", [
    ("SEQ_IID", "FM_STANDARD", 17), ("ROBOT_8DOF", "FM_STANDARD", 17),
    ("SEQ_DELAY", "EXTRA_MODEL", 17), ("SEQ_DELAY", "FM_STANDARD", 99),
    ("SEQ_DELAY", "FM_STANDARD", True), ("SEQ_DELAY", "FM_STANDARD", 17.0),
])
def test_reject_unselected_dataset_model_seed(triple):
    with pytest.raises(ValueError, match="outside PHASE2_SCOPE"):
        scope.validate_config(*triple)


def test_launch_resume_cannot_point_to_old_output_root(project):
    with pytest.raises(ValueError, match="dedicated output root"):
        scope.resolve_path("outputs/phase1_iid_robot/SEQ_IID__FM_STANDARD__seed17/attempt_001")


def test_manifest_tampering_and_artifact_free_completion_are_rejected(project):
    initialize()
    before = scope.MANIFEST_PATH.read_bytes()
    run = scope.load_manifest()["runs"][0]
    with pytest.raises(ValueError, match="Cannot mark COMPLETED"):
        scope.update_run(run["run_id"], "COMPLETED")
    assert scope.MANIFEST_PATH.read_bytes() == before
    changed = scope.read_json(scope.MANIFEST_PATH)
    changed["runs"][-1] = copy.deepcopy(changed["runs"][0])
    scope.write_json(scope.MANIFEST_PATH, changed)
    with pytest.raises(ValueError, match="75 fixed triples"):
        scope.load_manifest()


def test_attempt_and_failure_events_preserved_on_same_run_resume(project):
    initialize()
    run_id = scope.load_manifest()["runs"][0]["run_id"]
    attempt = scope.OUTPUT_ROOT / run_id / "attempt_001"
    attempt.mkdir(parents=True)
    scope.update_run(run_id, "RUNNING", attempt_dir=str(attempt))
    scope.update_run(run_id, "FAILED", reason="original device failure")
    scope.update_run(run_id, "RUNNING", attempt_dir=str(attempt))
    result = scope.get_run(run_id)
    assert len(result["attempts"]) == 1
    assert [e["status"] for e in result["events"]] == ["RUNNING", "FAILED", "RUNNING"]
    assert result["events"][1]["reason"] == "original device failure"
    assert len(scope.load_manifest()["runs"]) == 75


def test_per_dataset_protocol_hash_stable_when_another_task_locks(project):
    initialize()
    first = {"updates": 12000, "sampling_ensembles": 16, "sampling_particles": 64, "reason": "validation/cost choice"}
    scope.write_json(scope.PROTOCOL_PATH, {scope.DATASETS[0]: first})
    original = scope.dataset_protocol_digest(scope.DATASETS[0])
    scope.write_json(scope.PROTOCOL_PATH, {scope.DATASETS[0]: first, scope.DATASETS[1]: {**first, "updates": 24000}})
    assert scope.dataset_protocol_digest(scope.DATASETS[0]) == original


def completion_fixture(project, monkeypatch):
    """Synthetic artifact structure only, isolated from all formal manifests."""
    initialize()
    run = scope.load_manifest()["runs"][1]  # real ODE; no Heun fixture required
    attempt = scope.OUTPUT_ROOT / run["run_id"] / "attempt_001"
    attempt.mkdir(parents=True)
    run["attempt_dir"] = str(attempt)
    protocol = {"updates": 12000, "N": 32, "H": 17, "Q": 64, "E": 1, "sampling_ensembles": 16, "sampling_particles": 64}
    scope.write_json(scope.PROTOCOL_PATH, {run["dataset_id"]: protocol})
    spec_path = project / "data/fixture/DATA_SPEC.json"
    scope.write_json(spec_path, {"dataset_id": run["dataset_id"], "length": 64, "channels": 2, "state_dim": 128})
    monkeypatch.setattr(scope, "data_spec_path", lambda dataset: spec_path)
    source = project / "theory_fm_models.py"
    source.write_text("unit test source identity fixture", encoding="utf-8")
    snapshot = attempt / "source/theory_fm_models.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    source_hash = scope.sha256_file(source)
    cfg = {"dataset_id": run["dataset_id"], "model_id": run["model_id"], "seed": run["seed"], "run_id": run["run_id"], "kind": "formal",
        "steps": 12000, "N": 32, "H": 17, "Q": 64, "E": 1, "model_config": {"state_dim": 128}, "source_sha256": source_hash,
        "data_spec_sha256": scope.sha256_file(spec_path), "dataset_protocol_sha256": scope.payload_digest(protocol),
        "execution_source_hashes": {"theory_fm_models.py": source_hash}}
    scope.write_json(attempt / "config.json", cfg)
    checkpoint = {"step": 12000, "model": {"fixture_parameter": torch.tensor([1.0])}, "optimizer": {}, "python_rng": None,
        "numpy_rng": None, "torch_rng": torch.tensor([1], dtype=torch.uint8), "model_sha256": source_hash, "model_config": cfg["model_config"],
        "run_identity": {k: cfg[k] for k in ("dataset_id", "model_id", "seed", "kind", "run_id", "steps")},
        "data_spec_sha256": cfg["data_spec_sha256"], "dataset_protocol_sha256": cfg["dataset_protocol_sha256"], "execution_source_hashes": cfg["execution_source_hashes"]}
    for name in ("checkpoint_final.pt", "checkpoint_best.pt"):
        torch.save(checkpoint, attempt / name)
    with (attempt / "training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "grad_norm", "elapsed_seconds"])
        writer.writerows((step, 1.0, 1.0, step / 1000) for step in range(1, 12001))
    for name in ("evaluation.json", "sampling_diagnostics.json", "training_cost.json", "metric_provenance.json"):
        scope.write_json(attempt / name, {"fixture_numeric_value": 0.1})
    np.save(attempt / "samples.npy", np.zeros((16, 64, 128), dtype=np.float32))
    diagnostic_root = attempt / "checkpoint_checks/checkpoint_diagnostics"
    diagnostic_root.mkdir(parents=True)
    checks = {}
    for particles, steps in ((32, 32), (32, 64), (32, 128), (64, 64), (128, 64)):
        key = f"N{particles}_steps{steps}"
        filename = f"samples_{key}.npy"
        checks[key] = {"status": "COMPLETED", "samples": filename}
        np.save(diagnostic_root / filename, np.zeros((2, particles, 128), dtype=np.float32))
    scope.write_json(attempt / "checkpoint_checks.json", {"status": "COMPLETED", "dataset_id": run["dataset_id"], "model_id": run["model_id"],
        "training_updates": 0, "formal_configuration_count_added": 0, "checks": checks})
    return run, attempt


def test_d64_or_nonfinite_or_incomplete_diagnostics_cannot_complete_d128_run(project, monkeypatch):
    run, attempt = completion_fixture(project, monkeypatch)
    assert scope.completion_issues(run) == []
    np.save(attempt / "samples.npy", np.zeros((16, 64, 64), dtype=np.float32))
    assert any("128" in issue and "samples.npy" in issue for issue in scope.completion_issues(run))
    values = np.zeros((16, 64, 128), dtype=np.float32)
    values[0, 0, 0] = np.nan
    np.save(attempt / "samples.npy", values)
    assert any("finite float" in issue for issue in scope.completion_issues(run))
    np.save(attempt / "samples.npy", np.zeros((16, 64, 128), dtype=np.float32))
    scope.write_json(attempt / "checkpoint_checks.json", {})
    assert any("diagnostics are incomplete" in issue for issue in scope.completion_issues(run))


def test_reporting_pairs_seeds_and_keeps_three_tasks_separate(project):
    initialize()
    records = []
    for run in scope.load_manifest()["runs"]:
        value = scope.MODELS.index(run["model_id"]) + run["seed"] / 100
        records.append({"run": {**run, "status": "COMPLETED"}, "attempt": None, "config": {}, "evaluation": {}, "training_cost": {},
            "diagnostics": {}, "metrics": {"swd_raw": value}, "supplementary_metrics": {}, "issues": [],
            "primary_solver": "fixture", "primary_metrics_source": "isolated arithmetic fixture"})
    reporting.create_tables(records)
    table = reporting.REPORT_ROOT / "tables"
    with (table / "paired_comparisons.csv").open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 21 and all(row["n_training_seeds"] == "5" for row in pairs)
    selected = [r for r in pairs if r["model_id"] == "FM_COMPLEX_ODE" and r["reference_model_id"] == "FM_REAL_ODE"]
    assert len(selected) == 3 and all(float(row["mean"]) == pytest.approx(2.0) for row in selected)
    with (table / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 15 and all(r["complete_seed_group"] == "True" for r in summary)
    assert reporting.holm([(0, 0.01)]) == pytest.approx({0: 0.07})
    assert not reporting.moments([1.0, 2.0])["complete_seed_group"]


def test_actual_source_and_prior_failure_copies_preserved(project):
    attempt = scope.OUTPUT_ROOT / "SEQ_DELAY__FM_REAL_ODE__seed17/attempt_001"
    actual = attempt / "source/phase2_experiment/runner.py"
    actual.parent.mkdir(parents=True)
    actual.write_text("actual historical execution source", encoding="utf-8")
    scope.write_json(attempt / "failure_001.json", {"error": "original failure"})
    scope.write_json(attempt / "metric_provenance.json", {"registered_source": "new stage data"})
    (attempt / "training_before_resume.csv").write_text("preserved original rows", encoding="utf-8")
    reporting.copy_attempt_artifacts("SEQ_DELAY__FM_REAL_ODE__seed17", attempt)
    base = reporting.REPORT_ROOT / "runs/SEQ_DELAY__FM_REAL_ODE__seed17/attempt_001"
    assert (base / "source/phase2_experiment/runner.py").read_text(encoding="utf-8") == "actual historical execution source"
    assert (base / "failure_001.json").exists() and (base / "metric_provenance.json").exists()
    assert (reporting.REPORT_ROOT / "curves/SEQ_DELAY__FM_REAL_ODE__seed17/attempt_001/training_before_resume.csv").exists()
