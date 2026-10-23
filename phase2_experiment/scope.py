"""Fixed, independent 75-run scope for the newly authorized three-dataset phase.

Initialization requires an explicit dataset list and never touches Phase 1
scope, manifest, protocol, source, weights, logs, samples or reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCOPE_PATH = ROOT / "PHASE2_SCOPE.json"
MANIFEST_PATH = ROOT / "PHASE2_MANIFEST.json"
PROTOCOL_PATH = ROOT / "PHASE2_LOCKED_PROTOCOL.json"
OUTPUT_ROOT = ROOT / "outputs/phase2_3datasets"
DATASETS = ("SEQ_DELAY", "SEQ_ACCUM", "SEQ_MULTISCALE")
MODELS = ("FM_STANDARD", "FM_REAL_ODE", "FM_COMPLEX_SDE", "FM_COMPLEX_ODE", "FM_COMPLEX_VDC_ODE")
SEEDS = (17, 29, 43, 71, 101)
EXPECTED = tuple((d, m, s) for d in DATASETS for s in SEEDS for m in MODELS)
STATUSES = frozenset({"PENDING", "RUNNING", "COMPLETED", "FAILED", "BLOCKED"})


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_now() -> str:
    return datetime.now(timezone(timedelta(hours=8), name="Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def payload_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_config(dataset_id: str, model_id: str, seed: int) -> None:
    if type(seed) is not int or (dataset_id, model_id, seed) not in EXPECTED:
        raise ValueError(f"Formal configuration outside PHASE2_SCOPE: {dataset_id!r}, {model_id!r}, {seed!r}. Exactly 3 datasets × 5 models × 5 seeds are authorized.")


def run_identifier(dataset_id: str, model_id: str, seed: int) -> str:
    validate_config(dataset_id, model_id, seed)
    return f"{dataset_id}__{model_id}__seed{seed}"


def validate_scope() -> dict[str, Any]:
    scope = read_json(SCOPE_PATH)
    if scope.get("formal_configuration_count") != 75 or scope.get("train_from_scratch") is not True:
        raise ValueError("Phase 2 must preserve its independent 75-run denominator and from-scratch policy.")
    if scope.get("datasets") != list(DATASETS) or scope.get("models") != list(MODELS) or scope.get("training_seeds") != list(SEEDS):
        raise ValueError("Phase 2 dataset/model/seed whitelist changed.")
    configs = scope.get("configurations", [])
    triples = []
    for cfg in configs:
        triple = (cfg.get("dataset_id"), cfg.get("model_id"), cfg.get("seed"))
        validate_config(*triple)
        if cfg.get("run_id") != run_identifier(*triple):
            raise ValueError("Scope run_id does not match its configuration.")
        triples.append(triple)
        if cfg.get("queue_index") != len(triples):
            raise ValueError("Phase 2 scope queue indices changed.")
    if tuple(triples) != EXPECTED or len(set(triples)) != 75:
        raise ValueError("Phase 2 scope must preserve exactly 75 seed-grouped unique triples.")
    return scope


def load_manifest() -> dict[str, Any]:
    validate_scope()
    manifest = read_json(MANIFEST_PATH)
    runs = manifest.get("runs", [])
    triples = [(r.get("dataset_id"), r.get("model_id"), r.get("seed")) for r in runs]
    if manifest.get("formal_configuration_count") != 75 or tuple(triples) != EXPECTED or len({r.get("run_id") for r in runs}) != 75:
        raise ValueError("Phase 2 manifest must retain all 75 fixed triples and queue order.")
    for index, run in enumerate(runs, 1):
        validate_config(run["dataset_id"], run["model_id"], run["seed"])
        if run["run_id"] != run_identifier(run["dataset_id"], run["model_id"], run["seed"]) or run.get("status") not in STATUSES:
            raise ValueError("Phase 2 run identity/status is invalid.")
        if run.get("queue_index") != index:
            raise ValueError("Phase 2 manifest queue indices changed.")
    return manifest


def get_run(run_id: str) -> dict[str, Any]:
    for run in load_manifest()["runs"]:
        if run["run_id"] == run_id:
            return run
    raise ValueError(f"Run outside fixed Phase 2 manifest: {run_id}")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    result = (path if path.is_absolute() else ROOT / path).resolve()
    if not result.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"Phase 2 run artifact path outside its dedicated output root: {value}")
    return result


def active_attempt_path(run: dict[str, Any]) -> Path | None:
    for key in ("attempt_dir", "current_attempt_dir"):
        if run.get(key):
            return resolve_path(run[key])
    base = resolve_path(run.get("output_dir", f"outputs/phase2_3datasets/{run['run_id']}"))
    if base.name.startswith("attempt_"):
        return base
    attempts = sorted(base.glob("attempt_*")) if base.exists() else []
    return attempts[-1] if attempts else None


def dataset_protocol(dataset_id: str) -> dict[str, Any]:
    if dataset_id not in DATASETS:
        raise ValueError("Dataset outside Phase 2")
    protocol = read_json(PROTOCOL_PATH)[dataset_id]
    if not isinstance(protocol, dict) or protocol.get("updates") not in (12000, 24000, 48000):
        raise ValueError(f"No locked authorized formal protocol for {dataset_id}")
    return protocol


def dataset_protocol_digest(dataset_id: str) -> str:
    return payload_digest(dataset_protocol(dataset_id))


def data_spec_path(dataset_id: str) -> Path:
    from .data import data_dir
    return Path(data_dir(dataset_id)) / "DATA_SPEC.json"


def completion_issues(run: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    attempt = active_attempt_path(run)
    if attempt is None:
        return ["No preserved Phase 2 attempt directory"]
    required = ["config.json", "checkpoint_final.pt", "checkpoint_best.pt", "training.csv", "samples.npy", "evaluation.json",
                "sampling_diagnostics.json", "checkpoint_checks.json", "training_cost.json", "metric_provenance.json"]
    if run["model_id"] == "FM_STANDARD":
        required += ["evaluation_heun.json", "samples_heun.npy"]
    for name in required:
        path = attempt / name
        if not path.is_file() or not path.stat().st_size:
            issues.append(f"Missing/empty {name}")
    if issues:
        return issues
    try:
        import numpy as np
        import torch
        cfg = read_json(attempt / "config.json")
        for key in ("dataset_id", "model_id", "seed"):
            if cfg.get(key) != run[key]:
                issues.append(f"Configuration identity mismatch: {key}")
        if cfg.get("kind") != "formal":
            issues.append("Pilot/auxiliary configuration cannot count as a formal run")
        protocol = dataset_protocol(run["dataset_id"])
        spec_path = data_spec_path(run["dataset_id"])
        spec = read_json(spec_path)
        dimension = spec.get("state_dim", spec.get("dimension"))
        if dimension is None:
            dimension = spec["length"] * spec["channels"]
        if dimension != 128 or spec.get("length") != 64 or spec.get("channels") != 2:
            issues.append("Phase 2 task definition requires complete L64 C2 D128 states")
        steps = cfg.get("steps", cfg.get("updates"))
        if steps != protocol["updates"] or type(steps) is not int:
            issues.append("Formal update budget differs from this dataset's locked protocol")
        protocol_hash = cfg.get("dataset_protocol_sha256", cfg.get("locked_protocol_sha256"))
        if protocol_hash != dataset_protocol_digest(run["dataset_id"]):
            issues.append("Per-dataset locked protocol digest differs")
        if cfg.get("data_spec_sha256") != sha256_file(spec_path):
            issues.append("Frozen dataset specification digest differs")
        for key in ("N", "H", "Q", "E"):
            if key in protocol and cfg.get(key) != protocol[key]:
                issues.append(f"Shared dataset protocol differs: {key}")
        with (attempt / "training.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != steps or [int(row["step"]) for row in rows] != list(range(1, (steps or 0) + 1)):
            issues.append("Training log must preserve each formal update exactly once through its complete budget")
        for row in rows:
            if not all(math.isfinite(float(row[k])) for k in ("loss", "grad_norm", "elapsed_seconds")):
                issues.append("Training log contains nonfinite loss/gradient/time")
                break
        model_source = ROOT / "theory_fm_models.py"
        if cfg.get("source_sha256") != sha256_file(model_source):
            issues.append("Delivered model source digest differs")
        snapshots = cfg.get("execution_source_hashes", {})
        if not snapshots:
            issues.append("Actual execution source hashes missing")
        for name, expected in snapshots.items():
            snapshot = attempt / "source" / name
            if not snapshot.is_file() or sha256_file(snapshot) != expected:
                issues.append(f"Actual attempt source snapshot differs/missing: {name}")
        for filename in ("checkpoint_final.pt", "checkpoint_best.pt"):
            checkpoint = torch.load(attempt / filename, map_location="cpu", weights_only=False)
            if filename == "checkpoint_final.pt" and checkpoint.get("step") != steps:
                issues.append("Final checkpoint step does not match complete formal budget")
            if filename == "checkpoint_best.pt" and not 1 <= checkpoint.get("step", 0) <= steps:
                issues.append("Validation-best checkpoint outside this formal run")
            if checkpoint.get("model_sha256") != cfg.get("source_sha256"):
                issues.append(f"{filename} model source differs from configuration")
            expected_identity = {k: cfg[k] for k in ("dataset_id", "model_id", "seed", "kind", "run_id", "steps")}
            if checkpoint.get("run_identity") != expected_identity:
                issues.append(f"{filename} formal run identity differs/missing")
            if checkpoint.get("data_spec_sha256") != cfg.get("data_spec_sha256") or checkpoint.get("execution_source_hashes") != snapshots:
                issues.append(f"{filename} actual source/data provenance differs")
            if checkpoint.get("dataset_protocol_sha256") != protocol_hash:
                issues.append(f"{filename} per-dataset protocol provenance differs")
            if json.dumps(checkpoint.get("model_config"), sort_keys=True) != json.dumps(cfg.get("model_config"), sort_keys=True):
                issues.append(f"{filename} model configuration differs")
            for key in ("model", "optimizer", "python_rng", "numpy_rng", "torch_rng"):
                if key not in checkpoint:
                    issues.append(f"{filename} lacks preserved {key}")
            if not all(torch.isfinite(v).all() for v in checkpoint.get("model", {}).values() if isinstance(v, torch.Tensor)):
                issues.append(f"{filename} has nonfinite parameters")
        def check_finite_values(mapping):
            for value in mapping.values():
                if isinstance(value, dict):
                    check_finite_values(value)
                elif isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
                    raise ValueError("Evaluation contains nonfinite numeric metrics")
        evaluations = ["evaluation.json"] + (["evaluation_heun.json"] if run["model_id"] == "FM_STANDARD" else [])
        for filename in evaluations:
            evaluation = read_json(attempt / filename)
            metrics = evaluation.get("metrics", evaluation)
            if not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in metrics.values()):
                issues.append(f"{filename} lacks actual numerical evaluation")
            check_finite_values(evaluation)
        expected_shape = (protocol["sampling_ensembles"], protocol["sampling_particles"], dimension)
        for filename in ["samples.npy"] + (["samples_heun.npy"] if run["model_id"] == "FM_STANDARD" else []):
            samples = np.load(attempt / filename, mmap_mode="r", allow_pickle=False)
            if tuple(samples.shape) != expected_shape or not np.issubdtype(samples.dtype, np.floating) or not np.isfinite(samples).all():
                issues.append(f"{filename} must contain finite float ensembles with shape {expected_shape}")
        if run["model_id"] == "FM_STANDARD" and not (attempt / "heun_metrics/metric_provenance.json").is_file():
            issues.append("Same-checkpoint Heun metric provenance missing")
        read_json(attempt / "sampling_diagnostics.json")
        diagnostics = read_json(attempt / "checkpoint_checks.json")
        if diagnostics.get("status") != "COMPLETED" or diagnostics.get("dataset_id") != run["dataset_id"] or diagnostics.get("model_id") != run["model_id"]:
            issues.append("Required same-checkpoint diagnostics are incomplete or have different identity")
        if diagnostics.get("training_updates") != 0 or diagnostics.get("formal_configuration_count_added") != 0:
            issues.append("Same-checkpoint diagnostics must add no formal training/configurations")
        diagnostic_root = attempt / "checkpoint_checks/checkpoint_diagnostics"
        required_checks = ((32, 32), (32, 64), (32, 128), (64, 64), (128, 64))
        for particles, check_steps in required_checks:
            key = f"N{particles}_steps{check_steps}"
            evidence = diagnostics.get("checks", {}).get(key, {})
            path = diagnostic_root / evidence.get("samples", f"samples_{key}.npy")
            if evidence.get("status") != "COMPLETED" or not path.is_file() or not path.resolve().is_relative_to(diagnostic_root.resolve()):
                issues.append(f"Missing/incomplete diagnostic evidence: {key}")
                continue
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (2, particles, dimension) or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
                issues.append(f"Nonfinite/wrong-dimension diagnostic samples: {key}")
    except Exception as exc:
        issues.append(f"Artifact validation error: {type(exc).__name__}: {exc}")
    return issues


def update_run(run_id: str, status: str, **fields: Any) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"Invalid Phase 2 status: {status}")
    manifest = load_manifest()
    match = next((r for r in manifest["runs"] if r["run_id"] == run_id), None)
    if match is None:
        raise ValueError(f"Run outside fixed Phase 2 manifest: {run_id}")
    for key in {"run_id", "dataset_id", "model_id", "seed", "queue_index"}.intersection(fields):
        if fields[key] != match[key]:
            raise ValueError(f"Cannot mutate Phase 2 formal identity/order: {key}")
    if "reason" not in fields and status in ("RUNNING", "COMPLETED"):
        fields["reason"] = "Independent Phase 2 from-scratch run in progress" if status == "RUNNING" else "Independent Phase 2 formal evidence verified"
    prospective = dict(match, **fields, status=status)
    if status == "COMPLETED":
        issues = completion_issues(prospective)
        if issues:
            raise ValueError("Cannot mark COMPLETED: " + "; ".join(issues))
    event = {"at": now(), "previous_status": match["status"], "status": status, "reason": fields.get("reason", "")}
    match.update(fields)
    match.update(status=status, updated_at=now())
    attempt = active_attempt_path(match)
    if attempt is not None:
        record = next((a for a in match.setdefault("attempts", []) if isinstance(a, dict) and a.get("attempt_id") == attempt.name), None)
        if record is None:
            record = {"attempt_id": attempt.name, "directory": str(attempt.relative_to(ROOT)), "created_at": now()}
            match["attempts"].append(record)
        record.update(status=status, updated_at=now(), reason=fields.get("reason", ""))
    match.setdefault("events", []).append(event)
    manifest["updated_at"] = now()
    write_json(MANIFEST_PATH, manifest)
    return match


def verify_manifest_artifacts() -> dict[str, Any]:
    manifest = load_manifest()
    issues = {r["run_id"]: completion_issues(r) for r in manifest["runs"] if r["status"] == "COMPLETED"}
    issues = {k: v for k, v in issues.items() if v}
    counts = {s: sum(r["status"] == s for r in manifest["runs"]) for s in sorted(STATUSES)}
    guard = verify_phase1_guard()
    return {"ok": not issues and guard["ok"], "formal_configuration_count": 75, "counts": counts, "issues": issues,
            "scope_sha256": sha256_file(SCOPE_PATH), "phase1_guard": guard}


def verify_phase1_guard() -> dict[str, Any]:
    path = ROOT / "phase2_reference/PHASE1_GUARD.json"
    if not path.exists():
        return {"ok": False, "issues": ["Phase 1 protection record unavailable"], "files_checked": 0}
    guard = read_json(path)
    issues = []
    for name, expected in guard["files"].items():
        source = ROOT / name
        if not source.is_file() or sha256_file(source) != expected["sha256"]:
            issues.append(f"Protected Phase 1 file differs/missing: {name}")
    return {"ok": not issues, "issues": issues, "files_checked": len(guard["files"])}


def initialize_scope(datasets: list[str] | tuple[str, ...], *, authority_note: str) -> dict[str, Any]:
    """Called by the execution lead only after task definitions/selection settle."""
    if tuple(datasets) != DATASETS:
        raise ValueError("Selected datasets differ from the supported fixed Phase 2 task definitions; resolve selection before initializing.")
    if SCOPE_PATH.exists() or MANIFEST_PATH.exists():
        raise ValueError("Phase 2 scope/manifest exists; initialization refuses to erase progress.")
    configurations = [{"queue_index": i + 1, "dataset_id": d, "model_id": m, "seed": s, "run_id": run_identifier(d, m, s)} for i, (d, m, s) in enumerate(EXPECTED)]
    scope = {"schema_version": 1, "phase_id": "PHASE2_3_DATASETS", "created_at": now(), "authority": authority_note,
        "datasets": list(DATASETS), "models": list(MODELS), "training_seeds": list(SEEDS), "formal_configuration_count": 75,
        "configurations_per_dataset": 25, "train_from_scratch": True, "allow_old_checkpoints": False,
        "queue_order": "dataset, training seed, fixed model order", "forbid_default_full_queue": True,
        "formal_update_budget_options": [12000, 24000, 48000], "task_dimensions": {d: {"length": 64, "channels": 2, "state_dim": 128} for d in DATASETS},
        "prior_phase_is_immutable": True, "prior_phase_formal_denominator": 50, "nonformal_activities_count_towards_75": False,
        "configurations": configurations}
    write_json(SCOPE_PATH, scope)
    write_json(MANIFEST_PATH, {"schema_version": 1, "phase_id": "PHASE2_3_DATASETS", "created_at": now(), "updated_at": now(),
        "formal_configuration_count": 75, "runs": [dict(cfg, status="PENDING", attempts=[], events=[], created_at=now(), updated_at=now(),
            output_dir=f"outputs/phase2_3datasets/{cfg['run_id']}", reason="Awaiting selected-task correctness/pilot and per-dataset locked protocol") for cfg in configurations],
        "nonformal_registry_path": "PHASE2_AUXILIARY_RUNS.json", "deferred_registry_path": "PHASE2_DEFERRED_BY_USER.json"})
    if not PROTOCOL_PATH.exists():
        write_json(PROTOCOL_PATH, {"phase_id": "PHASE2_3_DATASETS", "model_order": list(MODELS), "seed_order": list(SEEDS), "datasets_pending_lock": list(DATASETS)})
    deferred = ["SEQ_IID", "ROBOT_8DOF", "TS_ETTM1", "TS_APPLIANCES", "TRAJ_LASA", "ROBOT_PLANAR", "OTHER_DATASETS", "EXTRA_CONTROLS", "ABLATIONS", "ROTATION_PATHS", "LENGTH_DATA_SIZE_J_NOISE_VARIANTS"]
    write_json(ROOT / "PHASE2_DEFERRED_BY_USER.json", {"status": "DEFERRED_BY_USER", "count_towards_75": False,
        "items": deferred, "reason": "Outside the newly authorized independent three-dataset queue; prior Phase 1 results remain under their original scope/status."})
    if not (ROOT / "PHASE2_AUXILIARY_RUNS.json").exists():
        write_json(ROOT / "PHASE2_AUXILIARY_RUNS.json", {"formal": False, "count_towards_75": False, "activities": []})
    create_statistical_analysis_plan()
    write_progress()
    return scope


def create_statistical_analysis_plan() -> dict[str, Any]:
    path = ROOT / "PHASE2_STATISTICAL_ANALYSIS_PLAN.json"
    if path.exists():
        return read_json(path)
    pairs = [("FM_COMPLEX_ODE", "FM_REAL_ODE"), ("FM_COMPLEX_SDE", "FM_COMPLEX_ODE"),
        ("FM_COMPLEX_VDC_ODE", "FM_COMPLEX_ODE"), *((m, "FM_STANDARD") for m in MODELS[1:])]
    plan = {"schema_version": 1, "phase_id": "PHASE2_3_DATASETS", "registered_before_formal_training": True,
        "created_at": now(), "datasets_analyzed_separately": True, "training_replication_unit": "training seed", "required_training_seeds": list(SEEDS),
        "primary_metric": "swd_raw", "secondary_metric": "mmd2_raw", "fixed_comparison_family_size": 7,
        "fixed_comparisons": [{"model_id": m, "reference_model_id": r} for m, r in pairs],
        "difference_definition": "model metric minus its specified reference model metric at the same training seed",
        "standard_reference": "same final FM_STANDARD checkpoint with Heun; Euler kept as explicit supplementary evaluation",
        "interval_method": "two-sided 95% Student t interval over available independent training seeds, incomplete groups explicitly labeled n of 5",
        "test_method": "two-sided paired Student t test; assumption-dependent descriptive evidence with five seeds",
        "multiplicity": "Holm over the fixed seven-comparison family separately for each dataset and metric; raw confidence intervals and adjusted p-values remain separate",
        "temporal_dependency_diagnostics": "Per-task metrics and dependency-breaking controls, channel/time axes and downstream ridge rules preregistered in each METRIC_SPEC. Additional diagnostics are descriptive.",
        "sampling_hierarchy": "Independent ensembles retained per checkpoint; interacting particles/projections/pair counts are not independent training repeats",
        "missing_policy": "Never impute or drop failed/blocked formal identities; denominator remains 75 and each dataset 25",
        "previous_phase": "Phase 1 results retain original labels/status and are excluded from Phase 2 numerical tables",
        "predetermined_winner": False}
    write_json(path, plan)
    return plan


def write_progress() -> dict[str, Any]:
    manifest = load_manifest()
    counts = {d: {s: sum(r["dataset_id"] == d and r["status"] == s for r in manifest["runs"]) for s in sorted(STATUSES)} for d in DATASETS}
    completed = sum(c["COMPLETED"] for c in counts.values())
    lines = ["# Phase 2 progress", "", f"更新时间: {local_now()}", "", f"正式配置完成 **{completed}/75**；本阶段独立于Phase 1原50项，原结果不改写。", ""]
    for d in DATASETS:
        lines.append(f"- {d}: {counts[d]['COMPLETED']}/25 COMPLETED；" + ", ".join(f"{s}={n}" for s, n in counts[d].items() if n))
    lines += ["", "仅75项固定配置；每任务L64、C2、D128完整状态，五模型共享数据划分与锁定协议。正式预算与收敛判断以实际协议、日志、验证曲线为据。", "",
        "恢复与汇总：", "", "```powershell", "python -m phase2_experiment.verify", "python -m phase2_experiment.runner resume",
        "python -m phase2_experiment.reporting --package", "```", "", "失败/受阻记录留在PHASE2_MANIFEST.json及各attempt；其他任务单列PHASE2_DEFERRED_BY_USER.json。完成本阶段后停止，不启动其他矩阵。"]
    (ROOT / "PHASE2_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"formal_configuration_count": 75, "completed": completed, "datasets": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.parse_args()
    result = verify_manifest_artifacts()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
