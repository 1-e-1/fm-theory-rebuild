"""Immutable Phase 1 training scope and artifact-checked manifest updates.

The constants below intentionally do not consult old full-experiment defaults.
Only the 50 user-authorized triples may enter the formal training queue.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCOPE_PATH = ROOT / "PHASE1_SCOPE.json"
MANIFEST_PATH = ROOT / "manifest.json"
DATASETS = ("SEQ_IID", "ROBOT_8DOF")
MODELS = (
    "FM_STANDARD", "FM_REAL_ODE", "FM_COMPLEX_SDE",
    "FM_COMPLEX_ODE", "FM_COMPLEX_VDC_ODE",
)
SEEDS = (17, 29, 43, 71, 101)
STATUSES = frozenset({"PENDING", "RUNNING", "COMPLETED", "FAILED", "BLOCKED"})
EXPECTED = tuple((d, m, s) for d in DATASETS for s in SEEDS for m in MODELS)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_identifier(dataset_id: str, model_id: str, seed: int) -> str:
    validate_config(dataset_id, model_id, seed)
    return f"{dataset_id}__{model_id}__seed{seed}"


def validate_config(dataset_id: str, model_id: str, seed: int) -> None:
    if type(seed) is not int or (dataset_id, model_id, seed) not in EXPECTED:
        raise ValueError(
            f"Formal configuration outside PHASE1_SCOPE: "
            f"dataset_id={dataset_id!r}, model_id={model_id!r}, seed={seed!r}. "
            "Only the fixed 2 datasets × 5 models × 5 seeds are authorized."
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_scope() -> dict[str, Any]:
    scope = read_json(SCOPE_PATH)
    triples = []
    for cfg in scope.get("configurations", []):
        triple = (cfg.get("dataset_id"), cfg.get("model_id"), cfg.get("seed"))
        validate_config(*triple)
        if cfg.get("run_id") != run_identifier(*triple):
            raise ValueError("Scope run_id does not match its configuration.")
        triples.append(triple)
    if tuple(triples) != EXPECTED or len(set(triples)) != 50:
        raise ValueError("Phase 1 scope must contain exactly the 50 fixed triples in seed-grouped order.")
    if scope.get("formal_configuration_count") != 50 or scope.get("train_from_scratch") is not True:
        raise ValueError("Scope denominator/from-scratch policy was changed.")
    return scope


def load_manifest() -> dict[str, Any]:
    validate_scope()
    manifest = read_json(MANIFEST_PATH)
    runs = manifest.get("runs", [])
    triples = [(r.get("dataset_id"), r.get("model_id"), r.get("seed")) for r in runs]
    if tuple(triples) != EXPECTED or len({r.get("run_id") for r in runs}) != 50:
        raise ValueError("Manifest must preserve all 50 fixed configurations and queue order.")
    for run in runs:
        validate_config(run["dataset_id"], run["model_id"], run["seed"])
        if run["run_id"] != run_identifier(run["dataset_id"], run["model_id"], run["seed"]):
            raise ValueError("Manifest run_id does not match its configuration.")
        if run.get("status") not in STATUSES:
            raise ValueError(f"Invalid manifest status: {run.get('status')}")
    if manifest.get("formal_configuration_count") != 50:
        raise ValueError("Manifest denominator must stay 50.")
    return manifest


def get_run(run_id: str) -> dict[str, Any]:
    for run in load_manifest()["runs"]:
        if run["run_id"] == run_id:
            return run
    raise ValueError(f"Run outside fixed manifest: {run_id}")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    result = (path if path.is_absolute() else ROOT / path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Run artifact path outside this project: {value}")
    return result


def active_attempt_path(run: dict[str, Any]) -> Path | None:
    for key in ("attempt_dir", "current_attempt_dir"):
        if run.get(key):
            return resolve_path(run[key])
    base = resolve_path(run.get("output_dir", f"outputs/phase1_iid_robot/{run['run_id']}"))
    attempts = sorted(base.glob("attempt_*")) if base.exists() else []
    return attempts[-1] if attempts else None


def completion_issues(run: dict[str, Any]) -> list[str]:
    """Check evidence before reporting a formal run as completed.

    The trainer/evaluator performs checkpoint numerical validation. This gate
    checks persisted artifacts, identity, floating samples, and final metrics.
    """
    issues: list[str] = []
    attempt = active_attempt_path(run)
    if attempt is None:
        return ["No preserved attempt directory"]
    required = ("config.json", "checkpoint_final.pt", "checkpoint_best.pt", "training.csv", "samples.npy", "evaluation.json", "sampling_diagnostics.json", "checkpoint_checks.json")
    if run["model_id"] == "FM_STANDARD":
        required += ("evaluation_heun.json", "samples_heun.npy")
    for name in required:
        path = attempt / name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"Missing/empty {name}")
    if issues:
        return issues
    try:
        cfg = read_json(attempt / "config.json")
        for key in ("dataset_id", "model_id", "seed"):
            if cfg.get(key) != run[key]:
                issues.append(f"Configuration identity mismatch: {key}")
        steps = cfg.get("steps", cfg.get("updates"))
        if type(steps) is not int or steps not in (12000, 24000, 48000):
            issues.append("Formal update budget is not one of 12000/24000/48000")
        with (attempt / "training.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != steps or [int(row["step"]) for row in rows] != list(range(1, (steps or 0) + 1)):
            issues.append("Training log must contain every formal update exactly once through configured steps")
        for row in rows:
            if not all(math.isfinite(float(row[k])) for k in ("loss", "grad_norm", "elapsed_seconds")):
                issues.append("Training log contains nonfinite loss/gradient/time")
                break
        import torch
        for filename in ("checkpoint_final.pt", "checkpoint_best.pt"):
            checkpoint = torch.load(attempt / filename, map_location="cpu", weights_only=False)
            if filename == "checkpoint_final.pt" and checkpoint.get("step") != steps:
                issues.append("Final checkpoint step does not match the configured formal budget")
            if filename == "checkpoint_best.pt" and not (1 <= checkpoint.get("step", 0) <= steps):
                issues.append("Validation-best checkpoint step outside the formal budget")
            for key in ("model", "optimizer", "python_rng", "numpy_rng", "torch_rng"):
                if key not in checkpoint:
                    issues.append(f"{filename} lacks preserved {key}")
            if not all(torch.isfinite(value).all() for value in checkpoint.get("model", {}).values() if isinstance(value, torch.Tensor)):
                issues.append(f"{filename} contains nonfinite model parameters")
        evaluation_files = ["evaluation.json"] + (["evaluation_heun.json"] if run["model_id"] == "FM_STANDARD" else [])
        def validate_evaluation_numbers(mapping):
            for value in mapping.values():
                if isinstance(value, dict):
                    validate_evaluation_numbers(value)
                elif isinstance(value, (float, int)) and not isinstance(value, bool) and not math.isfinite(float(value)):
                    raise ValueError("Final evaluation contains nonfinite numerical metrics")
        for filename in evaluation_files:
            evaluation = read_json(attempt / filename)
            metric_values = evaluation.get("metrics", evaluation)
            if not isinstance(metric_values, dict) or not any(
                isinstance(v, (float, int)) and not isinstance(v, bool) for v in metric_values.values()
            ):
                issues.append(f"{filename} has no numerical metrics")
            validate_evaluation_numbers(evaluation)
        import numpy as np
        sample_files = ["samples.npy"] + (["samples_heun.npy"] if run["model_id"] == "FM_STANDARD" else [])
        for filename in sample_files:
            samples = np.load(attempt / filename, mmap_mode="r", allow_pickle=False)
            if samples.size == 0 or not np.issubdtype(samples.dtype, np.floating):
                issues.append(f"{filename} must be a nonempty floating-point array")
            elif not np.isfinite(samples).all():
                issues.append(f"{filename} contains nonfinite values")
    except Exception as exc:
        issues.append(f"Artifact validation error: {type(exc).__name__}: {exc}")
    return issues


def update_run(run_id: str, status: str, **fields: Any) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"Invalid run status: {status}")
    manifest = load_manifest()
    match = next((r for r in manifest["runs"] if r["run_id"] == run_id), None)
    if match is None:
        raise ValueError(f"Run outside fixed manifest: {run_id}")
    protected = {"run_id", "dataset_id", "model_id", "seed", "queue_index"}
    for key in protected.intersection(fields):
        if fields[key] != match[key]:
            raise ValueError(f"Cannot change formal run identity/order: {key}")
    if "reason" not in fields and status in ("RUNNING", "COMPLETED"):
        fields["reason"] = "Formal from-scratch run in progress" if status == "RUNNING" else "Formal artifacts and final evaluation verified"
    prospective = dict(match, **fields, status=status)
    if status == "COMPLETED":
        issues = completion_issues(prospective)
        if issues:
            raise ValueError("Cannot mark COMPLETED: " + "; ".join(issues))
    event = {"at": now(), "previous_status": match["status"], "status": status}
    if fields.get("reason"):
        event["reason"] = fields["reason"]
    match.update(fields)
    match["status"] = status
    match["updated_at"] = now()
    attempt = active_attempt_path(match)
    if attempt is not None and (attempt.exists() or fields.get("attempt_dir")):
        attempts = match.setdefault("attempts", [])
        record = next((item for item in attempts if isinstance(item, dict) and item.get("attempt_id") == attempt.name), None)
        if record is None:
            record = {"attempt_id": attempt.name, "directory": str(attempt.relative_to(ROOT)), "created_at": now()}
            attempts.append(record)
        record.update(status=status, updated_at=now())
        if fields.get("reason"):
            record["reason"] = fields["reason"]
    match.setdefault("events", []).append(event)
    manifest["updated_at"] = now()
    write_json(MANIFEST_PATH, manifest)
    return match


def scope_digest() -> str:
    validate_scope()
    return hashlib.sha256(SCOPE_PATH.read_bytes()).hexdigest()


def verify_manifest_artifacts() -> dict[str, Any]:
    manifest = load_manifest()
    issues = {r["run_id"]: completion_issues(r) for r in manifest["runs"] if r["status"] == "COMPLETED"}
    issues = {k: v for k, v in issues.items() if v}
    counts = {status: sum(r["status"] == status for r in manifest["runs"]) for status in sorted(STATUSES)}
    return {"ok": not issues, "formal_configuration_count": 50, "counts": counts, "issues": issues,
            "scope_sha256": scope_digest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify_manifest_artifacts()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
