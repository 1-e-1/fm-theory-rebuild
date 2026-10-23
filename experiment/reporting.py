"""Fact-based Phase 1 aggregation and review-pack creation, including partial runs.

Usage: python -m experiment.reporting --package
Full checkpoints, logs and samples stay under outputs/phase1_iid_robot. The
review pack includes raw tables, source/provenance, training curves, diagnostics
and generated floating samples, and inventories every preserved full artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import stat
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .scope import DATASETS, MODELS, ROOT, SEEDS, active_attempt_path, completion_issues, load_manifest, now, read_json, resolve_path, verify_manifest_artifacts, write_json

REPORT_ROOT = ROOT / "review_pack_phase1_50"
ZIP_PATH = ROOT / "CHATGPT_REVIEW_PACK_PHASE1_50.zip"
OUTPUT_ROOT = ROOT / "outputs/phase1_iid_robot"
COMPARISONS = (
    ("FM_COMPLEX_ODE", "FM_REAL_ODE"),
    ("FM_COMPLEX_SDE", "FM_COMPLEX_ODE"),
    ("FM_COMPLEX_VDC_ODE", "FM_COMPLEX_ODE"),
    *((model, "FM_STANDARD") for model in MODELS[1:]),
)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def flatten_numbers(mapping: dict[str, Any], prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_numbers(value, name))
        elif finite_number(value):
            result[name] = float(value)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def moments(values: list[float]) -> dict[str, Any]:
    n = len(values)
    result: dict[str, Any] = {"n_training_seeds": n, "expected_training_seeds": 5,
        "complete_seed_group": n == 5, "mean": "", "std": "", "ci95_low": "", "ci95_high": ""}
    if n:
        result["mean"] = statistics.mean(values)
    if n >= 2:
        from scipy.stats import t
        sd = statistics.stdev(values)
        radius = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        result.update(std=sd, ci95_low=result["mean"] - radius, ci95_high=result["mean"] + radius)
    return result


def holm(pvalues: list[tuple[int, float]], total_family_size: int = 7) -> dict[int, float]:
    # Keep the pre-registered family size even when some comparisons are unavailable.
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (index, pvalue) in enumerate(sorted(pvalues, key=lambda item: item[1])):
        running = min(1.0, max(running, (total_family_size - rank) * pvalue))
        adjusted[index] = running
    return adjusted


def read_attempt(run: dict[str, Any]) -> dict[str, Any]:
    attempt = active_attempt_path(run)
    record: dict[str, Any] = {"run": run, "attempt": attempt, "config": {}, "evaluation": {}, "evaluation_heun": {}, "training_cost": {},
        "metrics": {}, "supplementary_metrics": {}, "diagnostics": {}, "issues": [], "primary_solver": "", "primary_metrics_source": ""}
    if attempt is None:
        return record
    for key, filename in (("config", "config.json"), ("evaluation", "evaluation.json"), ("evaluation_heun", "evaluation_heun.json"),
                          ("diagnostics", "sampling_diagnostics.json"), ("training_cost", "training_cost.json")):
        path = attempt / filename
        if path.exists():
            try:
                record[key] = read_json(path)
            except Exception as exc:
                record["issues"].append(f"{filename}: {type(exc).__name__}: {exc}")
    if run["status"] == "COMPLETED":
        record["issues"].extend(completion_issues(run))
        if not record["issues"]:
            standard = run["model_id"] == "FM_STANDARD"
            evaluation = record["evaluation_heun"] if standard else record["evaluation"]
            record["primary_metrics_source"] = "evaluation_heun.json" if standard else "evaluation.json"
            record["primary_solver"] = "Heun" if standard else ("Euler-Maruyama" if run["model_id"] == "FM_COMPLEX_SDE" else "PL-history Euler")
            raw_metrics = evaluation.get("metrics", evaluation)
            record["metrics"] = flatten_numbers(raw_metrics)
            if standard:
                record["supplementary_metrics"] = {"standard_euler." + k: v for k, v in flatten_numbers(record["evaluation"].get("metrics", record["evaluation"])).items()}
            if "metrics" not in evaluation:
                # Avoid presenting resource quantities/identity as quality metrics.
                for key in list(record["metrics"]):
                    if key.startswith(("compute.", "resources.", "timing.", "seed", "sample_count", "parameter_count")):
                        record["metrics"].pop(key)
    return record


def create_tables(records: list[dict[str, Any]]) -> None:
    directory = REPORT_ROOT / "tables"
    metric_names = sorted({k for rec in records for k in list(rec["metrics"]) + list(rec["supplementary_metrics"])})
    main_rows = []
    metric_rows = []
    compute_rows = []
    failure_rows = []
    for rec in records:
        run = rec["run"]
        attempt = rec["attempt"]
        common = {k: run[k] for k in ("run_id", "dataset_id", "model_id", "seed", "status")}
        common["queue_index"] = run["queue_index"]
        common["attempt_dir"] = str(attempt.relative_to(ROOT)) if attempt else ""
        common["reason"] = run.get("reason", "")
        row = dict(common, artifact_validation_issues="; ".join(rec["issues"]))
        row.update(primary_solver=rec["primary_solver"], primary_metrics_source=rec["primary_metrics_source"])
        row.update({metric: {**rec["metrics"], **rec["supplementary_metrics"]}.get(metric, "") for metric in metric_names})
        main_rows.append(row)
        metric_rows.extend(dict(common, metric=k, value=v) for k, v in rec["metrics"].items())
        compute = dict(common)
        cfg, evaluation = rec["config"], rec["evaluation"]
        compute["primary_quality_solver"] = rec["primary_solver"]
        for key in ("parameter_count", "parameter_counts", "parameters", "effective_batch", "query_batch", "N", "H", "Q", "reference_particles", "history_nodes", "sampling_steps", "solver", "dtype", "device", "updates", "steps", "microbatch"):
            if key in cfg:
                compute[key] = cfg[key]
        for section in ("compute", "resources", "timing"):
            value = evaluation.get(section)
            if isinstance(value, dict):
                compute.update({f"{section}.{k}": v for k, v in value.items()})
        for key in ("train_seconds", "training_seconds", "sampling_seconds", "peak_memory_bytes", "peak_memory_mb", "parameter_count", "drift_calls", "potential_evaluation_points", "spatial_derivative_calls"):
            if key in evaluation:
                compute[key] = evaluation[key]
        compute.update({"training." + k: v for k, v in rec["training_cost"].items()})
        compute.update({"sampling." + k: v for k, v in flatten_numbers(rec["diagnostics"]).items()})
        for section, section_value in rec["diagnostics"].items():
            if isinstance(section_value, dict):
                for key in ("solver", "initial_cell", "dtype", "device", "history_mode"):
                    if key in section_value:
                        compute[f"sampling.{section}.{key}"] = section_value[key]
        if attempt and (attempt / "training.csv").exists():
            with (attempt / "training.csv").open(newline="", encoding="utf-8-sig") as handle:
                last = None
                for last in csv.DictReader(handle):
                    pass
            if last:
                compute["recorded_training_step"] = last.get("step", "")
                compute["recorded_train_elapsed_seconds"] = last.get("elapsed_seconds", "")
        compute_rows.append(compute)
        if run["status"] != "COMPLETED" or rec["issues"]:
            failure_rows.append(dict(common, artifact_validation_issues="; ".join(rec["issues"])))
    basic_fields = ["run_id", "dataset_id", "model_id", "seed", "status", "queue_index", "attempt_dir", "reason", "artifact_validation_issues"]
    write_csv(directory / "main_per_seed.csv", main_rows, basic_fields + ["primary_solver", "primary_metrics_source"] + metric_names)
    write_csv(directory / "metric_values.csv", metric_rows, basic_fields[:-1] + ["metric", "value"])
    write_csv(directory / "compute_costs.csv", compute_rows)
    write_csv(directory / "failure_missing.csv", failure_rows, basic_fields)

    summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    lookup = {(rec["run"]["dataset_id"], rec["run"]["model_id"], rec["run"]["seed"]): rec["metrics"] for rec in records}
    for dataset in DATASETS:
        metrics = sorted({k for rec in records if rec["run"]["dataset_id"] == dataset for k in rec["metrics"]})
        for model in MODELS:
            if not metrics:
                summaries.append(dict(dataset_id=dataset, model_id=model, metric="NOT_AVAILABLE", **moments([])))
            for metric in metrics:
                values = [lookup[(dataset, model, seed)][metric] for seed in SEEDS if metric in lookup[(dataset, model, seed)]]
                summaries.append(dict(dataset_id=dataset, model_id=model, metric=metric, **moments(values)))
        for metric in metrics:
            pvalues = []
            for model, reference_model in COMPARISONS:
                differences = []
                for seed in SEEDS:
                    actual, reference = lookup[(dataset, model, seed)], lookup[(dataset, reference_model, seed)]
                    if metric in actual and metric in reference:
                        delta = actual[metric] - reference[metric]
                        differences.append(delta)
                        delta_rows.append({"dataset_id": dataset, "model_id": model, "reference_model_id": reference_model, "seed": seed,
                            "metric": metric, "model_value": actual[metric], "reference_value": reference[metric], "paired_difference": delta})
                row = {"dataset_id": dataset, "model_id": model, "reference_model_id": reference_model, "metric": metric,
                    "reference_solver": "Heun" if reference_model == "FM_STANDARD" else "PL-history Euler",
                    "difference_definition": "model minus " + reference_model, **moments(differences), "raw_p": "", "holm_p": "", "paired_effect_dz": ""}
                if len(differences) >= 2:
                    from scipy.stats import t
                    sd = statistics.stdev(differences)
                    mean = statistics.mean(differences)
                    if sd > 0:
                        test_t = mean / (sd / math.sqrt(len(differences)))
                        pvalue = float(2 * t.sf(abs(test_t), len(differences) - 1))
                        row["paired_effect_dz"] = mean / sd
                    else:
                        pvalue = 1.0 if mean == 0 else 0.0
                        row["paired_effect_dz"] = "undefined_zero_variance"
                    row["raw_p"] = pvalue
                    pvalues.append((len(comparisons), pvalue))
                comparisons.append(row)
            for index, adjusted in holm(pvalues, len(COMPARISONS)).items():
                comparisons[index]["holm_p"] = adjusted
    write_csv(directory / "summary.csv", summaries)
    comparison_fields = ["dataset_id", "model_id", "reference_model_id", "reference_solver", "metric", "difference_definition", "n_training_seeds", "expected_training_seeds", "complete_seed_group", "mean", "std", "ci95_low", "ci95_high", "raw_p", "holm_p", "paired_effect_dz"]
    write_csv(directory / "paired_comparisons.csv", comparisons, comparison_fields)
    write_csv(directory / "paired_per_seed.csv", delta_rows, ["dataset_id", "model_id", "reference_model_id", "seed", "metric", "model_value", "reference_value", "paired_difference"])

    robot_task: list[dict[str, Any]] = []
    robot_candidate: list[dict[str, Any]] = []
    for rec in records:
        if rec["run"]["dataset_id"] != "ROBOT_8DOF" or not rec["attempt"]:
            continue
        for filename, destination in (("robot_per_task.csv", robot_task), ("robot_per_candidate.csv", robot_candidate)):
            path = rec["attempt"] / filename
            if path.exists():
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    destination.extend(dict({k: rec["run"][k] for k in ("run_id", "model_id", "seed")}, **row) for row in csv.DictReader(handle))
    # Empty header-only tables represent absent robot observations, never zero performance.
    write_csv(directory / "robot_per_task.csv", robot_task, list(dict.fromkeys(["run_id", "model_id", "seed", "task_id", "status", "generated_candidate_count", "raw_feasible_count", "selection_status", "selected_candidate_id", "endpoint_error", "path_length", "jerk_rms"] + [k for row in robot_task for k in row])))
    write_csv(directory / "robot_per_candidate.csv", robot_candidate, list(dict.fromkeys(["run_id", "model_id", "seed", "task_id", "candidate_id", "feasible", "collision", "endpoint_error", "path_length", "jerk_rms"] + [k for row in robot_candidate for k in row])))


def copy_file(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        # copy2 preserves Windows read-only attributes from original snapshots;
        # unlock only the generated review copy before refreshing that copy.
        if destination.exists() and destination.resolve().is_relative_to(REPORT_ROOT):
            destination.chmod(destination.stat().st_mode | stat.S_IWRITE)
        shutil.copy2(source, destination)


def copy_attempt_artifacts(run_id: str, attempt: Path) -> None:
    """Preserve the actual attempt snapshot and every failure/resume record."""
    base = REPORT_ROOT / "runs" / run_id / attempt.name
    for filename in ("config.json", "evaluation.json", "evaluation_heun.json", "sampling_diagnostics.json", "checkpoint_checks.json",
                     "training_cost.json", "metric_provenance.json", "failure.json", "status.json", "orders.csv", "compute.json"):
        copy_file(attempt / filename, base / filename)
    for directory in (attempt, attempt / "heun_metrics"):
        if directory.exists():
            for path in directory.glob("*"):
                if path.is_file() and (path.suffix in (".npz", ".csv") or path.name in
                    ("metrics.json", "metric_definitions.json", "metric_provenance.json")) and not path.name.startswith("training"):
                    copy_file(path, base / path.relative_to(attempt))
    for path in attempt.glob("training*.csv"):
        copy_file(path, REPORT_ROOT / "curves" / run_id / attempt.name / path.name)
    for path in attempt.glob("failure*"):
        if path.is_file():
            copy_file(path, base / path.name)
    for directory_name in ("source", "checkpoint_checks"):
        directory = attempt / directory_name
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    copy_file(path, base / path.relative_to(attempt))
    for samples in attempt.glob("samples*.npy"):
        destination = REPORT_ROOT / "samples" / run_id / attempt.name
        if samples.stat().st_size <= 64 * 1024 * 1024:
            copy_file(samples, destination / samples.name)
        else:
            import numpy as np
            original = np.load(samples, mmap_mode="r", allow_pickle=False)
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / ("representative_" + samples.name), original[: min(16, len(original))])
            write_json(destination / (samples.name + ".provenance.json"), {"source": str(samples), "full_shape": list(original.shape),
                "selection": "first up to 16 entries of axis 0", "clipping_or_quantization": False})


def copy_review_artifacts(records: list[dict[str, Any]]) -> None:
    # Enumerate only this project; no historical directories or external assets.
    files = ["PHASE1_SCOPE.json", "manifest.json", "AGENTS.md", "PROGRESS.md", "MISSING_RESULTS.md", "DEFERRED_BY_USER.json", "AUXILIARY_RUNS.json",
             "ROBOT_RESOURCE_AUDIT.json", "STATISTICAL_ANALYSIS_PLAN.json", "MODEL_CHANGELOG.md", "MODEL_SPEC.json", "MODEL_CONTRACT.json",
             "DATA_SPEC.json", "LOCKED_PROTOCOL.json", "ENVIRONMENT.json", "EXPERIMENT_IMPLEMENTATION_LOG.md", "FILES_SHA256.json", "START_CODEX_PHASE1_50_RUNS.txt", "COMPUTE_ACCOUNTING_NOTES.md", "REPRODUCTION.md", "RESULT_OBSERVATIONS.md"]
    for name in files:
        copy_file(ROOT / name, REPORT_ROOT / name)
    for name in ("theory_fm_models.py", "MODEL_GUIDE.md", "CODEX_RUNBOOK.md"):
        copy_file(ROOT / name, REPORT_ROOT / "source" / name)
    for directory in ("experiment", "tests", "configs", "original_snapshot", "source_reference"):
        source_dir = ROOT / directory
        if source_dir.exists():
            for path in source_dir.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    copy_file(path, REPORT_ROOT / "source" / path.relative_to(ROOT))
    for name in ("MODEL_DIFF.patch", "MODEL_CHANGELOG.md"):
        copy_file(ROOT / name, REPORT_ROOT / "source" / name)
    for path in ROOT.glob("*.patch"):
        copy_file(path, REPORT_ROOT / "source" / path.name)
    # Frozen splits, train-only normalization, metric registration and reference
    # caches are small enough to deliver byte-for-byte in this phase's pack.
    data_directory = ROOT / "data/seq_iid"
    if data_directory.exists():
        for path in data_directory.glob("*"):
            if path.is_file() and path.suffix.lower() in (".json", ".npz"):
                copy_file(path, REPORT_ROOT / "data/seq_iid" / path.name)
    copy_file(OUTPUT_ROOT / "nonformal_registry.json", REPORT_ROOT / "nonformal_registry.json")
    for directory in (OUTPUT_ROOT / "pilots", OUTPUT_ROOT / "checks", ROOT / "outputs/pilots", ROOT / "outputs/checks"):
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix.lower() in (".csv", ".json", ".txt", ".log", ".md", ".npy", ".npz", ".py") and "__pycache__" not in path.parts:
                    copy_file(path, REPORT_ROOT / "nonformal" / path.relative_to(ROOT / "outputs"))
    attempt_rows = []
    failure_rows = []
    for rec in records:
        run_id = rec["run"]["run_id"]
        attempt_directories = sorted((OUTPUT_ROOT / run_id).glob("attempt_*"))
        if rec["attempt"] and rec["attempt"] not in attempt_directories:
            attempt_directories.append(rec["attempt"])
        for attempt in attempt_directories:
            if not attempt.is_dir():
                continue
            copy_attempt_artifacts(run_id, attempt)
            config_path = attempt / "config.json"
            cfg = read_json(config_path) if config_path.exists() else {}
            known = next((item for item in rec["run"].get("attempts", []) if isinstance(item, dict) and item.get("attempt_id") == attempt.name), {})
            attempt_rows.append({"run_id": run_id, "dataset_id": rec["run"]["dataset_id"], "model_id": rec["run"]["model_id"],
                "seed": rec["run"]["seed"], "attempt_id": attempt.name, "attempt_dir": str(attempt.relative_to(ROOT)),
                "is_current_attempt": attempt == rec["attempt"], "manifest_run_status": rec["run"]["status"],
                "registered_attempt_status": known.get("status", "UNREGISTERED_PRESERVED_ATTEMPT"),
                "source_sha256": cfg.get("source_sha256", ""), "execution_source_hashes": cfg.get("execution_source_hashes", {}),
                "actual_source_snapshot_files": sorted(p.name for p in (attempt / "source").glob("*.py")),
                "failure_record_present": (attempt / "failure.json").exists()})
            for failure_path in attempt.glob("failure*.json"):
                failure = read_json(failure_path)
                failure_rows.append({"run_id": run_id, "dataset_id": rec["run"]["dataset_id"], "model_id": rec["run"]["model_id"],
                    "seed": rec["run"]["seed"], "attempt_id": attempt.name, "failure_source": str(failure_path.relative_to(ROOT)),
                    "error": failure.get("error", failure.get("reason", "")), "traceback": failure.get("traceback", "")})
    write_csv(REPORT_ROOT / "tables/attempt_history.csv", attempt_rows, ["run_id", "dataset_id", "model_id", "seed", "attempt_id", "attempt_dir", "is_current_attempt",
        "manifest_run_status", "registered_attempt_status", "source_sha256", "execution_source_hashes", "actual_source_snapshot_files", "failure_record_present"])
    write_csv(REPORT_ROOT / "tables/attempt_failure_records.csv", failure_rows, ["run_id", "dataset_id", "model_id", "seed", "attempt_id", "failure_source", "error", "traceback"])


def build_inventory() -> None:
    rows = []
    all_outputs = ROOT / "outputs"
    if all_outputs.exists():
        for path in sorted(all_outputs.rglob("*")):
            if path.is_file():
                rows.append({"relative_path": str(path.relative_to(ROOT)), "absolute_path": str(path.resolve()), "bytes": path.stat().st_size})
    write_csv(REPORT_ROOT / "full_artifact_inventory.csv", rows, ["relative_path", "absolute_path", "bytes"])
    source_hashes = []
    for directory in (REPORT_ROOT / "source", REPORT_ROOT / "runs", REPORT_ROOT / "curves", REPORT_ROOT / "samples", REPORT_ROOT / "data", REPORT_ROOT / "nonformal"):
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    source_hashes.append({"path": str(path.relative_to(REPORT_ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    write_json(REPORT_ROOT / "review_artifact_hashes.json", source_hashes)


def write_summary(manifest: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    verified_completed = [rec for rec in records if rec["run"]["status"] == "COMPLETED" and not rec["issues"]]
    completed = len(verified_completed)
    status = Counter(rec["run"]["status"] for rec in records)
    summary = {"generated_at": now(), "formal_configuration_count": 50, "verified_completed": completed,
               "manifest_status_counts": dict(status), "datasets": {}, "artifact_issues": {r["run"]["run_id"]: r["issues"] for r in records if r["issues"]}}
    lines = ["# Phase 1 execution summary", "", f"Verified formal configurations completed: **{completed}/50**.", "",
             "The 50-run denominator is fixed. Tests/pilots/sampling checks and DEFERRED_BY_USER tasks are excluded from that denominator.", ""]
    for dataset in DATASETS:
        group = [rec for rec in records if rec["run"]["dataset_id"] == dataset]
        count = sum(rec in verified_completed for rec in group)
        counts = Counter(rec["run"]["status"] for rec in group)
        summary["datasets"][dataset] = {"verified_completed": count, "formal_configuration_count": 25, "manifest_status_counts": dict(counts)}
        lines.append(f"- **{dataset}: {count}/25 completed**; " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + ".")
    lines.extend(["", "Robot evidence and precise missing requirements are in `ROBOT_RESOURCE_AUDIT.json` / `MISSING_RESULTS.md`. No alternate robot, geometric proxy or expert fallback has been used. Empty robot tables indicate no observations; they do not indicate zero errors or zero collisions.", "",
        "Only artifact-verified COMPLETED runs contribute numerical metrics. `main_per_seed.csv` retains all 50 identities and states. FM_STANDARD main metrics use the same-checkpoint Heun reference; its Euler metrics remain in explicit `standard_euler.*` columns. Other main models use their original prescribed Euler/Euler-Maruyama history scheme. Solver/source columns and separate sampling costs identify these choices. `summary.csv` labels the available training-seed count out of five; an incomplete seed group is a partial statistic. `paired_per_seed.csv` retains the three research contrasts (complex ODE minus real ODE, complex SDE minus complex ODE, VDC ODE minus complex ODE), plus four model-minus-standard-Heun contrasts. The 95% Student t intervals and seven-comparison Holm family per dataset/metric are described in `STATISTICAL_ANALYSIS_PLAN.json`; particles and projections are not treated as training repeats.", "",
        f"Full formal checkpoint, training-log and floating-sample directory: `{OUTPUT_ROOT.resolve()}`. Nonformal pilot artifacts are separately retained under `{(ROOT / 'outputs/pilots').resolve()}`. `full_artifact_inventory.csv` provides concrete paths and sizes for all outputs. Review copies of curves, configuration, evaluation, sampling diagnostics and floating samples preserve numeric source arrays.", "",
        "Training state, checkpoint choice and formal budgets must be taken from the locked protocol and each preserved attempt. Reaching a budget does not demonstrate convergence. Original and validation-best/final weights must remain under the full output directory. Prior failures and retries are retained.", "",
        "Factual distribution-quality observations and metric means are in RESULT_OBSERVATIONS.md; reaching the training budget does not establish convergence. Exact frozen-protocol reproduction, reaggregation and checkpoint recovery commands are in REPRODUCTION.md.", "",
        "This package reports experiments and factual progress. SEQ_IID evidence is limited to independent sequence generation and establishes no long-memory claim. No model winner is predetermined. Completion of this phase does not authorize deferred experiments.", ""])
    if summary["artifact_issues"]:
        lines.extend(["Artifact validation issues (excluded from verified completed count):", ""])
        lines.extend(f"- `{run_id}`: {'; '.join(issues)}" for run_id, issues in summary["artifact_issues"].items())
    (REPORT_ROOT / "EXECUTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(REPORT_ROOT / "execution_summary.json", summary)
    # Publish a convenient current factual summary at the project root.
    copy_file(REPORT_ROOT / "EXECUTION_SUMMARY.md", ROOT / "EXECUTION_SUMMARY.md")
    commands = """# Run from D:\\work_space\\fm_theory_rebuild_v1; use the environment recorded in ENVIRONMENT.json.
# Inspect installed CLI options before relying on optional arguments:
python -m experiment.runner --help
python -m experiment.scope --verify
python -m experiment.verify
python -m experiment.reporting --package
# Resume the same fixed formal manifest, preserving attempts and RNG/optimizer state:
python -m experiment.runner resume
# Exact from-scratch reproduction: use the fresh-copy PowerShell procedure in REPRODUCTION.md.
# It copies frozen data and LOCKED_PROTOCOL.json, creates a fresh 50-row manifest,
# and trains the 25 available IID runs. Do not re-lock or initialize this completed project.
# Correctness, pilot and protocol-selection evidence is already preserved separately.
# Formal dataset/model/seed triples are validated at every launch/resume.
# BLOCKED robot runs require verified user robot assets and data before execution.
"""
    (REPORT_ROOT / "repro_commands.txt").write_text(commands, encoding="utf-8")
    (REPORT_ROOT / "README.md").write_text(
        "# Phase 1 review pack\n\nStart with EXECUTION_SUMMARY.md. The scope contains 50 formal runs. "
        "Tables are in tables/; source, original snapshots, configuration and model-change evidence are in source/. "
        "Full weights/logs/samples remain at the paths in full_artifact_inventory.csv. "
        "See RESULT_OBSERVATIONS.md for factual results and REPRODUCTION.md for executable commands. "
        "Do not treat blocked or deferred rows as successful experiments.\n", encoding="utf-8")
    return summary


def aggregate(package: bool = False) -> dict[str, Any]:
    from .nonformal_registry import refresh_registry
    refresh_registry()
    manifest = load_manifest()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    records = [read_attempt(run) for run in manifest["runs"]]
    create_tables(records)
    copy_review_artifacts(records)
    build_inventory()
    summary = write_summary(manifest, records)
    write_json(REPORT_ROOT / "artifact_verification.json", verify_manifest_artifacts())
    if package:
        temporary = ZIP_PATH.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(REPORT_ROOT.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=str(path.relative_to(REPORT_ROOT)))
        temporary.replace(ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise RuntimeError(f"Review pack ZIP verification failed: {corrupt}")
        summary["package"] = str(ZIP_PATH)
        summary["package_bytes"] = ZIP_PATH.stat().st_size
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true", help="Aggregate then create and verify CHATGPT_REVIEW_PACK_PHASE1_50.zip")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.package), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
