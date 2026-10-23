"""Independent Phase 2 aggregation; reads/writes only new-stage outputs and metadata.

Usage: python -m phase2_experiment.reporting --package
The independent denominator is 75. Prior Phase 1 metrics are never relabeled.
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
from .scope import DATASETS, MODELS, ROOT, SEEDS, OUTPUT_ROOT, active_attempt_path, completion_issues, load_manifest, now, read_json, verify_manifest_artifacts, write_json
REPORT_ROOT = ROOT / "review_pack_phase2_3datasets"
ZIP_PATH = ROOT / "CHATGPT_REVIEW_PACK_PHASE2_3_DATASETS.zip"
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
    metadata = ["PHASE2_SCOPE.json", "PHASE2_MANIFEST.json", "PHASE2_LOCKED_PROTOCOL.json", "PHASE2_PROGRESS.md",
        "PHASE2_AUTHORIZATION.md", "PHASE2_DEFERRED_BY_USER.json", "PHASE2_AUXILIARY_RUNS.json", "PHASE2_STATISTICAL_ANALYSIS_PLAN.json",
        "PHASE2_MODEL_CHANGELOG.md", "PHASE2_IMPLEMENTATION_LOG.md", "PHASE2_ENVIRONMENT.json", "PHASE2_REPRODUCTION.md",
        "PHASE2_MONITOR_SNAPSHOT.json", "PHASE2_PROCESS_STATE.json", "PHASE2_PROTOCOL_SELECTION.json",
        "PHASE2_DELIVERY_STATUS.json", "PHASE2_USER_SELECTION_CONFIRMATION.json"]
    for name in metadata:
        copy_file(ROOT / name, REPORT_ROOT / name)
    for name in ("theory_fm_models.py", "MODEL_GUIDE.md", "CODEX_RUNBOOK.md", "MODEL_CONTRACT.json"):
        copy_file(ROOT / name, REPORT_ROOT / "source" / name)
    for directory in ("phase2_experiment", "phase2_tests", "phase2_reference"):
        if (ROOT / directory).exists():
            for path in (ROOT / directory).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    copy_file(path, REPORT_ROOT / "source" / path.relative_to(ROOT))
    # The exact original model implementation remains a separate origin copy.
    original = ROOT / "original_snapshot"
    if original.exists():
        for path in original.glob("*"):
            if path.is_file():
                copy_file(path, REPORT_ROOT / "source/original_delivery" / path.name)
    copy_file(ROOT / "experiment/sampling.py", REPORT_ROOT / "source/experiment/sampling.py")
    from .data import data_dir
    for dataset in DATASETS:
        directory = Path(data_dir(dataset))
        if directory.exists():
            for path in directory.glob("*"):
                if path.is_file() and path.suffix.lower() in (".json", ".npz"):
                    copy_file(path, REPORT_ROOT / "data" / dataset / path.name)
    for directory in (ROOT / "outputs/phase2_pilots", ROOT / "outputs/phase2_checks"):
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix.lower() in (".csv", ".json", ".txt", ".log", ".md", ".npy", ".npz", ".py"):
                    copy_file(path, REPORT_ROOT / "nonformal" / path.relative_to(ROOT / "outputs"))
    attempt_rows, failure_rows = [], []
    for rec in records:
        run = rec["run"]
        directories = sorted((OUTPUT_ROOT / run["run_id"]).glob("attempt_*"))
        if rec["attempt"] and rec["attempt"] not in directories:
            directories.append(rec["attempt"])
        for attempt in directories:
            if not attempt.is_dir():
                continue
            copy_attempt_artifacts(run["run_id"], attempt)
            cfg = read_json(attempt / "config.json") if (attempt / "config.json").exists() else {}
            recorded = next((r for r in run.get("attempts", []) if isinstance(r, dict) and r.get("attempt_id") == attempt.name), {})
            attempt_rows.append({"run_id": run["run_id"], "dataset_id": run["dataset_id"], "model_id": run["model_id"], "seed": run["seed"],
                "attempt_id": attempt.name, "attempt_dir": str(attempt.relative_to(ROOT)), "is_current_attempt": attempt == rec["attempt"],
                "manifest_status": run["status"], "attempt_status": recorded.get("status", "PRESERVED_ATTEMPT"),
                "source_sha256": cfg.get("source_sha256", ""), "execution_source_hashes": cfg.get("execution_source_hashes", {}),
                "source_snapshot_files": sorted(str(p.relative_to(attempt / "source")) for p in (attempt / "source").rglob("*.py")),
                "failure_record_present": any(attempt.glob("failure*.json"))})
            for path in attempt.glob("failure*.json"):
                failure = read_json(path)
                failure_rows.append({"run_id": run["run_id"], "dataset_id": run["dataset_id"], "model_id": run["model_id"], "seed": run["seed"],
                    "attempt_id": attempt.name, "failure_source": str(path.relative_to(ROOT)), "error": failure.get("error", failure.get("reason", "")),
                    "traceback": failure.get("traceback", "")})
    write_csv(REPORT_ROOT / "tables/attempt_history.csv", attempt_rows, ["run_id", "dataset_id", "model_id", "seed", "attempt_id", "attempt_dir",
        "is_current_attempt", "manifest_status", "attempt_status", "source_sha256", "execution_source_hashes", "source_snapshot_files", "failure_record_present"])
    write_csv(REPORT_ROOT / "tables/attempt_failure_records.csv", failure_rows, ["run_id", "dataset_id", "model_id", "seed", "attempt_id", "failure_source", "error", "traceback"])


def build_inventory() -> None:
    rows = []
    for directory in (OUTPUT_ROOT, ROOT / "outputs/phase2_pilots", ROOT / "outputs/phase2_checks"):
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    rows.append({"relative_path": str(path.relative_to(ROOT)), "absolute_path": str(path.resolve()), "bytes": path.stat().st_size})
    write_csv(REPORT_ROOT / "full_artifact_inventory.csv", rows, ["relative_path", "absolute_path", "bytes"])
    from .scope import sha256_file
    hashes = []
    for directory in (REPORT_ROOT / "source", REPORT_ROOT / "runs", REPORT_ROOT / "curves", REPORT_ROOT / "samples", REPORT_ROOT / "data", REPORT_ROOT / "nonformal", REPORT_ROOT / "figures"):
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    hashes.append({"path": str(path.relative_to(REPORT_ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(REPORT_ROOT / "review_artifact_hashes.json", hashes)


def write_summary(records: list[dict[str, Any]], verification: dict[str, Any]) -> dict[str, Any]:
    verified = [r for r in records if r["run"]["status"] == "COMPLETED" and not r["issues"]]
    counts = Counter(r["run"]["status"] for r in records)
    summary = {"generated_at": now(), "phase_id": "PHASE2_3_DATASETS", "formal_configuration_count": 75,
        "verified_completed": len(verified), "manifest_status_counts": dict(counts), "datasets": {}, "artifact_verification_ok": verification["ok"],
        "phase1_guard": verification["phase1_guard"], "artifact_issues": {r["run"]["run_id"]: r["issues"] for r in records if r["issues"]}}
    lines = ["# Phase 2 execution summary", "", f"Artifact-verified formal runs completed: **{len(verified)}/75**.", "",
        "The new 75-run denominator contains three selected datasets, five delivered models and five training seeds. Prior Phase 1 results retain their original 25/50 status and are not relabeled or included in this denominator.", ""]
    for dataset in DATASETS:
        group = [r for r in records if r["run"]["dataset_id"] == dataset]
        done = sum(r["run"]["status"] == "COMPLETED" and not r["issues"] for r in group)
        status = Counter(r["run"]["status"] for r in group)
        summary["datasets"][dataset] = {"verified_completed": done, "formal_configuration_count": 25, "manifest_status_counts": dict(status)}
        lines.append(f"- **{dataset}: {done}/25 completed**; " + ", ".join(f"{s}={n}" for s, n in sorted(status.items())) + ".")
    lines += ["", "Every task retains the complete L=64, C=2, D=128 C-order flattened state. Observation time s, channel identity and generation time tau are distinct. Each task's DATA_SPEC records exact formulas and engineering choices. Train-only normalization and split membership are shared by all five models/seeds within that task; datasets are summarized separately.", "",
        "FM_STANDARD primary metrics use same-final-checkpoint Heun samples. Its Euler numbers are retained as standard_euler.* supplementary columns. Other models use their original prescribed history Euler/Euler-Maruyama solver. Solver identity and training/sampling/spatial-computation costs are recorded separately.", "",
        "The tables preserve all 75 identities. Numerical quality summaries contain only artifact-verified COMPLETED runs and report their available training-seed count out of five. Seven fixed paired comparisons comprise three research contrasts (complex ODE minus real ODE, complex SDE minus complex ODE, VDC ODE minus complex ODE) and four model-minus-standard-Heun contrasts. Mean, sample SD, 95% Student t interval, same-seed raw differences and seven-comparison Holm-adjusted tests are separate fields. Interacting particles, projected directions and metric pairs are not independent training repeats. Secondary diagnostic metrics remain descriptive and do not establish a model winner.", "",
        "Metric registration and numerical arrays cover the selected tasks' distribution and temporal-dependency evaluation, with their actual channel/observation axes described in METRIC_SPEC. No IID-only interpretation is imposed on these tasks. Validation pilots do not access formal test, do not initialize formal weights and are excluded from the 75-run denominator.", "",
        "Real-versus-real comparisons reuse a pre-registered fixed reference across model/training-seed runs. Zero training-seed SD for a shared reference reflects this reuse and does not establish a reference with no uncertainty; retained real-reference blocks/arrays document finite-sample variation.", "",
        f"Full formal weights/logs/floating samples: `{OUTPUT_ROOT.resolve()}`. Independent pilot artifacts: `{(ROOT / 'outputs/phase2_pilots').resolve()}`. Full artifact paths and sizes are in full_artifact_inventory.csv. Configurations, actual per-attempt source snapshots, all prior failure/resume records, raw curves, floating samples and metric provenance remain traceable.", "",
        "A formal checkpoint alone does not demonstrate convergence. Final/best weights, complete configured updates, samples and final evaluation are required for COMPLETED. Missing/failed/blocked and deferred items are retained explicitly. Other datasets, control models, ablations and parameter sweeps remain outside this phase. This is factual experiment reporting, not a paper; completion does not authorize further queues.", ""]
    if summary["artifact_issues"]:
        lines += ["Artifact issues excluded from verified completion:", ""]
        lines += [f"- {run}: {'; '.join(issues)}" for run, issues in summary["artifact_issues"].items()]
    (REPORT_ROOT / "EXECUTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(REPORT_ROOT / "execution_summary.json", summary)
    copy_file(REPORT_ROOT / "EXECUTION_SUMMARY.md", ROOT / "PHASE2_EXECUTION_SUMMARY.md")
    incomplete = [r for r in records if r["run"]["status"] != "COMPLETED" or r["issues"]]
    missing = ["# Phase 2 failures, missing evidence and pending work", "", f"{len(incomplete)} of 75 formal configurations are not artifact-verified complete.", ""]
    missing += [f"- {r['run']['run_id']}: {r['run']['status']}; {r['run'].get('reason', '')}; {'; '.join(r['issues'])}" for r in incomplete]
    missing += ["", "Prior failures are retained in tables/attempt_failure_records.csv and runs/<run_id>/<attempt_id>/failure*.json. Deferred work is separate in PHASE2_DEFERRED_BY_USER.json. Prior Phase 1 robot blockers remain under their original unchanged 50-run scope."]
    (REPORT_ROOT / "MISSING_RESULTS.md").write_text("\n".join(missing) + "\n", encoding="utf-8")
    commands = """# From D:\\work_space\\fm_theory_rebuild_v1; use PHASE2_ENVIRONMENT.json / recorded source versions.
python -m phase2_experiment.runner --help
python -m phase2_experiment.scope --verify
python -m phase2_experiment.verify
python -m phase2_experiment.reporting
python -m phase2_experiment.plots
python -m phase2_experiment.reporting --package
python -m phase2_experiment.runner resume
# Follow an already-running exact formal PID, then automatically verify and deliver:
python -m phase2_experiment.finish --attach
# Existing queue must be stopped for same-attempt recovery; no protocol/matrix changes:
python -m phase2_experiment.finish --resume
# Rebuild delivery only after 75 artifact-verified completions; starts no training:
python -m phase2_experiment.finish --delivery-only
# Reproduce only in an independent empty project copy with the recorded original model and Phase 2 source/config/data.
# Select/initialize the exact three-dataset scope before running; initialize_scope refuses to overwrite an existing manifest.
# Review runner CLI for explicit dataset, budget and pilot arguments; never call an all/full queue.
python -m phase2_experiment.runner prepare --dataset SEQ_DELAY
python -m phase2_experiment.runner prepare --dataset SEQ_ACCUM
python -m phase2_experiment.runner prepare --dataset SEQ_MULTISCALE
python -m phase2_experiment.runner pilot --dataset SEQ_DELAY --pilot-steps 200
python -m phase2_experiment.runner pilot --dataset SEQ_ACCUM --pilot-steps 200
python -m phase2_experiment.runner pilot --dataset SEQ_MULTISCALE --pilot-steps 200
# Lock each dataset only with its previously selected budget/rationale/N/H from PHASE2_LOCKED_PROTOCOL.json.
# Never select or alter a budget based on formal test scores. Example syntax (replace placeholders with recorded values):
# python -m phase2_experiment.runner lock --dataset <dataset> --updates <recorded_tier> --rationale <recorded_validation_and_cost_reason> --N <recorded_N> --H <recorded_H>
python -m phase2_experiment.runner run
# Formal training requires all independent selected-task pilots and each task's locked 12000/24000/48000-update protocol.
# Resume preserves attempt/checkpoint/RNG/optimizer provenance; Phase 1 files and outputs remain immutable.
"""
    (REPORT_ROOT / "repro_commands.txt").write_text(commands, encoding="utf-8")
    if (ROOT / "PHASE2_REPRODUCTION.md").exists():
        with (REPORT_ROOT / "repro_commands.txt").open("a", encoding="utf-8") as handle:
            handle.write("\n# Detailed source/data/checkpoint reproduction instructions: PHASE2_REPRODUCTION.md\n")
    (REPORT_ROOT / "README.md").write_text("# Independent Phase 2 review pack\n\nStart with EXECUTION_SUMMARY.md. The fixed denominator is 75, separately 25 per selected task. Raw tables, per-attempt sources, frozen data, curves, float samples, costs, failures and provenance are retained. Full weights/logs/samples are listed in full_artifact_inventory.csv. No prior Phase 1 result is relabeled.\n\nGenerate scientific figures after aggregation with `python -m phase2_experiment.plots`, then package with `python -m phase2_experiment.reporting --package`. Figures retain SVG/PNG, exact CSV/NPZ inputs and separate source/image hashes. Partial groups are labeled n/5; plots use final-evaluation artifacts and never open the original test NPZ.\n", encoding="utf-8")
    return summary


def aggregate(package: bool = False) -> dict[str, Any]:
    manifest = load_manifest()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    records = [read_attempt(run) for run in manifest["runs"]]
    create_tables(records)
    copy_review_artifacts(records)
    build_inventory()
    verification = verify_manifest_artifacts()
    summary = write_summary(records, verification)
    write_json(REPORT_ROOT / "artifact_verification.json", verification)
    if package:
        temporary = ZIP_PATH.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(REPORT_ROOT.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=str(path.relative_to(REPORT_ROOT)))
        temporary.replace(ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            if corrupt := archive.testzip():
                raise RuntimeError(f"Phase 2 ZIP verification failed: {corrupt}")
        summary.update(package=str(ZIP_PATH), package_bytes=ZIP_PATH.stat().st_size)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.package), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


