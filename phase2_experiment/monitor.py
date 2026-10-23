"""Read Phase 2 progress and atomically register verified nonformal pilots.

Never starts training, changes formal runs or opens test data. Independent
pilots remain outside the fixed 75 formal configurations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
import time
from typing import Any

from .scope import (DATASETS, ROOT, active_attempt_path, load_manifest, local_now,
                    now, read_json, run_identifier, sha256_file, validate_config)


def _atomic_text(path: Path, value: str) -> None:
    """Retry brief Windows sharing locks; retain failed temporaries as evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
            dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * (2 ** attempt))


@contextmanager
def _registry_lock():
    """Serialize monitor writers; artifact scanning happens outside this lock."""
    path = ROOT / ".PHASE2_AUXILIARY_RUNS.monitor.lock"
    deadline = time.monotonic() + 2
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("Another monitor is updating the auxiliary registry")
            time.sleep(.05)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _training_progress(path: Path) -> dict[str, Any]:
    result = {"last_flushed_step": None, "elapsed_seconds": None, "loss": None,
              "complete_log_rows": 0, "steps": []}
    if not path.is_file():
        return result
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            # A concurrent runner can be appending the last line.
            try:
                step, elapsed, loss = int(row["step"]), float(row["elapsed_seconds"]), float(row["loss"])
                if not (math.isfinite(elapsed) and math.isfinite(loss)) or row.get(None):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            result.update(last_flushed_step=step, elapsed_seconds=elapsed, loss=loss)
            result["complete_log_rows"] += 1
            result["steps"].append(step)
    return result


def _pid_alive(pid: Any) -> bool | None:
    """Read liveness; never send a Windows signal to another process."""
    if type(pid) is not int or pid <= 0:
        return None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False if ctypes.get_last_error() in (87, 1168) else None
    try:
        code = wintypes.DWORD()
        return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
    finally:
        kernel.CloseHandle(handle)


def _process_state() -> dict[str, Any] | None:
    path = ROOT / "PHASE2_PROCESS_STATE.json"
    if not path.is_file():
        return None
    result = read_json(path)
    result["pid_alive_at_snapshot"] = _pid_alive(result.get("pid"))
    return result


def _pilot_completion_issues(attempt: Path, config: dict[str, Any], progress: dict[str, Any]) -> list[str]:
    required = ("config.json", "pilot_result.json", "checkpoint_final.pt", "checkpoint_best.pt",
        "training.csv", "training_cost.json", "pilot_samples.npy", "sampling_diagnostics.json",
        "pilot_validation_diagnostics.json", "source/core_execution_source_hashes.json")
    issues = [f"missing {name}" for name in required if not (attempt / name).is_file()]
    # pilot_result is the runner's final write, after sampling and validation.
    # No active checkpoint is loaded until that final marker exists.
    if issues:
        return issues
    try:
        import numpy as np
        import torch
        steps = config["steps"]
        if type(steps) is not int or not 1 <= steps <= 2000:
            issues.append("invalid short-pilot budget")
        if progress["steps"] != list(range(1, steps + 1)):
            issues.append("training log does not contain the complete pilot budget")
        result = read_json(attempt / "pilot_result.json")
        identity = {key: config[key] for key in ("dataset_id", "model_id", "seed")}
        if any(result.get(key) != value for key, value in identity.items()):
            issues.append("pilot-result identity mismatch")
        if result.get("kind") != "pilot_not_formal" or result.get("test_accessed") is not False:
            issues.append("pilot result does not explicitly exclude formal-test access")
        for key in ("validation_loss", "final_validation_loss", "training_seconds"):
            if not math.isfinite(float(result[key])):
                issues.append(f"nonfinite {key}")
        validation = read_json(attempt / "pilot_validation_diagnostics.json")
        if validation.get("test_accessed") is not False or validation.get("small_sample_nonformal") is not True:
            issues.append("diagnostics are not explicitly nonformal/validation-only")
        if read_json(attempt / "training_cost.json").get("updates") != steps:
            issues.append("training cost budget mismatch")
        hashes = config.get("execution_source_hashes", {})
        if not hashes or read_json(attempt / "source/core_execution_source_hashes.json") != hashes:
            issues.append("core source hash registry mismatch")
        for name, digest in hashes.items():
            source = (attempt / "source" / name).resolve()
            if not source.is_relative_to((attempt / "source").resolve()) or not source.is_file() or sha256_file(source) != digest:
                issues.append(f"source snapshot mismatch: {name}")
        spec = (ROOT / config["data_spec_path"]).resolve()
        if not spec.is_relative_to((ROOT / "data/phase2_3datasets").resolve()) or not spec.is_file() or sha256_file(spec) != config["data_spec_sha256"]:
            issues.append("frozen task data specification mismatch")
        if config.get("model_config", {}).get("state_dim") != 128:
            issues.append("pilot does not use complete D128 state")
        cp_identity = {key: config[key] for key in ("dataset_id", "model_id", "seed", "kind", "run_id", "steps")}
        for name in ("checkpoint_final.pt", "checkpoint_best.pt"):
            checkpoint = torch.load(attempt / name, map_location="cpu", weights_only=False)
            step = checkpoint.get("step")
            valid_step = step == steps if name == "checkpoint_final.pt" else type(step) is int and 1 <= step <= steps
            if not valid_step:
                issues.append(f"{name} step mismatch")
            if checkpoint.get("run_identity") != cp_identity or checkpoint.get("model_sha256") != config["source_sha256"]:
                issues.append(f"{name} identity/source mismatch")
            if checkpoint.get("data_spec_sha256") != config["data_spec_sha256"] or checkpoint.get("execution_source_hashes") != hashes:
                issues.append(f"{name} provenance mismatch")
            if not checkpoint.get("model") or not all(torch.isfinite(value).all() for value in checkpoint["model"].values() if isinstance(value, torch.Tensor)):
                issues.append(f"{name} missing/nonfinite parameters")
        samples = np.load(attempt / "pilot_samples.npy", mmap_mode="r", allow_pickle=False)
        try:
            if samples.shape != (2, 32, 128) or samples.dtype.kind != "f" or not np.isfinite(samples).all():
                issues.append("pilot samples must be finite float [2,32,128]")
        finally:
            samples._mmap.close()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        issues.append(f"pilot artifact read failed: {exc}")
    return issues


def _completed_pilot_activity(attempt: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = [attempt / name for name in ("config.json", "pilot_result.json", "checkpoint_final.pt",
        "checkpoint_best.pt", "training.csv", "training_cost.json", "pilot_samples.npy", "sampling_diagnostics.json",
        "pilot_validation_diagnostics.json", "source/core_execution_source_hashes.json", "source/source_hashes.json")]
    evidence = [{"path": _relative(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths if path.is_file()]
    return {"activity_id": f"pilot__{config['run_id']}__{attempt.name}", "kind": "selected_task_validation_pilot",
        "status": "COMPLETED", "registered_at": now(), "dataset_id": config["dataset_id"], "model_id": config["model_id"],
        "seed": config["seed"], "run_id": config["run_id"], "attempt": attempt.name, "attempt_dir": _relative(attempt),
        "training_updates": config["steps"], "formal": False, "count_towards_75": False, "formal_configurations_added": 0,
        "formal_test_accessed": False, "validation_result_path": _relative(attempt / "pilot_result.json"),
        "source_snapshot_dir": _relative(attempt / "source"), "execution_source_hashes": config["execution_source_hashes"],
        "data_spec_path": config["data_spec_path"], "data_spec_sha256": config["data_spec_sha256"],
        "artifact_gate": "complete pilot checkpoint/log/source/samples/validation evidence verified read-only", "artifacts": evidence}


def _scan_pilots(process: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    pilot_root = (ROOT / "outputs/phase2_pilots").resolve()
    records, activities, warnings = [], [], []
    if not pilot_root.is_dir():
        return records, activities, warnings
    for path in sorted(pilot_root.glob("*__*__seed*/attempt_*/config.json")):
        try:
            if not path.resolve().is_relative_to(pilot_root):
                raise ValueError("pilot path leaves the dedicated pilot directory")
            config = read_json(path)
            validate_config(config["dataset_id"], config["model_id"], config["seed"])
            run_id = run_identifier(config["dataset_id"], config["model_id"], config["seed"])
            if config.get("phase") != "PHASE2_3DATASETS" or config.get("kind") != "pilot" or config.get("run_id") != run_id or path.parent.parent.name != run_id:
                raise ValueError("pilot config/path identity mismatch")
            if type(config.get("steps")) is not int or not 1 <= config["steps"] <= 2000:
                raise ValueError("pilot config has an invalid short-pilot budget")
            progress = _training_progress(path.parent / "training.csv")
            issues = _pilot_completion_issues(path.parent, config, progress)
            complete = not issues
            records.append({"run_id": run_id, "dataset_id": config["dataset_id"], "model_id": config["model_id"], "seed": config["seed"],
                "attempt": path.parent.name, "attempt_dir": _relative(path.parent), "planned_updates": config["steps"],
                "status": "COMPLETED" if complete else "INCOMPLETE_PRESERVED", "artifact_issues": issues,
                "config_mtime_ns": path.stat().st_mtime_ns, **{key: value for key, value in progress.items() if key != "steps"}})
            if complete:
                activities.append(_completed_pilot_activity(path.parent, config))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            warnings.append(f"Ignored unverified pilot {path}: {exc}")
    if process and process.get("mode") == "pilot" and process.get("status") == "RUNNING" and process.get("pid_alive_at_snapshot") is not False:
        candidates = [record for record in records if record["dataset_id"] == process.get("dataset_id")]
        if candidates:
            latest = max(candidates, key=lambda record: record["config_mtime_ns"])
            if latest["status"] != "COMPLETED":
                if process.get("pid_alive_at_snapshot") is True:
                    latest["status"] = "TRAINING" if (latest["last_flushed_step"] or 0) < latest["planned_updates"] else "SAMPLING_OR_VALIDATION"
                else:
                    latest["status"] = "IN_PROGRESS_PROCESS_STATE_UNCONFIRMED"
    return records, activities, warnings


def _plots_test_activity() -> dict[str, Any] | None:
    path = ROOT / "phase2_reference/PLOTS_TEST_EVIDENCE.json"
    if not path.is_file():
        return None
    evidence = read_json(path)
    if evidence.get("passed") != 3 or evidence.get("training_updates") != 0 or evidence.get("original_test_npz_opened") is not False:
        return None
    return {"activity_id": "phase2_plotting_correctness_checks", "kind": "correctness_and_engineering_checks",
        "status": "COMPLETED", "training_updates": 0, "formal_configurations_added": 0, "formal": False,
        "count_towards_75": False, "passed_tests": 3, "formal_test_accessed": False, "reports": [_relative(path)],
        "report_sha256": sha256_file(path), "test_command": evidence["test_command"], "registered_at": now()}


def _formal_check_activities(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Register saved same-checkpoint arrays only; never call the sampler."""
    import numpy as np
    activities, warnings = [], []
    expected = {"N32_steps32": (32, 32), "N32_steps64": (32, 64), "N32_steps128": (32, 128),
                "N64_steps64": (64, 64), "N128_steps64": (128, 64)}
    for run in manifest["runs"]:
        if run["status"] != "COMPLETED":
            continue
        try:
            attempt = active_attempt_path(run)
            check_path = attempt / "checkpoint_checks.json"
            diagnostics = read_json(check_path)
            config = read_json(attempt / "config.json")
            if (diagnostics.get("status") != "COMPLETED" or diagnostics.get("training_updates") != 0
                    or diagnostics.get("formal_configuration_count_added") != 0 or diagnostics.get("kind") != "same_checkpoint_auxiliary_evaluation"
                    or diagnostics.get("dataset_id") != run["dataset_id"] or diagnostics.get("model_id") != run["model_id"]
                    or set(diagnostics.get("checks", {})) != set(expected)):
                raise ValueError("same-checkpoint identity/zero-update contract differs")
            root = attempt / "checkpoint_checks/checkpoint_diagnostics"
            checks = []
            for key, (particles, steps) in expected.items():
                item = diagnostics["checks"][key]
                cost = item["cost"]
                sample = (root / item["samples"]).resolve()
                if not sample.is_relative_to(root.resolve()) or not sample.is_file():
                    raise ValueError(f"saved check path missing/outside diagnostic directory: {key}")
                if item.get("status") != "COMPLETED" or item.get("training_updates") != 0 or cost.get("N") != particles or cost.get("steps") != steps:
                    raise ValueError(f"saved N/steps/status differs: {key}")
                array = np.load(sample, mmap_mode="r", allow_pickle=False)
                try:
                    if array.shape != (2, particles, 128) or array.dtype.kind != "f" or not np.isfinite(array).all():
                        raise ValueError(f"check samples not finite float [2,N,128]: {key}")
                finally:
                    array._mmap.close()
                checks.append({"check_id": key, "N": particles, "steps": steps, "ensembles": 2,
                    "sampling_seed": cost.get("seed", diagnostics.get("seed")), "samples": _relative(sample),
                    "samples_sha256": sha256_file(sample), "cost": cost, "training_updates": 0})
            activities.append({"activity_id": f"same_checkpoint_checks__{run['run_id']}__{attempt.name}",
                "kind": "same_checkpoint_auxiliary_evaluation", "status": "COMPLETED", "registered_at": now(),
                "source_formal_run_id": run["run_id"], "dataset_id": run["dataset_id"], "model_id": run["model_id"],
                "training_seed": run["seed"], "attempt": attempt.name, "attempt_dir": _relative(attempt),
                "formal": False, "training_updates": 0, "formal_configurations_added": 0, "count_towards_75": False,
                "formal_test_accessed": False, "data_access": diagnostics.get("data_access"),
                "model_state_sha256": diagnostics["model_state_sha256"], "delivered_model_source_sha256": config["source_sha256"],
                "checkpoint": _relative(attempt / "checkpoint_final.pt"), "checkpoint_sha256": sha256_file(attempt / "checkpoint_final.pt"),
                "reports": [_relative(check_path)], "report_sha256": sha256_file(check_path), "saved_checks_count": 5, "checks": checks})
        except (OSError, KeyError, TypeError, ValueError) as exc:
            warnings.append(f"Saved same-checkpoint AUX not registered for {run['run_id']}: {exc}")
    return activities, warnings


def _cpu_review_activities() -> list[dict[str, Any]]:
    activities = []
    for name, activity_id, extra in (("REAL64_VALIDATION_REFERENCE.json", "finite_count_real64_validation_reference", ["REAL64_VALIDATION_REFERENCE.npz"]),
            ("PILOT_PROTOCOL_REVIEW.json", "validation_only_15pilot_protocol_review", ["PILOT_PROTOCOL_REVIEW_INITIAL.json", "review_pilots.py"])):
        path = ROOT / "phase2_reference" / name
        if not path.is_file():
            continue
        report = read_json(path)
        if any(report.get(key) != 0 for key in ("optimizer_updates", "extra_training_runs", "formal_configs_added")) or report.get("counts_toward_75") is not False or report.get("test_accessed") is not False:
            continue
        if name == "PILOT_PROTOCOL_REVIEW.json" and report.get("review_status") != "ALL_PILOTS_REVIEWED":
            continue
        paths = [path] + [ROOT / "phase2_reference" / filename for filename in extra]
        if any(not item.is_file() for item in paths):
            continue
        if name == "REAL64_VALIDATION_REFERENCE.json" and sha256_file(paths[1]) != report.get("array_sha256"):
            continue
        activities.append({"activity_id": activity_id, "kind": report["activity_kind"], "status": "COMPLETED",
            "registered_at": now(), "training_updates": 0, "formal_configurations_added": 0, "extra_training_runs": 0,
            "formal": False, "count_towards_75": False, "formal_test_accessed": False,
            "reports": [_relative(item) for item in paths], "artifacts": [{"path": _relative(item), "sha256": sha256_file(item)} for item in paths],
            "initial_snapshot_preserved": name == "PILOT_PROTOCOL_REVIEW.json", "pilot_results_count": report.get("pilot_results_count"),
            "formal_protocol_selected_by_this_activity": False, "actual_array_sources_opened": report.get("actual_array_sources_opened", [])})
    return activities


def _register_auxiliary(activities: list[dict[str, Any]]) -> dict[str, Any]:
    path = ROOT / "PHASE2_AUXILIARY_RUNS.json"
    with _registry_lock():
        registry = read_json(path) if path.is_file() else {"formal": False, "count_towards_75": False, "activities": []}
        if registry.get("formal") is not False or registry.get("count_towards_75") is not False or not isinstance(registry.get("activities"), list):
            raise ValueError("Auxiliary registry must preserve its nonformal identity")
        # Append only: never overwrite other authors' fields or prior activities.
        existing = {activity.get("activity_id") for activity in registry["activities"]}
        added = []
        for activity in activities:
            if activity["activity_id"] not in existing:
                registry["activities"].append(activity)
                existing.add(activity["activity_id"])
                added.append(activity["activity_id"])
        if added:
            _atomic_text(path, json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    return {"new_activity_ids": added, "total_registered_activities": len(registry["activities"])}


def snapshot() -> dict[str, Any]:
    manifest = load_manifest()
    statuses = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "BLOCKED")
    counts = {dataset: {status: sum(run["dataset_id"] == dataset and run["status"] == status for run in manifest["runs"])
                       for status in statuses} for dataset in DATASETS}
    completed = sum(count["COMPLETED"] for count in counts.values())
    process = _process_state()
    pilots, activities, warnings = _scan_pilots(process)
    plot_tests = _plots_test_activity()
    if plot_tests:
        activities.append(plot_tests)
    formal_checks, check_warnings = _formal_check_activities(manifest)
    activities.extend(formal_checks)
    activities.extend(_cpu_review_activities())
    warnings.extend(check_warnings)
    registration = _register_auxiliary(activities)
    running = []
    for run in manifest["runs"]:
        if run["status"] == "RUNNING":
            attempt = active_attempt_path(run)
            progress = _training_progress(attempt / "training.csv") if attempt else {}
            running.append({"run_id": run["run_id"], "attempt": str(attempt) if attempt else None,
                            **{key: value for key, value in progress.items() if key != "steps"}})
    initial_groups = {(pilot["dataset_id"], pilot["model_id"]) for pilot in pilots if pilot["status"] == "COMPLETED" and pilot["seed"] == 17}
    pilot_summary = {"initial_pilot_seed": 17, "initial_pilot_groups_expected": 15,
        "initial_pilot_groups_completed": len(initial_groups), "completed_attempts": sum(pilot["status"] == "COMPLETED" for pilot in pilots),
        "observed_attempts": len(pilots), "count_towards_75": False, "attempts": pilots}
    lines = ["# Phase 2 progress", "", f"更新时间: {local_now()}", "",
             f"正式配置完成 **{completed}/75**；本阶段独立于 Phase 1 原 50 项，原结果不改写。", ""]
    for dataset in DATASETS:
        lines.append(f"- {dataset}: {counts[dataset]['COMPLETED']}/25 COMPLETED；" + ", ".join(f"{status}={count}" for status, count in counts[dataset].items() if count))
    lines += ["", "独立验证 pilot（不计入 75 项正式配置）：", "",
        f"- 初始 seed17 pilot：**{len(initial_groups)}/15** 模型/任务群已完成；共 {pilot_summary['completed_attempts']} 个已完成 attempt，所有已观察 attempt 和历史记录保留。"]
    for dataset in DATASETS:
        task = [pilot for pilot in pilots if pilot["dataset_id"] == dataset]
        finished = len({pilot["model_id"] for pilot in task if pilot["status"] == "COMPLETED" and pilot["seed"] == 17})
        lines.append(f"- {dataset}: {finished}/5 初始 pilot 完成；{len(task)} 个已观察 attempt。")
    for pilot in pilots:
        elapsed = "尚无已落盘记录" if pilot["elapsed_seconds"] is None else f"{pilot['elapsed_seconds']:.2f}s"
        lines.append(f"- `{pilot['run_id']}/{pilot['attempt']}`: {pilot['status']}; step={pilot['last_flushed_step'] or 0}/{pilot['planned_updates']}; elapsed={elapsed}; loss={pilot['loss']}。")
    if process:
        lines += ["", f"监督进程记录：mode={process.get('mode')}; task={process.get('dataset_id')}; PID={process.get('pid')}; stored_status={process.get('status')}; 当前只读 PID 存活检查={process.get('pid_alive_at_snapshot')}。"]
    if running:
        lines += ["", "实际正式 RUNNING 日志的最近已落盘记录：", ""]
        for run in running:
            lines.append(f"- {run['run_id']}: step={run.get('last_flushed_step')}; elapsed_seconds={run.get('elapsed_seconds')}; loss={run.get('loss')}。")
    lines += ["", "pilot 的 COMPLETED 以配置、完整日志、final/best checkpoint、来源快照、浮点样本及 validation-only 结果核验；缺项或未结束只显示真实已落盘进度，不认作失败或正式完成。",
        "辅助注册见 PHASE2_AUXILIARY_RUNS.json；保留既有 47 项检查及 3 个 subtests，另登记 3 项 0-update 绘图测试。",
        "", "仅 75 项固定配置；每任务 L64、C2、D128，五模型共享数据划分与锁定协议。正式预算以锁定协议为据，pilot 不替代正式训练。", "",
        "恢复与汇总：", "", "```powershell", "python -m phase2_experiment.verify", "python -m phase2_experiment.monitor",
        "python -m phase2_experiment.runner resume", "python -m phase2_experiment.reporting", "python -m phase2_experiment.plots",
        "python -m phase2_experiment.reporting --package", "```", "", "失败/受阻记录留在 PHASE2_MANIFEST.json 及各 attempt；其他任务单列 PHASE2_DEFERRED_BY_USER.json。完成本阶段后停止，不启动其他矩阵。"]
    if warnings:
        lines += ["", "读取限制：", "", *[f"- {warning}" for warning in warnings]]
    _atomic_text(ROOT / "PHASE2_PROGRESS.md", "\n".join(lines) + "\n")
    result = {"at": now(), "formal_configuration_count": 75, "completed": completed, "datasets": counts,
        "running_log_evidence": running, "pilots": pilot_summary, "process_state": process,
        "auxiliary_registration": registration, "warnings": warnings, "training_started_by_monitor": False,
        "test_data_opened_by_monitor": False}
    _atomic_text(ROOT / "PHASE2_MONITOR_SNAPSHOT.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error("interval must be between 1 and 60 seconds")
    while True:
        result = snapshot()
        print(json.dumps(result, ensure_ascii=False), flush=True)
        terminal = sum(sum(counts[status] for status in ("COMPLETED", "FAILED", "BLOCKED")) for counts in result["datasets"].values())
        if not args.watch or terminal == 75:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
