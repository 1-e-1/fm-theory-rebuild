"""Supervise or attach to the fixed Phase 2 queue, then verify full delivery.

Only 75 artifact-verified formal runs and intact protection hashes permit
aggregation, figures and packaging. No initialization, budget selection,
mathematical change, extra matrix or training is performed in delivery-only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any
import zipfile

from . import monitor
from .scope import (DATASETS, ROOT, OUTPUT_ROOT, active_attempt_path, dataset_protocol,
                    load_manifest, now, read_json, sha256_file, validate_scope,
                    verify_manifest_artifacts, verify_phase1_guard, write_json)

_PROCESS_COVERAGE: dict[str, Any] = {}


def _core_guard() -> dict[str, Any]:
    path = ROOT / "phase2_reference/PRE_PILOT_FREEZE.json"
    if not path.is_file():
        return {"ok": False, "issues": ["Pre-pilot execution-core freeze unavailable"]}
    frozen = read_json(path)
    issues = [f"Execution core differs/missing: {name}" for name, digest in frozen["execution_core_hashes"].items()
              if not (ROOT / name).is_file() or sha256_file(ROOT / name) != digest]
    if sha256_file(ROOT / "PHASE2_SCOPE.json") != frozen["phase2_scope_sha256"]:
        issues.append("Independent 75-run scope differs from its frozen source")
    return {"ok": not issues, "files_checked": len(frozen["execution_core_hashes"]), "issues": issues}


def _preflight() -> dict[str, Any]:
    scope = validate_scope()
    guard, core = verify_phase1_guard(), _core_guard()
    if not guard["ok"] or not core["ok"]:
        raise RuntimeError(f"Protection gate failed: {guard}; {core}")
    protocols = {}
    for dataset in DATASETS:
        protocol = dataset_protocol(dataset)
        if protocol.get("updates") not in (12000, 24000, 48000) or protocol.get("state_dim") != 128:
            raise ValueError(f"Task protocol is not locked to an authorized formal budget/D128: {dataset}")
        protocols[dataset] = {key: protocol[key] for key in ("updates", "N", "H", "state_dim")}
    return {"formal_configuration_count": scope["formal_configuration_count"], "phase1_guard": guard,
            "frozen_core_guard": core, "protocols": protocols,
            "protocol_file_sha256": sha256_file(ROOT / "PHASE2_LOCKED_PROTOCOL.json")}


def _live_training_processes() -> list[dict[str, Any]]:
    """Check the registered process and relevant Python commands read-only."""
    global _PROCESS_COVERAGE
    observed = {}
    state = monitor._process_state()
    _PROCESS_COVERAGE = {"checked_at": now(), "registered_process_state": state,
        "direct_process_query": "NOT_RUN", "absence_verified": False, "warnings": []}
    if state and state.get("status") == "RUNNING" and state.get("pid_alive_at_snapshot") is not False:
        observed[state["pid"]] = {"pid": state["pid"], "source": "PHASE2_PROCESS_STATE.json",
            "mode": state.get("mode"), "started_at": state.get("started_at"),
            "command": state.get("command"), "alive": state.get("pid_alive_at_snapshot")}
        _PROCESS_COVERAGE["direct_process_query"] = "NOT_NEEDED_REGISTERED_LIVE_PROCESS"
        return list(observed.values())
    # Direct runner launches can bypass the supervisor metadata. Filter commands
    # before returning any process details; no filesystem search is performed.
    if os.name == "nt":
        script = "$ErrorActionPreference = 'Stop'; [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); $p = Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\"; $p | Where-Object { $_.CommandLine -match '(?i)-m\\s+phase2_experiment\\.(runner|orchestrate)\\s+(run|resume|pilot|pilots)(\\s|$)' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        try:
            process = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW)
            # PowerShell can emit non-terminating AccessDenied even with exit0.
            if process.returncode or process.stderr.strip():
                raise RuntimeError(process.stderr.strip() or f"PowerShell exit {process.returncode}")
            parsed = json.loads(process.stdout) if process.stdout.strip() else []
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            _PROCESS_COVERAGE["direct_process_query"] = "DENIED_OR_UNAVAILABLE"
            _PROCESS_COVERAGE["warnings"].append("Direct process enumeration is unavailable; absence of unregistered direct launches is not verified. " + str(exc)[:500])
            return list(observed.values())
        for row in ([parsed] if isinstance(parsed, dict) else parsed):
            pid = int(row["ProcessId"])
            if pid != os.getpid():
                observed.setdefault(pid, {"pid": pid, "source": "read-only process command query", "command": row["CommandLine"], "alive": True})
        _PROCESS_COVERAGE.update(direct_process_query="SUCCEEDED", absence_verified=not observed)
    else:
        _PROCESS_COVERAGE["warnings"].append("Direct process enumeration is not implemented for this platform; only the owned supervisor PID was checked")
    return list(observed.values())


def _attach_identity(state: dict[str, Any] | None) -> tuple[int, str, str]:
    if not state or state.get("mode") not in ("run", "resume") or state.get("status") != "RUNNING":
        raise ValueError("--attach requires the existing live formal run/resume process; pilots cannot be attached")
    if state.get("pid_alive_at_snapshot") is not True or type(state.get("pid")) is not int or not state.get("started_at"):
        raise ValueError("The exact registered formal process is not verified live")
    command = state.get("command", [])
    if not isinstance(command, list) or "phase2_experiment.runner" not in command or state["mode"] not in command:
        raise ValueError("Registered command is not the fixed Phase 2 formal runner")
    return state["pid"], state["started_at"], state["mode"]


def _persist(session: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    write_json(session / "finish_state.json", state)
    if state.get("writes_canonical_status"):
        write_json(ROOT / "PHASE2_DELIVERY_STATUS.json", state)


def _event(session: Path, state: dict[str, Any], kind: str, **fields: Any) -> None:
    item = {"at": now(), "kind": kind, **fields}
    state.setdefault("events", []).append(item)
    _persist(session, state)
    print(json.dumps(item, ensure_ascii=False), flush=True)


def _wait_attached(session: Path, state: dict[str, Any], interval: float) -> dict[str, Any]:
    process = monitor._process_state()
    identity = _attach_identity(process)
    state["attached_process"] = {"pid": identity[0], "started_at": identity[1], "mode": identity[2],
                                 "actual_command": process["command"], "console_log": process.get("console_log")}
    _event(session, state, "ATTACHED_EXISTING_FORMAL_PROCESS", **state["attached_process"])
    while monitor._pid_alive(identity[0]) is True:
        current = monitor._process_state()
        if current and (current.get("pid"), current.get("started_at"), current.get("mode")) != identity:
            raise RuntimeError("Registered process identity changed; refusing to follow or start another queue")
        progress = monitor.snapshot()
        _event(session, state, "ATTACHED_PROGRESS", tracked_pid=identity[0], formal_completed=progress["completed"], formal_total=75)
        time.sleep(interval)
    if monitor._pid_alive(identity[0]) is None:
        raise RuntimeError("Tracked process liveness became unverifiable; no process exit or experimental success is claimed")
    # Give the owning supervisor a brief opportunity to persist its actual exit.
    final = monitor._process_state()
    for _ in range(10):
        if final and (final.get("pid"), final.get("started_at"), final.get("mode")) == identity and final.get("status") in ("PROCESS_FINISHED", "PROCESS_FAILED"):
            break
        time.sleep(.2)
        final = monitor._process_state()
    matching = final and (final.get("pid"), final.get("started_at"), final.get("mode")) == identity
    result = {"process_done": True, "tracked_pid": identity[0], "tracked_started_at": identity[1],
              "process_status": final.get("status") if matching else "EXIT_OBSERVED_STATUS_UNAVAILABLE",
              "exit_code": final.get("exit_code") if matching else None,
              "actual_command": process["command"], "experimental_completion_claimed": False}
    _event(session, state, "PROCESS_DONE", **result)
    return result


def _launch_queue(session: Path, state: dict[str, Any], mode: str) -> dict[str, Any]:
    if live := _live_training_processes():
        raise RuntimeError(f"Live training process exists; refusing duplicate queue: {live}")
    state["process_coverage_before_launch"] = dict(_PROCESS_COVERAGE)
    if not _PROCESS_COVERAGE.get("absence_verified"):
        raise RuntimeError("Training launch blocked because absence of another direct queue cannot be verified. Existing full artifacts can still be delivered without process-enumeration permission")
    preflight = _preflight()
    if preflight["protocol_file_sha256"] != state["preflight"]["protocol_file_sha256"]:
        raise RuntimeError("Locked protocol changed during this execution; affected queue is paused")
    from .orchestrate import supervise
    command = [sys.executable, "-u", "-m", "phase2_experiment.runner", mode]
    _event(session, state, "QUEUE_COMMAND_STARTED", actual_command=command, queue_mode=mode, formal_configuration_count=75)
    exit_code = supervise(command, mode)
    result = {"process_done": True, "exit_code": exit_code, "actual_command": command,
              "process_status": "PROCESS_FINISHED" if exit_code == 0 else "PROCESS_FAILED", "experimental_completion_claimed": False}
    _event(session, state, "PROCESS_DONE", **result)
    return result


def _full_gate(expected_protocol_file_sha256: str | None = None) -> dict[str, Any]:
    verification = verify_manifest_artifacts()
    core = _core_guard()
    counts = verification["counts"]
    completed = counts.get("COMPLETED", 0)
    selection_path = ROOT / "PHASE2_PROTOCOL_SELECTION.json"
    registered_hash = read_json(selection_path).get("locked_protocol_sha256") if selection_path.is_file() else None
    protocol_path = ROOT / "PHASE2_LOCKED_PROTOCOL.json"
    actual_hash = sha256_file(protocol_path) if protocol_path.is_file() else None
    expected_hash = expected_protocol_file_sha256 or registered_hash
    protocol_ok = bool(expected_hash) and actual_hash == expected_hash and (registered_hash is None or actual_hash == registered_hash)
    ok = verification["ok"] and core["ok"] and protocol_ok and completed == 75 and sum(counts.values()) == 75
    return {"ok": bool(ok), "verified_completed": completed - len(verification.get("issues", {})),
            "formal_configuration_count": 75, "verification": verification, "frozen_core_guard": core,
            "protocol_file_guard": {"ok": protocol_ok, "expected_sha256": expected_hash, "actual_sha256": actual_hash,
                                    "selection_registered_sha256": registered_hash}}


def _recovery_eligibility() -> dict[str, Any]:
    """A retry cannot create a new attempt or train an unattempted configuration."""
    eligible, issues = [], []
    for run in load_manifest()["runs"]:
        if run["status"] == "COMPLETED":
            continue
        if run["status"] in ("PENDING", "BLOCKED"):
            issues.append(f"{run['run_id']}: {run['status']}, automatic recovery cannot launch an unattempted or blocked run")
            continue
        attempt = active_attempt_path(run)
        if attempt is None or not (attempt / "config.json").is_file() or not (attempt / "checkpoint_resume.pt").is_file():
            issues.append(f"{run['run_id']}: no preserved same-attempt resume checkpoint")
            continue
        config = read_json(attempt / "config.json")
        if any(config.get(key) != run[key] for key in ("run_id", "dataset_id", "model_id", "seed")) or config.get("kind") != "formal":
            issues.append(f"{run['run_id']}: preserved resume configuration identity mismatch")
            continue
        eligible.append({"run_id": run["run_id"], "attempt_dir": str(attempt), "checkpoint": str(attempt / "checkpoint_resume.pt")})
    return {"ok": bool(eligible) and not issues, "eligible": eligible, "issues": issues}


def _all_tensors_finite(value: Any) -> bool:
    import torch
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_tensors_finite(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_tensors_finite(child) for child in value)
    return True


def _resume_checkpoint_issues(run: dict[str, Any]) -> list[str]:
    import torch
    attempt = active_attempt_path(run)
    path = attempt / "checkpoint_resume.pt"
    if not path.is_file():
        return [f"{run['run_id']}: third/resume checkpoint missing"]
    config = read_json(attempt / "config.json")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    identity = {key: config[key] for key in ("dataset_id", "model_id", "seed", "kind", "run_id", "steps")}
    issues = []
    if checkpoint.get("step") != config["steps"] or checkpoint.get("run_identity") != identity:
        issues.append(f"{run['run_id']}: resume checkpoint step/identity differs from completed formal budget")
    if json.dumps(checkpoint.get("model_config"), sort_keys=True) != json.dumps(config.get("model_config"), sort_keys=True):
        issues.append(f"{run['run_id']}: resume checkpoint model configuration mismatch")
    for checkpoint_key, config_key in (("model_sha256", "source_sha256"), ("data_spec_sha256", "data_spec_sha256"),
            ("execution_source_hashes", "execution_source_hashes"), ("dataset_protocol_sha256", "dataset_protocol_sha256")):
        if checkpoint.get(checkpoint_key) != config.get(config_key):
            issues.append(f"{run['run_id']}: resume checkpoint provenance mismatch: {checkpoint_key}")
    for key in ("model", "optimizer", "python_rng", "numpy_rng", "torch_rng"):
        if key not in checkpoint:
            issues.append(f"{run['run_id']}: resume checkpoint lacks {key}")
    if str(config.get("device", "")).startswith("cuda") and not checkpoint.get("cuda_rng"):
        issues.append(f"{run['run_id']}: CUDA resume checkpoint lacks CUDA RNG state")
    if not _all_tensors_finite(checkpoint.get("model", {})) or not _all_tensors_finite(checkpoint.get("optimizer", {})):
        issues.append(f"{run['run_id']}: nonfinite resume model/optimizer")
    return issues


def _local_inventory(directory: Path) -> dict[str, Any]:
    """Hash all actual formal attempts, including weights and prior failures."""
    import numpy as np
    rows = []
    category_counts = {}
    issues = []
    manifest = load_manifest()
    for run in manifest["runs"]:
        current_attempt = active_attempt_path(run) if run["status"] == "COMPLETED" else None
        if run["status"] == "COMPLETED":
            try:
                issues.extend(_resume_checkpoint_issues(run))
            except Exception as exc:
                issues.append(f"{run['run_id']}: resume checkpoint read failed: {exc}")
        run_root = OUTPUT_ROOT / run["run_id"]
        if not run_root.is_dir():
            continue
        for path in sorted(run_root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(OUTPUT_ROOT.resolve()):
                issues.append(f"Artifact path leaves formal outputs: {path}")
                continue
            relative = path.relative_to(ROOT).as_posix()
            category = ("checkpoint" if path.suffix == ".pt" else "floating_generated_sample" if path.suffix == ".npy"
                else "training_log" if path.name.startswith("training") and path.suffix == ".csv"
                else "failure_record" if "failure" in path.name else "source_snapshot" if "source" in path.parts else "config_or_result")
            dtype, shape, finite = "", "", ""
            if path.suffix == ".npy":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                try:
                    dtype, shape = str(array.dtype), json.dumps(list(array.shape))
                    finite = bool(np.isfinite(array).all())
                    # Failed historical samples remain evidence; their invalid
                    # values must not invalidate a later complete attempt.
                    if (array.dtype.kind != "f" or not finite) and current_attempt is not None and resolved.is_relative_to(current_attempt.resolve()):
                        issues.append(f"Current completed generated array is not finite float: {relative}")
                finally:
                    array._mmap.close()
            rows.append({"run_id": run["run_id"], "dataset_id": run["dataset_id"], "model_id": run["model_id"],
                "seed": run["seed"], "attempt": path.relative_to(run_root).parts[0], "category": category,
                "relative_path": relative, "absolute_path": str(resolved), "bytes": path.stat().st_size,
                "sha256": sha256_file(path), "dtype": dtype, "shape": shape, "finite": finite})
            category_counts[category] = category_counts.get(category, 0) + 1
    fields = ["run_id", "dataset_id", "model_id", "seed", "attempt", "category", "relative_path", "absolute_path", "bytes", "sha256", "dtype", "shape", "finite"]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "full_weight_sample_log_inventory.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"ok": not issues, "issues": issues, "formal_output_root": str(OUTPUT_ROOT.resolve()),
            "inventory": str(target), "inventory_sha256": sha256_file(target), "files": len(rows),
            "bytes": sum(row["bytes"] for row in rows), "categories": category_counts,
            "current_run_checkpoint_contract": "final + validation-best + same-attempt resume for each of 75 runs",
            "full_weights_in_zip": False, "full_weights_retained_locally": True}


def verify_zip_contents(zip_path: Path, report_root: Path) -> dict[str, Any]:
    """Validate CRC and every archived member against the actual review files."""
    issues, rows = [], []
    expected = {path.relative_to(report_root).as_posix(): path for path in report_root.rglob("*") if path.is_file()}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            issues.append("ZIP has duplicate member names")
        if corrupt := archive.testzip():
            issues.append(f"ZIP CRC failure: {corrupt}")
        normalized = {name.replace("\\", "/") for name in names}
        if normalized != set(expected):
            issues.append(f"ZIP member set differs: missing={sorted(set(expected)-normalized)}, unexpected={sorted(normalized-set(expected))}")
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                issues.append(f"Unsafe ZIP member path: {name}")
                continue
            digest = hashlib.sha256()
            with archive.open(member) as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
            actual = digest.hexdigest()
            local_digest = sha256_file(expected[name]) if name in expected else None
            if actual != local_digest:
                issues.append(f"ZIP content SHA256 differs: {name}")
            rows.append({"path": name, "bytes": member.file_size, "CRC32": f"{member.CRC:08x}", "sha256": actual,
                         "matches_review_source": actual == local_digest})
    return {"checked_at": now(), "ok": not issues, "issues": issues, "archive": str(zip_path.resolve()),
            "archive_sha256": sha256_file(zip_path), "archive_bytes": zip_path.stat().st_size,
            "member_count": len(rows), "CRC_checked": True, "every_content_sha256_checked": True, "members": rows}


def _partial(session: Path, state: dict[str, Any], reason: str, gate: dict[str, Any] | None = None) -> dict[str, Any]:
    state.update(status="PARTIAL_OR_BLOCKED", reason=reason, experimental_completed=False, delivery_completed=False,
                 recovery_commands=["python -m phase2_experiment.finish --attach", "python -m phase2_experiment.finish --resume",
                                    "python -m phase2_experiment.finish --delivery-only", "python -m phase2_experiment.verify"])
    if gate is not None:
        state["final_gate"] = gate
    failures = [str(path.relative_to(ROOT)) for path in OUTPUT_ROOT.glob("*/attempt_*/failure*.json")]
    state["preserved_failure_records"] = failures
    _event(session, state, "TRUE_PARTIAL_STATUS_PRESERVED", reason=reason, failure_records=len(failures), package_created_by_this_session=False)
    write_json(session / "failure_or_missing.json", state)
    (session / "RESTORE_COMMANDS.txt").write_text("\n".join(state["recovery_commands"]) + "\n", encoding="utf-8")
    return state


def _deliver(session: Path, state: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    from . import plots, reporting
    if not gate["ok"] or _live_training_processes():
        raise RuntimeError("Delivery requires 75 verified configurations, intact guards and no live training")
    # Packaging is read-only with respect to experiments. A CIM permission gap
    # is recorded, not treated as evidence of a live queue or a failed experiment.
    state["process_coverage_at_delivery"] = dict(_PROCESS_COVERAGE)
    _event(session, state, "FULL_EXPERIMENT_GATE_PASSED", verified_completed=75, formal_total=75)
    summary = reporting.aggregate(package=False)
    if summary["verified_completed"] != 75 or not summary["artifact_verification_ok"]:
        raise RuntimeError("Aggregation did not retain 75 artifact-verified results")
    _event(session, state, "AGGREGATION_COMPLETE", verified_completed=summary["verified_completed"])
    figures = plots.generate_figures()
    if any(task["verified_completed"] != 25 for task in figures["datasets"].values()):
        raise RuntimeError("Figures do not cover all 25 verified configurations for each task")
    _event(session, state, "FIGURES_COMPLETE", figure_count=figures["figure_count"])
    inventory = _local_inventory(reporting.REPORT_ROOT)
    if not inventory["ok"]:
        raise RuntimeError("Full three-checkpoint/local-array inventory failed: " + "; ".join(inventory["issues"]))
    final = _full_gate(state["preflight"]["protocol_file_sha256"])
    if not final["ok"]:
        raise RuntimeError("Completion gate changed during read-only delivery")
    audit = {"checked_at": now(), "experimental_completed": True, "verified_completed": 75,
        "formal_configuration_count": 75, "gate": final, "local_inventory": inventory,
        "training_process_results": state.get("process_results", []), "figure_count": figures["figure_count"],
        "process_coverage_at_delivery": state["process_coverage_at_delivery"],
        "extra_formal_configurations": 0, "extra_training_during_delivery": 0,
        "zip_integrity_report_location": "PHASE2_ZIP_VALIDATION.json beside the final ZIP; archive digest is external to avoid self-reference"}
    write_json(reporting.REPORT_ROOT / "FINAL_ARTIFACT_VERIFICATION.json", audit)
    state.update(status="ARTIFACTS_COMPLETE_PACKAGING", experimental_completed=True, verified_completed=75, final_gate=final,
                 local_inventory=inventory, figure_count=figures["figure_count"], delivery_completed=False)
    _persist(session, state)
    write_json(reporting.REPORT_ROOT / "DELIVERY_STATE_AT_PACKAGING.json", state)
    package = reporting.aggregate(package=True)
    validation = verify_zip_contents(reporting.ZIP_PATH, reporting.REPORT_ROOT)
    write_json(ROOT / "PHASE2_ZIP_VALIDATION.json", validation)
    write_json(session / "zip_validation.json", validation)
    if not validation["ok"]:
        raise RuntimeError("Final ZIP validation failed: " + "; ".join(validation["issues"]))
    state.update(status="DELIVERY_COMPLETED", experimental_completed=True, delivery_completed=True,
                 package=str(reporting.ZIP_PATH), package_sha256=validation["archive_sha256"],
                 package_bytes=validation["archive_bytes"], zip_validation_report="PHASE2_ZIP_VALIDATION.json")
    _event(session, state, "FULL_DELIVERY_COMPLETED", verified_completed=75, formal_total=75,
           package=package["package"], archive_sha256=validation["archive_sha256"], CRC_checked=True)
    write_json(ROOT / "PHASE2_FINAL_DELIVERY_AUDIT.json", {"checked_at": now(), "state": state, "artifacts": audit,
        "zip_validation": {key: value for key, value in validation.items() if key != "members"}})
    return state


def execute(*, delivery_only: bool = False, attach: bool = False, resume: bool = False, interval: float = 30) -> dict[str, Any]:
    if sum((delivery_only, attach, resume)) > 1:
        raise ValueError("Select at most one of --delivery-only, --attach or --resume")
    if not 1 <= interval <= 60:
        raise ValueError("Monitoring interval must be 1..60 seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session = ROOT / "outputs/phase2_delivery" / stamp
    session.mkdir(parents=True)
    state = {"started_at": now(), "finish_pid": os.getpid(), "actual_invocation": list(sys.argv),
        "mode": "delivery-only" if delivery_only else "attach" if attach else "resume" if resume else "run",
        "status": "PREFLIGHT", "formal_configuration_count": 75, "session_dir": str(session),
        "experimental_completed": False, "delivery_completed": False, "process_results": [], "automatic_recovery_attempts": 0,
        "writes_canonical_status": False}
    lock_path = ROOT / "PHASE2_FINISH_LOCK.json"
    lock_owned = False
    try:
        state["preflight"] = _preflight()
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            previous = read_json(lock_path)
            if monitor._pid_alive(previous.get("pid")) is not False:
                return _partial(session, state, "Another live or unverifiable finish process owns the delivery lock")
            # Preserve a stale owner's record before reclaiming only this lock.
            shutil.copy2(lock_path, session / "stale_finish_lock.json")
            lock_path.unlink()
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started_at": state["started_at"], "session_dir": str(session)}, handle)
        lock_owned = True
        state["writes_canonical_status"] = True
        _persist(session, state)
        if delivery_only:
            if live := _live_training_processes():
                state["observed_live_training"] = live
                return _partial(session, state, "Delivery deferred while an existing training process is live; no queue launched")
        elif attach:
            state["process_results"].append(_wait_attached(session, state, interval))
        else:
            if live := _live_training_processes():
                state["observed_live_training"] = live
                return _partial(session, state, "Existing live training process; duplicate queue refused. Use --attach for the formal process")
            any_attempt = any(OUTPUT_ROOT.glob("*/attempt_*"))
            mode = "resume" if resume or any_attempt else "run"
            state["initial_queue_mode"] = mode
            state["process_results"].append(_launch_queue(session, state, mode))
        gate = _full_gate(state["preflight"].get("protocol_file_sha256"))
        # At most one automatic same-attempt recovery after a first run/attach.
        # An explicit/implicit initial resume already consumes this opportunity.
        if not gate["ok"] and not delivery_only and not resume and state.get("initial_queue_mode") != "resume":
            eligibility = _recovery_eligibility()
            state["recovery_eligibility"] = eligibility
            live = _live_training_processes() if eligibility["ok"] else []
            state["recovery_process_coverage"] = dict(_PROCESS_COVERAGE)
            if eligibility["ok"] and not live and _PROCESS_COVERAGE.get("absence_verified"):
                state["automatic_recovery_attempts"] = 1
                _event(session, state, "ONE_SAME_ATTEMPT_RECOVERY_STARTED", eligible=eligibility["eligible"])
                state["process_results"].append(_launch_queue(session, state, "resume"))
                gate = _full_gate(state["preflight"].get("protocol_file_sha256"))
            elif eligibility["ok"] and not live:
                _event(session, state, "AUTOMATIC_RECOVERY_BLOCKED_PROCESS_COVERAGE", warnings=_PROCESS_COVERAGE.get("warnings", []),
                       recovery_processes_started=0)
        if not gate["ok"]:
            return _partial(session, state, "Queue/process completion does not establish 75/75 artifact completion; automatic recovery exhausted or unavailable", gate)
        return _deliver(session, state, gate)
    except Exception as exc:
        state["exception"] = {"at": now(), "error": repr(exc), "traceback": traceback.format_exc()}
        return _partial(session, state, str(exc))
    finally:
        if lock_owned:
            lock_path.unlink(missing_ok=True)


def main() -> int:
    # Windows default GBK cannot represent every captured diagnostic character.
    # The foreground supervisor and persisted logs already use UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--delivery-only", action="store_true")
    modes.add_argument("--attach", action="store_true")
    modes.add_argument("--resume", action="store_true")
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    result = execute(delivery_only=args.delivery_only, attach=args.attach, resume=args.resume, interval=args.interval)
    print(json.dumps({key: value for key, value in result.items() if key != "events"}, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("delivery_completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
