"""Automatic delivery routing and completion claims without launching training."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile

import pytest

from phase2_experiment import finish
from phase2_experiment.scope import ROOT


@pytest.fixture
def workspace(monkeypatch):
    parent = ROOT / ".phase2_test_tmp"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="finish_", dir=parent) as folder:
        directory = Path(folder)
        monkeypatch.setattr(finish, "ROOT", directory)
        monkeypatch.setattr(finish, "OUTPUT_ROOT", directory / "outputs/phase2_3datasets")
        monkeypatch.setattr(finish, "_preflight", lambda: {"formal_configuration_count": 75})
        monkeypatch.setattr(finish, "_PROCESS_COVERAGE", {"absence_verified": True})
        yield directory


def test_verify_ok_with_pending_runs_cannot_claim_completion(monkeypatch):
    monkeypatch.setattr(finish, "verify_manifest_artifacts", lambda: {"ok": True, "counts": {"PENDING": 75}, "issues": {}})
    monkeypatch.setattr(finish, "_core_guard", lambda: {"ok": True})
    assert finish._full_gate()["ok"] is False
    monkeypatch.setattr(finish, "verify_manifest_artifacts", lambda: {"ok": True, "counts": {"COMPLETED": 75}, "issues": {}})
    assert finish._full_gate()["ok"] is True
    monkeypatch.setattr(finish, "_core_guard", lambda: {"ok": False})
    assert finish._full_gate()["ok"] is False


def test_attach_rejects_pilot_and_requires_exact_live_formal_identity():
    with pytest.raises(ValueError, match="pilots"):
        finish._attach_identity({"mode": "pilot", "status": "RUNNING"})
    state = {"mode": "run", "status": "RUNNING", "pid": 123, "pid_alive_at_snapshot": True,
        "started_at": "fixed_start", "command": ["python", "-m", "phase2_experiment.runner", "run"]}
    assert finish._attach_identity(state) == (123, "fixed_start", "run")
    state["pid_alive_at_snapshot"] = False
    with pytest.raises(ValueError, match="verified live"):
        finish._attach_identity(state)


def test_delivery_only_with_live_training_does_not_launch_or_package(workspace, monkeypatch):
    monkeypatch.setattr(finish, "_live_training_processes", lambda: [{"pid": 123}])
    monkeypatch.setattr(finish, "_launch_queue", lambda *args: pytest.fail("training launch forbidden"))
    monkeypatch.setattr(finish, "_deliver", lambda *args: pytest.fail("live/pending delivery forbidden"))
    result = finish.execute(delivery_only=True)
    assert result["status"] == "PARTIAL_OR_BLOCKED" and not result["experimental_completed"]
    assert not result["delivery_completed"] and result["process_results"] == []
    assert result["observed_live_training"] == [{"pid": 123}]


def test_default_refuses_duplicate_live_queue(workspace, monkeypatch):
    monkeypatch.setattr(finish, "_live_training_processes", lambda: [{"pid": 123}])
    monkeypatch.setattr(finish, "_launch_queue", lambda *args: pytest.fail("duplicate training launch forbidden"))
    result = finish.execute()
    assert "duplicate queue refused" in result["reason"]
    assert not result["delivery_completed"]


def test_process_exit_zero_is_separate_from_experimental_completion(workspace, monkeypatch):
    monkeypatch.setattr(finish, "_live_training_processes", lambda: [])
    monkeypatch.setattr(finish, "_launch_queue", lambda *args: {"process_done": True, "exit_code": 0})
    monkeypatch.setattr(finish, "_full_gate", lambda *args: {"ok": False, "verified_completed": 25, "formal_configuration_count": 75})
    monkeypatch.setattr(finish, "_recovery_eligibility", lambda: {"ok": False, "eligible": [], "issues": ["no same-attempt checkpoint"]})
    monkeypatch.setattr(finish, "_deliver", lambda *args: pytest.fail("partial experiment cannot be packaged"))
    result = finish.execute()
    assert result["process_results"][0]["exit_code"] == 0
    assert result["final_gate"]["verified_completed"] == 25 and not result["experimental_completed"]
    assert not result["delivery_completed"]


def test_automatic_recovery_runs_at_most_once_and_retains_partial_status(workspace, monkeypatch):
    launched = []
    monkeypatch.setattr(finish, "_live_training_processes", lambda: [])
    monkeypatch.setattr(finish, "_launch_queue", lambda session, state, mode: launched.append(mode) or {"process_done": True, "exit_code": 0})
    monkeypatch.setattr(finish, "_full_gate", lambda *args: {"ok": False, "verified_completed": 74, "formal_configuration_count": 75})
    monkeypatch.setattr(finish, "_recovery_eligibility", lambda: {"ok": True, "eligible": [{"run_id": "same_attempt"}], "issues": []})
    result = finish.execute()
    assert launched == ["run", "resume"] and result["automatic_recovery_attempts"] == 1
    assert not result["delivery_completed"] and not result["experimental_completed"]
    assert Path(result["session_dir"], "failure_or_missing.json").is_file()


def test_attach_path_follows_existing_process_without_new_launch(workspace, monkeypatch):
    monkeypatch.setattr(finish, "_wait_attached", lambda *args: {"process_done": True, "tracked_pid": 123, "exit_code": 0})
    monkeypatch.setattr(finish, "_full_gate", lambda *args: {"ok": True, "verified_completed": 75, "formal_configuration_count": 75})
    monkeypatch.setattr(finish, "_launch_queue", lambda *args: pytest.fail("attach cannot start a duplicate queue"))
    monkeypatch.setattr(finish, "_deliver", lambda session, state, gate: dict(state, delivery_completed=True, experimental_completed=True))
    result = finish.execute(attach=True)
    assert result["process_results"][0]["tracked_pid"] == 123 and result["delivery_completed"]
    assert result["automatic_recovery_attempts"] == 0


def test_changed_exact_attach_identity_is_not_followed(workspace, monkeypatch):
    initial = {"mode": "run", "status": "RUNNING", "pid": 123, "pid_alive_at_snapshot": True,
        "started_at": "fixed_start", "command": ["python", "-m", "phase2_experiment.runner", "run"]}
    changed = dict(initial, pid=456)
    states = iter([initial, changed])
    monkeypatch.setattr(finish.monitor, "_process_state", lambda: next(states))
    monkeypatch.setattr(finish.monitor, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(finish.monitor, "snapshot", lambda: pytest.fail("new PID must not be followed"))
    with pytest.raises(RuntimeError, match="identity changed"):
        finish._wait_attached(workspace, {"events": []}, interval=1)


def test_zip_crc_and_every_content_hash_are_checked(workspace):
    report = workspace / "review"
    report.mkdir()
    (report / "result.csv").write_text("seed,metric\n17,0.1\n", encoding="utf-8")
    (report / "float.npy").write_bytes(b"archive byte fixture")
    archive_path = workspace / "test.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in report.iterdir():
            archive.write(path, path.name)
    result = finish.verify_zip_contents(archive_path, report)
    assert result["ok"] and result["CRC_checked"] and result["every_content_sha256_checked"]
    assert result["member_count"] == 2 and all(row["matches_review_source"] for row in result["members"])
    (report / "result.csv").write_text("changed after export", encoding="utf-8")
    result = finish.verify_zip_contents(archive_path, report)
    assert not result["ok"] and any("SHA256 differs" in issue for issue in result["issues"])


def test_registered_live_process_never_requires_cim_query(monkeypatch):
    monkeypatch.setattr(finish.monitor, "_process_state", lambda: {"status": "RUNNING", "pid_alive_at_snapshot": True,
        "pid": 123, "mode": "run", "started_at": "fixed_start", "command": ["python", "-m", "phase2_experiment.runner", "run"]})
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: pytest.fail("registered live PID needs no CIM enumeration"))
    assert finish._live_training_processes()[0]["pid"] == 123
    assert finish._PROCESS_COVERAGE["direct_process_query"] == "NOT_NEEDED_REGISTERED_LIVE_PROCESS"


def test_cim_denied_does_not_claim_no_process_or_prevent_readonly_gate(workspace, monkeypatch):
    monkeypatch.setattr(finish.monitor, "_process_state", lambda: {"status": "PROCESS_FINISHED", "pid_alive_at_snapshot": False, "pid": 123})
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr="Get-CimInstance: AccessDenied"))
    assert finish._live_training_processes() == []
    assert finish._PROCESS_COVERAGE["direct_process_query"] == "DENIED_OR_UNAVAILABLE"
    assert finish._PROCESS_COVERAGE["absence_verified"] is False and finish._PROCESS_COVERAGE["warnings"]
    monkeypatch.setattr(finish, "verify_manifest_artifacts", lambda: {"ok": True, "counts": {"COMPLETED": 75}, "issues": {}})
    monkeypatch.setattr(finish, "_core_guard", lambda: {"ok": True})
    protocol = workspace / "PHASE2_LOCKED_PROTOCOL.json"
    protocol.write_text("{}", encoding="utf-8")
    assert finish._full_gate(finish.sha256_file(protocol))["ok"] is True
    with pytest.raises(RuntimeError, match="Training launch blocked"):
        finish._launch_queue(workspace, {"preflight": {}}, "resume")


def test_full_gate_rechecks_entire_locked_protocol_file(workspace, monkeypatch):
    monkeypatch.setattr(finish, "verify_manifest_artifacts", lambda: {"ok": True, "counts": {"COMPLETED": 75}, "issues": {}})
    monkeypatch.setattr(finish, "_core_guard", lambda: {"ok": True})
    protocol = workspace / "PHASE2_LOCKED_PROTOCOL.json"
    protocol.write_text('{"unchanged":true}', encoding="utf-8")
    digest = finish.sha256_file(protocol)
    (workspace / "PHASE2_PROTOCOL_SELECTION.json").write_text(json.dumps({"locked_protocol_sha256": digest}), encoding="utf-8")
    assert finish._full_gate(digest)["ok"] is True
    protocol.write_text('{"changed_after_preflight":true}', encoding="utf-8")
    gate = finish._full_gate(digest)
    assert not gate["ok"] and not gate["protocol_file_guard"]["ok"]
