"""Foreground supervisor for the fixed next-phase queue, with UTF-8 logs.

This starts only explicit selected-task pilots or the locked formal runner.
It never selects a protocol, initializes a scope, or launches old Phase 1 work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading

from .scope import DATASETS, ROOT, load_manifest, verify_phase1_guard, write_json
from .monitor import snapshot


def supervise(command: list[str], mode: str, dataset_id: str | None = None) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs = ROOT / "outputs" / ("phase2_pilots" if mode == "pilot" else "phase2_3datasets") / "console_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{mode}_{dataset_id or 'queue'}_{stamp}.log"
    state = {"mode": mode, "dataset_id": dataset_id, "command": command,
             "started_at": datetime.now(timezone.utc).isoformat(), "console_log": str(log_path),
             "status": "RUNNING", "formal_configuration_denominator": 75}
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(30):
            try:
                snapshot()
            except Exception as exc:
                # Training progress/checkpoints remain owned by the runner.
                print(json.dumps({"monitor_error": repr(exc)}), flush=True)

    watcher = threading.Thread(target=heartbeat, daemon=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(state, ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        state["pid"] = process.pid
        write_json(ROOT / "PHASE2_PROCESS_STATE.json", state)
        watcher.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        finally:
            stop.set()
            watcher.join(timeout=2)
    state.update(status="PROCESS_FINISHED" if code == 0 else "PROCESS_FAILED", exit_code=code,
                 ended_at=datetime.now(timezone.utc).isoformat())
    write_json(ROOT / "PHASE2_PROCESS_STATE.json", state)
    snapshot()
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilots", "run", "resume"))
    parser.add_argument("--pilot-steps", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.pilot_steps <= 2000:
        parser.error("pilot-steps must be 1..2000")
    load_manifest()
    guard = verify_phase1_guard()
    if not guard["ok"]:
        raise RuntimeError(f"Protected Phase 1 evidence changed: {guard['issues']}")
    if args.mode == "pilots":
        for dataset in DATASETS:
            code = supervise([sys.executable, "-u", "-m", "phase2_experiment.runner", "pilot",
                              "--dataset", dataset, "--pilot-steps", str(args.pilot_steps)], "pilot", dataset)
            if code:
                raise SystemExit(code)
    else:
        code = supervise([sys.executable, "-u", "-m", "phase2_experiment.runner", args.mode], args.mode)
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
