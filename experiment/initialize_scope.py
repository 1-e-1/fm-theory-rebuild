"""One-time initialization of the fixed Phase 1 scope; never overwrite progress."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from .scope import DATASETS, EXPECTED, MANIFEST_PATH, MODELS, ROOT, SCOPE_PATH, SEEDS, now, run_identifier, write_json


def main() -> None:
    if SCOPE_PATH.exists() or MANIFEST_PATH.exists():
        raise SystemExit("Scope/manifest already exists; initialization refuses to overwrite experiment progress.")
    configurations = [
        {"queue_index": i + 1, "dataset_id": d, "model_id": m, "seed": s,
         "run_id": run_identifier(d, m, s)}
        for i, (d, m, s) in enumerate(EXPECTED)
    ]
    source = ROOT / "START_CODEX_PHASE1_50_RUNS.txt"
    deferred_datasets = ["SEQ_DELAY", "SEQ_ACCUM", "SEQ_MULTISCALE", "TS_ETTM1", "TS_APPLIANCES", "TRAJ_LASA", "ROBOT_PLANAR", "OTHER_REAL_DATA"]
    deferred_models = ["EXTRA_CONTROL_MODELS", "FIXED_ORDER_ABLATIONS", "ROTATING_REFERENCE_PATH", "LENGTH_VARIANTS", "DATA_SIZE_VARIANTS", "J_VARIANTS", "NOISE_STRENGTH_VARIANTS", "EXTRA_CAPACITY_MATCHING"]
    scope = {
        "schema_version": 1, "phase_id": "PHASE1_50", "created_at": now(),
        "user_context_date": "2026-10-09", "timezone": "Asia/Shanghai",
        "authority": "START_CODEX_PHASE1_50_RUNS.txt and latest explicit user request",
        "authority_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "datasets": list(DATASETS), "models": list(MODELS), "training_seeds": list(SEEDS),
        "formal_configuration_count": 50, "configurations_per_dataset": 25,
        "train_from_scratch": True, "allow_old_checkpoints": False,
        "queue_order": "dataset, then training seed, then fixed model order",
        "forbid_default_full_queue": True, "formal_update_budget_options": [12000, 24000, 48000],
        "per_dataset_shared_data_split_and_protocol": True,
        "nonformal_activities": ["correctness tests", "short pilots on selected datasets", "same-checkpoint sampling checks"],
        "nonformal_activities_count_towards_50": False,
        "deferred_status": "DEFERRED_BY_USER", "deferred_datasets": deferred_datasets,
        "deferred_variants": deferred_models,
        "completion_required_artifacts": ["checkpoint", "configuration", "training log", "floating generated samples", "final evaluation"],
        "configurations": configurations,
    }
    write_json(SCOPE_PATH, scope)
    audit = {
        "schema_version": 1, "audited_at": now(), "dataset_id": "ROBOT_8DOF", "status": "BLOCKED",
        "current_project": str(ROOT), "current_project_robot_asset_files": [],
        "current_project_checks": "Full current-project file listing; no USD/USDA/USDC/URDF/robot expert-data files in delivered root.",
        "explicit_external_asset_paths_supplied": [],
        "historical_hints_existence_only": [{"path": p, "exists": Path(p).exists(), "contents_inspected": False} for p in
             [r"D:\work_space\cfm", r"D:\work_space\lunwen2.1", r"D:\conda_envs\env_isaaclab", str(__import__('pathlib').Path(__import__('os').environ.get('ISAACLAB_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'third_party/IsaacLab'))))]],
        "current_python": sys.executable,
        "current_python_dependency_probe": {"pxr": False, "isaaclab": False, "isaacsim": False},
        "dependency_probe_scope": "Default Python only; no assertion about uninspected separate Isaac environments.",
        "missing": [
            {"id": "actual_robot_usd", "path": None, "reason": "No actual right-arm USD entry path supplied or present in current project"},
            {"id": "referenced_assets", "path": None, "reason": "Referenced materials/geometry/textures cannot be resolved without USD"},
            {"id": "geometry_fk", "path": None, "reason": "Correct collision geometry and calibrated FK evidence unavailable"},
            {"id": "joint_tcp_limits", "path": None, "reason": "Joint order/TCP/base/unwrap/sourced limits/collision coverage unavailable"},
            {"id": "task_expert_data", "path": None, "reason": "Task conditions, task splits and >=8 valid experts/condition unavailable"},
        ],
        "no_substitution": True, "all_25_robot_runs_retained": True,
    }
    write_json(ROOT / "ROBOT_RESOURCE_AUDIT.json", audit)
    runs = []
    for cfg in configurations:
        robot = cfg["dataset_id"] == "ROBOT_8DOF"
        runs.append(dict(cfg, status="BLOCKED" if robot else "PENDING", attempts=[],
            output_dir=f"outputs/phase1_iid_robot/{cfg['run_id']}", created_at=now(), updated_at=now(),
            reason="Actual robot USD/geometry/FK/metadata/expert data unverified; see ROBOT_RESOURCE_AUDIT.json" if robot else "Awaiting correctness checks, shared data and locked formal protocol",
            events=[]))
    manifest = {"schema_version": 1, "phase_id": "PHASE1_50", "created_at": now(), "updated_at": now(),
                "formal_configuration_count": 50, "runs": runs,
                "nonformal_registry_path": "outputs/phase1_iid_robot/nonformal_registry.json",
                "deferred_registry_path": "DEFERRED_BY_USER.json"}
    write_json(MANIFEST_PATH, manifest)
    write_json(ROOT / "DEFERRED_BY_USER.json", {"status": "DEFERRED_BY_USER", "count_towards_50": False,
        "datasets": deferred_datasets, "variants": deferred_models,
        "reason": "Explicitly excluded by the latest Phase 1 scope; neither failures nor missing Phase 1 results"})
    write_json(ROOT / "outputs/phase1_iid_robot/nonformal_registry.json", {"formal": False, "count_towards_50": False, "activities": []})
    print("Initialized exactly 50 formal triples: SEQ_IID 25 PENDING; ROBOT_8DOF 25 BLOCKED.")


if __name__ == "__main__":
    main()
