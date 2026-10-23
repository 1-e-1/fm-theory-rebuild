"""Register selected-task pilots/checks independently of the 50 formal runs."""
from __future__ import annotations

from .scope import MODELS, ROOT, now, read_json, write_json


def refresh_registry() -> dict:
    source = ROOT / "AUXILIARY_RUNS.json"
    if not source.exists():
        return {"formal": False, "count_towards_50": False, "activities": [], "registry_source": "AUXILIARY_RUNS.json", "source_available": False}
    auxiliary = read_json(source)
    auxiliary.setdefault("pilots", {})["attempt_policy"] = {
        "attempt_001": "Initial independent 200-update train/validation pilot, preserved",
        "attempt_002": "200-update retest after checkpoint/resume/source-provenance interface fixes; first attempt retained; no warm-start of formal runs",
        "count_towards_formal_50": False,
    }
    pilot_attempts = []
    for model in MODELS:
        base = ROOT / "outputs/pilots" / f"SEQ_IID__{model}__seed17"
        for attempt in sorted(base.glob("attempt_*")) if base.exists() else []:
            config_path = attempt / "config.json"
            cfg = read_json(config_path) if config_path.exists() else {}
            result_path = attempt / "pilot_result.json"
            result = read_json(result_path) if result_path.exists() else {}
            pilot_attempts.append({
                "dataset_id": "SEQ_IID", "model_id": model, "seed": cfg.get("seed", 17),
                "attempt_id": attempt.name, "directory": str(attempt.relative_to(ROOT)),
                "configured_updates": cfg.get("steps", 200),
                "source_sha256": cfg.get("source_sha256"),
                "data_access": ["train", "validation"], "test_accessed": result.get("test_accessed", False),
                "state": "PILOT_ARTIFACTS_AVAILABLE" if result_path.exists() else "PILOT_PARTIAL_OR_PENDING",
                "retest_reason": "checkpoint/resume/source-provenance interface fixes" if attempt.name == "attempt_002" else None,
                "formal": False, "count_towards_50": False,
            })
    auxiliary["pilot_attempts"] = pilot_attempts
    auxiliary["registry_refreshed_at"] = now()
    write_json(source, auxiliary)
    registry = {"formal": False, "count_towards_50": False, "registry_source": "AUXILIARY_RUNS.json",
                "refreshed_at": now(), "activities": [
                    {"kind": "correctness_tests", "source": "AUXILIARY_RUNS.json#correctness_tests"},
                    {"kind": "resource_benchmark", "source": "AUXILIARY_RUNS.json#resource_benchmark", "optimizer_updates": 0},
                    *pilot_attempts,
                    {"kind": "same_checkpoint_sampling_checks", "source": "AUXILIARY_RUNS.json#same_checkpoint_checks", "additional_formal_configurations": 0},
                    {"kind": "robot_pilot", "status": "BLOCKED", "reason": "Actual robot resource blockade; no replacement robot"},
                ]}
    write_json(ROOT / "outputs/phase1_iid_robot/nonformal_registry.json", registry)
    return registry


if __name__ == "__main__":
    import json
    print(json.dumps(refresh_registry(), ensure_ascii=False, indent=2))
