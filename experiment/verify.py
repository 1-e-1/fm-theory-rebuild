"""Artifact verification CLI with Python-bool JSON normalization."""
import json
from experiment.scope import load_manifest, active_attempt_path, verify_manifest_artifacts
from experiment.runner import complete_artifacts


def main():
    result = verify_manifest_artifacts()
    evidence = {}
    for run in load_manifest()["runs"]:
        attempt = active_attempt_path(run)
        try:
            complete = bool(complete_artifacts(attempt)) if attempt else False
            evidence[run["run_id"]] = complete
            if run["status"] == "COMPLETED" and not complete:
                result["ok"] = False
                result["issues"][run["run_id"]] = ["Final-budget/source-context check failed"]
        except Exception as error:
            evidence[run["run_id"]] = False
            if run["status"] == "COMPLETED":
                result["ok"] = False
                result["issues"][run["run_id"]] = [repr(error)]
    result["complete_artifact_evidence"] = evidence
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
