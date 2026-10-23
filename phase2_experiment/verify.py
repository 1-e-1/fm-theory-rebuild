"""Check independent Phase 2 artifacts and immutable Phase 1 protection hashes."""
from __future__ import annotations
import json
from .scope import verify_manifest_artifacts


def main() -> int:
    result = verify_manifest_artifacts()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
