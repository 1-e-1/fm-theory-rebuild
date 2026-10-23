"""Guard the fixed formal denominator and fail-closed completion reporting."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from experiment.reporting import COMPARISONS, holm, moments
from experiment.scope import EXPECTED, get_run, load_manifest, update_run, validate_config, verify_manifest_artifacts


def test_scope_contains_exactly_seed_grouped_fifty_unique_triples():
    manifest = load_manifest()
    triples = [(r["dataset_id"], r["model_id"], r["seed"]) for r in manifest["runs"]]
    assert tuple(triples) == EXPECTED
    assert len(set(triples)) == len(triples) == 50
    assert sum(d == "SEQ_IID" for d, _, _ in triples) == 25
    assert sum(d == "ROBOT_8DOF" for d, _, _ in triples) == 25


@pytest.mark.parametrize("triple", [
    ("SEQ_DELAY", "FM_STANDARD", 17),
    ("SEQ_IID", "EXTRA_CONTROL", 17),
    ("SEQ_IID", "FM_STANDARD", 99),
    ("SEQ_IID", "FM_STANDARD", 17.0),
    ("SEQ_IID", "FM_STANDARD", True),
])
def test_unauthorized_formal_triples_are_rejected(triple):
    with pytest.raises(ValueError, match="outside PHASE1_SCOPE"):
        validate_config(*triple)


def test_completion_cannot_be_declared_without_actual_artifacts():
    manifest = deepcopy(load_manifest())
    run = manifest["runs"][0]
    # This test replaces only in-memory state and never edits a live manifest.
    run["attempt_dir"] = "outputs/phase1_iid_robot/nonexistent_scope_guard_test"
    with patch("experiment.scope.load_manifest", return_value=manifest), patch("experiment.scope.write_json") as persist:
        with pytest.raises(ValueError, match="Cannot mark COMPLETED"):
            update_run(run["run_id"], "COMPLETED")
        persist.assert_not_called()


def test_manifest_verification_retains_denominator_even_with_blockers():
    result = verify_manifest_artifacts()
    assert result["formal_configuration_count"] == 50
    assert sum(result["counts"].values()) == 50


def test_seed_statistics_do_not_treat_partial_group_as_five_repeats():
    partial = moments([1.0, 2.0])
    assert partial["n_training_seeds"] == 2
    assert partial["expected_training_seeds"] == 5
    assert not partial["complete_seed_group"]
    complete = moments([1.0, 1.0, 1.0, 1.0, 1.0])
    assert complete["complete_seed_group"]
    assert complete["std"] == 0
    assert complete["ci95_low"] == complete["ci95_high"] == 1


def test_holm_uses_registered_seven_comparison_family():
    assert len(COMPARISONS) == 7
    assert COMPARISONS[:3] == (("FM_COMPLEX_ODE", "FM_REAL_ODE"), ("FM_COMPLEX_SDE", "FM_COMPLEX_ODE"), ("FM_COMPLEX_VDC_ODE", "FM_COMPLEX_ODE"))
    result = holm([(0, 0.01), (1, 0.04), (2, 0.03), (3, 0.005)])
    assert result == pytest.approx({0: 0.06, 1: 0.16, 2: 0.15, 3: 0.035})
    # An unavailable comparison does not shrink the multiplicity family.
    assert holm([(0, 0.01)], 7) == pytest.approx({0: 0.07})
