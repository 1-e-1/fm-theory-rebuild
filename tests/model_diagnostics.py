"""Checks for same-checkpoint diagnostics; no optimization or real data access."""
import json
import numpy as np
import pytest
import torch
from theory_fm_models import ModelConfig, build_model
from experiment.diagnostics import checkpoint_diagnostics, couple_brownian, CURL_PAIRS


def test_brownian_fine_to_coarse_uses_sum_not_average():
    fine=torch.arange(128*2*4*2,dtype=torch.float64).reshape(128,2,4,2)
    for steps in (32,64,128):
        coarse=couple_brownian(fine,steps)
        torch.testing.assert_close(coarse.sum(0),fine.sum(0),rtol=0,atol=0)
        assert coarse.shape==(steps,2,4,2)
    with pytest.raises(ValueError):couple_brownian(fine,33)


def test_checkpoint_diagnostics_saves_recoverable_raw_arrays(tmp_path):
    model=build_model("FM_STANDARD",ModelConfig(state_dim=64,hidden_dim=4,hidden_layers=1))
    val=torch.randn(64,64,generator=torch.Generator().manual_seed(520))
    before={name:p.clone() for name,p in model.state_dict().items()}
    report=checkpoint_diagnostics(model,tmp_path,val)
    assert report["status"]=="COMPLETED" and report["training_updates"]==0
    assert len(report["checks"])==5 and report["field"]["true_labels_available"] is False
    assert model.training
    for name,p in model.state_dict().items():torch.testing.assert_close(p,before[name],rtol=0,atol=0)
    directory=tmp_path/"checkpoint_diagnostics"
    with np.load(directory/report["field"]["raw_npz"]) as arrays:
        assert arrays["queries"].shape==(1,4,64)
        assert arrays["reference_history"].shape==(1,17,32,64)
        assert arrays["trace_probes"].shape==(4,1,4,64)
        np.testing.assert_array_equal(arrays["curl_pairs"],CURL_PAIRS)
        assert arrays["has_true_labels"].item() is False
    timings={k:v["cost"]["sampling_seconds"] for k,v in report["checks"].items()}
    recovered=checkpoint_diagnostics(model,tmp_path,val)
    assert timings=={k:v["cost"]["sampling_seconds"] for k,v in recovered["checks"].items()}
    with torch.no_grad():next(model.parameters()).add_(0.01)
    with pytest.raises(ValueError):checkpoint_diagnostics(model,tmp_path,val)
