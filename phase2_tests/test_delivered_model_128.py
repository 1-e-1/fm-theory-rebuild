"""Actual delivered-model integration at the new complete-sequence dimension.

No optimizer step or formal configuration is created by these checks.
"""
import pytest
import torch

from theory_fm_models import FieldHistory, ModelConfig, build_model


MODELS = (
    "FM_STANDARD", "FM_REAL_ODE", "FM_COMPLEX_SDE",
    "FM_COMPLEX_ODE", "FM_COMPLEX_VDC_ODE",
)


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float64),
    pytest.param("cuda", torch.float32,
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_d128_complete_state_forward_parameter_gradient_and_causal_first_cell(model_id, device, dtype):
    torch.manual_seed(742019)
    cfg = ModelConfig(state_dim=128, hidden_dim=128, hidden_layers=3,
                      query_chunk=4, source_chunk=5)
    model = build_model(model_id, cfg).to(device=device, dtype=dtype)
    x = torch.randn(1, 4, 128, device=device, dtype=dtype)
    times = torch.tensor([0.0, 0.2, 0.4], dtype=torch.float64, device=device)
    sources = torch.randn(1, 3, 5, 128, device=device, dtype=dtype)
    history = FieldHistory(times, sources) if model.needs_history else None
    result = model(x, times[-1], history=history, create_graph=True)
    assert result.drift.shape == x.shape
    assert bool(torch.isfinite(result.drift).all())
    loss = result.drift.square().mean()
    loss.backward()
    active = [p for p in model.parameters() if p.requires_grad]
    assert active
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in active)
    initial = FieldHistory(times[:1], sources[:, :1]) if model.needs_history else None
    with torch.no_grad():
        average = model.cell_average(x, 0.0, 1.0 / 64, history=initial,
                                     create_graph=False)
    assert average.drift.shape == x.shape
    assert bool(torch.isfinite(average.drift).all())
