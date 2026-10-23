"""Independent label and solver checks on the actual experiment interfaces."""
from types import SimpleNamespace
import math
import numpy as np
import torch
from experiment.runner import training_batch
from experiment.sampling import sample


class ConstantDrift:
    def __init__(self, history=False, kappa=0, drift=.7):
        self.param = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(state_dim=64)
        self.spec = SimpleNamespace(family="shared_potential" if history else "direct")
        self.needs_history, self.kappa, self.drift = history, kappa, drift
        self.diffusion = math.sqrt(2 * kappa)
        self.calls = []
    def parameters(self):
        return iter([self.param])
    def eval(self):
        return self
    def __call__(self, x, t, history=None, create_graph=False):
        if self.needs_history:
            assert history.times[0] == 0
            torch.testing.assert_close(history.particles[:, -1], x)
            assert len(history.times) == len(self.calls) + 1
        self.calls.append((x.clone(), history))
        return SimpleNamespace(drift=torch.full_like(x, self.drift))
    def cell_average(self, x, left, right, history=None, create_graph=False):
        assert left == 0 and right > 0
        return self(x, left, history, create_graph)


def test_actual_solver_complete_drift_and_coupled_brownian():
    shape = (2, 256, 64)
    g = torch.Generator().manual_seed(930)
    increments = torch.randn((16, *shape), generator=g) / 4
    initial = torch.zeros(shape)
    model = ConstantDrift(kappa=.05)
    result, _ = sample(model, ensembles=2, particles=256, steps=16,
                       initial=initial, brownian=increments)
    expected = .7 + math.sqrt(.1) * increments.sum(0)
    np.testing.assert_allclose(result, expected.numpy(), atol=3e-6)
    residual = result - .7
    assert abs(residual.var() - .1) < .003
    # Doubling h by coupling adjacent fine increments yields the same Brownian law.
    coarse, _ = sample(ConstantDrift(kappa=.05), ensembles=2, particles=256,
                       steps=8, initial=initial,
                       brownian=increments.reshape(8, 2, *shape).sum(1))
    np.testing.assert_allclose(coarse, result, atol=3e-6)


def test_accepted_history_and_nonzero_initial_cell():
    model = ConstantDrift(history=True, drift=.7)
    result, info = sample(model, ensembles=2, particles=32, steps=8,
                          initial=torch.zeros(2, 32, 64))
    np.testing.assert_allclose(result, .7, atol=1e-6)
    assert info["history_source"] == "generated_accepted_states_only"
    assert len(model.calls) == 8


def test_training_path_and_complete_sde_label():
    class Capture:
        needs_history = True
        kappa = .05
        def __call__(self, x, tau, history, create_graph):
            self.x, self.tau, self.history = x, tau, history
            return SimpleNamespace(drift=torch.zeros_like(x))
    # Test tensors are operator inputs, not an extra research dataset or training.
    data = torch.randn(100, 64, generator=torch.Generator().manual_seed(120))
    seed = 600
    replay = torch.Generator().manual_seed(seed)
    qi = torch.randint(100, (64,), generator=replay)
    ri = torch.randint(100, (32,), generator=replay)
    y, yr = data[qi][None], data[ri][None]
    eps = torch.randn(y.shape, generator=replay)
    er = torch.randn(yr.shape, generator=replay)
    u = torch.rand((), dtype=torch.float64, generator=replay)
    coin = torch.rand((), generator=replay)
    tau = u if coin < .5 else u ** 4
    sigma = 1 - .95 * tau
    target = y - .95 * eps - .05 * eps / sigma.float()
    model = Capture()
    loss, time, raw = training_batch(data, model, generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(model.x, tau.float() * y + sigma.float() * eps)
    torch.testing.assert_close(model.history.particles[:, 0], er)
    torch.testing.assert_close(model.history.particles[:, -1], tau.float() * yr + sigma.float() * er)
    expected = target.square().mean() / (.5 + .125 * tau ** (-.75)).float()
    torch.testing.assert_close(loss, expected)
    assert time == float(tau)
    # Sources use a separately drawn endpoint/noise pair that persists across time.
    assert not torch.equal(model.history.particles[:, -1], model.x[:, :32])
