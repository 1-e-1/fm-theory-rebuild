"""Accepted empirical-law history, initial RL cell, and explicit ODE/SDE solvers."""
from __future__ import annotations
import time
import torch
from theory_fm_models import FieldHistory


def sample(model, *, ensembles=16, particles=64, steps=64, seed=8001,
           solver="euler", initial=None, brownian=None):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (ensembles, particles, model.cfg.state_dim)
    x = (torch.randn(shape, generator=generator, device=device) if initial is None
         else initial.to(device).clone())
    h = 1.0 / steps
    accepted = [x.detach().clone()]
    calls = 0
    started = time.perf_counter()
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for n in range(steps):
            t = n * h
            if model.needs_history:
                history = FieldHistory(torch.arange(n + 1, dtype=torch.float64,
                                                    device=device) * h,
                                       torch.stack(accepted, dim=1))
            else:
                history = None
            if n == 0:
                out = model.cell_average(x, 0.0, h, history=history, create_graph=False)
                calls += 1 if model.needs_history else 3
            else:
                out = model(x, t, history=history, create_graph=False)
                calls += 1
            proposal = x + h * out.drift
            if solver == "heun":
                if model.needs_history or model.kappa:
                    raise ValueError("Heun reference is only for the standard ODE checkpoint")
                # First cell already integrates the time dependence at frozen x.
                first = model(x, t, create_graph=False).drift if n == 0 else out.drift
                second = model(x + h * first, t + h, create_graph=False).drift
                proposal = x + .5 * h * (first + second)
                calls += 2 if n == 0 else 1
            if model.kappa:
                # Complete model drift already includes kappa*score: no second score.
                noise = (torch.randn(shape, generator=generator, device=device)
                         * (h ** .5) if brownian is None else brownian[n].to(device))
                proposal = proposal + model.diffusion * noise
            x = proposal.detach()
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"nonfinite accepted sampling state at step {n + 1}")
            accepted.append(x.clone())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # Count evaluated potential points and spatial reverse-mode calls, including
    # all historical fields at the SAME current query x.
    snapshots = steps * (steps + 1) // 2
    points = ensembles * particles * snapshots
    channels = 2 if model.spec.family == "two_potential" else 1
    cost = {"sampling_seconds": elapsed, "ensembles": ensembles, "N": particles,
            "steps": steps, "history_max": steps, "drift_calls": calls,
            "V_evaluation_points": points if model.needs_history else 0,
            "W_network_evaluation_points": 2 * particles * points if model.needs_history else 0,
            "spatial_reverse_calls": (snapshots * channels *
                                      ((particles + getattr(model.cfg, "query_chunk", particles) - 1)
                                       // getattr(model.cfg, "query_chunk", particles))) if model.needs_history else 0,
            "solver": solver, "initial_cell": ("Heun_endpoint_pair" if solver == "heun" else
                                                  "RL_PC_average" if model.needs_history else "Gauss3"),
            "peak_sampling_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
            "seed": seed, "history_source": "generated_accepted_states_only"}
    return x.cpu().numpy(), cost
