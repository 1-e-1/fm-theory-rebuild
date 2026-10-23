"""Checkpoint/RNG roundtrip checks; zero optimizer updates, no data splits read.

Small synthetic CPU float64 and (if available) GPU float32 inputs exercise the
trainer's actual random batch builder. These are correctness checks, not pilots
or formal training configurations.
"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiment.runner import save_checkpoint, training_batch, seed_all
from theory_fm_models import ModelConfig, build_model, optimization_groups


class CheckpointRoundtrip(unittest.TestCase):
    def check_device(self, device: str, dtype: torch.dtype) -> None:
        seed_all(1607)
        cfg = ModelConfig(state_dim=4, hidden_dim=8, hidden_layers=2, query_chunk=4, source_chunk=3)
        # Deterministic synthetic inputs; no project train/val/test read.
        data = torch.linspace(-1.8, 1.8, 80, dtype=dtype, device=device).reshape(20, 4)
        model = build_model("FM_COMPLEX_SDE", cfg).to(device=device, dtype=dtype)
        optimizer = torch.optim.AdamW(optimization_groups(model))

        def next_batch_snapshot(net):
            captured = {}
            def record_inputs(module, args, kwargs):
                captured["query"] = args[0].detach().clone()
                captured["tau_tensor"] = args[1].detach().clone()
                captured["history_times"] = kwargs["history"].times.detach().clone()
                captured["reference"] = kwargs["history"].particles.detach().clone()
            def record_output(module, args, kwargs, output):
                captured["drift"] = output.drift.detach().clone()
            before = net.register_forward_pre_hook(record_inputs, with_kwargs=True)
            after = net.register_forward_hook(record_output, with_kwargs=True)
            try:
                captured["python_random"] = random.random()
                captured["numpy_random"] = float(np.random.rand())
                loss, tau, raw_loss = training_batch(data, net, H=4, N=3, Q=4)
                captured["loss"] = loss.detach().clone()
                captured["tau"] = tau
                captured["raw_loss"] = raw_loss
            finally:
                before.remove()
                after.remove()
            return captured

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "roundtrip.pt"
            save_checkpoint(checkpoint_path, model, optimizer, 0, float("inf"), 0.0, cfg, 0)
            expected = next_batch_snapshot(model)
            # Construction and these independent draws deliberately disturb RNG.
            restored_model = build_model("FM_COMPLEX_SDE", cfg).to(device=device, dtype=dtype)
            restored_optimizer = torch.optim.AdamW(optimization_groups(restored_model))
            random.random()
            np.random.rand(20)
            torch.randn(20, device=device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            restored_model.load_state_dict(checkpoint["model"])
            restored_optimizer.load_state_dict(checkpoint["optimizer"])
            random.setstate(checkpoint["python_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
            observed = next_batch_snapshot(restored_model)
            self.assertEqual(checkpoint["step"], 0)
            self.assertEqual(checkpoint["model_config"], asdict(cfg))
            self.assertEqual(restored_optimizer.state_dict()["state"], {})
            self.assertEqual(restored_optimizer.state_dict()["param_groups"], optimizer.state_dict()["param_groups"])
            for key in expected:
                if isinstance(expected[key], torch.Tensor):
                    self.assertTrue(torch.equal(expected[key], observed[key]), f"{device}/{dtype}: {key} differs")
                else:
                    self.assertEqual(expected[key], observed[key], f"{device}/{dtype}: {key} differs")

    def test_cpu_float64_next_batch_drift_loss(self):
        self.check_device("cpu", torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_gpu_float32_next_batch_drift_loss(self):
        self.check_device("cuda", torch.float32)


if __name__ == "__main__":
    unittest.main()
