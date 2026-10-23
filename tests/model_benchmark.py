"""Resource-only forward/backward measurement; never a formal training run."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from theory_fm_models import ModelConfig, MAIN_MODELS, FieldHistory, build_model


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    cfg = ModelConfig(state_dim=64, hidden_dim=128, hidden_layers=3,
                      query_chunk=64, source_chunk=32)
    x = torch.randn(1, 64, 64, device="cuda")
    history = FieldHistory(torch.linspace(0, 0.7, 17, device="cuda", dtype=torch.float64),
                           torch.randn(1, 17, 32, 64, device="cuda"))
    results = []
    for name in MAIN_MODELS:
        m = build_model(name, cfg).cuda()
        durations = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(3):
            m.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tic = time.perf_counter()
            out = m(x, 0.7, history if m.needs_history else None)
            out.drift.square().mean().backward()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - tic)
        item = {"model_id": name, "seconds_forward_backward": durations,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "E": 1, "Q": 64, "N": 32, "H": 17, "D": 64,
                "tf32": False, "amp": False, "training_updates": 0}
        results.append(item)
        print(json.dumps(item), flush=True)
        del m
        torch.cuda.empty_cache()
    path = Path(__file__).with_name("model_benchmark_results.json")
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
