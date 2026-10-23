"""Phase-one-only trainer. No default all/full experiment entry point exists."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import numpy as np
import torch
from theory_fm_models import ModelConfig, FieldHistory, build_model, optimization_groups, parameter_counts
from experiment.sampling import sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase1_iid_robot"
MODELS = ["FM_STANDARD", "FM_REAL_ODE", "FM_COMPLEX_SDE", "FM_COMPLEX_ODE", "FM_COMPLEX_VDC_ODE"]
SEEDS = [17, 29, 43, 71, 101]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def config():
    return ModelConfig(state_dim=64, hidden_dim=128, hidden_layers=3,
                       query_chunk=64, source_chunk=32)


def load_data(split, device):
    from experiment.data import load_split
    return load_split(split, device=device)


def training_batch(data, model, *, H=17, N=32, Q=64, generator=None,
                   validation=False):
    device = data.device
    # Separate independent draws, never use the current query's paired endpoint
    # or its epsilon as reference law information.
    yi = torch.randint(len(data), (Q,), device=device, generator=generator)
    ri = torch.randint(len(data), (N,), device=device, generator=generator)
    y = data[yi].unsqueeze(0)
    eps = torch.randn(y.shape, device=device, generator=generator)
    yr = data[ri].unsqueeze(0)
    er = torch.randn(yr.shape, device=device, generator=generator)
    while True:
        u = torch.rand((), dtype=torch.float64, device=device, generator=generator)
        if bool(u > 0):
            break
    mixture = torch.rand((), device=device, generator=generator)
    tau = u if bool(mixture < .5) else u ** 4
    density = .5 + .125 * tau ** (-.75)
    sigma = 1 - .95 * tau
    x = tau.float() * y + sigma.float() * eps
    target = y - .95 * eps
    if model.kappa:
        target = target - model.kappa * eps / sigma.float()
    times = torch.linspace(0, 1, H, dtype=torch.float64, device=device) * tau
    sigma_ref = 1 - .95 * times
    refs = (times.float()[None, :, None, None] * yr[:, None]
            + sigma_ref.float()[None, :, None, None] * er[:, None])
    history = FieldHistory(times, refs) if model.needs_history else None
    out = model(x, tau, history=history, create_graph=not validation)
    raw = (out.drift - target).square().mean()
    return raw / density.float(), float(tau), float(raw.detach())


def validate(model, val, batches=4):
    g = torch.Generator(device=val.device).manual_seed(602113)
    vals = []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            loss, _, _ = training_batch(val, model, generator=g, validation=True)
            vals.append(float(loss))
    model.train()
    return float(np.mean(vals))


def save_checkpoint(path, model, optimizer, step, best, elapsed, cfg, clip_count):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "step": step, "best_validation": best, "elapsed_seconds": elapsed,
               "model_config": asdict(cfg), "clip_count": clip_count,
               "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
               "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
               "model_sha256": hashlib.sha256((ROOT / "theory_fm_models.py").read_bytes()).hexdigest()}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def train_one(model_id, seed, steps, *, pilot=False, resume=False):
    from experiment.scope import validate_config, update_run
    validate_config("SEQ_IID", model_id, seed)
    report = read_json(ROOT / "tests" / "model_correctness_report.json")
    source_hash = hashlib.sha256((ROOT / "theory_fm_models.py").read_bytes()).hexdigest()
    if report["failed"] or report["source_sha256"] != source_hash:
        raise RuntimeError("Current model source must pass the numerical correctness gate")
    run_id = f"SEQ_IID__{model_id}__seed{seed}"
    base = ROOT / "outputs" / "pilots" / run_id if pilot else OUT / run_id
    attempts = sorted(base.glob("attempt_*"))
    directory = attempts[-1] if resume and attempts else base / f"attempt_{len(attempts)+1:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    cfg = config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_id, cfg).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(optimization_groups(model))
    train = load_data("train", device)
    val = load_data("validation", device)
    run_config = {"dataset_id": "SEQ_IID", "model_id": model_id, "seed": seed,
                  "run_id": run_id, "kind": "pilot" if pilot else "formal", "steps": steps,
                  "model_config": asdict(cfg), "N": 32, "H": 17, "Q": 64, "E": 1,
                  "sigma_min": .05, "time_sampling": "0.5 Uniform + 0.5 U^4; inverse density weight; no truncation",
                  "parameter_counts": parameter_counts(model), "device": str(device),
                  "checkpoint_for_primary_evaluation": "final", "EMA": "not used",
                  "data_spec_sha256": hashlib.sha256((ROOT / "DATA_SPEC.json").read_bytes()).hexdigest(),
                  "source_sha256": source_hash,
                  "execution_source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [ROOT / "theory_fm_models.py", ROOT / "experiment" / "runner.py",
                                  ROOT / "experiment" / "sampling.py", ROOT / "experiment" / "data.py",
                                  ROOT / "experiment" / "metrics.py", ROOT / "experiment" / "diagnostics.py"]}}
    if not pilot:
        protocol = read_json(ROOT / "LOCKED_PROTOCOL.json")
        if steps != protocol["SEQ_IID"]["updates"]:
            raise ValueError("Formal steps must exactly match the locked common protocol")
        run_config["locked_protocol_sha256"] = hashlib.sha256((ROOT / "LOCKED_PROTOCOL.json").read_bytes()).hexdigest()
        update_run(run_id, "RUNNING", attempt=directory.name, attempt_dir=str(directory), output_dir=str(directory))
    if resume and (directory / "config.json").exists():
        old_config = read_json(directory / "config.json")
        for key in ["dataset_id", "model_id", "seed", "steps", "model_config", "N", "H", "Q",
                    "data_spec_sha256", "source_sha256", "execution_source_hashes", "locked_protocol_sha256"]:
            if old_config.get(key) != run_config.get(key):
                raise ValueError(f"Resume configuration/source mismatch at {key}")
    source_dir = directory / "source"
    source_dir.mkdir(exist_ok=True)
    for p in [ROOT / "theory_fm_models.py", *sorted((ROOT / "experiment").glob("*.py"))]:
        destination = source_dir / p.name
        if not destination.exists():
            shutil.copy2(p, destination)
    write_json(directory / "config.json", run_config)
    start_step, best, previous_elapsed, clip_count = 0, float("inf"), 0.0, 0
    resume_path = directory / "checkpoint_resume.pt"
    if resume and resume_path.exists():
        cp = torch.load(resume_path, map_location=device, weights_only=False)
        if cp["model_sha256"] != run_config["source_sha256"]:
            raise ValueError("Cannot resume a checkpoint against changed model source")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        start_step, best, previous_elapsed = cp["step"], cp["best_validation"], cp["elapsed_seconds"]
        clip_count = cp["clip_count"]
        random.setstate(cp["python_rng"])
        np.random.set_state(cp["numpy_rng"])
        torch.set_rng_state(cp["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in cp["cuda_rng"]])
    log_path = directory / "training.csv"
    if start_step and log_path.exists():
        shutil.copy2(log_path, directory / f"training_before_resume_{time.time_ns()}.csv")
        with log_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with log_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(row for row in rows if int(row["step"]) <= start_step)
    fields = ["step", "elapsed_seconds", "loss", "raw_loss", "val_loss", "grad_norm", "grad_norm_after_clip", "clipped", "tau", "orders_json"]
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with log_path.open("a" if start_step else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not start_step:
            writer.writeheader()
        for step in range(start_step + 1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, tau, raw = training_batch(train, model)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}; tau={tau}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient at step {step}")
            clipped = int(norm > 1)
            clip_count += clipped
            optimizer.step()
            elapsed = previous_elapsed + time.perf_counter() - started
            interval = 100 if pilot else 1000
            val_loss = ""
            if step % interval == 0 or step == steps:
                val_loss = validate(model, val)
                if val_loss < best:
                    best = val_loss
                    save_checkpoint(directory / "checkpoint_best.pt", model, optimizer, step, best, elapsed, cfg, clip_count)
            writer.writerow({"step": step, "elapsed_seconds": elapsed, "loss": float(loss.detach()),
                             "raw_loss": raw, "val_loss": val_loss, "grad_norm": float(norm),
                             "grad_norm_after_clip": float(norm) * min(1.0, 1.0 / (float(norm) + 1e-6)),
                             "clipped": clipped, "tau": tau,
                             "orders_json": json.dumps(model.order_summary()) if step % 100 == 0 or step == 1 else ""})
            if step % 100 == 0 or step == steps:
                f.flush()
                save_checkpoint(resume_path, model, optimizer, step, best, elapsed, cfg, clip_count)
                print(json.dumps({"kind": "pilot" if pilot else "formal", "run": run_id, "step": step,
                                  "of": steps, "seconds": round(elapsed, 2), "loss": float(loss.detach()),
                                  "val_loss": val_loss}), flush=True)
    elapsed = previous_elapsed + time.perf_counter() - started
    save_checkpoint(directory / "checkpoint_final.pt", model, optimizer, steps, best, elapsed, cfg, clip_count)
    write_json(directory / "training_cost.json", {"training_seconds": elapsed, "updates": steps,
               "peak_training_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
               "clip_count": clip_count, "parameter_counts": parameter_counts(model),
               "V_evaluation_points": steps * 17 * 64 if model.needs_history else 0,
               "W_network_evaluation_points": steps * 17 * 64 * 32 * 2 if model.needs_history else 0,
               "spatial_reverse_calls": steps * 17 * (2 if model.spec.family == "two_potential" else 1) if model.needs_history else 0})
    if pilot:
        arr, cost = sample(model, ensembles=2, particles=32, steps=32, seed=9011)
        np.save(directory / "pilot_samples.npy", arr)
        write_json(directory / "sampling_diagnostics.json", cost)
        # Validation-only generation diagnostics; no test access in pilot.
        write_json(directory / "pilot_result.json", {"validation_loss": best, "final_validation_loss": val_loss,
                   "generated_mean": float(arr.mean()), "generated_variance": float(arr.var()),
                   "validation_mean": float(val.mean()), "validation_variance": float(val.var()),
                   "training_seconds": elapsed, "sampling": cost, "test_accessed": False})
    else:
        evaluate_one(directory)
        update_run(run_id, "COMPLETED", attempt_dir=str(directory), output_dir=str(directory), updates=steps)
    return directory


def evaluate_one(directory):
    from experiment.metrics import evaluate_run
    directory = Path(directory)
    conf = read_json(directory / "config.json")
    from experiment.scope import validate_config
    validate_config(conf["dataset_id"], conf["model_id"], conf["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp = torch.load(directory / "checkpoint_final.pt", map_location=device, weights_only=False)
    assert_run_context(conf, cp)
    cfg = ModelConfig(**cp["model_config"])
    model = build_model(conf["model_id"], cfg).to(device)
    model.load_state_dict(cp["model"])
    protocol = read_json(ROOT / "LOCKED_PROTOCOL.json")["SEQ_IID"]
    arr, cost = sample(model, ensembles=protocol["sampling_ensembles"],
                       particles=protocol["sampling_particles"], steps=protocol["sampling_steps"], seed=801017)
    np.save(directory / "samples.npy", arr)
    metrics = evaluate_run(arr, directory)
    write_json(directory / "evaluation.json", metrics)
    costs = {"primary": cost}
    if conf["model_id"] == "FM_STANDARD":
        reference, extra = sample(model, ensembles=protocol["sampling_ensembles"],
                                  particles=protocol["sampling_particles"], steps=protocol["sampling_steps"],
                                  seed=801017, solver="heun")
        np.save(directory / "samples_heun.npy", reference)
        write_json(directory / "evaluation_heun.json", evaluate_run(reference, directory / "heun_metrics"))
        costs["heun_reference"] = extra
    write_json(directory / "sampling_diagnostics.json", costs)
    from experiment.diagnostics import checkpoint_diagnostics
    diagnostics = checkpoint_diagnostics(model, directory / "checkpoint_checks", load_data("validation", device))
    write_json(directory / "checkpoint_checks.json", diagnostics)


def assert_run_context(conf, cp):
    """A persisted final checkpoint may only be evaluated under its actual context."""
    from experiment.scope import validate_config
    validate_config(conf["dataset_id"], conf["model_id"], conf["seed"])
    if conf.get("kind") != "formal" or cp["step"] != conf["steps"]:
        raise ValueError("Final evaluation requires the complete formal budget")
    if cp["model_sha256"] != conf["source_sha256"]:
        raise ValueError("Checkpoint/model provenance differs from configuration")
    if json.dumps(cp["model_config"], sort_keys=True) != json.dumps(conf["model_config"], sort_keys=True):
        raise ValueError("Checkpoint model configuration mismatch")
    for name, digest in conf["execution_source_hashes"].items():
        source = ROOT / name if name == "theory_fm_models.py" else ROOT / "experiment" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Current execution source differs from this run: {name}")
    for key, name in [("data_spec_sha256", "DATA_SPEC.json"),
                      ("locked_protocol_sha256", "LOCKED_PROTOCOL.json")]:
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != conf[key]:
            raise ValueError(f"Frozen run context changed: {name}")
    protocol = read_json(ROOT / "LOCKED_PROTOCOL.json")[conf["dataset_id"]]
    for key in ["N", "H", "Q", "E"]:
        if conf[key] != protocol[key]:
            raise ValueError(f"Configuration differs from locked protocol: {key}")
    if conf["steps"] != protocol["updates"]:
        raise ValueError("Final checkpoint budget differs from locked protocol")


def lock_protocol():
    pilots = []
    for model_id in MODELS:
        dirs = sorted((ROOT / "outputs" / "pilots" / f"SEQ_IID__{model_id}__seed17").glob("attempt_*"))
        if not dirs or not (dirs[-1] / "pilot_result.json").exists():
            raise RuntimeError(f"Independent complete pilot missing: {model_id}")
        pilots.append({"model_id": model_id, **read_json(dirs[-1] / "pilot_result.json")})
    chosen = {"updates": 12000, "selection_rationale": "Lowest authorized formal tier; five validation-only pilots establish finite train/sampling operation and measured GPU cost. Pilot does not establish convergence; full curves retained.",
              "N": 32, "H": 17, "Q": 64, "E": 1, "hidden": 128, "hidden_layers": 3,
              "query_chunk": 64, "source_chunk": 32, "network_lr": 2e-4, "order_lr": 2e-5,
              "weight_decay": 1e-4, "grad_clip": 1, "sigma_min": .05, "kappa": .05,
              "tau_min": None, "time_distribution": ".5 Uniform + .5 U^4; inverse-density weighting",
              "sampling_ensembles": 16, "sampling_particles": 64, "sampling_steps": 64,
              "checkpoint": "final", "best_validation_checkpoint_retained": True,
              "solver": "first cell_average_PC; then PL node explicit Euler / Euler-Maruyama",
              "standard_reference": "same checkpoint and initial samples; Heun primary statistical baseline; Euler retained as supplement",
              "same_checkpoint_checks": {"steps": [32, 64, 128], "particles": [32, 64, 128],
                                         "ensembles": 2, "extra_training_updates": 0}, "pilots": pilots}
    write_json(ROOT / "LOCKED_PROTOCOL.json", {"SEQ_IID": chosen, "ROBOT_8DOF": {"status": "BLOCKED", "updates": None},
               "model_order": MODELS, "seed_order": SEEDS, "scope": "exactly 50 manifest entries; no other tasks"})


def run_queue():
    from experiment.scope import update_run, validate_config
    protocol = read_json(ROOT / "LOCKED_PROTOCOL.json")
    for seed in SEEDS:
        for model_id in MODELS:
            validate_config("SEQ_IID", model_id, seed)
            run_id = f"SEQ_IID__{model_id}__seed{seed}"
            dirs = sorted((OUT / run_id).glob("attempt_*"))
            if dirs and complete_artifacts(dirs[-1]):
                update_run(run_id, "COMPLETED", attempt_dir=str(dirs[-1]), output_dir=str(dirs[-1]))
                continue
            try:
                if dirs and (dirs[-1] / "checkpoint_final.pt").exists():
                    evaluate_one(dirs[-1])
                    update_run(run_id, "COMPLETED", attempt_dir=str(dirs[-1]), output_dir=str(dirs[-1]))
                else:
                    train_one(model_id, seed, protocol["SEQ_IID"]["updates"], resume=bool(dirs))
            except Exception as error:
                failure = {"error": repr(error), "traceback": traceback.format_exc()}
                dirs = sorted((OUT / run_id).glob("attempt_*"))
                if dirs:
                    write_json(dirs[-1] / "failure.json", failure)
                update_run(run_id, "FAILED", reason=str(error))
                print(json.dumps({"failed": run_id, **failure}), flush=True)


def complete_artifacts(directory):
    required = ["config.json", "checkpoint_final.pt", "checkpoint_best.pt", "training.csv",
                "samples.npy", "evaluation.json", "sampling_diagnostics.json", "checkpoint_checks.json"]
    if not all((directory / f).exists() for f in required):
        return False
    conf = read_json(directory / "config.json")
    if conf["model_id"] == "FM_STANDARD" and not all((directory / f).exists()
                                                    for f in ["samples_heun.npy", "evaluation_heun.json"]):
        return False
    from experiment.scope import validate_config
    validate_config(conf["dataset_id"], conf["model_id"], conf["seed"])
    cp = torch.load(directory / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    assert_run_context(conf, cp)
    evaluation = read_json(directory / "evaluation.json")
    numeric = [v for v in evaluation.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    with (directory / "training.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return (cp["step"] == conf["steps"] == read_json(ROOT / "LOCKED_PROTOCOL.json")["SEQ_IID"]["updates"]
            and cp["model_sha256"] == conf["source_sha256"]
            and len(rows) == conf["steps"] and int(rows[-1]["step"]) == conf["steps"]
            and len(numeric) > 0 and np.isfinite(numeric).all()
            and np.isfinite(np.load(directory / "samples.npy")).all())


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("command", choices=["prepare", "pilot", "lock", "run", "resume", "evaluate", "verify"])
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--seed", type=int, choices=SEEDS, default=17)
    p.add_argument("--pilot-steps", type=int, default=200)
    p.add_argument("--directory")
    args = p.parse_args()
    os.chdir(ROOT)
    if args.command == "prepare":
        from experiment.data import prepare_data
        print(json.dumps(prepare_data(), indent=2))
    elif args.command == "pilot":
        if not 1 <= args.pilot_steps <= 2000:
            p.error("pilot steps must be between 1 and 2000")
        for model in [args.model] if args.model else MODELS:
            train_one(model, args.seed, args.pilot_steps, pilot=True)
    elif args.command == "lock":
        lock_protocol()
    elif args.command in ("run", "resume"):
        run_queue()
    elif args.command == "evaluate":
        evaluate_one(args.directory)
    elif args.command == "verify":
        print(json.dumps({str(d): complete_artifacts(d) for d in OUT.glob("*/attempt_*")}, indent=2))


if __name__ == "__main__":
    main()
