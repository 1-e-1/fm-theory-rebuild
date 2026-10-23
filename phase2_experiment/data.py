"""Frozen C=2 synthetic tasks for the separately authorized Phase 2.

CODEX_RUNBOOK §7 supplies the mechanism, DELAY C=2 and position delay 16.
The exact distributions/matrices/noise below are reviewed implementation choices,
not equations attributed to the theoretical paper. No Phase 1 file is modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

DATASETS = ("SEQ_DELAY", "SEQ_ACCUM", "SEQ_MULTISCALE")
LENGTH, CHANNELS, STATE_DIM = 64, 2, 128
SPLIT_SIZES = {"train": 10000, "validation": 2000, "test": 5000}
DATA_SEEDS = {"SEQ_DELAY": 20261021, "SEQ_ACCUM": 20261022, "SEQ_MULTISCALE": 20261023}
METRIC_SEEDS = {"SEQ_DELAY": 20261121, "SEQ_ACCUM": 20261122, "SEQ_MULTISCALE": 20261123}
DELAY_TRANSFORM = np.asarray([[1.0, .25], [-.2, .9]], dtype=np.float64)
ACCUM_TRANSFORM = np.asarray([[1.0, .5], [-.4, .9]], dtype=np.float64)
RIDGE_GRID = np.asarray([1e-5, 1e-4, 1e-3, 1e-2, .1, 1., 10., 100.])


def workspace(root: Path | str | None = None) -> Path:
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]


def validate_dataset(dataset_id: str) -> None:
    if dataset_id not in DATASETS:
        raise ValueError(f"Dataset outside the three-task Phase 2: {dataset_id!r}")


def data_dir(dataset_id: str, root: Path | str | None = None) -> Path:
    validate_dataset(dataset_id)
    return workspace(root) / "data" / "phase2_3datasets" / dataset_id.lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def flatten(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states)
    if states.shape[-2:] != (LENGTH, CHANNELS):
        raise ValueError("Expected [...,64,2]")
    return states.reshape(*states.shape[:-2], STATE_DIM, order="C")


def unflatten(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states)
    if states.shape[-1] != STATE_DIM:
        raise ValueError("Expected [...,128]")
    return states.reshape(*states.shape[:-1], LENGTH, CHANNELS, order="C")


def task_parameters(dataset_id: str) -> dict[str, Any]:
    validate_dataset(dataset_id)
    if dataset_id == "SEQ_DELAY":
        return {"early_positions": list(range(8)), "late_positions": list(range(16, 24)),
                "position_delay": 16, "blank_gap_length": 8, "transform": DELAY_TRANSFORM.tolist(),
                "early_mixture_weights": [.25, .5, .25], "early_mixture_means": [-1.5, 0., 1.5],
                "early_mixture_standard_deviations": [.35, .55, .35], "late_noise_sd": .15,
                "other_position_noise_sd": .6,
                "formula": "Y[:,16+k,:]=Y[:,k,:]@B.T+0.15*eta[:,k,:], k=0..7; delay is index translation 16, not gap 16"}
    if dataset_id == "SEQ_ACCUM":
        return {"early_positions": list(range(16)), "late_positions": list(range(48, 64)),
                "transform": ACCUM_TRANSFORM.tolist(), "early_noise_sd": 1., "response_amplitude": 1.5,
                "late_noise_sd": .15, "other_position_noise_sd": .6,
                "formula": "z=sum(Y[:,0:16,:],axis=1)/sqrt(16); signal=1.5*tanh(z@A.T); Y[:,48:64,:]=signal[:,None,:]+0.15*eta"}
    angle = 2 * np.pi / 12
    transition = np.zeros((4, 4))
    transition[0, 0], transition[1, 1] = np.exp(-1 / 2), np.exp(-1 / 24)
    transition[2:, 2:] = .95 * np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return {"state_dim": 4, "observed_channels": 2, "transition": transition.tolist(),
            "observation": [[1., .8, .7, 0.], [.5, -.8, 0., .7]],
            "process_noise_standard_deviations": [.18, .06, .12, .12], "observation_noise_sd": .10,
            "fast_decay_scale_observations": 2, "slow_decay_scale_observations": 24,
            "oscillation_period_observations": 12, "oscillation_damping_per_observation": .95,
            "burn_in": 512, "initial_state": [0., 0., 0., 0.], "observation_interval": 1.,
            "downstream_input_positions": list(range(32)), "downstream_output_positions": list(range(32, 64)),
            "formula": "z_(s+1)=T z_s+diag(process_sd)*eta_s; Y_s=C z_s+0.10*epsilon_s; independent realization per sequence, not windows of one long record"}


def generate_raw(dataset_id: str, count: int, rng: np.random.Generator) -> np.ndarray:
    """One independent realization per row; exposed for mechanism tests."""
    p = task_parameters(dataset_id)
    if dataset_id in ("SEQ_DELAY", "SEQ_ACCUM"):
        raw = .6 * rng.standard_normal((count, LENGTH, CHANNELS))
        if dataset_id == "SEQ_DELAY":
            component = rng.choice(3, size=(count, 8, 2), p=[.25, .5, .25])
            mu, sd = np.asarray([-1.5, 0., 1.5]), np.asarray([.35, .55, .35])
            early = mu[component] + sd[component] * rng.standard_normal(component.shape)
            raw[:, :8] = early
            raw[:, 16:24] = early @ DELAY_TRANSFORM.T + .15 * rng.standard_normal(early.shape)
        else:
            early = rng.standard_normal((count, 16, 2))
            signal = 1.5 * np.tanh((early.sum(axis=1) / 4) @ ACCUM_TRANSFORM.T)
            raw[:, :16] = early
            raw[:, 48:] = signal[:, None, :] + .15 * rng.standard_normal((count, 16, 2))
        return raw.astype(np.float32)
    transition, observation = np.asarray(p["transition"]), np.asarray(p["observation"])
    process_sd = np.asarray(p["process_noise_standard_deviations"])
    state = np.zeros((count, 4), dtype=np.float64)
    output = np.empty((count, LENGTH, CHANNELS), dtype=np.float32)
    for index in range(p["burn_in"] + LENGTH):
        state = state @ transition.T + process_sd * rng.standard_normal(state.shape)
        if index >= p["burn_in"]:
            output[:, index - p["burn_in"]] = state @ observation.T + .10 * rng.standard_normal((count, 2))
    return output


def prepare_data(dataset_id: str, root: Path | str | None = None) -> dict[str, Any]:
    """Create once; reuse verifies every frozen file. Metric fitting excludes test."""
    directory = data_dir(dataset_id, root)
    spec_path = directory / "DATA_SPEC.json"
    p = task_parameters(dataset_id)
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        expected = {"dataset_id": dataset_id, "data_seed": DATA_SEEDS[dataset_id], "length": LENGTH,
                    "channels": CHANNELS, "state_dim": STATE_DIM, "parameters": p, "split_sizes": SPLIT_SIZES}
        for key, value in expected.items():
            if spec.get(key) != value:
                raise RuntimeError(f"Frozen data definition differs at {key}; refusing regeneration")
        for file in spec["files"]:
            path = directory / file["filename"]
            if not path.is_file() or sha256_file(path) != file["sha256"]:
                raise RuntimeError(f"Missing/changed frozen data file: {path}")
        return spec
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(DATA_SEEDS[dataset_id])
    raw = {name: generate_raw(dataset_id, count, rng) for name, count in SPLIT_SIZES.items()}
    mean, scale = flatten(raw["train"]).astype(np.float64).mean(axis=0), flatten(raw["train"]).astype(np.float64).std(axis=0)
    if not np.all(scale > 0) or not np.isfinite(scale).all():
        raise RuntimeError("Invalid train standardization")
    files, hashes = [], {}
    for name, values in raw.items():
        standardized = ((flatten(values).astype(np.float64) - mean) / scale).astype(np.float32)
        path = directory / f"{name}.npz"
        np.savez(path, raw=values, standardized=standardized)
        files.append(path)
        hashes[name] = {"raw": hashlib.sha256(values.tobytes()).hexdigest(), "standardized": hashlib.sha256(standardized.tobytes()).hexdigest()}
    np.savez(directory / "stats.npz", mean=mean, scale=scale)
    files.append(directory / "stats.npz")
    mrng = np.random.default_rng(METRIC_SEEDS[dataset_id])
    directions = mrng.standard_normal((256, STATE_DIM)); directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    train_indices, val_indices = mrng.choice(10000, 512, replace=False), mrng.choice(2000, 512, replace=False)
    train_standard = (flatten(raw["train"]).astype(np.float64) - mean) / scale
    val_standard = (flatten(raw["validation"]).astype(np.float64) - mean) / scale
    selected = np.concatenate([train_standard[train_indices], val_standard[val_indices]])
    norm = np.sum(selected * selected, axis=1)
    d2 = np.maximum(norm[:, None] + norm[None, :] - 2 * selected @ selected.T, 0)
    median = float(np.sqrt(np.median(d2[np.triu_indices(len(selected), k=1)])))
    early = p.get("early_positions", p.get("downstream_input_positions"))
    late = p.get("late_positions", p.get("downstream_output_positions"))
    early_indices = np.asarray([s * CHANNELS + c for s in early for c in range(CHANNELS)])
    late_indices = np.asarray([s * CHANNELS + c for s in late for c in range(CHANNELS)])
    joint_dim = len(early_indices) + len(late_indices)
    joint_directions = mrng.standard_normal((256, joint_dim)); joint_directions /= np.linalg.norm(joint_directions, axis=1, keepdims=True)
    np.savez(directory / "metric_parameters.npz", directions=directions, bandwidths=median * np.asarray([.5, 1., 2.]),
             bandwidth_train_indices=train_indices, bandwidth_validation_indices=val_indices,
             early_indices=early_indices, late_indices=late_indices, joint_directions=joint_directions,
             ridge_grid=RIDGE_GRID, matched_trtr_train_index_order=mrng.permutation(10000))
    files.append(directory / "metric_parameters.npz")
    metric_spec = {"dataset_id": dataset_id, "metric_seed": METRIC_SEEDS[dataset_id],
        "fit_data": "train and validation only; formal test is not read to determine projections, bandwidths, task parameters, ridge grid or feature indices",
        "swd": "sqrt(mean of 256 exact empirical projected W2-squared integrals), unit directions in complete train-standardized D=128 space",
        "mmd2": "mean of 3 RBF exp(-distance_squared/(2*bandwidth_squared)); generated-generated term uses distinct independent ensembles only; reference off-diagonal term; negative estimates retained",
        "bandwidth_selection": "median Euclidean distance in fixed 512 train + 512 validation realizations",
        "bandwidths": (median * np.asarray([.5, 1., 2.])).tolist(),
        "channel_alignment": "C-order reshape N,128 -> N,64,2; ACF and PSD use observation axis, never flattened adjacent coordinates",
        "acf": "Pearson position correlations across realizations, average within channel over time pairs, lags1..32; cross-channel lag correlation also retained",
        "psd": "one-sided rFFT periodogram of per-sequence time-demeaned raw channels, abs(FFT)^2/64; double interior bins; observation interval1; mean over sequences",
        "psd_log_epsilon": 1e-10, "time_difference": "raw first observation-time differences within each channel; empirical marginal W1",
        "early_late_joint": "fixed 256 joint-space projection directions; original pairing versus late re-pairing with unchanged early/late empirical marginals; DELAY and ACCUM only",
        "generated_shuffle": "rotate late ensembles by one, permute particles within source ensemble using fixed metric seed; never pair with the same ensemble",
        "true_shuffle": "fixed RNG derangement of test late rows; preserves marginal arrays exactly",
        "downstream_task": {"input_positions": early, "output_positions": late, "kind": "supervised within-realization ridge prediction, evaluation only; does not change complete-sequence generator task"},
        "ridge": {"lambda_grid": RIDGE_GRID.tolist(), "objective": "mean training squared residual + lambda*||W||^2, unpenalized fitted intercept; same shared real-train standardization",
                  "selection": "minimum real-validation mean squared error; ties choose first/smallest lambda; no test-based selection",
                  "TSTR": "fit generated complete-realization pairs, select lambda only on real validation, score real test",
                  "TRTR": "matched generated sequence count from one fixed train index order, plus full10000 real-train baseline; both select lambda on same real validation and score real test"},
        "sigma_min": .05, "smooth_reference": "shared train-standardized test + .05 independent N(0,I); fixed metric RNG seed",
        "shared_pure_metric_source": "experiment.metrics.empirical_wasserstein/cross_ensemble_mmd2; no Phase1 data or checkpoint reused",
        "shared_metric_source_sha256": sha256_file(Path(__file__).resolve().parents[1] / "experiment" / "metrics.py"),
        "statistics_unit": "training seed; ensemble particles, coordinate values and projection directions are not independent training repeats"}
    write_json(directory / "METRIC_SPEC.json", metric_spec); files.append(directory / "METRIC_SPEC.json")
    spec = {"schema_version": 1, "dataset_id": dataset_id, "data_seed": DATA_SEEDS[dataset_id], "metric_seed": METRIC_SEEDS[dataset_id],
            "length": LENGTH, "channels": CHANNELS, "state_dim": STATE_DIM, "split_sizes": SPLIT_SIZES,
            "source": "new synthetic independent realizations generated for Phase2, no old results/data relabelled", "parameters": p,
            "task_book_basis": "CODEX_RUNBOOK.md §7.A mechanisms and DELAY L64/C2/delay16; §9 required sequence and early-late evaluations",
            "implementation_choices": "all numeric mixture/matrix/noise/SSM parameters above, ACCUM/MULTISCALE C2, independent data seeds, metric projection and ridge rules are explicitly registered choices; not equations from the theory paper",
            "flatten": "reversible C-order [...,64,2] to [...,128]; no compression; each complete sequence is one generation particle",
            "normalization": "per-coordinate mean/population SD fitted on train only, shared by all models and training seeds",
            "dtype": "raw/standardized float32; normalization float64", "numpy_version": np.__version__,
            "split_policy": "distinct independent realizations; shared train10000/validation2000/test5000, not overlapping windows",
            "probability_path": {"sigma_min": .05, "ODE_target": "Y-.95*epsilon", "SDE_complete_target": "Y-.95*epsilon-kappa*epsilon/(1-.95*tau)"},
            "array_sha256": hashes,
            "files": [{"filename": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files], "directory": str(directory)}
    write_json(spec_path, spec)
    return spec


def load_split(dataset_id: str, name: str, device: str | torch.device = "cpu", root: Path | str | None = None) -> torch.Tensor:
    name = "validation" if name == "val" else name
    if name not in SPLIT_SIZES:
        raise ValueError("Unknown split")
    with np.load(data_dir(dataset_id, root) / f"{name}.npz", allow_pickle=False) as archive:
        values = archive["standardized"].copy()
    return torch.from_numpy(values).to(device=device, dtype=torch.float32)


def load_stats(dataset_id: str, root: Path | str | None = None) -> dict[str, np.ndarray]:
    with np.load(data_dir(dataset_id, root) / "stats.npz", allow_pickle=False) as archive:
        return {"mean": archive["mean"].copy(), "scale": archive["scale"].copy()}


def unstandardize(samples: np.ndarray, dataset_id: str, root: Path | str | None = None) -> np.ndarray:
    stats = load_stats(dataset_id, root)
    return np.asarray(samples, dtype=np.float64) * stats["scale"] + stats["mean"]
