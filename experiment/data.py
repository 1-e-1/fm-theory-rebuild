"""Frozen SEQ_IID data shared by all phase-1 models and training seeds.

The task book specifies independent observation positions and a non-Gaussian
mixture marginal, but not the channel count or mixture parameters. Those choices
are explicitly recorded below and in DATA_SPEC.json; they are not paper claims.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DATA_SEED = 20261009
METRIC_SEED = 20261010
LENGTH = 64
CHANNELS = 1
STATE_DIM = LENGTH * CHANNELS
SPLIT_SIZES = {"train": 10000, "validation": 2000, "test": 5000}
MIXTURE_WEIGHTS = (0.25, 0.50, 0.25)
MIXTURE_MEANS = (-2.0, 0.0, 2.0)
MIXTURE_SCALES = (0.35, 0.60, 0.35)


def workspace(root: Path | str | None = None) -> Path:
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]


def data_dir(root: Path | str | None = None) -> Path:
    return workspace(root) / "data" / "seq_iid"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def flatten(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.shape[-2:] != (LENGTH, CHANNELS):
        raise ValueError(f"Expected trailing shape {(LENGTH, CHANNELS)}, got {samples.shape}")
    return samples.reshape(*samples.shape[:-2], STATE_DIM, order="C")


def unflatten(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected state dimension {STATE_DIM}, got {samples.shape}")
    return samples.reshape(*samples.shape[:-1], LENGTH, CHANNELS, order="C")


def prepare_data(root: Path | str | None = None) -> dict[str, Any]:
    """Create the one shared synthetic dataset, or validate an existing copy.

    Metric directions/bandwidths use train and validation only. Preparing the
    dataset creates test realizations but does not evaluate them or use them to
    fit normalization, select bandwidths, or choose a training protocol.
    """
    directory = data_dir(root)
    spec_path = directory / "DATA_SPEC.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        expected = {
            "dataset_id": "SEQ_IID", "data_seed": DATA_SEED, "length": LENGTH,
            "channels": CHANNELS, "state_dim": STATE_DIM, "split_sizes": SPLIT_SIZES,
            "mixture_weights": list(MIXTURE_WEIGHTS), "mixture_means": list(MIXTURE_MEANS),
            "mixture_standard_deviations": list(MIXTURE_SCALES),
        }
        for key, value in expected.items():
            if spec.get(key) != value:
                raise RuntimeError(f"Existing data specification conflicts at {key}; refusing replacement")
        for record in spec["files"]:
            path = directory / record["filename"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Missing or changed frozen data file: {path}")
        return spec

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(DATA_SEED)
    raw_splits: dict[str, np.ndarray] = {}
    means = np.asarray(MIXTURE_MEANS, dtype=np.float64)
    scales = np.asarray(MIXTURE_SCALES, dtype=np.float64)
    for name, count in SPLIT_SIZES.items():
        shape = (count, LENGTH, CHANNELS)
        component = rng.choice(3, size=shape, p=MIXTURE_WEIGHTS)
        raw_splits[name] = (means[component] + scales[component] * rng.standard_normal(shape)).astype(np.float32)
    training = flatten(raw_splits["train"]).astype(np.float64)
    mean = training.mean(axis=0)
    scale = training.std(axis=0, ddof=0)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("Invalid training-only standardization scale")
    files = []
    arrays = {}
    for name, raw in raw_splits.items():
        standard = ((flatten(raw).astype(np.float64) - mean) / scale).astype(np.float32)
        path = directory / f"{name}.npz"
        np.savez(path, raw=raw, standardized=standard)
        files.append({"filename": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
        arrays[name] = {"raw_sha256": array_sha256(raw), "standardized_sha256": array_sha256(standard)}
    stats_path = directory / "stats.npz"
    np.savez(stats_path, mean=mean, scale=scale)
    files.append({"filename": stats_path.name, "sha256": sha256_file(stats_path), "bytes": stats_path.stat().st_size})

    # These choices are made before formal test evaluation and never optimized.
    metric_rng = np.random.default_rng(METRIC_SEED)
    directions = metric_rng.standard_normal((256, STATE_DIM))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    train_indices = metric_rng.choice(SPLIT_SIZES["train"], 512, replace=False)
    validation_indices = metric_rng.choice(SPLIT_SIZES["validation"], 512, replace=False)
    bandwidth_sample = np.concatenate([
        ((flatten(raw_splits["train"])[train_indices].astype(np.float64) - mean) / scale),
        ((flatten(raw_splits["validation"])[validation_indices].astype(np.float64) - mean) / scale),
    ])
    norms = np.einsum("ij,ij->i", bandwidth_sample, bandwidth_sample)
    distance_squared = np.maximum(norms[:, None] + norms[None, :] - 2 * bandwidth_sample @ bandwidth_sample.T, 0)
    median_distance = float(np.sqrt(np.median(distance_squared[np.triu_indices(len(bandwidth_sample), k=1)])))
    bandwidths = median_distance * np.asarray([0.5, 1.0, 2.0])
    metric_path = directory / "metric_parameters.npz"
    np.savez(metric_path, directions=directions, bandwidths=bandwidths,
             bandwidth_train_indices=train_indices, bandwidth_validation_indices=validation_indices)
    files.append({"filename": metric_path.name, "sha256": sha256_file(metric_path), "bytes": metric_path.stat().st_size})
    metric_spec = {
        "metric_seed": METRIC_SEED, "swd_projection_count": 256,
        "swd": "sqrt(mean_direction integral_0^1 (empirical_quantile_generated - empirical_quantile_reference)^2 dp)",
        "swd_quantile_integration": "exact piecewise-constant empirical quantiles, including unequal sample sizes",
        "projection_space": "shared train-standardized complete D=64 vector", "projection_norm": 1,
        "bandwidth_selection": "median Euclidean pair distance of fixed 512 train + 512 validation samples",
        "median_distance": median_distance, "bandwidth_factors": [0.5, 1.0, 2.0],
        "rbf_bandwidths": bandwidths.tolist(), "rbf": "exp(-||x-y||^2/(2*bandwidth^2)), averaged over three bandwidths",
        "mmd2_generated_generated": "average kernel over pairs from distinct independently initialized ensembles only",
        "mmd2_reference_reference": "all off-diagonal pairs of iid test reference",
        "mmd2_allow_negative": True,
        "test_use": "formal final evaluation only; all 5000 test realizations are retained as reference",
        "smooth_reference": "train-standardized test + 0.05*N(0,I), fixed independent metric RNG",
        "within_ensemble_caveat": "particles may be dependent; ensemble rows are diagnostics and are not independent training seeds",
        "real_real_reference": "two disjoint test blocks each reshaped [E,N,D]; finite sample reference, not an oracle lower bound",
        "acf": "mean correlations across observation-position pairs at lag 1..32; per-position centering across realizations",
        "marginal_amplitude_space": "original raw data units; inverse shared train standardization",
    }
    metric_spec_path = directory / "METRIC_SPEC.json"
    write_json(metric_spec_path, metric_spec)
    files.append({"filename": metric_spec_path.name, "sha256": sha256_file(metric_spec_path), "bytes": metric_spec_path.stat().st_size})
    spec = {
        "version": 1, "dataset_id": "SEQ_IID", "source": "new synthetic data generated in this project; no old experiment reused",
        "data_seed": DATA_SEED, "rng": "numpy.default_rng / PCG64", "numpy_version": np.__version__, "length": LENGTH,
        "channels": CHANNELS, "state_dim": STATE_DIM, "split_sizes": SPLIT_SIZES,
        "formula": "Y[n,s,0] = mu[Z[n,s,0]] + sd[Z[n,s,0]] * eta[n,s,0]; Z iid categorical(w), eta iid N(0,1)",
        "mixture_weights": list(MIXTURE_WEIGHTS), "mixture_means": list(MIXTURE_MEANS),
        "mixture_standard_deviations": list(MIXTURE_SCALES),
        "independence": "all realization indices, observation positions and component/noise draws independent; no latent shared sequence component",
        "implementation_choices_not_fixed_by_task_book": ["C=1", "three Gaussian mixture with the parameters above", "data_seed=20261009", "per-coordinate population-SD train normalization"],
        "flatten": "C-order reversible reshape [...,L=64,C=1] <-> [...,D=64], no projection or learned compression",
        "standardization": "per flattened coordinate mean and population SD fit using train only, shared by all models/seeds",
        "raw_dtype": "float32", "standardized_dtype": "float32", "stats_dtype": "float64",
        "probability_path": {"X_tau": "tau*Y + (1-0.95*tau)*epsilon", "sigma_min": 0.05,
                             "ODE_label": "Y-0.95*epsilon", "SDE_label": "Y-0.95*epsilon-kappa*epsilon/sigma(tau)"},
        "split_policy": "distinct independent realizations generated once, shared across all five models and five training seeds",
        "pilot_access_policy": "load_split opens only the named train/validation NPZ; formal test evaluation must call evaluate_run explicitly",
        "array_hashes": arrays, "files": files, "directory": str(directory),
    }
    write_json(spec_path, spec)
    write_json(workspace(root) / "DATA_SPEC.json", spec)
    return spec


def load_split(name: str, device: str | torch.device = "cpu", root: Path | str | None = None) -> torch.Tensor:
    """Read only the requested split and return complete standardized states."""
    if name == "val":
        name = "validation"
    if name not in SPLIT_SIZES:
        raise ValueError(f"Unknown split: {name!r}")
    path = data_dir(root) / f"{name}.npz"
    with np.load(path, allow_pickle=False) as archive:
        states = archive["standardized"]
    return torch.from_numpy(states.copy()).to(device=device, dtype=torch.float32)


def load_stats(root: Path | str | None = None) -> dict[str, np.ndarray]:
    with np.load(data_dir(root) / "stats.npz", allow_pickle=False) as archive:
        return {"mean": archive["mean"].copy(), "scale": archive["scale"].copy()}


def unstandardize(samples: np.ndarray, root: Path | str | None = None) -> np.ndarray:
    stats = load_stats(root)
    return np.asarray(samples, dtype=np.float64) * stats["scale"] + stats["mean"]
