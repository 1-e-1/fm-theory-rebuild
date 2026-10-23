"""Channel-aligned formal metrics for three separately authorized C2 tasks.

Test data is opened only by evaluate_run. The audited Phase1 Wasserstein and
cross-ensemble MMD pure mathematics is reused without its dataset loaders.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiment.metrics import empirical_wasserstein, cross_ensemble_mmd2
from phase2_experiment.data import (LENGTH, CHANNELS, STATE_DIM, METRIC_SEEDS,
    DELAY_TRANSFORM, ACCUM_TRANSFORM, data_dir, load_stats, task_parameters,
    sha256_file, write_json, unstandardize)


def channel_correlation_acf(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Realization-centered correlations with explicit observation/channel axes."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (LENGTH, CHANNELS) or len(raw) < 2:
        raise ValueError("Expected [N>=2,64,2] raw complete sequences")
    states = raw.reshape(len(raw), STATE_DIM)
    centered = states - states.mean(axis=0)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    denom = norms[:, None] * norms[None, :]
    corr = np.divide(centered.T @ centered, denom, out=np.zeros((STATE_DIM, STATE_DIM)), where=denom > 0)
    lag_corr = np.empty((CHANNELS, CHANNELS, 33))
    for first in range(CHANNELS):
        for second in range(CHANNELS):
            for lag in range(33):
                t = np.arange(LENGTH - lag)
                lag_corr[first, second, lag] = corr[t * CHANNELS + first, (t + lag) * CHANNELS + second].mean()
    acf = np.stack([lag_corr[c, c, 1:] for c in range(CHANNELS)])
    return corr, acf, lag_corr


def power_spectrum(raw: np.ndarray) -> np.ndarray:
    centered = raw - raw.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2 / LENGTH
    power[:, 1:-1] *= 2
    return power.mean(axis=0)


def ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    """Objective mean squared residual + penalty*||W||²; free intercept."""
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc / len(x) + penalty * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, xc.T @ yc / len(x))
    return weights, ym - xm @ weights


def select_ridge(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray,
                 grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    scores, fitted = [], []
    for penalty in grid:
        fit = ridge_fit(x, y, float(penalty))
        fitted.append(fit)
        scores.append(float(np.mean((xv @ fit[0] + fit[1] - yv) ** 2)))
    selected = int(np.argmin(scores))
    return *fitted[selected], float(grid[selected]), np.asarray(scores)


def structural_residual(raw: np.ndarray, dataset_id: str) -> np.ndarray:
    if dataset_id == "SEQ_DELAY":
        return raw[:, 16:24] - raw[:, :8] @ DELAY_TRANSFORM.T
    if dataset_id == "SEQ_ACCUM":
        signal = 1.5 * np.tanh((raw[:, :16].sum(axis=1) / 4) @ ACCUM_TRANSFORM.T)
        return raw[:, 48:] - signal[:, None, :]
    raise ValueError("No deterministic structural residual is specified for the SSM")


def shuffle_generated_late(samples: np.ndarray, dataset_id: str, seed: int) -> np.ndarray:
    """Late rows come from an independent ensemble; empirical marginals unchanged."""
    result = np.asarray(samples).copy()
    positions = task_parameters(dataset_id)["late_positions"]
    rng = np.random.default_rng(seed)
    for ensemble in range(len(samples)):
        source = samples[(ensemble + 1) % len(samples)]
        permutation = rng.permutation(len(source))
        result[ensemble][:, positions, :] = source[permutation][:, positions, :]
    return result


def _derangement(count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("Pair destruction needs at least two rows")
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(count)
        if np.all(order != np.arange(count)):
            return order


def _swd(a: np.ndarray, b: np.ndarray, directions: np.ndarray) -> tuple[float, np.ndarray]:
    squared = empirical_wasserstein(a @ directions.T, b @ directions.T, power=2)
    return float(np.sqrt(squared.mean())), squared


def _kernel_sum(a: np.ndarray, b: np.ndarray, bandwidths: np.ndarray) -> float:
    normb = np.sum(b * b, axis=1)
    total = 0.
    for start in range(0, len(a), 256):
        block = a[start:start + 256]
        d2 = np.maximum(np.sum(block * block, axis=1)[:, None] + normb[None, :] - 2 * block @ b.T, 0)
        total += sum(float(np.exp(-d2 / (2 * bw * bw)).sum()) for bw in bandwidths) / len(bandwidths)
    return total


def _references(dataset_id: str, root) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    directory = data_dir(dataset_id, root)
    with np.load(directory / "test.npz", allow_pickle=False) as archive:
        test = archive["standardized"].astype(np.float64)
        raw = archive["raw"].astype(np.float64)
    source_hash = sha256_file(directory / "test.npz")
    cache = directory / "formal_references.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            if str(archive["source_sha256"].item()) != source_hash:
                raise RuntimeError("Changed formal reference source")
            smooth = archive["smooth_standardized"].copy()
    else:
        noise = np.random.default_rng(METRIC_SEEDS[dataset_id] + 1).standard_normal(test.shape)
        smooth = test + .05 * noise
        np.savez(cache, smooth_standardized=smooth, smoothing_noise=noise, source_sha256=np.asarray(source_hash))
    provenance = {"dataset_id": dataset_id, "reference_source": str(directory / "test.npz"),
                  "reference_source_sha256": source_hash, "smooth_reference": str(cache),
                  "smooth_reference_sha256": sha256_file(cache), "smoothing_seed": METRIC_SEEDS[dataset_id] + 1,
                  "metric_parameters_sha256": sha256_file(directory / "metric_parameters.npz")}
    return test, smooth, raw, provenance


def _reference_kernel_means(test, smooth, bandwidths, dataset_id, root, provenance):
    path = data_dir(dataset_id, root) / "formal_reference_kernel_means.json"
    identity = {k: provenance[k] for k in ("reference_source_sha256", "smooth_reference_sha256", "metric_parameters_sha256")}
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8"))
        if cache["identity"] != identity:
            raise RuntimeError("Changed reference kernel cache inputs")
        return cache["raw"], cache["smooth"]
    means = [(_kernel_sum(r, r, bandwidths) - len(r)) / (len(r) * (len(r) - 1)) for r in (test, smooth)]
    write_json(path, {"identity": identity, "raw": means[0], "smooth": means[1]})
    return tuple(means)


def _basic(raw: np.ndarray, reference: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    flat, real_flat = raw.reshape(len(raw), STATE_DIM), reference.reshape(len(reference), STATE_DIM)
    corr, acf, cross = channel_correlation_acf(raw)
    real_corr, real_acf, real_cross = channel_correlation_acf(reference)
    psd, real_psd = power_spectrum(raw), power_spectrum(reference)
    w1 = empirical_wasserstein(flat, real_flat, power=1)
    diff = np.diff(raw, axis=1).reshape(-1, CHANNELS)
    real_diff = np.diff(reference, axis=1).reshape(-1, CHANNELS)
    difference_w1 = empirical_wasserstein(diff, real_diff, power=1)
    rms = np.sqrt(np.mean(flat * flat, axis=1)); real_rms = np.sqrt(np.mean(real_flat * real_flat, axis=1))
    peak = np.abs(flat).max(axis=1); real_peak = np.abs(real_flat).max(axis=1)
    offdiag = ~np.eye(STATE_DIM, dtype=bool)
    crosschannel = ~np.eye(CHANNELS, dtype=bool)
    metrics = {"marginal_w1_mean": float(w1.mean()), "marginal_w1_max": float(w1.max()),
        "position_mean_rmse": float(np.sqrt(np.mean((flat.mean(axis=0) - real_flat.mean(axis=0)) ** 2))),
        "position_variance_rmse": float(np.sqrt(np.mean((flat.var(axis=0) - real_flat.var(axis=0)) ** 2))),
        "raw_mean": float(flat.mean()), "raw_variance": float(flat.var()),
        "sequence_rms_w1": float(empirical_wasserstein(rms, real_rms, power=1)[0]),
        "sequence_max_abs_w1": float(empirical_wasserstein(peak, real_peak, power=1)[0]),
        "absolute_amplitude_w1": float(empirical_wasserstein(np.abs(flat).reshape(-1), np.abs(real_flat).reshape(-1), power=1)[0]),
        "acf_error_rms": float(np.sqrt(np.mean((acf - real_acf) ** 2))), "acf_rms": float(np.sqrt(np.mean(acf * acf))),
        "cross_channel_lag_corr_error_rms": float(np.sqrt(np.mean((cross[crosschannel] - real_cross[crosschannel]) ** 2))),
        "cross_position_corr_error_rms": float(np.sqrt(np.mean((corr[offdiag] - real_corr[offdiag]) ** 2))),
        "cross_position_corr_offdiag_rms": float(np.sqrt(np.mean(corr[offdiag] ** 2))),
        "psd_relative_l2_error": float(np.linalg.norm(psd - real_psd) / (np.linalg.norm(real_psd) + 1e-30)),
        "log_psd_rmse": float(np.sqrt(np.mean((np.log10(psd + 1e-10) - np.log10(real_psd + 1e-10)) ** 2))),
        "time_difference_w1": float(difference_w1.mean())}
    arrays = {"position_mean": flat.mean(axis=0), "position_variance": flat.var(axis=0), "marginal_w1": w1,
              "correlation": corr, "acf": acf, "cross_channel_lag_correlation": cross,
              "reference_correlation": real_corr, "reference_acf": real_acf,
              "reference_cross_channel_lag_correlation": real_cross, "psd": psd, "reference_psd": real_psd,
              "psd_frequency": np.fft.rfftfreq(LENGTH), "time_difference_w1_by_channel": difference_w1,
              "sequence_rms": rms, "sequence_max_abs": peak}
    return metrics, arrays


def _downstream(samples: np.ndarray, test: np.ndarray, params: dict, dataset_id: str, root) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, Any]]:
    directory = data_dir(dataset_id, root)
    with np.load(directory / "train.npz", allow_pickle=False) as archive: train = archive["standardized"].astype(np.float64)
    with np.load(directory / "validation.npz", allow_pickle=False) as archive: validation = archive["standardized"].astype(np.float64)
    early, late, grid = params["early_indices"], params["late_indices"], params["ridge_grid"]
    indexes = params["matched_trtr_train_index_order"][:len(samples)]
    if len(indexes) != len(samples): raise ValueError("Generated count exceeds available matched real-train count")
    xval, yval = validation[:, early], validation[:, late]
    xte, yte = test[:, early], test[:, late]
    stats = load_stats(dataset_id, root)
    metrics, arrays, records = {}, {}, {}
    for label, source in (("tstr", samples), ("trtr_matched", train[indexes]), ("trtr_full", train)):
        w, intercept, penalty, scores = select_ridge(source[:, early], source[:, late], xval, yval, grid)
        prediction = xte @ w + intercept
        mse = float(np.mean((prediction - yte) ** 2))
        raw_mse = float(np.mean(((prediction - yte) * stats["scale"][late]) ** 2))
        metrics.update({f"{label}_mse": mse, f"{label}_raw_mse": raw_mse})
        arrays.update({f"{label}_weights": w, f"{label}_intercept": intercept, f"{label}_validation_mse_grid": scores,
                       f"{label}_test_predictions_standardized": prediction})
        records[label] = {"train_count": len(source), "lambda": penalty, "validation_mse": float(scores.min()),
                          "test_mse": mse, "raw_test_mse": raw_mse}
    metrics["tstr_minus_trtr_matched_mse"] = metrics["tstr_mse"] - metrics["trtr_matched_mse"]
    arrays.update({"ridge_grid": grid, "downstream_test_targets_standardized": yte, "matched_trtr_train_indices": indexes})
    return metrics, arrays, records


def pilot_validation_diagnostics(samples_standardized: np.ndarray, dataset_id: str,
                                 root: Path | str | None = None) -> dict[str, float | bool]:
    """Small-sample validation-only diagnostics; no test/reference cache access."""
    samples = np.asarray(samples_standardized, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != STATE_DIM or min(samples.shape[:2]) < 2 or not np.isfinite(samples).all():
        raise ValueError("Expected finite pilot [E>=2,N>=2,128]")
    directory = data_dir(dataset_id, root)
    with np.load(directory / "validation.npz", allow_pickle=False) as archive:
        validation = archive["standardized"].astype(np.float64)
        raw_validation = archive["raw"].astype(np.float64)
    with np.load(directory / "metric_parameters.npz", allow_pickle=False) as archive:
        directions = archive["directions"].copy()
        early, late = archive["early_indices"].copy(), archive["late_indices"].copy()
        joint_directions = archive["joint_directions"].copy()
    raw_samples = unstandardize(samples, dataset_id, root).reshape(*samples.shape[:2], LENGTH, CHANNELS)
    raw = raw_samples.reshape(-1, LENGTH, CHANNELS)
    basic, _ = _basic(raw, raw_validation)
    result: dict[str, float | bool] = {"validation_" + key: value for key, value in basic.items()}
    result["validation_swd_raw"] = _swd(samples.reshape(-1, STATE_DIM), validation, directions)[0]
    smooth = validation + .05 * np.random.default_rng(METRIC_SEEDS[dataset_id] + 4).standard_normal(validation.shape)
    result["validation_swd_smooth"] = _swd(samples.reshape(-1, STATE_DIM), smooth, directions)[0]
    if dataset_id in ("SEQ_DELAY", "SEQ_ACCUM"):
        joint = np.r_[early, late]
        result["validation_early_late_joint_swd_raw"] = _swd(samples.reshape(-1, STATE_DIM)[:, joint], validation[:, joint], joint_directions)[0]
        paired = structural_residual(raw, dataset_id)
        shuffled = structural_residual(shuffle_generated_late(raw_samples, dataset_id, METRIC_SEEDS[dataset_id] + 2).reshape(-1, LENGTH, CHANNELS), dataset_id)
        result["generated_paired_structure_mse_raw"] = float(np.mean(paired ** 2))
        result["generated_shuffled_structure_mse_raw"] = float(np.mean(shuffled ** 2))
        result["generated_structure_pairing_gain_raw"] = float(np.mean(shuffled ** 2) - np.mean(paired ** 2))
        result["validation_real_structure_mse_raw"] = float(np.mean(structural_residual(raw_validation, dataset_id) ** 2))
    result.update({"pilot_sample_count": float(samples.shape[0] * samples.shape[1]),
                   "pilot_validation_count": float(len(validation)), "test_accessed": False,
                   "small_sample_nonformal": True})
    if not all(np.isfinite(v) for v in result.values() if not isinstance(v, bool)):
        raise FloatingPointError("Nonfinite validation-only pilot diagnostics")
    return result


def evaluate_run(samples_standardized: np.ndarray, output_dir: Path | str,
                 dataset_id: str, root: Path | str | None = None) -> dict[str, float]:
    """Explicit final test evaluation. Never call this function from a pilot."""
    samples = np.asarray(samples_standardized, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != STATE_DIM or min(samples.shape[:2]) < 2 or not np.isfinite(samples).all():
        raise ValueError("Expected finite [E>=2,N>=2,D=128]")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    test, smooth, real_raw, provenance = _references(dataset_id, root)
    e, n, _ = samples.shape
    if 2 * e * n > len(test): raise ValueError("Insufficient disjoint real blocks for requested E,N")
    with np.load(data_dir(dataset_id, root) / "metric_parameters.npz", allow_pickle=False) as archive:
        params = {k: archive[k].copy() for k in archive.files}
    directions, bandwidths = params["directions"], params["bandwidths"]
    reference_kernel = _reference_kernel_means(test, smooth, bandwidths, dataset_id, root, provenance)
    flat = samples.reshape(-1, STATE_DIM)
    raw_samples = unstandardize(samples, dataset_id, root).reshape(e, n, LENGTH, CHANNELS)
    raw = raw_samples.reshape(-1, LENGTH, CHANNELS)
    smooth_raw = unstandardize(smooth, dataset_id, root).reshape(-1, LENGTH, CHANNELS)
    metrics, arrays = _basic(raw, real_raw)
    smooth_basic, smooth_arrays = _basic(raw, smooth_raw)
    metrics.update({key + "_smooth_reference": value for key, value in smooth_basic.items()
                    if key not in {"raw_mean", "raw_variance", "acf_rms", "cross_position_corr_offdiag_rms"}})
    per_mmd = {}
    for index, (label, reference) in enumerate((("raw", test), ("smooth", smooth))):
        swd, squared = _swd(flat, reference, directions)
        mmd, rows, terms = cross_ensemble_mmd2(samples, reference, bandwidths, reference_kernel[index])
        metrics.update({f"swd_{label}": swd, f"mmd2_{label}": mmd})
        metrics.update({f"mmd2_{label}_{key}": val for key, val in terms.items()})
        arrays[f"swd_{label}_per_direction_squared"] = squared; per_mmd[label] = rows
    ensemble_rows = []
    for ensemble in range(e):
        basic, ensemble_arrays = _basic(raw_samples[ensemble], real_raw)
        basic.update({"ensemble_id": ensemble, "particle_count": n,
                      "swd_raw": _swd(samples[ensemble], test, directions)[0], "swd_smooth": _swd(samples[ensemble], smooth, directions)[0],
                      "mmd2_cross_other_ensembles_raw": float(per_mmd["raw"][ensemble]),
                      "mmd2_cross_other_ensembles_smooth": float(per_mmd["smooth"][ensemble])})
        ensemble_rows.append(basic)
        arrays[f"ensemble{ensemble}_acf"] = ensemble_arrays["acf"]
        arrays[f"ensemble{ensemble}_psd"] = ensemble_arrays["psd"]
    with (output / "metrics_per_ensemble.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ensemble_rows[0])); writer.writeheader(); writer.writerows(ensemble_rows)
    if dataset_id in ("SEQ_DELAY", "SEQ_ACCUM"):
        early, late = params["early_indices"], params["late_indices"]
        joint_indices = np.r_[early, late]
        joint_directions = params["joint_directions"]
        generated_shuffled_raw = shuffle_generated_late(raw_samples, dataset_id, METRIC_SEEDS[dataset_id] + 2).reshape(-1, LENGTH, CHANNELS)
        stats = load_stats(dataset_id, root)
        generated_shuffled = (generated_shuffled_raw.reshape(-1, STATE_DIM) - stats["mean"]) / stats["scale"]
        true_shuffled = test.copy(); order = _derangement(len(test), METRIC_SEEDS[dataset_id] + 3)
        true_shuffled[:, late] = test[order][:, late]
        real_shuffled_raw = unstandardize(true_shuffled, dataset_id, root).reshape(-1, LENGTH, CHANNELS)
        metrics["early_late_joint_swd_raw"], arrays["early_late_joint_swd_raw_per_direction_squared"] = _swd(flat[:, joint_indices], test[:, joint_indices], joint_directions)
        metrics["early_late_joint_swd_smooth"], arrays["early_late_joint_swd_smooth_per_direction_squared"] = _swd(flat[:, joint_indices], smooth[:, joint_indices], joint_directions)
        metrics["generated_shuffled_joint_swd_raw"] = _swd(generated_shuffled[:, joint_indices], test[:, joint_indices], joint_directions)[0]
        metrics["true_shuffled_joint_swd_raw"] = _swd(true_shuffled[:, joint_indices], test[:, joint_indices], joint_directions)[0]
        generated_corr = arrays["correlation"][np.ix_(early, late)]; real_corr = arrays["reference_correlation"][np.ix_(early, late)]
        metrics["early_late_correlation_error_rms"] = float(np.sqrt(np.mean((generated_corr - real_corr) ** 2)))
        residuals = {"generated_paired": structural_residual(raw, dataset_id),
                     "generated_shuffled": structural_residual(generated_shuffled_raw, dataset_id),
                     "real_paired": structural_residual(real_raw, dataset_id), "real_shuffled": structural_residual(real_shuffled_raw, dataset_id)}
        for name, residual in residuals.items():
            arrays[f"{name}_structural_residual"] = residual
            metrics[f"{name}_structure_mse_raw"] = float(np.mean(residual ** 2))
        metrics["generated_structure_pairing_gain_raw"] = metrics["generated_shuffled_structure_mse_raw"] - metrics["generated_paired_structure_mse_raw"]
        metrics["real_structure_pairing_gain_raw"] = metrics["real_shuffled_structure_mse_raw"] - metrics["real_paired_structure_mse_raw"]
        arrays.update(early_late_generated_correlation=generated_corr, early_late_real_correlation=real_corr,
                      real_late_shuffle_order=order, generated_shuffled_raw=generated_shuffled_raw)
    downstream, ridge_arrays, ridge_records = _downstream(flat, test, params, dataset_id, root)
    metrics.update(downstream)
    block = e * n
    real_a, real_b = test[:block].reshape(e, n, STATE_DIM), test[block:2 * block].reshape(e, n, STATE_DIM)
    smooth_a, smooth_b = smooth[:block].reshape(e, n, STATE_DIM), smooth[block:2 * block].reshape(e, n, STATE_DIM)
    metrics.update({"real_real_swd_raw_same_total_count": _swd(real_a.reshape(-1, STATE_DIM), real_b.reshape(-1, STATE_DIM), directions)[0],
                    "real_real_swd_smooth_same_total_count": _swd(smooth_a.reshape(-1, STATE_DIM), smooth_b.reshape(-1, STATE_DIM), directions)[0],
                    "real_real_mmd2_raw": cross_ensemble_mmd2(real_a, real_b.reshape(-1, STATE_DIM), bandwidths)[0],
                    "real_real_mmd2_smooth": cross_ensemble_mmd2(smooth_a, smooth_b.reshape(-1, STATE_DIM), bandwidths)[0],
                    "generated_ensemble_count": float(e), "generated_particles_per_ensemble": float(n),
                    "generated_sample_count": float(block), "test_reference_count": float(len(test))})
    baseline_rows = [{"block_id": i, "samples_per_block": n,
                      "swd_raw": _swd(real_a[i], real_b[i], directions)[0], "swd_smooth": _swd(smooth_a[i], smooth_b[i], directions)[0]} for i in range(e)]
    with (output / "real_real_per_block.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0])); writer.writeheader(); writer.writerows(baseline_rows)
    np.save(output / "samples_standardized.npy", np.asarray(samples_standardized))
    np.save(output / "samples_raw.npy", raw_samples)
    np.savez_compressed(output / "metrics_arrays.npz", **arrays, **{f"smooth_reference_{k}": v for k, v in smooth_arrays.items()})
    np.savez_compressed(output / "ridge_arrays.npz", **ridge_arrays)
    np.savez(output / "real_real_reference_blocks.npz", raw_standardized=np.stack([real_a, real_b]), smooth_standardized=np.stack([smooth_a, smooth_b]))
    write_json(output / "ridge_diagnostics.json", ridge_records)
    write_json(output / "metrics.json", metrics)
    write_json(output / "metric_provenance.json", {**provenance, "ensembles": e, "particles_per_ensemble": n,
        "metric_spec_sha256": sha256_file(data_dir(dataset_id, root) / "METRIC_SPEC.json"),
        "generated_standardized_sha256": sha256_file(output / "samples_standardized.npy"),
        "generated_raw_sha256": sha256_file(output / "samples_raw.npy"),
        "reference_blocks": {"A": [0, block], "B": [block, 2 * block]},
        "TSTR_TRTR": "lambda chosen only with real validation; paired within-sequence downstream prediction, never unrelated trajectory MSE",
        "within_ensemble": "may be dependent; per-ensemble MMD rows share cross terms and are not independent training repeats"})
    return metrics
