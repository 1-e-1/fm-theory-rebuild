"""Formal SEQ_IID distribution evaluation with independent ensemble accounting.

No routine in this module is called by data preparation or a validation pilot.
``evaluate_run`` explicitly opens formal test data. Negative MMD² estimates are
retained, and generated-generated kernel pairs always cross ensembles.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiment.data import (LENGTH, STATE_DIM, METRIC_SEED, data_dir,
                             load_stats, sha256_file, unstandardize, write_json)


def empirical_wasserstein(a: np.ndarray, b: np.ndarray, power: int = 2) -> np.ndarray:
    """Exact empirical quantile integral, independently for every column.

    Returns the integral of |Fa^-1-Fb^-1|**power, without a root. It supports
    unequal sample counts and never treats projections as independent seeds.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        raise ValueError("Wasserstein expects nonempty [sample,feature] arrays with equal feature counts")
    a = np.sort(a, axis=0)
    b = np.sort(b, axis=0)
    if len(a) == len(b):
        return np.mean(np.abs(a - b) ** power, axis=0)
    edges = np.unique(np.concatenate([np.arange(len(a) + 1) / len(a), np.arange(len(b) + 1) / len(b)]))
    midpoint = (edges[:-1] + edges[1:]) / 2
    ia = np.minimum((midpoint * len(a)).astype(np.int64), len(a) - 1)
    ib = np.minimum((midpoint * len(b)).astype(np.int64), len(b) - 1)
    differences = np.abs(a[ia] - b[ib]) ** power
    return np.diff(edges) @ differences


def _kernel_sum(a: np.ndarray, b: np.ndarray, bandwidths: np.ndarray, block_size: int = 256) -> float:
    """Sum the mean multi-bandwidth RBF kernel without an NxM allocation."""
    result = 0.0
    b = np.asarray(b, dtype=np.float64)
    bnorm = np.einsum("ij,ij->i", b, b)
    for offset in range(0, len(a), block_size):
        block = np.asarray(a[offset:offset + block_size], dtype=np.float64)
        anorm = np.einsum("ij,ij->i", block, block)
        distances = np.maximum(anorm[:, None] + bnorm[None, :] - 2 * block @ b.T, 0)
        for bandwidth in bandwidths:
            result += float(np.exp(-distances / (2 * bandwidth * bandwidth)).sum(dtype=np.float64)) / len(bandwidths)
    return result


def cross_ensemble_mmd2(samples: np.ndarray, reference: np.ndarray,
                        bandwidths: np.ndarray, reference_kernel_mean: float | None = None) -> tuple[float, np.ndarray, dict[str, float]]:
    """Independent-ensemble generated term + iid off-diagonal real term.

    A row for ensemble e uses every other ensemble in its generated-generated
    term. These rows share terms and are diagnostics, not independent MMD
    replicates. Averaging the rows gives the pooled estimate exactly.
    """
    samples = np.asarray(samples, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if samples.ndim != 3 or len(samples) < 2 or samples.shape[2] != reference.shape[1]:
        raise ValueError("MMD requires samples [E>=2,N,D] and reference [M,D]")
    ensembles, count, _ = samples.shape
    if len(reference) < 2:
        raise ValueError("MMD reference requires at least two samples")
    if reference_kernel_mean is None:
        reference_kernel_mean = (_kernel_sum(reference, reference, bandwidths) - len(reference)) / (len(reference) * (len(reference) - 1))
    gg_per_ensemble = np.zeros(ensembles, dtype=np.float64)
    for first in range(ensembles):
        for second in range(first + 1, ensembles):
            pair = _kernel_sum(samples[first], samples[second], bandwidths) / (count * count)
            gg_per_ensemble[first] += pair / (ensembles - 1)
            gg_per_ensemble[second] += pair / (ensembles - 1)
    gr_per_ensemble = np.asarray([
        _kernel_sum(ensemble, reference, bandwidths) / (count * len(reference)) for ensemble in samples
    ])
    per_ensemble = gg_per_ensemble + reference_kernel_mean - 2 * gr_per_ensemble
    details = {"gg_cross_ensemble": float(gg_per_ensemble.mean()),
               "rr_off_diagonal": float(reference_kernel_mean), "gr": float(gr_per_ensemble.mean())}
    return float(per_ensemble.mean()), per_ensemble, details


def correlation_acf(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Across-realization positional correlations and mean diagonal-lag ACF."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != STATE_DIM or len(states) < 2:
        raise ValueError("Correlation expects [N>=2,64] complete sequence states")
    centered = states - states.mean(axis=0)
    norms = np.sqrt(np.einsum("ij,ij->j", centered, centered))
    denominators = norms[:, None] * norms[None, :]
    correlation = np.divide(centered.T @ centered, denominators,
                            out=np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64), where=denominators > 0)
    acf = np.asarray([np.diagonal(correlation, offset=lag).mean() for lag in range(1, LENGTH // 2 + 1)])
    return correlation, acf


def _basic_metrics(raw: np.ndarray, reference_raw: np.ndarray, reference_corr: np.ndarray,
                   reference_acf: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    mean = raw.mean(axis=0)
    variance = raw.var(axis=0, ddof=0)
    reference_mean = reference_raw.mean(axis=0)
    reference_variance = reference_raw.var(axis=0, ddof=0)
    marginal_w1 = empirical_wasserstein(raw, reference_raw, power=1)
    marginal_w2_squared = empirical_wasserstein(raw, reference_raw, power=2)
    corr, acf = correlation_acf(raw)
    off_diagonal = ~np.eye(STATE_DIM, dtype=bool)
    rms = np.sqrt(np.mean(raw * raw, axis=1))
    max_abs = np.max(np.abs(raw), axis=1)
    reference_rms = np.sqrt(np.mean(reference_raw * reference_raw, axis=1))
    reference_max_abs = np.max(np.abs(reference_raw), axis=1)
    metrics = {
        "marginal_w1_mean": float(marginal_w1.mean()), "marginal_w1_max": float(marginal_w1.max()),
        "marginal_w2_rms": float(np.sqrt(marginal_w2_squared.mean())),
        "position_mean_rmse": float(np.sqrt(np.mean((mean - reference_mean) ** 2))),
        "position_variance_rmse": float(np.sqrt(np.mean((variance - reference_variance) ** 2))),
        "raw_mean": float(raw.mean()), "raw_variance": float(raw.var(ddof=0)),
        "absolute_amplitude_w1": float(empirical_wasserstein(np.abs(raw).reshape(-1), np.abs(reference_raw).reshape(-1), power=1)[0]),
        "sequence_rms_w1": float(empirical_wasserstein(rms, reference_rms, power=1)[0]),
        "sequence_max_abs_w1": float(empirical_wasserstein(max_abs, reference_max_abs, power=1)[0]),
        "cross_position_corr_offdiag_rms": float(np.sqrt(np.mean(corr[off_diagonal] ** 2))),
        "cross_position_corr_offdiag_max_abs": float(np.abs(corr[off_diagonal]).max()),
        "cross_position_corr_error_rms": float(np.sqrt(np.mean((corr[off_diagonal] - reference_corr[off_diagonal]) ** 2))),
        "acf_rms": float(np.sqrt(np.mean(acf ** 2))), "acf_max_abs": float(np.abs(acf).max()),
        "acf_error_rms": float(np.sqrt(np.mean((acf - reference_acf) ** 2))),
    }
    arrays = {"position_mean": mean, "position_variance": variance, "marginal_w1": marginal_w1,
              "marginal_w2_squared": marginal_w2_squared, "correlation": corr, "acf": acf,
              "sequence_rms": rms, "sequence_max_abs": max_abs}
    return metrics, arrays


def _formal_reference(root: Path | str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    directory = data_dir(root)
    with np.load(directory / "test.npz", allow_pickle=False) as archive:
        standardized = archive["standardized"].astype(np.float64)
        raw = archive["raw"].reshape(-1, STATE_DIM).astype(np.float64)
    cache_path = directory / "formal_references.npz"
    source_hash = sha256_file(directory / "test.npz")
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            if str(archive["source_sha256"].item()) != source_hash:
                raise RuntimeError("Formal reference cache source hash differs from frozen test data")
            smooth = archive["smooth_standardized"].copy()
    else:
        rng = np.random.default_rng(METRIC_SEED + 1)
        noise = rng.standard_normal(standardized.shape)
        smooth = standardized + 0.05 * noise
        np.savez(cache_path, smooth_standardized=smooth, source_sha256=np.asarray(source_hash),
                 smoothing_noise=noise, smoothing_seed=np.asarray(METRIC_SEED + 1))
    return standardized, smooth, raw, {"reference_source": str(directory / "test.npz"),
                                      "reference_source_sha256": source_hash,
                                      "smooth_reference_path": str(cache_path),
                                      "smooth_reference_sha256": sha256_file(cache_path)}


def _reference_kernel_means(raw_standardized: np.ndarray, smooth_standardized: np.ndarray,
                            bandwidths: np.ndarray, root: Path | str | None,
                            reference_provenance: dict[str, Any]) -> tuple[float, float]:
    cache = data_dir(root) / "formal_reference_kernel_means.json"
    hashes = {"raw": reference_provenance["reference_source_sha256"],
              "smooth": reference_provenance["smooth_reference_sha256"],
              "metric_parameters": sha256_file(data_dir(root) / "metric_parameters.npz")}
    if cache.exists():
        content = json.loads(cache.read_text(encoding="utf-8"))
        if content["hashes"] != hashes:
            raise RuntimeError("Formal reference kernel cache no longer matches its data/metric hashes")
        return content["raw_mean"], content["smooth_mean"]
    means = []
    for reference in (raw_standardized, smooth_standardized):
        count = len(reference)
        means.append((_kernel_sum(reference, reference, bandwidths) - count) / (count * (count - 1)))
    write_json(cache, {"hashes": hashes, "raw_mean": means[0], "smooth_mean": means[1]})
    return means[0], means[1]


def evaluate_run(samples_standardized: np.ndarray, output_dir: Path | str,
                 root: Path | str | None = None) -> dict[str, float]:
    """Evaluate and archive one formal checkpoint's independent ensembles.

    Input is [E,N,D=64] in the shared training-standardized space. This is the
    only high-level entry point; invoking it is explicit permission to read test.
    """
    samples = np.asarray(samples_standardized, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != STATE_DIM or samples.shape[0] < 2 or samples.shape[1] < 2:
        raise ValueError("Formal evaluation requires [E>=2,N>=2,D=64]")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Nonfinite generated samples are a formal evaluation failure")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_reference, smooth_reference, reference_raw_units, provenance = _formal_reference(root)
    ensembles, count, _ = samples.shape
    if 2 * ensembles * count > len(raw_reference):
        raise ValueError("Insufficient test data for two disjoint real-real reference blocks at this E,N")
    with np.load(data_dir(root) / "metric_parameters.npz", allow_pickle=False) as archive:
        directions = archive["directions"].copy()
        bandwidths = archive["bandwidths"].copy()
    raw_rr_mean, smooth_rr_mean = _reference_kernel_means(raw_reference, smooth_reference, bandwidths, root, provenance)
    generated = samples.reshape(-1, STATE_DIM)
    generated_raw = unstandardize(samples, root)
    flat_raw = generated_raw.reshape(-1, STATE_DIM)
    reference_corr, reference_acf = correlation_acf(reference_raw_units)
    reference_smooth_raw = unstandardize(smooth_reference, root)
    reference_smooth_corr, reference_smooth_acf = correlation_acf(reference_smooth_raw)
    metrics, basic_arrays = _basic_metrics(flat_raw, reference_raw_units, reference_corr, reference_acf)
    smooth_basic, smooth_arrays = _basic_metrics(flat_raw, reference_smooth_raw, reference_smooth_corr, reference_smooth_acf)
    metrics.update({key + "_smooth_reference": value for key, value in smooth_basic.items()
                    if key not in {"raw_mean", "raw_variance", "cross_position_corr_offdiag_rms", "cross_position_corr_offdiag_max_abs", "acf_rms", "acf_max_abs"}})
    generated_projection = generated @ directions.T
    raw_projection = raw_reference @ directions.T
    smooth_projection = smooth_reference @ directions.T
    swd_raw_directions = empirical_wasserstein(generated_projection, raw_projection, power=2)
    swd_smooth_directions = empirical_wasserstein(generated_projection, smooth_projection, power=2)
    metrics.update({"swd_raw": float(np.sqrt(swd_raw_directions.mean())),
                    "swd_smooth": float(np.sqrt(swd_smooth_directions.mean())),
                    "generated_ensemble_count": float(ensembles), "generated_particles_per_ensemble": float(count),
                    "generated_sample_count": float(ensembles * count), "test_reference_count": float(len(raw_reference))})
    mmd_raw, per_mmd_raw, mmd_raw_terms = cross_ensemble_mmd2(samples, raw_reference, bandwidths, raw_rr_mean)
    mmd_smooth, per_mmd_smooth, mmd_smooth_terms = cross_ensemble_mmd2(samples, smooth_reference, bandwidths, smooth_rr_mean)
    metrics.update({"mmd2_raw": mmd_raw, "mmd2_smooth": mmd_smooth})
    for name, terms in (("raw", mmd_raw_terms), ("smooth", mmd_smooth_terms)):
        metrics.update({f"mmd2_{name}_{key}": value for key, value in terms.items()})

    ensemble_rows = []
    ensemble_arrays: dict[str, list[np.ndarray]] = {}
    for index in range(ensembles):
        row, row_arrays = _basic_metrics(generated_raw[index], reference_raw_units, reference_corr, reference_acf)
        row.update({"ensemble_id": index, "particle_count": count,
                    "swd_raw": float(np.sqrt(empirical_wasserstein(samples[index] @ directions.T, raw_projection, power=2).mean())),
                    "swd_smooth": float(np.sqrt(empirical_wasserstein(samples[index] @ directions.T, smooth_projection, power=2).mean())),
                    "mmd2_cross_other_ensembles_raw": float(per_mmd_raw[index]),
                    "mmd2_cross_other_ensembles_smooth": float(per_mmd_smooth[index])})
        ensemble_rows.append(row)
        for name, array in row_arrays.items():
            ensemble_arrays.setdefault(name, []).append(array)
    with (output / "metrics_per_ensemble.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ensemble_rows[0]))
        writer.writeheader()
        writer.writerows(ensemble_rows)

    # Disjoint real blocks expose finite-sample noise at the actual particle
    # budget. They are a reference distribution, not a quality guarantee.
    block_count = ensembles * count
    real_a = raw_reference[:block_count].reshape(ensembles, count, STATE_DIM)
    real_b = raw_reference[block_count:2 * block_count].reshape(ensembles, count, STATE_DIM)
    real_a_smooth = smooth_reference[:block_count].reshape(ensembles, count, STATE_DIM)
    real_b_smooth = smooth_reference[block_count:2 * block_count].reshape(ensembles, count, STATE_DIM)
    baseline_rows = []
    for index in range(ensembles):
        first_raw = unstandardize(real_a[index], root)
        second_raw = unstandardize(real_b[index], root)
        second_corr, second_acf = correlation_acf(second_raw)
        row, _ = _basic_metrics(first_raw, second_raw, second_corr, second_acf)
        row.update({"block_id": index, "samples_per_block": count,
                    "swd_raw": float(np.sqrt(empirical_wasserstein(real_a[index] @ directions.T, real_b[index] @ directions.T, power=2).mean())),
                    "swd_smooth": float(np.sqrt(empirical_wasserstein(real_a_smooth[index] @ directions.T, real_b_smooth[index] @ directions.T, power=2).mean()))})
        baseline_rows.append(row)
    with (output / "real_real_per_block.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)
    baseline_mmd_raw, _, _ = cross_ensemble_mmd2(real_a, real_b.reshape(-1, STATE_DIM), bandwidths)
    baseline_mmd_smooth, _, _ = cross_ensemble_mmd2(real_a_smooth, real_b_smooth.reshape(-1, STATE_DIM), bandwidths)
    metrics.update({"real_real_swd_raw_same_total_count": float(np.sqrt(empirical_wasserstein(real_a.reshape(-1, STATE_DIM) @ directions.T, real_b.reshape(-1, STATE_DIM) @ directions.T, power=2).mean())),
                    "real_real_swd_smooth_same_total_count": float(np.sqrt(empirical_wasserstein(real_a_smooth.reshape(-1, STATE_DIM) @ directions.T, real_b_smooth.reshape(-1, STATE_DIM) @ directions.T, power=2).mean())),
                    "real_real_mmd2_raw": baseline_mmd_raw, "real_real_mmd2_smooth": baseline_mmd_smooth,
                    "real_test_corr_offdiag_rms": float(np.sqrt(np.mean(reference_corr[~np.eye(STATE_DIM, dtype=bool)] ** 2))),
                    "real_test_acf_rms": float(np.sqrt(np.mean(reference_acf ** 2)))})

    np.save(output / "samples_standardized.npy", np.asarray(samples_standardized))
    np.save(output / "samples_raw.npy", generated_raw)
    np.savez(output / "metrics_arrays.npz", **basic_arrays,
             **{f"smooth_reference_{name}": array for name, array in smooth_arrays.items()},
             **{f"ensemble_{name}": np.stack(arrays) for name, arrays in ensemble_arrays.items()},
             swd_raw_per_direction_squared=swd_raw_directions,
             swd_smooth_per_direction_squared=swd_smooth_directions,
             reference_position_mean=reference_raw_units.mean(axis=0),
             reference_position_variance=reference_raw_units.var(axis=0),
             reference_correlation=reference_corr, reference_acf=reference_acf)
    np.savez(output / "real_real_reference_blocks.npz", raw_standardized=np.stack([real_a, real_b]),
             smooth_standardized=np.stack([real_a_smooth, real_b_smooth]))
    write_json(output / "metrics.json", metrics)
    write_json(output / "metric_provenance.json", {
        **provenance, "metric_spec": str(data_dir(root) / "METRIC_SPEC.json"),
        "metric_parameters_sha256": sha256_file(data_dir(root) / "metric_parameters.npz"),
        "ensemble_count": ensembles, "particles_per_ensemble": count,
        "particle_dependence": "within-ensemble particles not independent; seed-level summaries must use training seeds",
        "mmd2_rows": "each row shares every other ensemble; not independent replicates",
        "real_real_reference_indices": {"A": [0, block_count], "B": [block_count, 2 * block_count]},
        "smoothing_seed": METRIC_SEED + 1,
        "generated_standardized_sha256": sha256_file(output / "samples_standardized.npy"),
        "generated_raw_sha256": sha256_file(output / "samples_raw.npy"),
    })
    return metrics
