"""CPU-only scientific figures from artifact-verified phase-1 outputs.

Usage: python -m experiment.plots [--output-dir PATH] [--tables-dir PATH]
No model is built, no optimizer is created, and no data is generated. Only
manifest COMPLETED entries are plotted. Run reporting before this command to
refresh its tables, and package after this command to include the figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("FM_STANDARD", "FM_REAL_ODE", "FM_COMPLEX_SDE", "FM_COMPLEX_ODE", "FM_COMPLEX_VDC_ODE")
SEEDS = (17, 29, 43, 71, 101)
LABELS = {"FM_STANDARD": "Standard (Heun)", "FM_REAL_ODE": "Real ODE",
          "FM_COMPLEX_SDE": "Complex SDE", "FM_COMPLEX_ODE": "Complex ODE",
          "FM_COMPLEX_VDC_ODE": "Complex VDC ODE"}
COLORS = {"FM_STANDARD": "#0072B2", "FM_REAL_ODE": "#D55E00", "FM_COMPLEX_SDE": "#009E73",
          "FM_COMPLEX_ODE": "#CC79A7", "FM_COMPLEX_VDC_ODE": "#E69F00"}
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Causal arithmetic mean; first points use their available prefix only."""
    values = np.asarray(values, dtype=np.float64)
    if window < 1:
        raise ValueError("Moving-average window must be positive")
    sums = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(ends - window, 0)
    return (sums[ends] - sums[starts]) / (ends - starts)


def _seed_moments(values: list[float]) -> dict[str, Any]:
    n = len(values)
    result = {"n_training_seeds": n, "expected_training_seeds": 5,
              "complete_seed_group": n == 5, "mean": "", "std": "", "ci95_low": "", "ci95_high": ""}
    if n:
        result["mean"] = float(np.mean(values))
    if n >= 2:
        # Exact two-sided Student t critical values for the only possible df.
        # Same method as the preregistered seed-level reporting protocol.
        critical = {1: 12.706204736432095, 2: 4.302652729911275,
                    3: 3.182446305284263, 4: 2.7764451051977987}[n - 1]
        sd = float(np.std(values, ddof=1))
        radius = critical * sd / math.sqrt(n)
        result.update(std=sd, ci95_low=result["mean"] - radius, ci95_high=result["mean"] + radius)
    return result


def _corr_acf(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = states - states.mean(axis=0)
    norms = np.sqrt(np.einsum("ij,ij->j", centered, centered))
    denom = norms[:, None] * norms[None, :]
    corr = np.divide(centered.T @ centered, denom, out=np.zeros((64, 64)), where=denom > 0)
    acf = np.asarray([np.diagonal(corr, offset=lag).mean() for lag in range(1, 33)])
    return corr, acf


def _records(root: Path, tables_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    manifest = _json(root / "manifest.json")
    if manifest.get("formal_configuration_count") != 50 or len(manifest["runs"]) != 50:
        raise ValueError("Plotting requires the fixed 50-entry phase-1 manifest")
    table_path = tables_dir / "main_per_seed.csv"
    table_rows: dict[str, dict[str, str]] = {}
    warnings = []
    inputs = [root / "manifest.json"]
    if table_path.exists():
        with table_path.open(newline="", encoding="utf-8-sig") as handle:
            table_rows = {row["run_id"]: row for row in csv.DictReader(handle)}
        inputs.append(table_path)
    else:
        warnings.append("No summary table found; completed output JSONs are the numerical source")
    compute_table = tables_dir / "compute_costs.csv"
    if compute_table.exists():
        inputs.append(compute_table)
    records = []
    for run in manifest["runs"]:
        if run["dataset_id"] != "SEQ_IID" or run["status"] != "COMPLETED":
            continue
        if run["model_id"] not in MODELS or run["seed"] not in SEEDS:
            raise ValueError("Completed run lies outside the phase-1 model/seed scope")
        attempt_value = run.get("attempt_dir", run.get("current_attempt_dir", run.get("output_dir")))
        if not attempt_value:
            raise ValueError(f"Completed run has no attempt path: {run['run_id']}")
        attempt = Path(attempt_value)
        if not attempt.is_absolute():
            attempt = root / attempt
        attempt = attempt.resolve()
        if not attempt.is_relative_to(root):
            raise ValueError("Completed artifact path lies outside this project")
        config_path = attempt / "config.json"
        cfg = _json(config_path)
        if any(cfg[key] != run[key] for key in ("dataset_id", "model_id", "seed")):
            raise ValueError(f"Completed output identity mismatch: {run['run_id']}")
        standard = run["model_id"] == "FM_STANDARD"
        metric_dir = attempt / "heun_metrics" if standard else attempt
        evaluation_path = attempt / ("evaluation_heun.json" if standard else "evaluation.json")
        required = [attempt / "checkpoint_final.pt", attempt / "training.csv", evaluation_path,
                    metric_dir / "samples_raw.npy", metric_dir / "metrics_arrays.npz",
                    attempt / "training_cost.json", attempt / "sampling_diagnostics.json"]
        if any(not path.is_file() for path in required):
            raise ValueError(f"Completed output lacks plot evidence: {run['run_id']}")
        evaluation = _json(evaluation_path)
        metrics = evaluation.get("metrics", evaluation)
        if not all(math.isfinite(float(metrics[key])) for key in ("swd_raw", "mmd2_raw")):
            raise ValueError(f"Completed output has nonfinite main quality metrics: {run['run_id']}")
        diagnostics = _json(attempt / "sampling_diagnostics.json")
        primary_cost = diagnostics["heun_reference" if standard else "primary"]
        costs = _json(attempt / "training_cost.json")
        table = table_rows.get(run["run_id"], {})
        if table.get("status") != "COMPLETED":
            warnings.append(f"Table not refreshed for {run['run_id']}; plot uses completed output JSON")
        elif any(table.get(key) and not np.isclose(float(table[key]), float(metrics[key]), rtol=1e-12, atol=1e-14)
                 for key in ("swd_raw", "mmd2_raw")):
            raise ValueError(f"Summary table disagrees with completed primary evaluation: {run['run_id']}")
        record = {"run": run, "attempt": attempt, "config": cfg, "metric_dir": metric_dir,
                  "metrics": metrics, "train_seconds": float(costs["training_seconds"]),
                  "sample_seconds": float(primary_cost["sampling_seconds"]),
                  "solver": "Heun" if standard else "Euler-Maruyama" if run["model_id"] == "FM_COMPLEX_SDE" else "PL-history Euler"}
        records.append(record)
        inputs.extend([config_path, *required[1:]])
        for name in ("metric_provenance.json", "real_real_reference_blocks.npz"):
            if (metric_dir / name).exists():
                inputs.append(metric_dir / name)
    return records, warnings, list(dict.fromkeys(inputs))


def generate_figures(root: Path | str | None = None, output_dir: Path | str | None = None,
                     tables_dir: Path | str | None = None, moving_window: int = 100) -> dict[str, Any]:
    """Generate vector/raster figures plus exact source arrays; no training."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root = Path(root).resolve() if root is not None else ROOT
    output = Path(output_dir).resolve() if output_dir is not None else root / "review_pack_phase1_50" / "figures"
    tables = Path(tables_dir).resolve() if tables_dir is not None else root / "review_pack_phase1_50" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    sources = output / "sources"
    sources.mkdir(exist_ok=True)
    records, warnings, input_paths = _records(root, tables)
    grouped = {model: sorted([r for r in records if r["run"]["model_id"] == model], key=lambda r: r["run"]["seed"]) for model in MODELS}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "svg.fonttype": "none", "figure.dpi": 110,
                         "savefig.dpi": 200, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.fontsize": 8})
    figures = []
    def save(fig, stem: str, source_files: list[Path], note: str) -> None:
        fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(output / f"{stem}.png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        figures.append({"stem": stem, "svg": f"{stem}.svg", "png": f"{stem}.png",
                        "source_files": [str(p.relative_to(output)) for p in source_files], "interpretation": note})

    # Separate learning plots preserve the different SDE/ODE regression targets.
    for model in MODELS:
        fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.4))
        rows = []
        arrays = {}
        for rec in grouped[model]:
            seed = rec["run"]["seed"]
            with (rec["attempt"] / "training.csv").open(newline="", encoding="utf-8-sig") as handle:
                log = list(csv.DictReader(handle))
            steps = np.asarray([int(row["step"]) for row in log])
            batch = np.asarray([float(row["loss"]) for row in log])
            smooth = moving_average(batch, moving_window)
            validation = np.asarray([float(row["val_loss"]) if row.get("val_loss") else np.nan for row in log])
            color = SEED_COLORS[SEEDS.index(seed)]
            axes[0].plot(steps, smooth, color=color, linewidth=1.1, label=f"seed {seed}")
            finite = np.isfinite(validation)
            axes[1].plot(steps[finite], validation[finite], color=color, marker="o", markersize=3,
                         linewidth=1.1, label=f"seed {seed}")
            arrays.update({f"seed{seed}_step": steps, f"seed{seed}_weighted_batch_loss": batch,
                           f"seed{seed}_causal_moving_average": smooth, f"seed{seed}_validation_loss": validation})
            rows.extend({"model_id": model, "seed": seed, "step": int(steps[i]), "weighted_batch_loss": float(batch[i]),
                         "weighted_batch_causal_mean": float(smooth[i]), "raw_batch_loss": row["raw_loss"],
                         "validation_loss": row.get("val_loss", ""), "tau": row.get("tau", ""),
                         "elapsed_seconds": row["elapsed_seconds"]} for i, row in enumerate(log))
        for ax in axes:
            ax.set_xlabel("Optimizer update")
            ax.set_ylabel("Importance-weighted drift MSE")
            ax.grid(alpha=.2)
            if grouped[model]:
                ax.set_yscale("log")
                ax.legend(frameon=False)
            else:
                ax.text(.5, .5, "No completed runs (0/5)", transform=ax.transAxes, ha="center")
        axes[0].set_title(f"Training: causal {moving_window}-update arithmetic mean")
        axes[1].set_title("Fixed validation batches")
        target = "SDE target u + kappa*score; do not compare its loss to ODE target losses" if model == "FM_COMPLEX_SDE" else "ODE target u; losses shown for this model's seeds"
        fig.suptitle(f"{model} — completed seeds {len(grouped[model])}/5\n{target}", y=1.06)
        fig.tight_layout(w_pad=2.2)
        csv_path = sources / f"learning_{model}.csv"
        npz_path = sources / f"learning_{model}.npz"
        _write_csv(csv_path, rows, ["model_id", "seed", "step", "weighted_batch_loss", "weighted_batch_causal_mean",
                                  "raw_batch_loss", "validation_loss", "tau", "elapsed_seconds"])
        np.savez_compressed(npz_path, **arrays)
        save(fig, f"learning_{model}", [csv_path, npz_path],
             "Each curve is one completed training seed; importance-weighted arithmetic smoothing. SDE and ODE labels differ. Budget does not establish convergence.")

    # Quality summaries use training seeds, not particles or directions.
    quality_rows = []
    moments_rows = []
    for rec in records:
        quality_rows.append({"dataset_id": "SEQ_IID", "model_id": rec["run"]["model_id"], "seed": rec["run"]["seed"],
                             "run_id": rec["run"]["run_id"], "primary_solver": rec["solver"],
                             "swd_raw": float(rec["metrics"]["swd_raw"]), "mmd2_raw": float(rec["metrics"]["mmd2_raw"]),
                             "training_seconds": rec["train_seconds"], "primary_sampling_seconds": rec["sample_seconds"]})
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.0))
    for axis, metric in zip(axes, ("swd_raw", "mmd2_raw")):
        for x, model in enumerate(MODELS):
            group = grouped[model]
            values = [float(rec["metrics"][metric]) for rec in group]
            moments = _seed_moments(values)
            moments_rows.append({"model_id": model, "metric": metric, **moments})
            for rec, value in zip(group, values):
                jitter = (SEEDS.index(rec["run"]["seed"]) - 2) * .065
                axis.scatter(x + jitter, value, color=COLORS[model], s=24, alpha=.75, zorder=3)
            if values:
                axis.scatter(x, moments["mean"], marker="D", color="black", s=32, zorder=5)
            if len(values) >= 2:
                radius = moments["ci95_high"] - moments["mean"]
                axis.errorbar(x, moments["mean"], yerr=radius, color="black", capsize=4, linewidth=1.2, zorder=4)
        axis.set_xticks(range(5), [LABELS[m].replace(" ", "\n", 1) + f"\nn={len(grouped[m])}/5" for m in MODELS])
        axis.grid(axis="y", alpha=.2)
        axis.set_xlim(-.6, 4.6)
        axis.set_title("SWD₂" if metric == "swd_raw" else "MMD² (negative estimates retained)")
        axis.set_ylabel("SWD₂ (standardized complete sequence)" if metric == "swd_raw" else "Cross-ensemble MMD²")
        if metric == "mmd2_raw":
            axis.axhline(0, color="gray", linestyle=":", linewidth=.8)
    fig.suptitle("SEQ_IID: training-seed scatter, mean and 95% Student t interval\nPartial n/5 groups are descriptive interim results; n<2 has no interval", y=1.03)
    fig.tight_layout(w_pad=2.2)
    quality_csv = sources / "quality_per_training_seed.csv"
    moments_csv = sources / "quality_seed_moments.csv"
    _write_csv(quality_csv, quality_rows, ["dataset_id", "model_id", "seed", "run_id", "primary_solver", "swd_raw", "mmd2_raw", "training_seconds", "primary_sampling_seconds"])
    _write_csv(moments_csv, moments_rows, ["model_id", "metric", "n_training_seeds", "expected_training_seeds", "complete_seed_group", "mean", "std", "ci95_low", "ci95_high"])
    quality_npz = sources / "quality_per_training_seed.npz"
    np.savez(quality_npz, model_id=np.asarray([r["model_id"] for r in quality_rows]),
             seed=np.asarray([r["seed"] for r in quality_rows], dtype=np.int64),
             swd_raw=np.asarray([r["swd_raw"] for r in quality_rows]), mmd2_raw=np.asarray([r["mmd2_raw"] for r in quality_rows]),
             training_seconds=np.asarray([r["training_seconds"] for r in quality_rows]),
             primary_sampling_seconds=np.asarray([r["primary_sampling_seconds"] for r in quality_rows]))
    save(fig, "quality_seed_scatter", [quality_csv, moments_csv, quality_npz],
         "95% intervals resample no particles: five independent training seeds are the replication unit. Standard main values are Heun. Partial groups are not final five-seed comparisons.")

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    for row, metric in enumerate(("swd_raw", "mmd2_raw")):
        for column, cost in enumerate(("training_seconds", "primary_sampling_seconds")):
            axis = axes[row, column]
            for model in MODELS:
                values = [r for r in quality_rows if r["model_id"] == model and r[cost] > 0]
                axis.scatter([r[cost] for r in values], [r[metric] for r in values], color=COLORS[model], s=32, alpha=.8,
                             label=f"{LABELS[model]} (n={len(grouped[model])}/5)")
            if quality_rows:
                axis.set_xscale("log")
            axis.set_xlabel(("Training" if column == 0 else "Primary sampling") + " wall time (seconds, log scale)")
            axis.set_ylabel("SWD₂" if row == 0 else "MMD²")
            axis.grid(alpha=.2)
    fig.suptitle("SEQ_IID: generation quality and measured cost per training seed\nSampling cost uses the solver that produced the corresponding quality metric", y=1.01)
    fig.tight_layout(w_pad=2.2, h_pad=2.0)
    fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(.5, -.065), frameon=False)
    save(fig, "quality_walltime", [quality_csv, quality_npz],
         "Each point is a completed checkpoint. Time includes measured internal computation, not only NFE. No Pareto or superiority claim is inferred from partial groups.")

    # Held-out distribution diagnostics use archived raw generated samples and
    # archived real-reference blocks; no new test evaluation is run here.
    if records:
        stats_path = root / "data" / "seq_iid" / "stats.npz"
        with np.load(stats_path, allow_pickle=False) as stats:
            mean, scale = stats["mean"], stats["scale"]
        input_paths.append(stats_path)
        reference_path = records[0]["metric_dir"] / "real_real_reference_blocks.npz"
        with np.load(reference_path, allow_pickle=False) as archive:
            real_blocks = archive["raw_standardized"].astype(np.float64)
        reference = real_blocks.reshape(-1, 64) * scale + mean
        matched_reference = real_blocks[0].reshape(-1, 64) * scale + mean
        reference_corr, reference_acf = _corr_acf(matched_reference)
        real_rms = float(np.sqrt(np.mean(reference_corr[~np.eye(64, dtype=bool)] ** 2)))
        marginal_edges = np.linspace(-5.6, 5.6, 113)
        centers = .5 * (marginal_edges[:-1] + marginal_edges[1:])
        hist_rows = []
        diagnostic_rows = []
        arrays: dict[str, np.ndarray] = {"hist_edges": marginal_edges, "hist_centers": centers,
                                       "real_reference_raw": reference, "real_matched_reference_raw": matched_reference,
                                       "real_matched_correlation": reference_corr, "real_matched_acf": reference_acf}
        generated_groups = {}
        for model in MODELS:
            raw_list = []
            corr_list, acf_list = [], []
            for rec in grouped[model]:
                sample_path = rec["metric_dir"] / "samples_raw.npy"
                generated = np.load(sample_path, allow_pickle=False).astype(np.float64)
                if generated.ndim != 3 or generated.shape[2] != 64 or not np.isfinite(generated).all():
                    raise ValueError("IID figure input must be finite [ensemble,particle,64] raw samples")
                with np.load(rec["metric_dir"] / "metrics_arrays.npz", allow_pickle=False) as archive:
                    corr_list.append(archive["correlation"].copy())
                    acf_list.append(archive["acf"].copy())
                raw_list.append(generated.reshape(-1, 64))
                seed = rec["run"]["seed"]
                arrays[f"{model}_seed{seed}_raw"] = generated
                arrays[f"{model}_seed{seed}_correlation"] = corr_list[-1]
                arrays[f"{model}_seed{seed}_acf"] = acf_list[-1]
                diagnostic_rows.extend({"model_id": model, "seed": seed, "lag": lag + 1, "acf": float(value),
                                        "matched_real_acf": float(reference_acf[lag]),
                                        "corr_offdiag_rms": float(rec["metrics"]["cross_position_corr_offdiag_rms"]),
                                        "matched_real_corr_offdiag_rms": real_rms,
                                        "particle_count": len(raw_list[-1]), "complete_seed_group": len(grouped[model]) == 5}
                                       for lag, value in enumerate(acf_list[-1]))
            if raw_list:
                generated_groups[model] = {"raw": np.concatenate(raw_list), "correlation": np.stack(corr_list), "acf": np.stack(acf_list)}

        fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.8))
        for axis, position in zip(axes, (0, "pooled")):
            def density(states):
                values = states[:, 0] if position == 0 else states.reshape(-1)
                counts, _ = np.histogram(values, bins=marginal_edges)
                # Divide by all observations, preserving missing plotted tail mass.
                return counts / (len(values) * np.diff(marginal_edges)), float(np.mean((values < marginal_edges[0]) | (values > marginal_edges[-1])))
            real_density, real_tail = density(reference)
            axis.plot(centers, real_density, color="black", linestyle="--", linewidth=1.5, label=f"Held-out real ({len(reference)} sequences)")
            hist_rows.extend({"series": "real_reference", "position": position, "bin_left": marginal_edges[i], "bin_right": marginal_edges[i + 1], "density": value,
                              "out_of_range_fraction": real_tail, "n_training_seeds": "not applicable"} for i, value in enumerate(real_density))
            arrays[f"real_hist_{position}"] = real_density
            for model, values in generated_groups.items():
                observed, tail = density(values["raw"])
                axis.plot(centers, observed, color=COLORS[model], linewidth=1.2,
                          label=f"{LABELS[model]} (n={len(grouped[model])}/5)")
                hist_rows.extend({"series": model, "position": position, "bin_left": marginal_edges[i], "bin_right": marginal_edges[i + 1], "density": value,
                                  "out_of_range_fraction": tail, "n_training_seeds": len(grouped[model])} for i, value in enumerate(observed))
                arrays[f"{model}_hist_{position}"] = observed
            axis.set_title("Observation position 0" if position == 0 else "All observation positions (descriptive pooling)")
            axis.set_xlabel("Raw marginal value")
            axis.set_ylabel("Density (all-observation normalization)")
            axis.grid(alpha=.2)
        fig.suptitle("SEQ_IID: non-Gaussian marginal diagnostics\nSamples pooled only for visualization; particle counts are not training replicates", y=1.03)
        fig.tight_layout(w_pad=2.2)
        fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=3,
                   bbox_to_anchor=(.5, -.14), frameon=False)
        histogram_csv = sources / "iid_marginal_histograms.csv"
        diagnostic_csv = sources / "iid_acf_per_training_seed.csv"
        diagnostic_npz = sources / "iid_diagnostics.npz"
        _write_csv(histogram_csv, hist_rows, ["series", "position", "bin_left", "bin_right", "density", "out_of_range_fraction", "n_training_seeds"])
        _write_csv(diagnostic_csv, diagnostic_rows, ["model_id", "seed", "lag", "acf", "matched_real_acf", "corr_offdiag_rms", "matched_real_corr_offdiag_rms", "particle_count", "complete_seed_group"])
        np.savez_compressed(diagnostic_npz, **arrays)
        save(fig, "iid_marginal_diagnostics", [histogram_csv, diagnostic_npz],
             "Raw units; fixed mixture-scale plot range [-5.6,5.6]. Histogram densities divide by all observations; out-of-range mass is preserved in CSV. Real reference is the archived disjoint held-out blocks, not a new data split.")

        fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.8))
        lags = np.arange(1, 33)
        axes[0].plot(lags, reference_acf, color="black", linestyle="--", linewidth=1.5, label=f"Real, matched count {len(matched_reference)}")
        axes[0].axhline(0, color="gray", linewidth=.7)
        for index, model in enumerate(MODELS):
            if model not in generated_groups:
                continue
            values = generated_groups[model]
            acf = values["acf"]
            axes[0].plot(lags, acf.mean(axis=0), color=COLORS[model], label=f"{LABELS[model]} (n={len(grouped[model])}/5)")
            if len(acf) >= 2:
                axes[0].fill_between(lags, acf.min(axis=0), acf.max(axis=0), color=COLORS[model], alpha=.12)
            offdiag = ~np.eye(64, dtype=bool)
            rms_values = np.sqrt(np.mean(values["correlation"][:, offdiag] ** 2, axis=1))
            for rec, rms_value in zip(grouped[model], rms_values):
                jitter = (SEEDS.index(rec["run"]["seed"]) - 2) * .065
                axes[1].scatter(index + jitter, rms_value, color=COLORS[model], s=25, alpha=.8)
            axes[1].scatter(index, rms_values.mean(), marker="D", color="black", s=30)
        axes[0].set_xlabel("Observation-position lag")
        axes[0].set_ylabel("ACF (across-realization positional correlation)")
        axes[0].grid(alpha=.2)
        axes[1].axhline(real_rms, color="black", linestyle="--", linewidth=1.2, label=f"Real, matched count {len(matched_reference)}")
        axes[1].set_xticks(range(5), [LABELS[m].replace(" ", "\n", 1) + f"\nn={len(grouped[m])}/5" for m in MODELS])
        axes[1].set_ylabel("RMS off-diagonal position correlation")
        axes[1].grid(axis="y", alpha=.2)
        fig.suptitle("SEQ_IID: dependence diagnostics and matched finite-sample real reference\nACF ribbon is the min–max range across completed training seeds, not a particle confidence interval", y=1.04)
        fig.tight_layout(w_pad=2.2)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
                   bbox_to_anchor=(.5, -.14), frameon=False)
        save(fig, "iid_dependence_diagnostics", [diagnostic_csv, diagnostic_npz],
             "Real finite-sample correlation uses the same sequence count as one generated checkpoint. A nonzero sample correlation alone does not establish spurious population dependence; IID results do not establish long-memory advantages.")

        fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.8))
        fig.subplots_adjust(hspace=.42, wspace=.28, top=.85, bottom=.08, left=.07, right=.9)
        matrices = [("Held-out real (matched count)", reference_corr)]
        matrices.extend((f"{LABELS[m]} (n={len(grouped[m])}/5)", generated_groups[m]["correlation"].mean(axis=0) if m in generated_groups else None) for m in MODELS)
        image = None
        for axis, (title, matrix) in zip(axes.flat, matrices):
            if matrix is None:
                axis.text(.5, .5, "No completed runs", ha="center", transform=axis.transAxes)
                axis.set_xlim(0, 64)
                axis.set_ylim(64, 0)
            else:
                masked = np.ma.array(matrix, mask=np.eye(64, dtype=bool))
                image = axis.imshow(masked, vmin=-.35, vmax=.35, cmap="RdBu_r", interpolation="nearest")
            axis.set_title(title)
            axis.set_xlabel("Observation position")
            axis.set_ylabel("Observation position")
        if image is not None:
            fig.colorbar(image, ax=list(axes.flat), fraction=.024, pad=.02, label="Mean position correlation (diagonal masked)")
        fig.suptitle("SEQ_IID: complete position-correlation matrices\nModel panels average completed training-seed matrices; fixed color scale ±0.35", y=.99)
        save(fig, "iid_position_correlation", [diagnostic_csv, diagnostic_npz],
             "Diagonal entries masked. Pointwise averages across training seeds are descriptive, not tests. Full unclipped matrices remain in NPZ; color saturation does not discard numbers.")

    summary = {"completed_seq_iid": len(records), "formal_denominator": 50,
               "per_model_completed_training_seeds": {m: len(grouped[m]) for m in MODELS},
               "backend": "matplotlib Agg / CPU", "matplotlib_version": matplotlib.__version__,
               "moving_average_window": moving_window, "warnings": warnings,
               "figures": figures, "output_directory": str(output),
               "input_files": [{"path": str(p), "sha256": _hash(p), "bytes": p.stat().st_size} for p in input_paths],
               "generated_files": [{"path": str(p.relative_to(output)), "sha256": _hash(p), "bytes": p.stat().st_size}
                                   for p in sorted(output.rglob("*")) if p.is_file() and p.name not in {"FIGURE_SOURCES.json", "FIGURE_README.md"}],
               "script_sha256": _hash(Path(__file__))}
    _write_json(output / "FIGURE_SOURCES.json", summary)
    (output / "FIGURE_README.md").write_text(
        "# Phase 1 scientific figures\n\n"
        "All figures are standalone SVG and PNG artifacts. Their CSV/NPZ source arrays are in sources/. "
        "FIGURE_SOURCES.json maps figures to source arrays and hashes every completed output input. "
        "No training or new formal evaluation is performed by this module.\n\n"
        "The training replication unit is the training seed. Partial groups are marked n/5 and are interim descriptions; "
        "n<2 has no interval. Five-seed quality plots use two-sided 95% Student t intervals. "
        "FM_STANDARD quality and raw samples use the same-checkpoint Heun reference. "
        "SDE learning loss targets u+kappa*score and is not directly comparable to ODE learning losses.\n\n"
        "Learning curves use a causal arithmetic moving average of weighted batch losses, with the available prefix "
        "for the first points. Validation losses are the actual logged fixed validation batches. "
        "IID marginals and dependence plots are descriptive diagnostics; independent real blocks give a matched "
        "finite-sample correlation reference. Particle counts, projections, pooled observation values and ACF lags "
        "are not independent training repetitions.\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--tables-dir")
    parser.add_argument("--moving-window", type=int, default=100)
    args = parser.parse_args()
    if args.moving_window < 1:
        parser.error("--moving-window must be positive")
    summary = generate_figures(output_dir=args.output_dir, tables_dir=args.tables_dir, moving_window=args.moving_window)
    print(json.dumps({key: summary[key] for key in ("completed_seq_iid", "formal_denominator", "per_model_completed_training_seeds", "output_directory", "warnings")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
