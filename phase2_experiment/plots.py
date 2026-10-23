"""CPU figures from verified Phase 2 final artifacts; never train or evaluate.

Run reporting first, then this command, then reporting --package. No model is
built and no test NPZ is opened. Fixed reference arrays come from saved final
evaluation artifacts, and uncertainty uses independent training seeds only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .scope import DATASETS, MODELS, ROOT, SEEDS, active_attempt_path, completion_issues, load_manifest, now, read_json, sha256_file, write_json
from .reporting import COMPARISONS, moments, write_csv

LABELS = {"FM_STANDARD": "Standard (Heun)", "FM_REAL_ODE": "Real ODE", "FM_COMPLEX_SDE": "Complex SDE",
          "FM_COMPLEX_ODE": "Complex ODE", "FM_COMPLEX_VDC_ODE": "Complex VDC ODE"}
COLORS = {"FM_STANDARD": "#0072B2", "FM_REAL_ODE": "#D55E00", "FM_COMPLEX_SDE": "#009E73",
          "FM_COMPLEX_ODE": "#CC79A7", "FM_COMPLEX_VDC_ODE": "#E69F00"}
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666")


def moving_average(values: np.ndarray, window: int = 100) -> np.ndarray:
    if window < 1:
        raise ValueError("Smoothing window must be positive")
    values = np.asarray(values, dtype=np.float64)
    sums = np.r_[0., np.cumsum(values)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(end - window, 0)
    return (sums[end] - sums[start]) / (end - start)


def pointwise_seed_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """[training seed,...] pointwise t interval; no particle/projection repeats."""
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    if len(values) < 2:
        return mean, None, None
    from scipy.stats import t
    radius = float(t.ppf(.975, len(values) - 1)) * values.std(axis=0, ddof=1) / math.sqrt(len(values))
    return mean, mean - radius, mean + radius


def _records(dataset_id: str) -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    records, warnings, inputs = [], [], [ROOT / "PHASE2_MANIFEST.json", ROOT / "PHASE2_SCOPE.json"]
    for run in load_manifest()["runs"]:
        if run["dataset_id"] != dataset_id or run["status"] != "COMPLETED":
            continue
        issues = completion_issues(run)
        if issues:
            warnings.append(f"{run['run_id']} excluded: {'; '.join(issues)}")
            continue
        attempt = active_attempt_path(run)
        standard = run["model_id"] == "FM_STANDARD"
        metric_dir = attempt / "heun_metrics" if standard else attempt
        evaluation_path = attempt / ("evaluation_heun.json" if standard else "evaluation.json")
        paths = [attempt / "config.json", attempt / "training.csv", attempt / "training_cost.json", attempt / "sampling_diagnostics.json",
            evaluation_path, metric_dir / "metrics_arrays.npz", metric_dir / "samples_raw.npy", metric_dir / "metric_provenance.json",
            metric_dir / "ridge_diagnostics.json", metric_dir / "ridge_arrays.npz", metric_dir / "real_real_reference_blocks.npz"]
        absent = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
        if absent:
            warnings.append(f"{run['run_id']} lacks plot evidence: {absent}")
            continue
        evaluation = read_json(evaluation_path)
        metrics = evaluation.get("metrics", evaluation)
        if not all(math.isfinite(float(metrics[k])) for k in ("swd_raw", "mmd2_raw", "tstr_mse")):
            warnings.append(f"{run['run_id']} has nonfinite plot metrics")
            continue
        diagnostics = read_json(attempt / "sampling_diagnostics.json")
        cost = diagnostics["heun_reference" if standard else "primary"]
        training_cost = read_json(attempt / "training_cost.json")
        with np.load(metric_dir / "metrics_arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        records.append({"run": run, "attempt": attempt, "metric_dir": metric_dir, "metrics": metrics, "arrays": arrays,
            "training_seconds": float(training_cost["training_seconds"]), "sampling_seconds": float(cost["sampling_seconds"]),
            "solver": "Heun" if standard else "Euler-Maruyama" if run["model_id"] == "FM_COMPLEX_SDE" else "PL-history Euler"})
        inputs.extend(paths)
    records.sort(key=lambda r: (MODELS.index(r["run"]["model_id"]), r["run"]["seed"]))
    return records, warnings, list(dict.fromkeys(inputs))


def _export_sources(output: Path, stem: str, rows: list[dict[str, Any]], arrays: dict[str, np.ndarray], fields: list[str] | None = None) -> list[Path]:
    directory = output / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    csv_path, npz_path = directory / f"{stem}.csv", directory / f"{stem}.npz"
    write_csv(csv_path, rows, fields)
    np.savez_compressed(npz_path, **arrays)
    return [csv_path, npz_path]


def _error_point(ax, x, stat, color, marker="o", label=None):
    if not stat["n_training_seeds"]:
        return
    if stat["n_training_seeds"] >= 2:
        ax.errorbar(x, stat["mean"], yerr=[[stat["mean"] - stat["ci95_low"]], [stat["ci95_high"] - stat["mean"]]],
                    fmt=marker, color=color, capsize=3, markersize=5, label=label)
    else:
        ax.plot(x, stat["mean"], marker=marker, color=color, label=label)


def generate_figures(dataset_id: str | None = None, output_dir: Path | str | None = None, moving_window: int = 100) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from .data import data_dir

    selected = [dataset_id] if dataset_id else list(DATASETS)
    if any(d not in DATASETS for d in selected):
        raise ValueError("Plot dataset outside the fixed Phase 2 scope")
    output = Path(output_dir).resolve() if output_dir else ROOT / "review_pack_phase2_3datasets/figures"
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Figure output must stay within this workspace")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "svg.fonttype": "none", "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 8})
    figures, warnings, input_paths, dataset_states = [], [], [], {}

    def save(fig, dataset, stem, sources, note):
        full_stem = f"{dataset.lower()}__{stem}"
        fig.savefig(output / f"{full_stem}.svg", bbox_inches="tight")
        fig.savefig(output / f"{full_stem}.png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        figures.append({"dataset_id": dataset, "stem": full_stem, "svg": full_stem + ".svg", "png": full_stem + ".png",
            "source_files": [str(p.relative_to(output)) for p in sources], "interpretation": note})

    for dataset in selected:
        directory = Path(data_dir(dataset))
        registered_paths = [directory / "DATA_SPEC.json", directory / "METRIC_SPEC.json", directory / "metric_parameters.npz", directory / "stats.npz"]
        input_paths.extend(path for path in registered_paths if path.exists())
        spec = read_json(directory / "DATA_SPEC.json") if (directory / "DATA_SPEC.json").exists() else {}
        metric_spec = read_json(directory / "METRIC_SPEC.json") if (directory / "METRIC_SPEC.json").exists() else {}
        records, skipped, paths = _records(dataset)
        warnings.extend(skipped)
        input_paths.extend(paths)
        grouped = {model: [r for r in records if r["run"]["model_id"] == model] for model in MODELS}
        dataset_states[dataset] = {"verified_completed": len(records), "expected_formal_runs": 25,
            "models": {m: {"available_training_seeds": [r["run"]["seed"] for r in grouped[m]], "n_of_5": len(grouped[m])} for m in MODELS},
            "registered_data_spec": str(directory / "DATA_SPEC.json"), "length": spec.get("length"), "channels": spec.get("channels"),
            "state_dim": spec.get("state_dim"), "test_npz_opened_by_plotter": False}
        if not records:
            dataset_states[dataset]["status"] = "SKIPPED_NO_VERIFIED_COMPLETED_RUNS"
            continue
        dataset_states[dataset]["status"] = "FIGURES_FROM_FINAL_EVALUATIONS"
        prefix = dataset.lower()

        # Five separate learning figures preserve distinct ODE/SDE labels.
        for model, group in grouped.items():
            if not group:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
            rows, arrays = [], {}
            for rec in group:
                seed = rec["run"]["seed"]
                with (rec["attempt"] / "training.csv").open(newline="", encoding="utf-8-sig") as handle:
                    log = list(csv.DictReader(handle))
                steps = np.asarray([int(row["step"]) for row in log])
                loss = np.asarray([float(row["loss"]) for row in log])
                smooth = moving_average(loss, moving_window)
                mask = np.asarray([bool(row.get("val_loss")) for row in log])
                val = np.asarray([float(row["val_loss"]) for row in log if row.get("val_loss")])
                color = SEED_COLORS[SEEDS.index(seed)]
                stride = max(1, len(steps) // 1600)
                axes[0].plot(steps[::stride], smooth[::stride], color=color, alpha=.9, lw=1, label=f"seed {seed}")
                axes[1].plot(steps[mask], val, "o-", color=color, lw=1, ms=3, label=f"seed {seed}")
                arrays.update({f"seed{seed}_steps": steps, f"seed{seed}_batch_loss": loss, f"seed{seed}_moving_mean": smooth,
                    f"seed{seed}_validation_steps": steps[mask], f"seed{seed}_validation_loss": val})
                rows.extend({"dataset_id": dataset, "model_id": model, "seed": seed, "step": int(step), "batch_loss": float(raw),
                    "causal_moving_mean": float(mean), "validation_loss": row.get("val_loss", "")} for step, raw, mean, row in zip(steps, loss, smooth, log))
            for ax in axes:
                ax.set_xlabel("Optimizer update")
                ax.set_ylabel("Weighted drift regression loss")
                ax.set_yscale("log")
                ax.grid(alpha=.2)
                ax.legend(ncol=2)
            axes[1].legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -.23), borderaxespad=0)
            axes[0].set_title(f"Train: causal mean of up to {moving_window} updates")
            axes[1].set_title("Fixed validation protocol")
            fig.suptitle(f"{dataset} · {LABELS[model]} · n={len(group)}/5 training seeds")
            fig.tight_layout()
            sources = _export_sources(output, f"{prefix}__learning_{model.lower()}", rows, arrays)
            save(fig, dataset, f"learning_{model.lower()}", sources, "Full numeric curves retained. ODE and complete-SDE drift losses are different targets and are not ranked across model families. No convergence claim from reaching budget.")

        # Distribution metrics: raw points and seed-level Student t intervals.
        quality_metrics = [("swd_raw", "SWD to raw real reference"), ("mmd2_raw", "MMD² to raw real reference"),
                           ("swd_smooth", "SWD to shared smoothed reference"), ("mmd2_smooth", "MMD² to shared smoothed reference")]
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.1))
        rows, arrays = [], {}
        for ax, (key, label) in zip(axes.flat, quality_metrics):
            for index, model in enumerate(MODELS):
                group = grouped[model]
                values = [float(r["metrics"][key]) for r in group]
                offsets = np.linspace(-.16, .16, len(group)) if len(group) > 1 else np.zeros(len(group))
                ax.scatter(index + offsets, values, s=19, color=COLORS[model], alpha=.65)
                stat = moments(values)
                _error_point(ax, index, stat, COLORS[model])
                arrays[f"{key}_{model}_values"] = np.asarray(values)
                rows.extend({"dataset_id": dataset, "model_id": model, "seed": r["run"]["seed"], "metric": key,
                    "value": value, "solver": r["solver"], **stat} for r, value in zip(group, values))
            if key.startswith("mmd2"):
                ax.axhline(0, color="#777777", lw=.7, linestyle="--")
            ax.set_title(label)
            ax.set_xticks(range(5), [f"{LABELS[m]}\nn={len(grouped[m])}/5" for m in MODELS], rotation=20, ha="right")
            ax.set_xlim(-.4, len(MODELS) - .6)
            ax.set_ylabel(key)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"{dataset} · quality · mean and 95% training-seed t interval")
        fig.tight_layout()
        sources = _export_sources(output, f"{prefix}__quality", rows, arrays)
        save(fig, dataset, "quality", sources, "Points are individual training seeds; intervals use training seeds only. Negative MMD estimates are retained. Standard FM uses same-checkpoint Heun. Partial groups are explicitly n/5.")

        # Each point is one actual run, with mean/CI in both quality and cost.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
        rows, arrays = [], {}
        for ax, cost_key, unit, divisor in ((axes[0], "training_seconds", "Training wall-clock hours", 3600),
                                           (axes[1], "sampling_seconds", "Primary-solver sampling seconds", 1)):
            for model, group in grouped.items():
                if not group:
                    continue
                x = np.asarray([r[cost_key] / divisor for r in group])
                y = np.asarray([r["metrics"]["swd_raw"] for r in group])
                ax.scatter(x, y, color=COLORS[model], s=21, alpha=.55)
                xm, ym = moments(x.tolist()), moments(y.tolist())
                error = {}
                if len(group) >= 2:
                    error = {"xerr": [[xm["mean"] - xm["ci95_low"]], [xm["ci95_high"] - xm["mean"]]],
                             "yerr": [[ym["mean"] - ym["ci95_low"]], [ym["ci95_high"] - ym["mean"]]]}
                ax.errorbar(xm["mean"], ym["mean"], fmt="o", color=COLORS[model], capsize=3, ms=6,
                    label=f"{LABELS[model]} (n={len(group)}/5)", **error)
                arrays[f"{cost_key}_{model}"] = x
                arrays[f"swd_{cost_key}_{model}"] = y
                rows.extend({"dataset_id": dataset, "model_id": model, "seed": r["run"]["seed"], "cost_quantity": cost_key,
                    "cost_original_seconds": r[cost_key], "display_divisor": divisor, "swd_raw": r["metrics"]["swd_raw"], "solver": r["solver"]} for r in group)
            ax.set_xlabel(unit)
            ax.set_ylabel("SWD to raw real reference")
            ax.grid(alpha=.2)
        axes[1].legend(loc="best", fontsize=8)
        fig.suptitle(f"{dataset} · measured cost and quality · training-seed intervals")
        fig.tight_layout()
        sources = _export_sources(output, f"{prefix}__quality_cost", rows, arrays)
        save(fig, dataset, "quality_cost", sources, "Actual per-run training and primary-solver sampling wall-clock costs; mean and marginal 95% t intervals across training seeds. Same-checkpoint diagnostic cost is separate in raw tables.")

        # First available run's saved real reference is fixed, not five repeats.
        first = records[0]
        reference = first["arrays"]
        with np.load(directory / "metric_parameters.npz", allow_pickle=False) as archive:
            early, late = archive["early_indices"].copy(), archive["late_indices"].copy()
        with np.load(directory / "stats.npz", allow_pickle=False) as archive:
            mean, scale = archive["mean"].copy(), archive["scale"].copy()

        # No original test NPZ is opened: illustrations use saved reference blocks.
        with np.load(first["metric_dir"] / "real_real_reference_blocks.npz", allow_pickle=False) as archive:
            fixed_reference = archive["raw_standardized"][0, 0, :4].astype(np.float64)
        real_sequences = (fixed_reference * scale + mean).reshape(-1, 64, 2)
        columns = [("Fixed real block", real_sequences, None)]
        rows, arrays = [], {"fixed_real_examples": real_sequences}
        for model in MODELS:
            group = grouped[model]
            if group:
                rec = group[0]  # lowest available seed, fixed first particles
                raw = np.load(rec["metric_dir"] / "samples_raw.npy", mmap_mode="r", allow_pickle=False)
                # Detach the small illustration slice before closing the Windows
                # file mapping; no plotted or exported array should own a handle.
                example = np.array(raw[0, :4], copy=True)
                raw._mmap.close()
                del raw
                columns.append((f"{LABELS[model]}\nseed {rec['run']['seed']}", example, model))
                arrays[f"{model}_seed{rec['run']['seed']}_examples"] = example
        fig, axes = plt.subplots(2, len(columns), figsize=(3.05 * len(columns), 5.25), squeeze=False, sharex=True, sharey="row")
        input_positions = metric_spec.get("downstream_task", {}).get("input_positions", [])
        output_positions = metric_spec.get("downstream_task", {}).get("output_positions", [])
        for index, (label, sequences, model) in enumerate(columns):
            for channel in range(2):
                ax = axes[channel, index]
                for sample_index, sequence in enumerate(sequences):
                    ax.plot(np.arange(64), sequence[:, channel], lw=.9, alpha=.72)
                    rows.extend({"dataset_id": dataset, "series": label.replace("\n", " "), "channel": channel,
                        "example_index": sample_index, "observation_position": position, "raw_value": float(value)} for position, value in enumerate(sequence[:, channel]))
                for positions, color in ((input_positions, "#0072B2"), (output_positions, "#D55E00")):
                    if positions:
                        ax.axvspan(min(positions) - .5, max(positions) + .5, color=color, alpha=.055)
                ax.set_title(label if channel == 0 else f"Channel {channel}")
                ax.set_xlabel("Observation position s")
                if index == 0:
                    ax.set_ylabel(f"Channel {channel}, raw units")
                ax.grid(alpha=.15)
        fig.suptitle(f"{dataset} · fixed representative complete two-channel sequences")
        fig.tight_layout()
        sources = _export_sources(output, f"{prefix}__sequence_examples", rows, arrays)
        save(fig, dataset, "sequence_examples", sources, "Representative rule: lowest available training seed, first ensemble, first four particles; fixed saved real-reference block. Examples are illustrative, not paired prediction errors. Blue/red spans identify registered downstream input/output observation windows.")

        # ACF and PSD stay on the observation axis of each physical data channel.
        for kind, reference_key, x_key, xlabel in (("acf", "reference_acf", None, "Observation lag"),
                                                  ("psd", "reference_psd", "psd_frequency", "Cycles per observation interval")):
            fig, axes = plt.subplots(1, 2, figsize=(11.1, 3.9))
            x = np.arange(1, 33) if kind == "acf" else reference[x_key]
            rows, arrays = [], {"x": x, "fixed_reference": reference[reference_key]}
            for channel, ax in enumerate(axes):
                fixed = reference[reference_key][channel] if kind == "acf" else reference[reference_key][:, channel]
                ax.plot(x, fixed, color="#202020", lw=1.8, linestyle="--", label="Fixed real reference")
                rows.extend({"dataset_id": dataset, "model_id": "FIXED_REAL_REFERENCE", "seed": "", "channel": channel,
                    "x": float(coordinate), "value": float(value)} for coordinate, value in zip(x, fixed))
                for model, group in grouped.items():
                    if not group:
                        continue
                    values = np.stack([r["arrays"][kind][channel] if kind == "acf" else r["arrays"][kind][:, channel] for r in group])
                    avg, low, high = pointwise_seed_interval(values)
                    ax.plot(x, avg, color=COLORS[model], lw=1.2, label=f"{LABELS[model]} n={len(group)}/5")
                    if low is not None:
                        ax.fill_between(x, low, high, color=COLORS[model], alpha=.10)
                    arrays[f"channel{channel}_{model}_per_seed"] = values
                    rows.extend({"dataset_id": dataset, "model_id": model, "seed": r["run"]["seed"], "channel": channel,
                        "x": float(coordinate), "value": float(value)} for r, vector in zip(group, values) for coordinate, value in zip(x, vector))
                ax.set_title(f"Channel {channel}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Position correlation" if kind == "acf" else "One-sided raw-channel power")
                if kind == "psd":
                    ax.set_yscale("symlog", linthresh=1e-10)
                ax.grid(alpha=.2)
            axes[1].legend(loc="best", fontsize=7)
            fig.suptitle(f"{dataset} · {'within-channel ACF' if kind == 'acf' else 'time-axis PSD'} · pointwise 95% seed t intervals")
            fig.tight_layout()
            sources = _export_sources(output, f"{prefix}__{kind}", rows, arrays)
            save(fig, dataset, kind, sources, "Saved evaluation arrays use correct [observation position, channel] axes. Bands are pointwise training-seed intervals, not simultaneous bands or particle-count intervals. Fixed real reference is not treated as repeated seed evidence.")

        fig, axes = plt.subplots(1, 2, figsize=(11.1, 3.9))
        rows, arrays = [], {"lags": np.arange(33), "fixed_reference": reference["reference_cross_channel_lag_correlation"]}
        for ax, (c1, c2) in zip(axes, ((0, 1), (1, 0))):
            fixed = reference["reference_cross_channel_lag_correlation"][c1, c2]
            ax.plot(np.arange(33), fixed, "--", color="#202020", lw=1.8, label="Fixed real reference")
            rows.extend({"model_id": "FIXED_REAL_REFERENCE", "seed": "", "first_channel": c1, "second_channel": c2,
                "lag": lag, "correlation": float(value)} for lag, value in enumerate(fixed))
            for model, group in grouped.items():
                if not group:
                    continue
                values = np.stack([r["arrays"]["cross_channel_lag_correlation"][c1, c2] for r in group])
                avg, low, high = pointwise_seed_interval(values)
                ax.plot(np.arange(33), avg, color=COLORS[model], label=f"{LABELS[model]} n={len(group)}/5")
                if low is not None:
                    ax.fill_between(np.arange(33), low, high, color=COLORS[model], alpha=.10)
                arrays[f"channels{c1}_{c2}_{model}_per_seed"] = values
                rows.extend({"model_id": model, "seed": r["run"]["seed"], "first_channel": c1, "second_channel": c2,
                    "lag": lag, "correlation": float(value)} for r, vector in zip(group, values) for lag, value in enumerate(vector))
            ax.set_title(f"corr[channel {c1} at s, channel {c2} at s + lag]")
            ax.set_xlabel("Observation lag")
            ax.set_ylabel("Position correlation")
            ax.grid(alpha=.2)
        axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -.22), borderaxespad=0)
        fig.suptitle(f"{dataset} · directed cross-channel lag structure")
        fig.tight_layout()
        sources = _export_sources(output, f"{prefix}__cross_channel_lags", rows, arrays)
        save(fig, dataset, "cross_channel_lags", sources, "Directed cross-channel correlations preserve observation-time alignment. Pointwise bands aggregate independent training seeds only.")

        # Registered early/late matrix exists for all tasks; no SSM residual invented.
        matrices = [("Fixed real reference", reference["reference_correlation"][np.ix_(early, late)], None)]
        rows, arrays = [], {"early_flat_indices": early, "late_flat_indices": late}
        for model, group in grouped.items():
            if group:
                values = np.stack([r["arrays"]["correlation"][np.ix_(early, late)] for r in group])
                matrices.append((f"{LABELS[model]}\nn={len(group)}/5", values.mean(axis=0), model))
                arrays[f"{model}_per_seed_early_late_correlation"] = values
                rows.extend({"model_id": model, "seed": r["run"]["seed"], "early_flat_index": int(e), "late_flat_index": int(l),
                    "early_observation": int(e // 2), "early_channel": int(e % 2), "late_observation": int(l // 2), "late_channel": int(l % 2),
                    "correlation": float(matrix[i, j])} for r, matrix in zip(group, values) for i, e in enumerate(early) for j, l in enumerate(late))
        arrays["fixed_real_early_late_correlation"] = matrices[0][1]
        rows.extend({"model_id": "FIXED_REAL_REFERENCE", "seed": "", "early_flat_index": int(e), "late_flat_index": int(l),
            "early_observation": int(e // 2), "early_channel": int(e % 2), "late_observation": int(l // 2), "late_channel": int(l % 2),
            "correlation": float(matrices[0][1][i, j])} for i, e in enumerate(early) for j, l in enumerate(late))
        fig, axes = plt.subplots(1, len(matrices), figsize=(3.05 * len(matrices), 3.8), squeeze=False)
        for ax, (label, matrix, model) in zip(axes.flat, matrices):
            image = ax.imshow(matrix.T, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", aspect="auto")
            ax.set_title(label)
            ax.set_ylabel("Registered late coordinate index")
        fig.suptitle(f"{dataset} · registered early/late dependence (C-order coordinates)")
        fig.supxlabel("Registered early coordinate index", y=.04, fontsize=9)
        fig.subplots_adjust(top=.78, bottom=.2, wspace=.36, right=.90)
        fig.colorbar(image, ax=list(axes.flat), label="Position correlation", shrink=.8)
        sources = _export_sources(output, f"{prefix}__early_late_correlation", rows, arrays)
        save(fig, dataset, "early_late_correlation", sources, "Mean saved correlation matrix over available independent training seeds; coordinates map s*2+channel as exported. Multiscale has no invented deterministic residual target.")

        if dataset in ("SEQ_DELAY", "SEQ_ACCUM"):
            fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.1))
            rows, arrays = [], {}
            for index, model in enumerate(MODELS):
                group = grouped[model]
                for key, shift, marker, linestyle in (("generated_paired_structure_mse_raw", -.12, "o", "paired"),
                    ("generated_shuffled_structure_mse_raw", .12, "s", "late shuffled")):
                    values = [r["metrics"][key] for r in group]
                    _error_point(axes[0], index + shift, moments(values), COLORS[model], marker=marker)
                    arrays[f"{key}_{model}"] = np.asarray(values)
                    rows.extend({"model_id": model, "seed": r["run"]["seed"], "metric": key, "value": value} for r, value in zip(group, values))
                for key, shift, marker in (("early_late_joint_swd_raw", -.12, "o"), ("generated_shuffled_joint_swd_raw", .12, "s")):
                    values = [r["metrics"][key] for r in group]
                    _error_point(axes[1], index + shift, moments(values), COLORS[model], marker=marker)
                    arrays[f"{key}_{model}"] = np.asarray(values)
                    rows.extend({"model_id": model, "seed": r["run"]["seed"], "metric": key, "value": value} for r, value in zip(group, values))
            for key, ax in (("real_paired_structure_mse_raw", axes[0]), ("true_shuffled_joint_swd_raw", axes[1])):
                value = float(first["metrics"][key])
                ax.axhline(value, color="#333333", linestyle="--", label=f"Fixed reference: {key}")
                arrays[key] = np.asarray(value)
                rows.append({"model_id": "FIXED_REAL_REFERENCE", "seed": "", "metric": key, "value": value})
            for ax in axes:
                ax.plot([], [], "o", color="#555555", label="Original pairing")
                ax.plot([], [], "s", color="#555555", label="Late pairing destroyed")
                ax.set_xticks(range(5), [f"{LABELS[m]}\nn={len(grouped[m])}/5" for m in MODELS], rotation=20, ha="right")
                ax.set_xlim(-.4, len(MODELS) - .6)
                ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -.35), borderaxespad=0)
                ax.grid(axis="y", alpha=.2)
            axes[0].set_title("Registered structural residual")
            axes[0].set_ylabel("Residual mean square, raw units")
            axes[1].set_title("Joint early/late distribution")
            axes[1].set_ylabel("Joint SWD to real paired reference")
            fig.suptitle(f"{dataset} · pairing and dependency-breaking control · seed t intervals")
            fig.tight_layout()
            sources = _export_sources(output, f"{prefix}__pairing_structure", rows, arrays)
            save(fig, dataset, "pairing_structure", sources, "Existing evaluation metrics only: late re-pairing preserves marginal samples and breaks early/late dependence. Same-seed original versus shuffled numbers are retained; reference lines are fixed, not repeated independent seeds.")

        fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.1))
        rows, arrays = [], {}
        for ax, key in zip(axes, ("tstr_mse", "tstr_raw_mse")):
            for index, model in enumerate(MODELS):
                group = grouped[model]
                values = [r["metrics"][key] for r in group]
                _error_point(ax, index, moments(values), COLORS[model])
                ax.scatter(index + (np.linspace(-.14, .14, len(group)) if len(group) > 1 else np.zeros(len(group))), values, color=COLORS[model], alpha=.6, s=18)
                arrays[f"{key}_{model}"] = np.asarray(values)
                rows.extend({"model_id": model, "seed": r["run"]["seed"], "metric": key, "value": value,
                    "n_training_seeds": len(group), "expected_training_seeds": 5} for r, value in zip(group, values))
            suffix = "raw_mse" if key.endswith("raw_mse") else "mse"
            for label, color, style in (("trtr_matched", "#333333", "--"), ("trtr_full", "#777777", ":")):
                value = float(first["metrics"][f"{label}_{suffix}"])
                ax.axhline(value, color=color, linestyle=style, label=f"Fixed {label.replace('_', ' ')}")
                rows.append({"model_id": "FIXED_REAL_REFERENCE", "seed": "", "metric": f"{label}_{suffix}", "value": value,
                    "n_training_seeds": "", "expected_training_seeds": ""})
                arrays[f"{label}_{suffix}"] = np.asarray(value)
            ax.set_xticks(range(5), [f"{LABELS[m]}\nn={len(grouped[m])}/5" for m in MODELS], rotation=20, ha="right")
            ax.set_xlim(-.4, len(MODELS) - .6)
            ax.set_ylabel("Real-test prediction MSE,\nraw units" if suffix == "raw_mse" else "Real-test prediction MSE,\nshared standardized units")
            ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.35), borderaxespad=0)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"{dataset} · TSTR versus matched-count / full-count TRTR")
        fig.tight_layout()
        sources = _export_sources(output, f"{prefix}__tstr_trtr", rows, arrays)
        save(fig, dataset, "tstr_trtr", sources, "Read existing final ridge evaluation, never fit or select a new model. Lambda was selected only using shared real validation. TSTR intervals use independent generator training seeds; repeated fixed TRTR baselines have no seed-based uncertainty claim.")

        # Paired forest: only actual matched seeds, no filled missing comparisons.
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.1))
        rows, arrays = [], {}
        lookup = {(r["run"]["model_id"], r["run"]["seed"]): r for r in records}
        any_pairs = False
        pair_labels = [f"{LABELS[m]} − {LABELS[r]}" for m, r in COMPARISONS]
        for ax, metric in zip(axes, ("swd_raw", "mmd2_raw")):
            for index, (model, reference_model) in enumerate(COMPARISONS):
                seeds = [seed for seed in SEEDS if (model, seed) in lookup and (reference_model, seed) in lookup]
                differences = [lookup[(model, seed)]["metrics"][metric] - lookup[(reference_model, seed)]["metrics"][metric] for seed in seeds]
                if not differences:
                    continue
                any_pairs = True
                stat = moments(differences)
                if len(seeds) >= 2:
                    ax.errorbar(stat["mean"], index, xerr=[[stat["mean"] - stat["ci95_low"]], [stat["ci95_high"] - stat["mean"]]],
                        fmt="o", color=COLORS[model], capsize=3)
                else:
                    ax.plot(stat["mean"], index, "o", color=COLORS[model])
                ax.annotate(f"n={len(seeds)}/5", (stat["mean"], index), xytext=(5, 5), textcoords="offset points", fontsize=7)
                arrays[f"{metric}_{model}_minus_{reference_model}"] = np.asarray(differences)
                rows.extend({"dataset_id": dataset, "model_id": model, "reference_model_id": reference_model, "seed": seed,
                    "metric": metric, "paired_difference": difference, **stat} for seed, difference in zip(seeds, differences))
            ax.axvline(0, color="#666666", lw=.8, linestyle="--")
            ax.set_yticks(range(7), pair_labels)
            ax.invert_yaxis()
            ax.set_title(metric)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useOffset=False)
            ax.set_xlabel("Model minus specified reference\nn=1: point only; n≥2: 95% seed t interval")
            ax.grid(axis="x", alpha=.2)
        if any_pairs:
            fig.suptitle(f"{dataset} · seven registered paired training-seed contrasts")
            fig.tight_layout()
            sources = _export_sources(output, f"{prefix}__paired_contrasts", rows, arrays)
            save(fig, dataset, "paired_contrasts", sources, "Raw 95% intervals over available paired training seeds, separate from Holm-adjusted tests in paired_comparisons.csv. Missing seed pairs are not imputed. An interval crossing zero does not prove equivalence.")
        else:
            plt.close(fig)

    input_paths.append(Path(__file__))
    unique_inputs = list(dict.fromkeys(path.resolve() for path in input_paths if path.is_file()))
    source_hashes = [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in unique_inputs]
    write_json(output / "FIGURE_SOURCE_HASHES.json", source_hashes)
    exported = []
    for item in figures:
        for name in [item["svg"], item["png"], *item["source_files"]]:
            path = output / name
            exported.append({"path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(output / "FIGURE_FILE_HASHES.json", exported)
    result = {"generated_at": now(), "status": "FIGURES_GENERATED" if figures else "SKIPPED_NO_VERIFIED_COMPLETED_RUNS",
        "datasets": dataset_states, "figure_count": len(figures), "figures": figures, "warnings": warnings,
        "training_updates": 0, "models_built": 0, "test_npz_opened": False, "protocol_modified": False,
        "uncertainty_unit": "independent training seeds; pointwise t intervals, no particle/projection repeat count", "output_dir": str(output)}
    write_json(output / "FIGURE_MANIFEST.json", result)
    (output / "README.md").write_text("# Phase 2 scientific figures\n\nEach figure has SVG/PNG and numerical CSV/NPZ sources. FIGURE_SOURCE_HASHES.json hashes original inputs; FIGURE_FILE_HASHES.json hashes exported images/arrays. Only artifact-verified COMPLETED runs are plotted, with each dataset separate and partial n/5 explicit. No training or test-NPZ evaluation occurs. Standard FM uses Heun. Fixed references are shared data, not independent training-seed repeats. Run reporting, then plots, then reporting --package to include the figures.\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--output-dir")
    parser.add_argument("--moving-window", type=int, default=100)
    args = parser.parse_args()
    result = generate_figures(args.dataset, args.output_dir, args.moving_window)
    print(json.dumps({k: v for k, v in result.items() if k != "figures"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
