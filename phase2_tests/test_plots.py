"""Plot-only correctness: no test evaluation, seed intervals and full exports."""
import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from phase2_experiment import plots
from phase2_experiment.scope import ROOT, write_json


def test_causal_plot_smoothing_and_seed_intervals():
    assert plots.moving_average(np.array([2., 4., 8.]), 2) == pytest.approx([2., 3., 6.])
    mean, low, high = plots.pointwise_seed_interval(np.array([[1., 2.]]))
    assert mean == pytest.approx([1., 2.]) and low is None and high is None
    mean, low, high = plots.pointwise_seed_interval(np.array([[1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.]]))
    assert mean == pytest.approx([1., 2.]) and low == pytest.approx(mean) and high == pytest.approx(mean)


def test_no_completed_plotting_does_not_open_test_npz(monkeypatch):
    parent = ROOT / ".phase2_test_tmp"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plots_empty_", dir=parent) as temporary:
        monkeypatch.setattr(plots, "_records", lambda dataset: ([], [], []))
        original = np.load
        opened = []
        def checked_load(path, *args, **kwargs):
            opened.append(str(path))
            assert Path(path).name != "test.npz"
            return original(path, *args, **kwargs)
        monkeypatch.setattr(np, "load", checked_load)
        result = plots.generate_figures("SEQ_DELAY", Path(temporary) / "figures")
        assert result["status"] == "SKIPPED_NO_VERIFIED_COMPLETED_RUNS"
        assert result["figure_count"] == result["models_built"] == result["training_updates"] == 0
        assert not result["test_npz_opened"] and not opened


def test_partial_final_artifact_figures_export_csv_npz_svg_png(monkeypatch):
    """Synthetic plotting fixtures stay in temporary folders, never formal scope."""
    parent = ROOT / ".phase2_test_tmp"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plots_fixture_", dir=parent) as temporary:
        base = Path(temporary)
        records = []
        for model in ("FM_STANDARD", "FM_REAL_ODE"):
            for seed in (17, 29):
                attempt = base / f"{model}_{seed}"
                metric_dir = attempt / "heun_metrics" if model == "FM_STANDARD" else attempt
                metric_dir.mkdir(parents=True)
                with (attempt / "training.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["step", "loss", "val_loss"])
                    writer.writerows([(1, 2., ""), (2, 1., .8)])
                np.save(metric_dir / "samples_raw.npy", np.zeros((2, 4, 64, 2)))
                np.savez(metric_dir / "real_real_reference_blocks.npz", raw_standardized=np.zeros((2, 2, 4, 128)))
                array = {"correlation": np.eye(128), "reference_correlation": np.eye(128), "acf": np.zeros((2, 32)),
                    "reference_acf": np.zeros((2, 32)), "psd": np.ones((33, 2)), "reference_psd": np.ones((33, 2)),
                    "psd_frequency": np.fft.rfftfreq(64), "cross_channel_lag_correlation": np.zeros((2, 2, 33)),
                    "reference_cross_channel_lag_correlation": np.zeros((2, 2, 33))}
                keys = ["swd_raw", "swd_smooth", "mmd2_raw", "mmd2_smooth", "generated_paired_structure_mse_raw",
                    "generated_shuffled_structure_mse_raw", "early_late_joint_swd_raw", "generated_shuffled_joint_swd_raw",
                    "real_paired_structure_mse_raw", "true_shuffled_joint_swd_raw", "tstr_mse", "tstr_raw_mse",
                    "trtr_matched_mse", "trtr_full_mse", "trtr_matched_raw_mse", "trtr_full_raw_mse"]
                metrics = {key: .1 + seed / 1000 for key in keys}
                records.append({"run": {"dataset_id": "SEQ_DELAY", "model_id": model, "seed": seed}, "attempt": attempt,
                    "metric_dir": metric_dir, "arrays": array, "metrics": metrics, "training_seconds": float(seed),
                    "sampling_seconds": seed / 10, "solver": "fixture"})
        monkeypatch.setattr(plots, "_records", lambda dataset: (records, [], []))
        result = plots.generate_figures("SEQ_DELAY", base / "figures", moving_window=2)
        assert result["figure_count"] == 12
        assert result["datasets"]["SEQ_DELAY"]["models"]["FM_STANDARD"]["n_of_5"] == 2
        assert result["training_updates"] == result["models_built"] == 0 and not result["test_npz_opened"]
        for figure in result["figures"]:
            assert (base / "figures" / figure["svg"]).is_file()
            assert (base / "figures" / figure["png"]).is_file()
            assert len(figure["source_files"]) == 2
            assert all((base / "figures" / name).exists() for name in figure["source_files"])
