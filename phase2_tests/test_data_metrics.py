"""Independent synthetic correctness checks; no formal data/test or GPU read."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from phase2_experiment.data import flatten, unflatten, load_split, generate_raw, task_parameters
from phase2_experiment.metrics import (channel_correlation_acf, power_spectrum,
    structural_residual, shuffle_generated_late, select_ridge, pilot_validation_diagnostics, evaluate_run)


class Phase2DataMetricTests(unittest.TestCase):
    def test_complete_sample_layout_and_named_only_loader(self):
        raw = np.arange(3 * 64 * 2, dtype=np.float32).reshape(3, 64, 2)
        np.testing.assert_array_equal(unflatten(flatten(raw)), raw)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "data" / "phase2_3datasets" / "seq_delay"
            directory.mkdir(parents=True)
            np.savez(directory / "train.npz", standardized=flatten(raw))
            actual = load_split("SEQ_DELAY", "train", root=temporary)
            np.testing.assert_array_equal(actual.numpy(), flatten(raw))
            self.assertFalse((directory / "test.npz").exists())

    def test_channel_axis_acf_does_not_interleave_channels(self):
        first = np.array([-2., -1., 1., 2.])
        second = np.array([1., -2., 2., -1.])
        raw = np.stack([np.tile(first[:, None], (1, 64)), np.tile(second[:, None], (1, 64))], axis=-1)
        corr, acf, cross = channel_correlation_acf(raw)
        np.testing.assert_allclose(acf, 1., atol=1e-14)
        np.testing.assert_allclose(cross[0, 1], 0., atol=1e-14)
        self.assertEqual(corr.shape, (128, 128))

    def test_periodogram_frequency_and_parseval(self):
        time = np.arange(64)
        raw = np.tile(np.stack([np.sin(2 * np.pi * time / 8), np.cos(2 * np.pi * time / 16)], axis=-1)[None], (4, 1, 1))
        spectrum = power_spectrum(raw)
        np.testing.assert_array_equal(np.argmax(spectrum, axis=0), [8, 4])
        np.testing.assert_allclose(spectrum.sum(axis=0) / 64, [.5, .5], rtol=0, atol=1e-14)

    def test_delay_registered_matrix_noise_and_pair_destruction(self):
        raw = generate_raw("SEQ_DELAY", 2000, np.random.default_rng(123))
        matrix = np.array([[1., .25], [-.2, .9]])
        residual = raw[:, 16:24].astype(float) - raw[:, :8].astype(float) @ matrix.T
        self.assertAlmostEqual(float(np.mean(residual ** 2)), .15 ** 2, delta=.0007)
        np.testing.assert_allclose(structural_residual(raw.astype(float), "SEQ_DELAY"), residual)
        shaped = raw.reshape(4, 500, 64, 2)
        shuffled = shuffle_generated_late(shaped, "SEQ_DELAY", 193).reshape(-1, 64, 2)
        for channel in range(2):
            np.testing.assert_array_equal(np.sort(shuffled[:, 16:24, channel].reshape(-1)), np.sort(raw[:, 16:24, channel].reshape(-1)))
        self.assertGreater(float(np.mean(structural_residual(shuffled, "SEQ_DELAY") ** 2)), 1.)
        np.testing.assert_array_equal(shuffled[:, :8], raw[:, :8])

    def test_accum_registered_sum_scale_tanh_and_noise(self):
        raw = generate_raw("SEQ_ACCUM", 2000, np.random.default_rng(124)).astype(float)
        matrix = np.array([[1., .5], [-.4, .9]])
        signal = 1.5 * np.tanh((raw[:, :16].sum(axis=1) / np.sqrt(16)) @ matrix.T)
        residual = raw[:, 48:] - signal[:, None]
        self.assertAlmostEqual(float(np.mean(residual ** 2)), .15 ** 2, delta=.0005)
        np.testing.assert_allclose(structural_residual(raw, "SEQ_ACCUM"), residual, atol=1e-15)
        shuffled = shuffle_generated_late(raw.reshape(4, 500, 64, 2), "SEQ_ACCUM", 194).reshape(-1, 64, 2)
        self.assertGreater(float(np.mean(structural_residual(shuffled, "SEQ_ACCUM") ** 2)), .8)

    def test_ssm_stationary_covariance_after_independent_burn_in(self):
        parameters = task_parameters("SEQ_MULTISCALE")
        transition, observation = np.asarray(parameters["transition"]), np.asarray(parameters["observation"])
        self.assertLess(float(np.abs(np.linalg.eigvals(transition)).max()), 1.)
        q = np.diag(np.asarray(parameters["process_noise_standard_deviations"]) ** 2)
        latent_covariance = solve_discrete_lyapunov(transition, q)
        expected = observation @ latent_covariance @ observation.T + .1 ** 2 * np.eye(2)
        raw = generate_raw("SEQ_MULTISCALE", 1500, np.random.default_rng(125)).astype(float)
        np.testing.assert_allclose(np.cov(raw[:, 0].T, ddof=0), expected, rtol=.10, atol=.015)
        np.testing.assert_allclose(np.cov(raw.reshape(-1, 2).T, ddof=0), expected, rtol=.10, atol=.015)
        self.assertEqual(raw.shape, (1500, 64, 2))

    def test_ridge_validation_selection_without_test_argument(self):
        rng = np.random.default_rng(149)
        weights = np.array([[1., -.3], [.4, .8], [-.2, .7]])
        x = rng.normal(size=(100, 3)); y = x @ weights + np.array([.5, -.2])
        xv = rng.normal(size=(40, 3)); yv = xv @ weights + np.array([.5, -.2])
        grid = np.array([1e-8, .01, 10.])
        learned, intercept, selected, scores = select_ridge(x, y, xv, yv, grid)
        self.assertEqual(selected, 1e-8)
        self.assertLess(float(np.mean((xv @ learned + intercept - yv) ** 2)), 1e-14)
        self.assertEqual(scores.shape, (3,))

    def test_pilot_diagnostics_operates_when_test_does_not_exist(self):
        rng = np.random.default_rng(150)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "data" / "phase2_3datasets" / "seq_delay"
            directory.mkdir(parents=True)
            raw = generate_raw("SEQ_DELAY", 80, rng).astype(float)
            np.savez(directory / "validation.npz", standardized=raw.reshape(80, 128), raw=raw)
            np.savez(directory / "stats.npz", mean=np.zeros(128), scale=np.ones(128))
            directions = rng.normal(size=(256, 128)); directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            joint_directions = rng.normal(size=(256, 32)); joint_directions /= np.linalg.norm(joint_directions, axis=1, keepdims=True)
            early = np.arange(16); late = np.arange(32, 48)
            np.savez(directory / "metric_parameters.npz", directions=directions, joint_directions=joint_directions, early_indices=early, late_indices=late)
            result = pilot_validation_diagnostics(raw[:16].reshape(2, 8, 128), "SEQ_DELAY", root=temporary)
            self.assertFalse(result["test_accessed"])
            self.assertTrue(result["small_sample_nonformal"])
            self.assertTrue(np.isfinite(result["validation_swd_raw"]))
            self.assertFalse((directory / "test.npz").exists())

    def test_formal_evaluator_three_isolated_fixtures_and_matched_ridge(self):
        for dataset in ("SEQ_DELAY", "SEQ_ACCUM", "SEQ_MULTISCALE"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                rng = np.random.default_rng(171)
                directory = Path(temporary) / "data" / "phase2_3datasets" / dataset.lower()
                directory.mkdir(parents=True)
                raw_splits = {name: generate_raw(dataset, count, rng).astype(float)
                              for name, count in (("train", 80), ("validation", 40), ("test", 64))}
                for name, raw in raw_splits.items():
                    np.savez(directory / f"{name}.npz", standardized=raw.reshape(len(raw), 128), raw=raw)
                np.savez(directory / "stats.npz", mean=np.zeros(128), scale=np.ones(128))
                parameters = task_parameters(dataset)
                early_positions = parameters.get("early_positions", parameters.get("downstream_input_positions"))
                late_positions = parameters.get("late_positions", parameters.get("downstream_output_positions"))
                early = np.asarray([2*s+c for s in early_positions for c in range(2)])
                late = np.asarray([2*s+c for s in late_positions for c in range(2)])
                dirs = rng.normal(size=(256, 128)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                joint = rng.normal(size=(256, len(early)+len(late))); joint /= np.linalg.norm(joint, axis=1, keepdims=True)
                np.savez(directory / "metric_parameters.npz", directions=dirs, bandwidths=np.array([4., 8., 16.]),
                         early_indices=early, late_indices=late, joint_directions=joint, ridge_grid=np.array([1e-4, .1, 10.]),
                         matched_trtr_train_index_order=np.arange(80))
                (directory / "METRIC_SPEC.json").write_text('{"isolated_unit_fixture":true}')
                output = Path(temporary) / "evaluation"
                samples = raw_splits["train"][:8].reshape(2, 4, 128).astype(np.float32)
                result = evaluate_run(samples, output, dataset, root=temporary)
                self.assertTrue(all(np.isfinite(v) for v in result.values()))
                self.assertTrue((output / "metrics_per_ensemble.csv").is_file())
                self.assertTrue((output / "metric_provenance.json").is_file())
                with np.load(output / "metrics_arrays.npz", allow_pickle=False) as archive:
                    self.assertEqual(archive["acf"].shape, (2, 32))
                    self.assertEqual(archive["psd"].shape, (33, 2))
                with np.load(output / "ridge_arrays.npz", allow_pickle=False) as archive:
                    self.assertEqual(len(archive["matched_trtr_train_indices"]), 8)
                self.assertEqual(np.load(output / "samples_raw.npy").shape, (2, 4, 64, 2))


if __name__ == "__main__":
    unittest.main()
