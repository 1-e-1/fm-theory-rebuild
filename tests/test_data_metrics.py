"""Independent analytic checks of SEQ_IID layout and metric estimators.

These tests use hand-constructed arrays and never open formal test data.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment.data import flatten, unflatten, load_split
from experiment.metrics import empirical_wasserstein, cross_ensemble_mmd2, correlation_acf, evaluate_run


class DataMetricChecks(unittest.TestCase):
    def test_flatten_unflatten_preserves_whole_sample(self):
        original = np.arange(3 * 64, dtype=np.float32).reshape(3, 64, 1)
        self.assertEqual(flatten(original).shape, (3, 64))
        np.testing.assert_array_equal(unflatten(flatten(original)), original)

    def test_loader_only_requires_named_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "data" / "seq_iid"
            directory.mkdir(parents=True)
            states = np.arange(4 * 64, dtype=np.float32).reshape(4, 64)
            np.savez(directory / "train.npz", standardized=states)
            actual = load_split("train", root=temporary)
            np.testing.assert_array_equal(actual.numpy(), states)
            self.assertFalse((directory / "test.npz").exists())

    def test_exact_empirical_quantiles_unequal_counts(self):
        # a quantile is 0 then 2; b is always 1, hence both W1 and W2^2 are 1.
        np.testing.assert_allclose(empirical_wasserstein(np.array([0., 2.]), np.array([1.]), 1), [1.], rtol=0, atol=1e-15)
        np.testing.assert_allclose(empirical_wasserstein(np.array([0., 2.]), np.array([1.]), 2), [1.], rtol=0, atol=1e-15)
        # Boundaries 1/3,1/2,2/3 give W2^2=1/3+1/6+4/6+1/3=1.5.
        np.testing.assert_allclose(empirical_wasserstein(np.array([0., 0., 3.]), np.array([1., 2.]), 2), [1.5], rtol=0, atol=1e-15)

    def test_mmd_uses_only_different_ensembles_and_keeps_negative(self):
        generated = np.array([[[0.], [0.]], [[2.], [2.]]])
        real = np.array([[1.], [3.]])
        observed, rows, terms = cross_ensemble_mmd2(generated, real, np.array([1.]))
        gg = np.exp(-2.)
        rr = np.exp(-2.)
        gr = (3 * np.exp(-0.5) + np.exp(-4.5)) / 4
        expected = gg + rr - 2 * gr
        self.assertLess(observed, 0)
        self.assertAlmostEqual(observed, expected, places=14)
        self.assertAlmostEqual(terms["gg_cross_ensemble"], gg, places=14)
        self.assertAlmostEqual(float(rows.mean()), observed, places=14)
        permuted, _, _ = cross_ensemble_mmd2(generated[::-1, ::-1], real[::-1], np.array([1.]))
        self.assertAlmostEqual(permuted, observed, places=14)

    def test_acf_detects_shared_observation_dependence(self):
        # Identical random scalar at every observation position gives corr/ACF 1.
        correlated = np.tile(np.array([-2., -1., 1., 2.])[:, None], (1, 64))
        corr, acf = correlation_acf(correlated)
        np.testing.assert_allclose(corr, 1., rtol=0, atol=1e-15)
        np.testing.assert_allclose(acf, 1., rtol=0, atol=1e-15)

    def test_formal_evaluator_archives_float_samples_and_ensemble_rows(self):
        # Use an isolated synthetic fixture, never the project's formal test.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "data" / "seq_iid"
            directory.mkdir(parents=True)
            rng = np.random.default_rng(9081)
            reference = rng.normal(size=(12, 64))
            generated = rng.normal(size=(2, 3, 64)).astype(np.float32)
            np.savez(directory / "test.npz", standardized=reference, raw=reference[..., None])
            np.savez(directory / "stats.npz", mean=np.zeros(64), scale=np.ones(64))
            directions = rng.normal(size=(256, 64))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            np.savez(directory / "metric_parameters.npz", directions=directions, bandwidths=np.array([4., 8., 16.]))
            output = Path(temporary) / "result"
            metrics = evaluate_run(generated, output, root=temporary)
            self.assertTrue(np.isfinite(metrics["swd_raw"]))
            self.assertTrue(np.isfinite(metrics["mmd2_raw"]))
            np.testing.assert_array_equal(np.load(output / "samples_standardized.npy"), generated)
            self.assertEqual(len((output / "metrics_per_ensemble.csv").read_text().splitlines()), 3)
            self.assertTrue((output / "metric_provenance.json").is_file())
            self.assertTrue((output / "real_real_reference_blocks.npz").is_file())


if __name__ == "__main__":
    unittest.main()
