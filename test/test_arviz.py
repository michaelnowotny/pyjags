import unittest

import numpy as np


class TestConvertPyjagsSamplesToArviz(unittest.TestCase):
    """Test the PyJAGS-to-ArviZ array shape conversion."""

    def test_scalar_variable(self):
        """Scalar variable: shape (1, iterations, chains) -> (chains, draws)."""
        from pyjags.arviz import _convert_pyjags_samples_to_arviz

        iterations, chains = 100, 4
        samples = {"mu": np.random.randn(1, iterations, chains)}
        result = _convert_pyjags_samples_to_arviz(samples)

        # Singleton param dim is squeezed out.
        self.assertEqual(result["mu"].shape, (chains, iterations))

    def test_vector_variable(self):
        """Vector variable: shape (3, iterations, chains) -> (chains, draws, 3)."""
        from pyjags.arviz import _convert_pyjags_samples_to_arviz

        param_dim, iterations, chains = 3, 100, 4
        samples = {"beta": np.random.randn(param_dim, iterations, chains)}
        result = _convert_pyjags_samples_to_arviz(samples)

        self.assertEqual(result["beta"].shape, (chains, iterations, param_dim))

    def test_matrix_variable(self):
        """Matrix variable: shape (3, 5, iterations, chains) -> (chains, draws, 3, 5)."""
        from pyjags.arviz import _convert_pyjags_samples_to_arviz

        d1, d2, iterations, chains = 3, 5, 100, 4
        samples = {"Sigma": np.random.randn(d1, d2, iterations, chains)}
        result = _convert_pyjags_samples_to_arviz(samples)

        self.assertEqual(result["Sigma"].shape, (chains, iterations, d1, d2))

    def test_values_preserved(self):
        """Verify that actual data values are correctly transposed."""
        from pyjags.arviz import _convert_pyjags_samples_to_arviz

        # shape: (2, 3, 2) = (param_dim=2, iterations=3, chains=2)
        arr = np.arange(12).reshape(2, 3, 2).astype(float)
        samples = {"x": arr}
        result = _convert_pyjags_samples_to_arviz(samples)

        # result shape: (chains=2, draws=3, param_dim=2)
        # result[chain, draw, param] == arr[param, draw, chain]
        for chain in range(2):
            for draw in range(3):
                for param in range(2):
                    self.assertEqual(
                        result["x"][chain, draw, param],
                        arr[param, draw, chain],
                    )

    def test_rejects_1d_array(self):
        """Arrays with fewer than 2 dimensions should raise ValueError."""
        from pyjags.arviz import _convert_pyjags_samples_to_arviz

        with self.assertRaises(ValueError):
            _convert_pyjags_samples_to_arviz({"x": np.array([1.0, 2.0])})

    def test_multiple_variables(self):
        """Multiple variables with different shapes."""
        from pyjags.arviz import _convert_pyjags_samples_to_arviz

        iterations, chains = 50, 3
        samples = {
            "mu": np.random.randn(1, iterations, chains),
            "beta": np.random.randn(5, iterations, chains),
            "Sigma": np.random.randn(2, 2, iterations, chains),
        }
        result = _convert_pyjags_samples_to_arviz(samples)

        self.assertEqual(result["mu"].shape, (chains, iterations))
        self.assertEqual(result["beta"].shape, (chains, iterations, 5))
        self.assertEqual(result["Sigma"].shape, (chains, iterations, 2, 2))


class TestFromPyjags(unittest.TestCase):
    """Test the full from_pyjags conversion to ArviZ DataTree."""

    def _make_samples(self, iterations=100, chains=4):
        return {
            "mu": np.random.randn(1, iterations, chains),
            "sigma": np.random.randn(1, iterations, chains),
        }

    def test_basic_posterior(self):
        """Basic posterior conversion returns a DataTree with correct dims."""
        from pyjags.arviz import from_pyjags

        samples = self._make_samples()
        idata = from_pyjags(samples)

        # Should have a posterior group
        posterior = idata["posterior"]
        self.assertIn("mu", posterior.data_vars)
        self.assertIn("sigma", posterior.data_vars)
        self.assertEqual(dict(posterior.sizes)["chain"], 4)
        self.assertEqual(dict(posterior.sizes)["draw"], 100)

    def test_ess_and_rhat_work(self):
        """ESS and Rhat should work on the converted data."""
        import arviz as az
        from pyjags.arviz import from_pyjags

        samples = self._make_samples(iterations=500, chains=4)
        idata = from_pyjags(samples)

        ess = az.ess(idata)
        rhat = az.rhat(idata)

        # ESS should be positive
        for var in ess.data_vars:
            self.assertGreater(float(ess[var]), 0)

        # Rhat should be close to 1 for iid draws
        for var in rhat.data_vars:
            self.assertAlmostEqual(float(rhat[var]), 1.0, delta=0.1)

    def test_log_likelihood_string(self):
        """Log-likelihood extraction with a string variable name."""
        from pyjags.arviz import from_pyjags

        iterations, chains = 100, 4
        samples = {
            "mu": np.random.randn(1, iterations, chains),
            "loglik": np.random.randn(10, iterations, chains),
        }
        idata = from_pyjags(samples, log_likelihood="loglik")

        # loglik should be in log_likelihood group, not posterior
        self.assertNotIn("loglik", idata["posterior"].data_vars)
        self.assertIn("loglik", idata["log_likelihood"].data_vars)
        self.assertIn("mu", idata["posterior"].data_vars)

    def test_log_likelihood_list(self):
        """Log-likelihood extraction with a list of variable names."""
        from pyjags.arviz import from_pyjags

        iterations, chains = 100, 4
        samples = {
            "mu": np.random.randn(1, iterations, chains),
            "ll_y1": np.random.randn(5, iterations, chains),
            "ll_y2": np.random.randn(3, iterations, chains),
        }
        idata = from_pyjags(samples, log_likelihood=["ll_y1", "ll_y2"])

        self.assertIn("ll_y1", idata["log_likelihood"].data_vars)
        self.assertIn("ll_y2", idata["log_likelihood"].data_vars)
        self.assertNotIn("ll_y1", idata["posterior"].data_vars)

    def test_log_likelihood_mapping(self):
        """Log-likelihood extraction with a name mapping."""
        from pyjags.arviz import from_pyjags

        iterations, chains = 100, 4
        samples = {
            "mu": np.random.randn(1, iterations, chains),
            "loglik_obs": np.random.randn(10, iterations, chains),
        }
        idata = from_pyjags(
            samples, log_likelihood={"y": "loglik_obs"}
        )

        self.assertIn("y", idata["log_likelihood"].data_vars)
        self.assertNotIn("loglik_obs", idata["posterior"].data_vars)

    def test_prior(self):
        """Prior samples should go into a prior group."""
        from pyjags.arviz import from_pyjags

        iterations, chains = 100, 4
        posterior = {"mu": np.random.randn(1, iterations, chains)}
        prior = {"mu": np.random.randn(1, 50, chains)}
        idata = from_pyjags(posterior, prior=prior)

        self.assertIn("mu", idata["posterior"].data_vars)
        self.assertIn("mu", idata["prior"].data_vars)

    def test_warmup_saved(self):
        """Warmup iterations should be split into warmup_posterior."""
        from pyjags.arviz import from_pyjags

        iterations, chains = 200, 4
        warmup = 50
        samples = {"mu": np.random.randn(1, iterations, chains)}
        idata = from_pyjags(
            samples, save_warmup=True, warmup_iterations=warmup
        )

        self.assertEqual(dict(idata["posterior"].sizes)["draw"], 150)
        self.assertEqual(dict(idata["warmup_posterior"].sizes)["draw"], 50)

    def test_warmup_discarded(self):
        """Warmup iterations should be discarded when save_warmup is False."""
        from pyjags.arviz import from_pyjags

        iterations, chains = 200, 4
        warmup = 50
        samples = {"mu": np.random.randn(1, iterations, chains)}
        idata = from_pyjags(
            samples, save_warmup=False, warmup_iterations=warmup
        )

        self.assertEqual(dict(idata["posterior"].sizes)["draw"], 150)
        # warmup_posterior should not exist
        self.assertNotIn("warmup_posterior", [c for c in idata.children])

    def test_pyjags_public_api(self):
        """from_pyjags should be importable from the top-level pyjags package."""
        import pyjags
        self.assertTrue(callable(pyjags.from_pyjags))


class TestSplitWarmup(unittest.TestCase):
    """Test the warmup splitting helper."""

    def test_split_at_boundary(self):
        from pyjags.arviz import _split_warmup

        arr = np.arange(60).reshape(1, 20, 3).astype(float)
        warmup, actual = _split_warmup({"x": arr}, warmup_iterations=5)

        self.assertEqual(warmup["x"].shape, (1, 5, 3))
        self.assertEqual(actual["x"].shape, (1, 15, 3))
        np.testing.assert_array_equal(warmup["x"], arr[:, :5, :])
        np.testing.assert_array_equal(actual["x"], arr[:, 5:, :])


if __name__ == "__main__":
    unittest.main()