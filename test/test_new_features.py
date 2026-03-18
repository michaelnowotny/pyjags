# Copyright (C) 2026 Michael Nowotny
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""Tests for v2.3.0 features: seed, repr, iter_sample, check_model,
sampler diagnostics, pathlib support, from_pyjags enhancements."""

import pathlib
import tempfile

import numpy as np
import pytest

import pyjags

MODEL_CODE = """
model {
    mu ~ dnorm(0, 0.001)
    sigma ~ dunif(0, 1000)
    for (i in 1:N) {
        x[i] ~ dnorm(mu, pow(sigma, -2))
    }
}
"""


@pytest.fixture
def model_data():
    np.random.seed(123)
    return {"x": np.random.normal(5.0, 2.0, 20), "N": 20}


@pytest.fixture
def model(model_data):
    return pyjags.Model(
        code=MODEL_CODE,
        data=model_data,
        chains=2,
        adapt=100,
        progress_bar=False,
    )


class TestSeedParameter:
    def test_seed_produces_reproducible_results(self, model_data):
        m1 = pyjags.Model(
            code=MODEL_CODE,
            data=model_data,
            chains=2,
            adapt=100,
            progress_bar=False,
            seed=42,
        )
        s1 = m1.sample(500, vars=["mu"])

        m2 = pyjags.Model(
            code=MODEL_CODE,
            data=model_data,
            chains=2,
            adapt=100,
            progress_bar=False,
            seed=42,
        )
        s2 = m2.sample(500, vars=["mu"])

        np.testing.assert_array_equal(s1["mu"], s2["mu"])

    def test_different_seeds_produce_different_results(self, model_data):
        m1 = pyjags.Model(
            code=MODEL_CODE,
            data=model_data,
            chains=2,
            adapt=100,
            progress_bar=False,
            seed=42,
        )
        s1 = m1.sample(500, vars=["mu"])

        m2 = pyjags.Model(
            code=MODEL_CODE,
            data=model_data,
            chains=2,
            adapt=100,
            progress_bar=False,
            seed=99,
        )
        s2 = m2.sample(500, vars=["mu"])

        assert not np.array_equal(s1["mu"], s2["mu"])

    def test_seed_with_init_raises_on_rng_conflict(self, model_data):
        with pytest.raises(ValueError, match="Cannot specify both"):
            pyjags.Model(
                code=MODEL_CODE,
                data=model_data,
                chains=2,
                progress_bar=False,
                seed=42,
                init={".RNG.name": "base::Mersenne-Twister"},
            )

    def test_seed_with_non_rng_init_works(self, model_data):
        m = pyjags.Model(
            code=MODEL_CODE,
            data=model_data,
            chains=2,
            adapt=100,
            progress_bar=False,
            seed=42,
            init={"mu": 5.0},
        )
        assert m.chains == 2

    def test_seed_with_many_chains(self, model_data):
        m = pyjags.Model(
            code=MODEL_CODE,
            data=model_data,
            chains=8,
            adapt=100,
            progress_bar=False,
            seed=42,
        )
        samples = m.sample(100, vars=["mu"])
        assert samples["mu"].shape[-1] == 8


class TestModelRepr:
    def test_repr_contains_chains(self, model):
        r = repr(model)
        assert "chains=2" in r

    def test_repr_contains_variables(self, model):
        r = repr(model)
        assert "variables=" in r

    def test_repr_contains_adapted(self, model):
        r = repr(model)
        assert "adapted=" in r

    def test_repr_contains_iteration(self, model):
        r = repr(model)
        assert "iteration=" in r

    def test_repr_format(self, model):
        r = repr(model)
        assert r.startswith("Model(")
        assert r.endswith(")")


class TestSamplerDiagnostics:
    def test_samplers_returns_list(self, model):
        samplers = model.samplers
        assert isinstance(samplers, list)

    def test_is_adapted_returns_bool(self, model):
        assert isinstance(model.is_adapted, bool)

    def test_iteration_returns_int(self, model):
        assert isinstance(model.iteration, int)

    def test_iteration_increases_after_sampling(self, model):
        iter_before = model.iteration
        model.sample(100, vars=["mu"])
        assert model.iteration > iter_before


class TestIterSample:
    def test_yields_sampling_state(self, model):
        for state in model.iter_sample(iterations=200, chunk_size=100, vars=["mu"]):
            assert isinstance(state, pyjags.SamplingState)
            assert isinstance(state.samples, dict)
            assert "mu" in state.samples
            break

    def test_accumulates_iterations(self, model):
        iterations_seen = []
        for state in model.iter_sample(iterations=300, chunk_size=100, vars=["mu"]):
            iterations_seen.append(state.iteration)

        assert iterations_seen == [100, 200, 300]

    def test_samples_grow(self, model):
        sizes = []
        for state in model.iter_sample(iterations=300, chunk_size=100, vars=["mu"]):
            sizes.append(state.samples["mu"].shape[-2])

        assert sizes == [100, 200, 300]

    def test_ess_is_computed_lazily(self, model):
        for state in model.iter_sample(iterations=500, chunk_size=500, vars=["mu"]):
            ess = state.ess
            assert isinstance(ess, dict)
            assert "mu" in ess
            assert ess["mu"] > 0

    def test_rhat_is_computed_lazily(self, model):
        for state in model.iter_sample(iterations=500, chunk_size=500, vars=["mu"]):
            rhat = state.rhat
            assert isinstance(rhat, dict)
            assert "mu" in rhat
            assert rhat["mu"] > 0

    def test_early_break(self, model):
        total = 0
        for state in model.iter_sample(iterations=10000, chunk_size=100, vars=["mu"]):
            total = state.iteration
            if total >= 200:
                break
        assert total == 200


class TestSampleMore:
    def test_extends_samples(self, model):
        s1 = model.sample(100, vars=["mu"])
        s2 = model.sample_more(100, s1, vars=["mu"])
        assert s2["mu"].shape[-2] == 200

    def test_preserves_chains(self, model):
        s1 = model.sample(100, vars=["mu"])
        s2 = model.sample_more(100, s1, vars=["mu"])
        assert s2["mu"].shape[-1] == s1["mu"].shape[-1]


class TestCheckModel:
    def test_valid_model_returns_true(self):
        assert pyjags.check_model(code=MODEL_CODE) is True

    def test_invalid_model_raises(self):
        from pyjags.console import JagsError

        with pytest.raises(JagsError):
            pyjags.check_model(code="model { not valid jags }")

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bug", delete=False) as f:
            f.write(MODEL_CODE)
            f.flush()
            assert pyjags.check_model(file=f.name) is True

    def test_pathlib_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bug", delete=False) as f:
            f.write(MODEL_CODE)
            f.flush()
            assert pyjags.check_model(file=pathlib.Path(f.name)) is True

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            pyjags.check_model()


class TestPathlibSupport:
    def test_model_from_pathlib(self, model_data):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bug", delete=False) as f:
            f.write(MODEL_CODE)
            f.flush()
            m = pyjags.Model(
                file=pathlib.Path(f.name),
                data=model_data,
                chains=2,
                adapt=100,
                progress_bar=False,
            )
            assert m.chains == 2


class TestFromPyjagsEnhancements:
    def test_observed_data_group(self):
        samples = {"mu": np.random.randn(1, 100, 2)}
        obs = {"y": np.array([1.0, 2.0, 3.0])}
        idata = pyjags.from_pyjags(samples, observed_data=obs)
        assert "observed_data" in [str(n) for n in idata.children]

    def test_constant_data_group(self):
        samples = {"mu": np.random.randn(1, 100, 2)}
        const = {"N": np.array(20)}
        idata = pyjags.from_pyjags(samples, constant_data=const)
        assert "constant_data" in [str(n) for n in idata.children]

    def test_metadata_attributes(self):
        samples = {"mu": np.random.randn(1, 100, 2)}
        idata = pyjags.from_pyjags(samples)
        assert idata.attrs.get("inference_library") == "pyjags"

    def test_backward_compatible(self):
        samples = {"mu": np.random.randn(1, 100, 2)}
        idata = pyjags.from_pyjags(samples)
        assert "posterior" in [str(n) for n in idata.children]


class TestWarnConvergence:
    def test_no_warning_on_good_samples(self, model):
        import warnings

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            model.sample(1000, vars=["mu"], warn_convergence=True)

        convergence_warnings = [
            w for w in record if "R-hat" in str(w.message) or "ESS" in str(w.message)
        ]
        assert len(convergence_warnings) == 0, (
            f"Unexpected warnings: {convergence_warnings}"
        )

    def test_default_no_warning(self, model):
        # Default warn_convergence=False should never warn
        model.sample(100, vars=["mu"])


class TestPyTyped:
    def test_py_typed_exists(self):
        import importlib.resources

        files = importlib.resources.files("pyjags")
        py_typed = files / "py.typed"
        assert py_typed.is_file()
