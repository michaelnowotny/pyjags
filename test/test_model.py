# Copyright (C) 2015-2016 Tomasz Miasko
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import os.path
import sys
import unittest

import numpy as np

import pyjags


class ParametrizedTestCase(unittest.TestCase):
    def __init__(self, method_name, **params):
        super(ParametrizedTestCase, self).__init__(method_name)
        self.__dict__.update(params)


class TestModel(unittest.TestCase):
    def model(self, *args, **kwargs):
        """Create new model instance."""
        return pyjags.Model(*args, **kwargs)

    def test_creating_model_from_string(self):
        # No exceptions should be thrown.
        self.model(code="model { y ~ dbern(1) }")
        self.model(code=b"model { y ~ dbern(1) }")

    def test_creating_model_from_file(self):
        path = os.path.dirname(__file__)
        path = os.path.join(path, "model.jags")
        self.model(file=path)

    def test_model_is_required(self):
        with self.assertRaises(ValueError):
            self.model()

    def test_random_number_generator_seed(self):
        code = """
        model {
            x ~ dbern(0.5)
        }
        """
        init = {".RNG.name": "base::Wichmann-Hill", ".RNG.seed": 1}
        n = 100
        s1 = self.model(code, init=init).sample(n)
        s2 = self.model(code, init=init).sample(n)
        np.testing.assert_equal(
            s1, s2, "Using seed should be give deterministic samples."
        )

    def test_selecting_random_number_generators(self):
        expected_names = [
            "base::Marsaglia-Multicarry",
            "base::Marsaglia-Multicarry",
            "base::Mersenne-Twister",
            "base::Mersenne-Twister",
            "base::Wichmann-Hill",
        ]
        code = "model { x ~ dbern(0.5) }"
        init = [{".RNG.name": name, ".RNG.seed": 0} for name in expected_names]
        chains = len(init)

        model = self.model(code, init=init, chains=chains, adapt=10)
        model.sample(20)

        actual_names = [
            v for state in model.state for k, v in state.items() if k == ".RNG.name"
        ]
        self.assertEqual(expected_names, actual_names)

    def test_empty_array(self):
        # This used to throw an exception.
        code = "model { x ~ dbern(1) }"
        data = {"x": []}
        self.model(code, data=data)

    def test_invalid_length_of_initial_value_list_throws_exception(self):
        model = "model { x ~ dbern(1) }"
        with self.assertRaises(ValueError):
            self.model(model, chains=2, init=[dict(x=1)])

    def test_invalid_type_of_init_value_list(self):
        code = "model { x ~ dbern(1) }"
        with self.assertRaises(ValueError):
            self.model(code, chains=2, init=1234)

    def test_model_variables(self):
        code = """
        data {
            a <- 1
        }
        model {
            b <- 2
            for (i in 1:3) {
                x[i] ~ dbeta(1, 1)
            }
        }
        """

        model = self.model(code)
        self.assertEqual({"a", "b", "x"}, set(model.variables))

    def test_model_data(self):
        code = """
        model {
            for (i in 1:N) {
                x[i] ~ dbin(p, n[i])
            }
            p ~ dbeta(1, 1)
        }
        """

        N = 100
        n = np.random.randint(1, 11, N)
        x = np.random.binomial(n, 0.10, N)
        x = np.ma.masked_array(x, np.random.choice([0, 1], size=N))
        data = dict(n=n, x=x, N=N)

        m = self.model(code, data=data)
        model_data = m.data

        self.assertEqual(data.keys(), model_data.keys())
        for var in data.keys():
            np.testing.assert_equal(data[var], model_data[var])

    def test_model_parameters(self):
        code = """
        model {
            for (i in 1:10) {
                x[i] ~ dnorm(mu[i], 1)
                mu[i] ~ dunif(-1, 1)
            }
        }
        """
        chains = 3
        model = self.model(code, data=dict(x=np.zeros(10)), chains=chains)

        parameters = model.parameters
        self.assertEqual(chains, len(parameters))
        names = set(parameters[0].keys())
        self.assertEqual({"mu", ".RNG.name", ".RNG.state"}, names)

    def test_samples_shape(self):
        code = """
        model {
            for (i in 1:3) {
                for (j in 1:5) {
                    x[i, j] ~ dnorm(mu[i], 1)

                }
                mu[i] ~ dunif(-1, 1)
            }
        }
        """

        chains = 7
        iterations = 17
        data = {"x": np.zeros((3, 5))}

        m = self.model(code, data=data, chains=chains)
        s = m.sample(iterations)

        self.assertEqual(s["x"].shape, (3, 5, iterations, chains))
        self.assertEqual(s["mu"].shape, (3, iterations, chains))

    def test_missing_input_data(self):
        code = """
        model {
            for (i in 1:length(x)) {
                x[i] ~ dbern(0.5)
            }
        }"""

        data = {"x": np.ma.masked_outside([0, 1, -1], 0, 1)}
        c = 2
        m = self.model(code, data=data, chains=c)
        n = 100
        s = m.sample(n, vars=["x"])

        x1 = s["x"][0, :, :]
        x2 = s["x"][1, :, :]
        x3 = s["x"][2, :, :]

        # Observed values, samples should be constant
        np.testing.assert_equal([0] * n * c, x1.flatten())
        np.testing.assert_equal([1] * n * c, x2.flatten())
        # Missing value, samples should vary between 0 and 1
        self.assertIn(0, x3)
        self.assertIn(1, x3)

    @unittest.skipIf(pyjags.version() < (4, 0, 0), "Not supported before JAGS 4.0.0")
    def test_missing_sample_data(self):
        code = """
        model {
            x[1] ~ dnorm(0, 10)
            x[3] ~ dnorm(0, 15)
        }"""

        m = self.model(code, chains=2)
        s = m.sample(10, vars=["x"])

        x = s["x"]
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        # x[2] should be completely masked
        self.assertTrue(np.all(x2.mask))
        # x[1] and x[3] should be plain unmasked numpy arrays
        self.assertFalse(np.ma.is_mask(x1))
        self.assertFalse(np.ma.is_mask(x3))

    def test_unused_variables_throws_exception(self):
        code = "model { x ~ dbern(0.5) }"

        with self.assertRaises(ValueError):
            self.model(code, data=dict(x=1, y=2))

        with self.assertRaises(ValueError):
            self.model(code, init=dict(x=1, y=2))


class TestModelWithoutProgressBar(TestModel):
    def model(self, *args, **kwargs):
        return pyjags.Model(*args, progress_bar=False, **kwargs)


if sys.version_info[0] > 2:

    class TestModelWithThreads(TestModel):
        def model(self, *args, **kwargs):
            return pyjags.Model(*args, threads=3, **kwargs)

    class TestModelWithChainsPerThread(TestModel):
        def model(self, *args, **kwargs):
            return pyjags.Model(*args, threads=3, chains_per_thread=2, **kwargs)


if __name__ == "__main__":
    unittest.main()
