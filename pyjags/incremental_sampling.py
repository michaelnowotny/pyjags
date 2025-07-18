# Copyright (C) 2015-2016 Tomasz Miasko
#               2020 Michael Nowotny
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import arviz as az
import numpy as np
import typing as tp

from .chain_utilities import merge_consecutive_chains

from .model import Model


class EffectiveSampleSizeCriterion:
    def __init__(
        self, minimum_ess: int, variable_names: tp.Optional[tp.List[str]] = None
    ):
        """
        This class implements a minimum effective sample size criterion to be
        used with sample_until.

        Parameters
        ----------
        minimum_ess: the minimum effective sample size required
        variable_names: the names of the variables to consider
        """
        self._minimum_ess = minimum_ess
        self._variable_names = variable_names

    @property
    def variable_names(self) -> tp.Optional[tp.List[str]]:
        return self._variable_names

    @property
    def minimum_ess(self) -> int:
        return self._minimum_ess

    def __call__(self, samples: tp.Dict[str, np.ndarray], verbose: bool) -> bool:
        idata = az.from_pyjags(samples)
        ess = az.ess(idata, var_names=self.variable_names)

        minimum_ess = min(
            value["data"] for key, value in ess.to_dict()["data_vars"].items()
        )

        if verbose:
            print(f"minimum ess = {minimum_ess}")

        return minimum_ess >= self.minimum_ess


class RHatDeviationCriterion:
    def __init__(
        self,
        maximum_rhat_deviation: float,
        variable_names: tp.Optional[tp.List[str]] = None,
    ):
        """
        This class implements a maximum rhat deviation criterion to be used with
        sample_until.

        Parameters
        ----------
        maximum_rhat_deviation: the maximum allowed deviation of rhat from 1
        variable_names: the names of the variables to consider
        """
        self._maximum_rhat_deviation = maximum_rhat_deviation
        self._variable_names = variable_names

    @property
    def variable_names(self) -> tp.Optional[tp.List[str]]:
        return self._variable_names

    @property
    def maximum_rhat_deviation(self) -> float:
        return self._maximum_rhat_deviation

    def __call__(self, samples: tp.Dict[str, np.ndarray], verbose: bool) -> bool:
        idata = az.from_pyjags(samples)
        rhat = az.rhat(idata, var_names=self.variable_names)

        maximum_rhat_deviation = max(
            abs(value["data"] - 1.0)
            for key, value in rhat.to_dict()["data_vars"].items()
        )

        if verbose:
            print(f"maximum rhat deviation = {maximum_rhat_deviation}")

        return maximum_rhat_deviation <= self.maximum_rhat_deviation


class EffectiveSampleSizeAndRHatCriterion:
    def __init__(
        self,
        minimum_ess: int,
        maximum_rhat_deviation: float,
        variable_names: tp.Optional[tp.List[str]] = None,
    ):
        """
        This class implements a combined minimum effective sample size and
        maximum rhat deviation criterion to be used with sample_until.

        Parameters
        ----------
        minimum_ess: the minimum effective sample size required
        maximum_rhat_deviation: the maximum allowed deviation of rhat from 1
        variable_names: the names of the variables to consider
        """
        self._minimum_ess = minimum_ess
        self._maximum_rhat_deviation = maximum_rhat_deviation
        self._variable_names = variable_names

    @property
    def variable_names(self) -> tp.Optional[tp.List[str]]:
        return self._variable_names

    @property
    def minimum_ess(self) -> int:
        return self._minimum_ess

    @property
    def maximum_rhat_deviation(self) -> float:
        return self._maximum_rhat_deviation

    def __call__(self, samples: tp.Dict[str, np.ndarray], verbose: bool) -> bool:
        idata = az.from_pyjags(samples)
        ess = az.ess(idata, var_names=self.variable_names)

        minimum_ess = min(
            value["data"] for key, value in ess.to_dict()["data_vars"].items()
        )

        rhat = az.rhat(idata, var_names=self.variable_names)

        maximum_rhat_deviation = max(
            abs(value["data"] - 1.0)
            for key, value in rhat.to_dict()["data_vars"].items()
        )
        if verbose:
            print(f"minimum ess = {minimum_ess}")
            print(f"maximum rhat deviation = {maximum_rhat_deviation}")

        return (
            minimum_ess >= self.minimum_ess
            and maximum_rhat_deviation <= self.maximum_rhat_deviation
        )


IterationFunctionType = tp.Callable[[tp.Dict[str, np.ndarray], bool, int], None]


def sample_until(
    model: Model,
    criterion: tp.Callable[[tp.Dict[str, np.ndarray], bool], bool],
    previous_samples: tp.Optional[tp.Dict[str, np.ndarray]] = None,
    chunk_size: int = 5000,
    max_iterations: int = 250000,
    vars: tp.Sequence[str] = None,
    thin: int = 1,
    monitor_type: str = "trace",
    verbose: bool = False,
    iteration_function: tp.Optional[IterationFunctionType] = None,
) -> tp.Dict[str, np.ndarray]:
    """
    This function progressively samples from a model until a criterion is met.

    Parameters
    ----------
    model: a PyJAGS model
    criterion: a function evaluating a samples dictionary and returning a bool
    previous_samples: an existing sample dictionary to incorporate
    chunk_size: the number of iterations to sample each step
    max_iterations: the maximum number of iterations to sample
    vars: a list of variables to monitor
    thin: a positive integer specifying thinning interval
    monitor_type
    verbose: whether to output step information
    iteration_function: A function to be called at the end of each iteration with
                        arguments:
                        1. dictionary of samples so far
                        2. boolean indicating whether the criterion has been met
                        3. integer of iterations so far
                        returning None

    Returns
    -------
    a dictionary of samples
    """

    if chunk_size > max_iterations:
        raise ValueError("chunk_size must be less than or equal to max_iterations")

    # if previous_samples is not None:
    #     print(f'chain_length at the beginning of sample_until = '
    #           f'{get_chain_length(previous_samples)}')

    if previous_samples is not None and criterion(previous_samples, verbose):
        return previous_samples

    iterations_left = max_iterations
    while True:
        iterations = min(iterations_left, chunk_size)

        new_samples = model.sample(
            iterations=iterations, vars=vars, thin=thin, monitor_type=monitor_type
        )

        if previous_samples is None:
            previous_samples = new_samples
        else:
            previous_samples = merge_consecutive_chains((previous_samples, new_samples))
            # print(f'chain_length at the after merging in sample_until = '
            #       f'{get_chain_length(previous_samples)}')

        iterations_left -= iterations

        criterion_satisfied = criterion(previous_samples, verbose)

        if iteration_function is not None:
            iteration_function(
                previous_samples, criterion_satisfied, max_iterations - iterations_left
            )

        if criterion_satisfied:
            break
        elif iterations_left <= 1:
            print(
                "maximum number of iterations reached without satisfying the criterion"
            )
            break

    return previous_samples
