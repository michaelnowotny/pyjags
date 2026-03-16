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

"""Convergence criteria and adaptive sampling until convergence.

Provides pluggable convergence criteria based on effective sample size
(ESS) and the Gelman-Rubin R-hat statistic, plus the :func:`sample_until`
driver that repeatedly draws samples until criteria are satisfied.
"""

import typing as tp

import arviz as az
import numpy as np

from .arviz import from_pyjags
from .chain_utilities import merge_consecutive_chains
from .model import Model


class EffectiveSampleSizeCriterion:
    """Convergence criterion based on minimum effective sample size.

    Sampling is considered converged when the ESS of all monitored
    variables exceeds a specified threshold.

    Parameters
    ----------
    minimum_ess : float
        Minimum effective sample size required for convergence.
    variable_names : list of str, optional
        Variables to monitor. If ``None``, all variables are checked.
    """

    def __init__(self, minimum_ess: int, variable_names: list[str] | None = None):
        """Initialize the ESS convergence criterion.

        Parameters
        ----------
        minimum_ess : int
            Minimum effective sample size required for convergence.
        variable_names : list[str], optional
            Variables to monitor. If ``None``, all variables are checked.
        """
        self._minimum_ess = minimum_ess
        self._variable_names = variable_names

    @property
    def variable_names(self) -> list[str] | None:
        """Names of the variables to monitor.

        Returns
        -------
        list[str] or None
            Variable names, or ``None`` if all variables are monitored.
        """
        return self._variable_names

    @property
    def minimum_ess(self) -> int:
        """Minimum effective sample size threshold.

        Returns
        -------
        int
            The ESS threshold that must be exceeded for convergence.
        """
        return self._minimum_ess

    def __call__(self, samples: dict[str, np.ndarray], verbose: bool) -> bool:
        """Evaluate whether the ESS criterion is satisfied.

        Parameters
        ----------
        samples : dict[str, numpy.ndarray]
            Sample dictionary with shape
            ``(*variable_dims, iterations, chains)``.
        verbose : bool
            If ``True``, print the current minimum ESS.

        Returns
        -------
        bool
            ``True`` if ESS exceeds the threshold for all monitored
            variables.
        """
        idata = from_pyjags(samples)
        ess = az.ess(idata, var_names=self.variable_names)

        minimum_ess = min(float(ess[var]) for var in ess.data_vars)

        if verbose:
            print(f"minimum ess = {minimum_ess}")

        return minimum_ess >= self.minimum_ess


class RHatDeviationCriterion:
    """Convergence criterion based on the Gelman-Rubin R-hat statistic.

    Sampling is considered converged when the R-hat of all monitored
    variables is within a specified tolerance of 1.0.

    Parameters
    ----------
    maximum_rhat_deviation : float
        Maximum allowed deviation of R-hat from 1.0 (e.g., 0.01 means
        R-hat must be below 1.01).
    variable_names : list of str, optional
        Variables to monitor. If ``None``, all variables are checked.
    """

    def __init__(
        self,
        maximum_rhat_deviation: float,
        variable_names: list[str] | None = None,
    ):
        """Initialize the R-hat convergence criterion.

        Parameters
        ----------
        maximum_rhat_deviation : float
            Maximum allowed deviation of R-hat from 1.0.
        variable_names : list[str], optional
            Variables to monitor. If ``None``, all variables are checked.
        """
        self._maximum_rhat_deviation = maximum_rhat_deviation
        self._variable_names = variable_names

    @property
    def variable_names(self) -> list[str] | None:
        """Names of the variables to monitor.

        Returns
        -------
        list[str] or None
            Variable names, or ``None`` if all variables are monitored.
        """
        return self._variable_names

    @property
    def maximum_rhat_deviation(self) -> float:
        """Maximum allowed deviation of R-hat from 1.0.

        Returns
        -------
        float
            The R-hat deviation threshold.
        """
        return self._maximum_rhat_deviation

    def __call__(self, samples: dict[str, np.ndarray], verbose: bool) -> bool:
        """Evaluate whether the R-hat criterion is satisfied.

        Parameters
        ----------
        samples : dict[str, numpy.ndarray]
            Sample dictionary with shape
            ``(*variable_dims, iterations, chains)``.
        verbose : bool
            If ``True``, print the current maximum R-hat deviation.

        Returns
        -------
        bool
            ``True`` if R-hat is within the allowed deviation of 1.0
            for all monitored variables.
        """
        idata = from_pyjags(samples)
        rhat = az.rhat(idata, var_names=self.variable_names)

        maximum_rhat_deviation = max(
            abs(float(rhat[var]) - 1.0) for var in rhat.data_vars
        )

        if verbose:
            print(f"maximum rhat deviation = {maximum_rhat_deviation}")

        return maximum_rhat_deviation <= self.maximum_rhat_deviation


class EffectiveSampleSizeAndRHatCriterion:
    """Combined convergence criterion requiring both ESS and R-hat thresholds.

    Sampling is considered converged when *both* the effective sample size
    exceeds ``minimum_ess`` and the R-hat is within ``maximum_rhat_deviation``
    of 1.0 for all monitored variables.

    Parameters
    ----------
    minimum_ess : float
        Minimum effective sample size required.
    maximum_rhat_deviation : float
        Maximum allowed deviation of R-hat from 1.0.
    variable_names : list of str, optional
        Variables to monitor. If ``None``, all variables are checked.
    """

    def __init__(
        self,
        minimum_ess: int,
        maximum_rhat_deviation: float,
        variable_names: list[str] | None = None,
    ):
        """Initialize the combined ESS and R-hat convergence criterion.

        Parameters
        ----------
        minimum_ess : int
            Minimum effective sample size required.
        maximum_rhat_deviation : float
            Maximum allowed deviation of R-hat from 1.0.
        variable_names : list[str], optional
            Variables to monitor. If ``None``, all variables are checked.
        """
        self._minimum_ess = minimum_ess
        self._maximum_rhat_deviation = maximum_rhat_deviation
        self._variable_names = variable_names

    @property
    def variable_names(self) -> list[str] | None:
        """Names of the variables to monitor.

        Returns
        -------
        list[str] or None
            Variable names, or ``None`` if all variables are monitored.
        """
        return self._variable_names

    @property
    def minimum_ess(self) -> int:
        """Minimum effective sample size threshold.

        Returns
        -------
        int
            The ESS threshold that must be exceeded for convergence.
        """
        return self._minimum_ess

    @property
    def maximum_rhat_deviation(self) -> float:
        """Maximum allowed deviation of R-hat from 1.0.

        Returns
        -------
        float
            The R-hat deviation threshold.
        """
        return self._maximum_rhat_deviation

    def __call__(self, samples: dict[str, np.ndarray], verbose: bool) -> bool:
        """Evaluate whether both ESS and R-hat criteria are satisfied.

        Parameters
        ----------
        samples : dict[str, numpy.ndarray]
            Sample dictionary with shape
            ``(*variable_dims, iterations, chains)``.
        verbose : bool
            If ``True``, print the current minimum ESS and maximum
            R-hat deviation.

        Returns
        -------
        bool
            ``True`` if ESS exceeds the threshold *and* R-hat is within
            the allowed deviation of 1.0 for all monitored variables.
        """
        idata = from_pyjags(samples)
        ess = az.ess(idata, var_names=self.variable_names)

        minimum_ess = min(float(ess[var]) for var in ess.data_vars)

        rhat = az.rhat(idata, var_names=self.variable_names)

        maximum_rhat_deviation = max(
            abs(float(rhat[var]) - 1.0) for var in rhat.data_vars
        )
        if verbose:
            print(f"minimum ess = {minimum_ess}")
            print(f"maximum rhat deviation = {maximum_rhat_deviation}")

        return (
            minimum_ess >= self.minimum_ess
            and maximum_rhat_deviation <= self.maximum_rhat_deviation
        )


IterationFunctionType = tp.Callable[[dict[str, np.ndarray], bool, int], None]


def sample_until(
    model: Model,
    criterion: tp.Callable[[dict[str, np.ndarray], bool], bool],
    previous_samples: dict[str, np.ndarray] | None = None,
    chunk_size: int = 5000,
    max_iterations: int = 250000,
    vars: tp.Sequence[str] | None = None,
    thin: int = 1,
    monitor_type: str = "trace",
    verbose: bool = False,
    iteration_function: IterationFunctionType | None = None,
) -> dict[str, np.ndarray]:
    """Progressively sample from a model until a convergence criterion is met.

    Draws samples in chunks of *chunk_size* iterations, merging them
    with any *previous_samples*, and evaluates *criterion* after each
    chunk.  Stops when the criterion is satisfied or *max_iterations*
    is reached.

    Parameters
    ----------
    model : Model
        A compiled PyJAGS model instance.
    criterion : callable
        A callable with signature ``(samples, verbose) -> bool`` that
        returns ``True`` when convergence has been reached.  The
        built-in criterion classes in this module can be used directly.
    previous_samples : dict[str, numpy.ndarray], optional
        An existing sample dictionary to incorporate.  If provided and
        the criterion is already satisfied, these samples are returned
        immediately.
    chunk_size : int
        Number of iterations to draw in each sampling step.
    max_iterations : int
        Maximum total number of iterations to draw before stopping.
    vars : list of str, optional
        Variables to monitor.  Defaults to all model variables.
    thin : int
        A positive integer specifying the thinning interval.
    monitor_type : str
        JAGS monitor type (default ``'trace'``).
    verbose : bool
        If ``True``, the criterion may print diagnostic information
        after each chunk.
    iteration_function : callable, optional
        A callback invoked after each chunk with signature
        ``(samples, criterion_satisfied, iterations_so_far) -> None``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Accumulated sample dictionary with shape
        ``(*variable_dims, total_iterations, n_chains)``.

    Raises
    ------
    ValueError
        If *chunk_size* exceeds *max_iterations*.
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
