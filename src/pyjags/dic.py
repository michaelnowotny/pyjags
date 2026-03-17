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

"""Deviance Information Criterion (DIC) computation and comparison."""

import numbers

import numpy as np

from .model import Model
from .modules import load_module


class DiffDIC:
    """Difference between two DIC values for model comparison.

    Created by subtracting one :class:`DIC` instance from another.

    Attributes
    ----------
    delta : numpy.ndarray
        Array of DIC differences (one per penalty type).

    Parameters
    ----------
    delta : numpy.ndarray or number
        Array (or scalar) of DIC differences.
    """

    def __init__(self, delta):
        """Initialize a DiffDIC instance.

        Parameters
        ----------
        delta : numpy.ndarray or number
            Array (or scalar) of DIC differences between two models.

        Raises
        ------
        TypeError
            If *delta* is neither a numpy array nor a number.
        """
        self._delta = delta

        if isinstance(self.delta, np.ndarray):
            self._n = len(self.delta)
        elif isinstance(self.delta, numbers.Number):
            self._n = 1
        else:
            raise TypeError(
                f"delta must either be a numpy array or a number "
                f"but is of type {type(delta)}"
            )

    @property
    def delta(self):
        """DIC difference values.

        Returns
        -------
        numpy.ndarray or number
            The DIC difference(s).
        """
        return self._delta

    def __repr__(self):
        """Return a human-readable summary of the DIC difference."""
        result = f"Difference: {np.sum(self.delta)}\n"
        result += f"Sample standard error: {np.sqrt(self._n) * np.std(self.delta)}"
        return result

    def __str__(self):
        """Return the string representation of the DIC difference."""
        return self.__repr__()


class DIC:
    """Deviance Information Criterion for a fitted JAGS model.

    DIC combines a measure of model fit (deviance) with a complexity
    penalty.  Lower DIC values indicate a better trade-off between fit
    and complexity.

    Attributes
    ----------
    deviance : numpy.ndarray
        Mean posterior deviance.
    penalty : numpy.ndarray
        Complexity penalties (one per penalty type).
    type : str
        Name of the penalty type (``'pD'`` or ``'popt'``).

    Parameters
    ----------
    deviance : numpy.ndarray
        Mean posterior deviance values.
    penalty : numpy.ndarray
        Complexity penalty values.
    type : str
        Penalty type, either ``'pD'`` or ``'popt'``.
    """

    def __init__(self, deviance, penalty, type):
        """Initialize a DIC instance.

        Parameters
        ----------
        deviance : numpy.ndarray
            Mean posterior deviance values.
        penalty : numpy.ndarray
            Complexity penalty values.
        type : str
            Penalty type, either ``'pD'`` or ``'popt'``.
        """
        self._deviance = deviance
        self._penalty = penalty
        self._type = type

    @property
    def deviance(self):
        """Mean posterior deviance.

        Returns
        -------
        numpy.ndarray
            Deviance values.
        """
        return self._deviance

    @property
    def penalty(self):
        """Complexity penalty values.

        Returns
        -------
        numpy.ndarray
            Penalty values for the selected penalty type.
        """
        return self._penalty

    @property
    def type(self):
        """Penalty type name.

        Returns
        -------
        str
            ``'pD'`` or ``'popt'``.
        """
        return self._type

    def construct_report(self, digits=2) -> str:
        """Build a human-readable DIC summary string.

        Parameters
        ----------
        digits : int
            Number of decimal places in the formatted output.

        Returns
        -------
        str
            Multi-line string with mean deviance, penalty, and
            penalized deviance.
        """
        result = ""
        deviance = np.sum(self.deviance)
        psum = np.sum(self.penalty)
        result += "Mean deviance: {:.{}f}\n".format(deviance, digits)
        result += "penalty: {:.{}f}\n".format(psum, digits)
        result += "Penalized deviance: {:.{}f}".format(deviance + psum, digits)

        return result

    def __sub__(self, other):
        """Subtract another DIC to compute the difference for model comparison.

        Parameters
        ----------
        other : DIC
            The DIC of the model to compare against.

        Returns
        -------
        DiffDIC
            The element-wise difference in penalized deviance.

        Raises
        ------
        TypeError
            If *other* is not a :class:`DIC` instance.
        ValueError
            If the two DIC objects use different penalty types.
        """
        if not isinstance(other, DIC):
            raise TypeError("The second object must be of type DIC.")

        if self.type != other.type:
            raise ValueError("incompatible dic object: different penalty types")

        delta = self.deviance + self.penalty - other.deviance - other.penalty

        return DiffDIC(delta)

    def __repr__(self):
        """Return a human-readable DIC summary report."""
        return self.construct_report()

    def __str__(self):
        """Return the string representation of the DIC object."""
        return self.__repr__()


def dic_samples(model, n_iter, thin=1, type="pD"):
    """Draw samples from a model and compute the Deviance Information Criterion.

    Parameters
    ----------
    model : Model
        A compiled PyJAGS model instance with at least 2 chains.
    n_iter : int
        A positive integer specifying the number of iterations to sample.
    thin : int
        A positive integer specifying the thinning interval.
    type : str
        Penalty type, either ``'pD'`` or ``'popt'``.

    Returns
    -------
    DIC
        A :class:`DIC` object containing the deviance, penalty, and
        penalty type.

    Raises
    ------
    ValueError
        If *model* is not a valid JAGS model, has fewer than 2 chains,
        *n_iter* is not a positive integer, or *type* is not one of
        ``'pD'`` or ``'popt'``.
    """
    if not isinstance(model, Model):
        raise ValueError("Invalid JAGS model")

    if model.chains == 1:
        raise ValueError("2 or more parallel chains required")

    if not isinstance(n_iter, int) or n_iter <= 0:
        raise ValueError("n_iter must be a positive integer")

    load_module(name="dic")

    if type not in ("pD", "popt"):
        raise ValueError(f"type must either be pD or popt but is {type}")
    pdtype = type
    model.console.setMonitors(names=("deviance", pdtype), thin=thin, type="mean")

    model.update(iterations=n_iter)

    # this returns a dictionary
    dev = model.console.dumpMonitors(type="mean", flat=True)

    model.console.clearMonitor(name="deviance", type="mean")
    model.console.clearMonitor(name=pdtype, type="mean")

    return DIC(deviance=dev["deviance"], penalty=dev[type], type=type)
