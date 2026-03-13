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

import numbers

import numpy as np

from .model import Model
from .modules import load_module


class DiffDIC:
    def __init__(self, delta):
        """
        This class stores the difference in the DIC between two models.

        Parameters
        ----------
        delta: Numpy array with the difference in the DIC for each observations
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
        return self._delta

    def __str__(self):

        result = f"Difference: {np.sum(self.delta)}\n"
        result += f"Sample standard error: {np.sqrt(self._n) * np.std(self.delta)}"
        return result

    def __repr__(self):
        return self.__str__()


class DIC:
    def __init__(self, deviance, penalty, type):
        """
        This class stores the Deviance Information Criterion (DIC) sampled from
        a PyJAGS model.

        Parameters
        ----------
        deviance
        penalty
        type: either 'pD' or 'popt'
        """
        self._deviance = deviance
        self._penalty = penalty
        self._type = type

    @property
    def deviance(self):
        return self._deviance

    @property
    def penalty(self):
        return self._penalty

    @property
    def type(self):
        return self._type

    def construct_report(self, digits=2) -> str:
        result = ""
        deviance = np.sum(self.deviance)
        psum = np.sum(self.penalty)
        result += "Mean deviance: {:.{}f}\n".format(deviance, digits)
        result += "penalty: {:.{}f}\n".format(psum, digits)
        result += "Penalized deviance: {:.{}f}".format(deviance + psum, digits)

        return result

    def __sub__(self, other):
        if not isinstance(other, DIC):
            raise TypeError("The second object must be of type DIC.")

        if self.type != other.type:
            raise ValueError("incompatible dic object: different penalty types")

        delta = self.deviance + self.penalty - other.deviance - other.penalty

        return DiffDIC(delta)

    def __str__(self):
        return self.construct_report()

    def __repr__(self):
        return self.__str__()


def dic_samples(model, n_iter, thin=1, type="pD"):
    """
    This function draws samples from a model and computes the
    Deviance Information Criterion (DIC).

    Parameters
    ----------
    model: a PyJAGS model instance
    n_iter: the number of iterations to sample for
    thin: a positive integer specifying thinning interval
    type: either 'pD' or 'popt'

    Returns
    -------
    the Deviance Information Criterion (DIC)
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
