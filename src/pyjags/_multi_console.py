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

"""Internal wrapper managing multiple JAGS Console instances for parallel chains."""

import numpy as np

from .console import Console


class MultiConsole:
    """Wrapper managing multiple JAGS Console instances for parallel chain execution.

    Each Console instance handles one or more chains. ``MultiConsole``
    distributes operations (compilation, initialization, sampling) across
    all consoles, optionally using threads for parallel execution.

    Parameters
    ----------
    chains : int
        Total number of chains to run.
    chains_per_thread : int
        Maximum number of chains assigned to each Console instance.
    """

    def __init__(self, chains, chains_per_thread):
        """Initialize a MultiConsole distributing chains across Console instances.

        Parameters
        ----------
        chains : int
            Total number of MCMC chains to run.
        chains_per_thread : int
            Maximum number of chains assigned to a single Console
            instance. Additional Console instances are created as needed
            to cover all chains.
        """
        # Multiple consoles that emulate a single JAGS console.
        self.consoles = []
        self.chains_per_console = []
        # Map from outer chain number to inner console and its inner chain number.
        # Uses JAGS indexing from 1.
        self.chains = {}

        outer_chain = 1
        while chains > 0:
            console = Console()
            console_chains = min(chains_per_thread, chains)

            self.consoles.append(console)
            self.chains_per_console.append(console_chains)

            for inner_chain in range(1, console_chains + 1):
                self.chains[outer_chain] = (console, inner_chain)
                outer_chain += 1

            chains -= chains_per_thread

    def checkModel(self, path):
        """Check model syntax on all Console instances.

        Parameters
        ----------
        path : str
            Filesystem path to the JAGS model file.
        """
        for c in self.consoles:
            c.checkModel(path)

    def compile(self, data, chains, generate_data):
        """Compile the model on all Console instances.

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            Observed data to pass to JAGS, already converted via
            :func:`dict_to_jags`.
        chains : int
            Total number of chains. Must equal the number of chains this
            MultiConsole was initialised with.
        generate_data : bool
            If ``True``, execute the ``data`` block in the model to
            generate additional data.
        """
        assert chains == len(self.chains)
        for console, chains in zip(self.consoles, self.chains_per_console, strict=True):
            console.compile(data, chains, generate_data)

    def setRNGname(self, name, chain):
        """Set the random number generator name for a specific chain.

        Parameters
        ----------
        name : str
            Name of the JAGS RNG factory (e.g.
            ``'base::Mersenne-Twister'``).
        chain : int
            Chain number (1-indexed).
        """
        console, chain = self.chains[chain]
        console.setRNGname(name, chain)

    def setParameters(self, data, chain):
        """Set initial parameter values for a specific chain.

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            Dictionary mapping parameter names to their initial values.
        chain : int
            Chain number (1-indexed).
        """
        console, chain = self.chains[chain]
        console.setParameters(data, chain)

    def setMonitor(self, name, thin, type):
        """Add a monitor for a single variable on all Console instances.

        Parameters
        ----------
        name : str
            Variable name to monitor.
        thin : int
            Thinning interval for the monitor.
        type : str
            Monitor type (e.g. ``'trace'``, ``'mean'``, ``'variance'``).
        """
        for c in self.consoles:
            c.setMonitor(name, thin, type)

    def setMonitors(self, names, thin, type):
        """Add monitors for multiple variables on all Console instances.

        Parameters
        ----------
        names : list[str]
            Variable names to monitor.
        thin : int
            Thinning interval for each monitor.
        type : str
            Monitor type (e.g. ``'trace'``, ``'mean'``, ``'variance'``).
        """
        for name in names:
            self.setMonitor(name, thin, type)

    def clearMonitor(self, name, type):
        """Remove a monitor for a variable from all Console instances.

        Parameters
        ----------
        name : str
            Variable name whose monitor should be removed.
        type : str
            Monitor type to clear (e.g. ``'trace'``).
        """
        for c in self.consoles:
            c.clearMonitor(name, type)

    def dumpMonitors(self, type, flat):
        """Retrieve monitored samples from all Console instances.

        Samples from each Console are concatenated along the last axis
        (the chain dimension) so that the result has all chains combined.

        Parameters
        ----------
        type : str
            Monitor type to dump (e.g. ``'trace'``).
        flat : bool
            If ``True``, return flat (1-D) arrays; if ``False``, return
            arrays with their natural variable dimensions.

        Returns
        -------
        dict[str, numpy.ndarray]
            Dictionary mapping variable names to sample arrays with shape
            ``(*variable_dims, iterations, chains)``.
        """
        ds = [c.dumpMonitors(type, flat) for c in self.consoles]
        return {
            k: np.concatenate([d[k] for d in ds], axis=-1)
            for k in set(k for d in ds for k in d)
        }

    def initialize(self):
        """Initialize all Console instances after compilation."""
        for c in self.consoles:
            c.initialize()

    def isAdapting(self):
        """Return ``True`` if any Console is still in adaptation mode."""
        return any(c.isAdapting() for c in self.consoles)

    def checkAdaptation(self):
        """Return ``True`` if any Console reports successful adaptation."""
        return any(c.checkAdaptation() for c in self.consoles)

    def variableNames(self):
        """Return variable names defined in the model.

        Returns
        -------
        list[str]
            Variable names from the first Console instance (all
            consoles share the same model specification).
        """
        return self.consoles[0].variableNames()

    def dumpState(self, type, chain):
        """Dump the current state of a specific chain.

        Parameters
        ----------
        type : int
            State type flag (e.g. ``DUMP_ALL``, ``DUMP_PARAMETERS``,
            ``DUMP_DATA``).
        chain : int
            Chain number (1-indexed).

        Returns
        -------
        dict[str, numpy.ndarray]
            Dictionary mapping variable names to their current values
            in the specified chain.
        """
        console, chain = self.chains[chain]
        return console.dumpState(type, chain)

    def dumpSamplers(self):
        """Return sampler information from the first Console instance.

        Returns
        -------
        list[list[str]]
            A list of sampler descriptions, where each inner list
            contains the node name and its assigned sampler method.
        """
        return self.consoles[0].dumpSamplers()

    def iter(self):
        """Return the current iteration count from the first Console.

        Returns
        -------
        int
            Number of iterations completed so far.
        """
        return self.consoles[0].iter()
