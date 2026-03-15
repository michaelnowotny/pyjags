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

"""Type stubs for the pyjags.console C++ extension module."""

import enum

import numpy as np
import numpy.typing as npt

class JagsError(Exception):
    """Exception raised by the JAGS engine for model errors."""

    ...

class DumpType(enum.IntEnum):
    """Flags controlling which parts of the model state to dump."""

    DUMP_DATA: int
    DUMP_PARAMETERS: int
    DUMP_ALL: int

DUMP_DATA: int
DUMP_PARAMETERS: int
DUMP_ALL: int

class FactoryType(enum.IntEnum):
    """Enumerates factory types in a JAGS model."""

    SAMPLER_FACTORY: int
    MONITOR_FACTORY: int
    RNG_FACTORY: int

SAMPLER_FACTORY: int
MONITOR_FACTORY: int
RNG_FACTORY: int

class Console:
    """Low-level wrapper around the JAGS Console class.

    Each ``Console`` instance manages a single JAGS model with one or
    more chains. Most users should use :class:`pyjags.Model` instead.
    """

    def __init__(self) -> None: ...
    def checkModel(self, path: str) -> None:
        """Load a model from *path* and check its syntactic correctness.

        Parameters
        ----------
        path : str
            Filesystem path to a BUGS model file.

        Raises
        ------
        JagsError
            If the model contains syntax errors.
        """
        ...

    def compile(
        self, data: dict[str, npt.NDArray[np.float64]], chains: int, generate_data: bool
    ) -> None:
        """Compile the model with observed data.

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            Observed data as a dictionary of numpy arrays.
        chains : int
            Number of parallel chains.
        generate_data : bool
            Whether to generate data from ``data {}`` blocks.

        Raises
        ------
        JagsError
            If compilation fails.
        """
        ...

    def setParameters(
        self, parameters: dict[str, npt.NDArray[np.float64]], chain: int
    ) -> None:
        """Set initial parameter values for a chain.

        Parameters
        ----------
        parameters : dict[str, numpy.ndarray]
            Parameter name-to-value mapping.
        chain : int
            Chain index (1-based).

        Raises
        ------
        JagsError
            If the parameters are invalid.
        """
        ...

    def setRNGname(self, name: str, chain: int) -> None:
        """Set the RNG algorithm for a chain.

        Parameters
        ----------
        name : str
            RNG name (e.g., ``"base::Mersenne-Twister"``).
        chain : int
            Chain index (1-based).

        Raises
        ------
        JagsError
            If the RNG name is invalid.
        """
        ...

    def initialize(self) -> None:
        """Initialize all chains.

        Raises
        ------
        JagsError
            If initialization fails.
        """
        ...

    def update(self, iterations: int) -> None:
        """Advance the MCMC sampler by *iterations* steps.

        The GIL is released during sampling to allow multithreading.

        Parameters
        ----------
        iterations : int
            Number of MCMC iterations to run.

        Raises
        ------
        JagsError
            If the update fails.
        """
        ...

    def setMonitor(self, name: str, thin: int, type: str) -> None:
        """Start monitoring a node.

        Parameters
        ----------
        name : str
            Node name to monitor.
        thin : int
            Thinning interval (keep every *thin*-th sample).
        type : str
            Monitor type (e.g., ``"trace"``).

        Raises
        ------
        JagsError
            If the node cannot be monitored.
        """
        ...

    def setMonitors(self, names: list[str], thin: int, type: str) -> None:
        """Start monitoring multiple nodes.

        Parameters
        ----------
        names : list[str]
            Node names to monitor.
        thin : int
            Thinning interval.
        type : str
            Monitor type.

        Raises
        ------
        JagsError
            If any node cannot be monitored.
        """
        ...

    def clearMonitor(self, name: str, type: str) -> None:
        """Stop monitoring a node.

        Parameters
        ----------
        name : str
            Node name.
        type : str
            Monitor type.

        Raises
        ------
        JagsError
            If the monitor cannot be cleared.
        """
        ...

    def dumpState(
        self, type: DumpType, chain: int
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Dump the model state for a chain.

        Parameters
        ----------
        type : DumpType
            Which parts of the state to dump.
        chain : int
            Chain index (1-based).

        Returns
        -------
        dict[str, numpy.ndarray]
            State dictionary. May include ``".RNG.name"`` as a string.

        Raises
        ------
        JagsError
            If the state cannot be dumped.
        """
        ...

    def iter(self) -> int:
        """Return the current iteration number of the model.

        Returns
        -------
        int
            Current iteration count.
        """
        ...

    def variableNames(self) -> list[str]:
        """Return the names of all variables in the model.

        Returns
        -------
        list[str]
            Variable names.
        """
        ...

    def nchain(self) -> int:
        """Return the number of chains.

        Returns
        -------
        int
            Chain count.
        """
        ...

    def dumpMonitors(self, type: str, flat: bool) -> dict[str, npt.NDArray[np.float64]]:
        """Dump all monitored values.

        Parameters
        ----------
        type : str
            Monitor type (e.g., ``"trace"``).
        flat : bool
            Whether to flatten multi-dimensional nodes.

        Returns
        -------
        dict[str, numpy.ndarray]
            Monitored values keyed by variable name.

        Raises
        ------
        JagsError
            If monitors cannot be dumped.
        """
        ...

    def dumpSamplers(self) -> list[list[str]]:
        """Dump sampler names and the nodes they sample.

        Returns
        -------
        list[list[str]]
            Each inner list contains the sampler name followed by the
            names of the nodes it samples.
        """
        ...

    def adaptOff(self) -> None:
        """Turn off adaptive mode.

        Raises
        ------
        JagsError
            If adaptation cannot be disabled.
        """
        ...

    def checkAdaptation(self) -> bool:
        """Check whether adaptation is complete.

        Returns
        -------
        bool
            ``True`` if adaptation has converged.

        Raises
        ------
        JagsError
            If the check fails.
        """
        ...

    def isAdapting(self) -> bool:
        """Return whether the model is still in adaptive mode.

        Returns
        -------
        bool
            ``True`` if the model is adapting.
        """
        ...

    def clearModel(self) -> None:
        """Clear the compiled model, releasing all resources."""
        ...

    @staticmethod
    def loadModule(name: str) -> None:
        """Load a JAGS module by name.

        Parameters
        ----------
        name : str
            Module name (e.g., ``"glm"``).

        Raises
        ------
        JagsError
            If the module cannot be loaded.
        """
        ...

    @staticmethod
    def unloadModule(name: str) -> None:
        """Unload a JAGS module by name.

        Parameters
        ----------
        name : str
            Module name.

        Raises
        ------
        JagsError
            If the module cannot be unloaded.
        """
        ...

    @staticmethod
    def listModules() -> list[str]:
        """Return the names of all loaded modules.

        Returns
        -------
        list[str]
            Module names.
        """
        ...

    @staticmethod
    def listFactories(type: FactoryType) -> list[tuple[str, bool]]:
        """Return loaded factories and their active status.

        Parameters
        ----------
        type : FactoryType
            Factory category to list.

        Returns
        -------
        list[tuple[str, bool]]
            Each tuple contains ``(factory_name, is_active)``.
        """
        ...

    @staticmethod
    def setFactoryActive(name: str, type: FactoryType, active: bool) -> None:
        """Activate or deactivate a factory.

        Parameters
        ----------
        name : str
            Factory name.
        type : FactoryType
            Factory category.
        active : bool
            Whether to activate (``True``) or deactivate (``False``).

        Raises
        ------
        JagsError
            If the factory cannot be found.
        """
        ...

    @staticmethod
    def na() -> float:
        """Return the JAGS sentinel value for missing data.

        Returns
        -------
        float
            ``JAGS_NA`` value.
        """
        ...

    @staticmethod
    def version() -> str:
        """Return the JAGS library version string.

        Returns
        -------
        str
            Version string (e.g., ``"4.3.2"``).
        """
        ...

    @staticmethod
    def parallel_rngs(factory: str, chains: int) -> list[dict[str, object]]:
        """Generate independent RNG states for parallel chain execution.

        Parameters
        ----------
        factory : str
            RNG factory name (e.g., ``"base::Mersenne-Twister"``).
        chains : int
            Number of chains.

        Returns
        -------
        list[dict[str, object]]
            Each dict contains ``".RNG.name"`` (str) and
            ``".RNG.state"`` (list[int]).

        Raises
        ------
        JagsError
            If the factory is not found or not active.
        """
        ...
