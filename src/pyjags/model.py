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

"""High-level interface to JAGS for Bayesian model specification and MCMC sampling.

This module provides the :class:`Model` class, the primary public API for
compiling JAGS models, running adaptation, and drawing posterior samples.
It also provides :class:`SamplingState` for generator-based sampling and
:func:`check_model` for syntax validation.
"""

__all__ = ["Model", "SamplingState", "check_model"]

import collections.abc
import contextlib
import os
import re
import sys
import tempfile
import typing as tp
import warnings

import numpy as np

from ._multi_console import MultiConsole
from ._rng import seed_to_chain_inits as _seed_to_chain_inits
from .console import DUMP_ALL, DUMP_DATA, DUMP_PARAMETERS, Console, JagsError
from .modules import load_module
from .progressbar import const_time_partition, progress_bar_factory

# Special value indicating missing data in JAGS.
JAGS_NA = -sys.float_info.max * (1 - 1e-15)


def dict_to_jags(src):
    """Convert a Python dictionary to a format suitable for JAGS.

    Prepares data for consumption by the JAGS C++ engine:

    * Returned arrays have at least one dimension.
    * Empty arrays are removed from the dictionary.
    * Masked values are replaced with ``JAGS_NA``.

    Parameters
    ----------
    src : dict[str, array_like]
        Dictionary mapping variable names to array-like values.
        Values may be numpy arrays, scalars, or masked arrays.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary with the same keys (minus empty arrays) and values
        converted to numpy arrays suitable for JAGS.
    """
    dst = {}
    for k, v in src.items():
        if np.ma.is_masked(v):
            v = np.ma.array(data=v, dtype=np.double, ndmin=1, fill_value=JAGS_NA)
            v = np.ma.filled(v)
        else:
            v = np.atleast_1d(v)
        if not np.size(v):
            continue
        dst[k] = v
    return dst


def dict_from_jags(src):
    """Convert a dictionary returned from JAGS to Python-friendly format.

    Arrays containing ``JAGS_NA`` sentinel values are converted to
    ``numpy.ma.MaskedArray`` so that missing data is handled transparently.

    Parameters
    ----------
    src : dict[str, numpy.ndarray]
        Dictionary mapping variable names to numpy arrays as returned
        by the JAGS C++ engine.

    Returns
    -------
    dict[str, numpy.ndarray | numpy.ma.MaskedArray]
        Dictionary with the same keys. Values that contained ``JAGS_NA``
        are replaced by masked arrays; other values are passed through
        unchanged.
    """
    dst = {}
    for k, v in src.items():
        mask = v == JAGS_NA
        # Don't mask if it not necessary
        if np.any(mask):
            v = np.ma.masked_equal(v, JAGS_NA, copy=False)
        dst[k] = v
    return dst


def check_locale_compatibility():
    """Verify that the current locale uses a period as the decimal separator.

    JAGS requires a locale where the decimal point character is ``'.'``.
    If the active locale uses a different character (e.g. ``','`` in some
    European locales), this function raises ``ValueError`` with instructions
    on how to set a compatible locale.

    Raises
    ------
    ValueError
        If the current locale's decimal point is not ``'.'``.
    """
    import locale
    import textwrap

    if locale.localeconv().get("decimal_point") != ".":
        msg = """\
        JAGS requires locale that uses period as decimal point character. The standard C locale would be one possible choice. It can be configured as follows:
        > import locale
        > locale.setlocale(locale.LC_ALL, 'C')"""
        raise ValueError(textwrap.dedent(msg))


def _annotate_jags_error(error, code):
    """Re-raise a JagsError with source context if a line number is mentioned.

    When JAGS reports a syntax error like "syntax error on line 3", this
    function annotates the error message with the surrounding model code
    lines so the user can see what went wrong.
    """
    msg = str(error)
    if code is None:
        return error
    match = re.search(r"line (\d+)", msg)
    if not match:
        return error
    if isinstance(code, bytes):
        code = code.decode("utf-8", errors="replace")
    lineno = int(match.group(1))
    lines = code.splitlines()
    context_start = max(0, lineno - 3)
    context_end = min(len(lines), lineno + 2)
    context_lines = []
    for i, line in enumerate(lines[context_start:context_end], start=context_start):
        marker = "-->" if i + 1 == lineno else "   "
        context_lines.append(f"  {marker} {i + 1}: {line}")
    context = "\n".join(context_lines)
    return JagsError(f"{msg}\n\nModel code:\n{context}")


@contextlib.contextmanager
def model_path(file=None, code=None, encoding="utf-8"):
    """Context manager that yields a filesystem path to a JAGS model file.

    If *file* is given, its path is yielded directly. If *code* is given,
    the code is written to a temporary file and the temporary path is
    yielded. The temporary file is deleted on context exit.

    Parameters
    ----------
    file : str or path-like, optional
        Path to an existing model file on disk.
    code : str or bytes, optional
        Model code to write to a temporary file.
    encoding : str, optional
        Encoding used when *code* is a ``str``. Default is ``'utf-8'``.

    Yields
    ------
    str
        Filesystem path to the model file.

    Raises
    ------
    ValueError
        If neither *file* nor *code* is provided.
    """
    if file:
        yield str(file) if isinstance(file, os.PathLike) else file
    elif code:
        if isinstance(code, str):
            code = code.encode(encoding=encoding)
        # TODO use separate delete to support Windows?
        with tempfile.NamedTemporaryFile() as fh:
            fh.write(code)
            fh.flush()
            yield fh.name
    else:
        raise ValueError("Either model name or model text must be provided.")


class SamplingState:
    """State yielded by :meth:`Model.iter_sample` at each chunk boundary.

    Attributes
    ----------
    samples : dict[str, numpy.ndarray]
        Accumulated samples so far, with shape
        ``(*variable_dims, iterations, chains)``.
    iteration : int
        Total number of iterations sampled so far.
    """

    def __init__(
        self,
        samples: dict[str, np.ndarray],
        iteration: int,
    ):
        """Create a new SamplingState.

        Parameters
        ----------
        samples : dict[str, numpy.ndarray]
            Accumulated MCMC samples so far. Keys are variable names and
            values are arrays with shape
            ``(*variable_dims, iterations, chains)``.
        iteration : int
            Total number of iterations sampled so far (before thinning).
        """
        self.samples = samples
        self.iteration = iteration
        self._ess: dict[str, float] | None = None
        self._rhat: dict[str, float] | None = None
        self._chain_div: dict[str, np.ndarray] | None = None
        self._chain_div_computed: bool = False

    def _compute_diagnostics(self) -> None:
        """Compute effective sample size and R-hat for all variables.

        Converts the accumulated samples to an ``arviz.InferenceData``
        object and computes per-variable ESS and R-hat diagnostics.
        Results are cached in ``_ess`` and ``_rhat``.
        """
        import arviz as az

        from .arviz import from_pyjags

        idata = from_pyjags(self.samples)
        ess_ds = az.ess(idata)
        rhat_ds = az.rhat(idata)
        self._ess = {str(var): float(ess_ds[var]) for var in ess_ds.data_vars}
        self._rhat = {str(var): float(rhat_ds[var]) for var in rhat_ds.data_vars}

    @property
    def ess(self) -> dict[str, float]:
        """Per-variable effective sample size (computed lazily)."""
        if self._ess is None:
            self._compute_diagnostics()
        return self._ess  # type: ignore[return-value]

    @property
    def rhat(self) -> dict[str, float]:
        """Per-variable R-hat statistic (computed lazily)."""
        if self._rhat is None:
            self._compute_diagnostics()
        return self._rhat  # type: ignore[return-value]

    @property
    def chain_divergence(self) -> dict[str, np.ndarray] | None:
        """Pairwise chain divergence matrix via energy distance.

        Requires the ``divergence`` package.  Returns ``None`` if it is
        not installed.  Computed lazily on first access.

        Returns
        -------
        dict[str, numpy.ndarray] or None
            Mapping of variable names to ``(n_chains, n_chains)``
            pairwise energy distance matrices, or ``None`` if
            ``divergence`` is not available.
        """
        if not self._chain_div_computed:
            self._chain_div_computed = True
            try:
                from .diagnostics import _import_divergence

                div = _import_divergence()
                from .arviz import from_pyjags

                idata = from_pyjags(self.samples)
                self._chain_div = div.chain_divergence(idata)
            except ImportError:
                self._chain_div = None
        return self._chain_div


def check_model(
    code: str | bytes | None = None,
    *,
    file: str | os.PathLike | None = None,
    encoding: str = "utf-8",
) -> bool:
    """Validate JAGS model syntax without compiling.

    Parameters
    ----------
    code : str or bytes, optional
        Model code to validate.
    file : str or path-like, optional
        Path to a model file to validate.
    encoding : str
        Encoding for model code strings (default ``'utf-8'``).

    Returns
    -------
    bool
        ``True`` if the model syntax is valid.

    Raises
    ------
    JagsError
        If the model contains syntax errors, with annotated source context.
    ValueError
        If neither *code* nor *file* is provided.
    """
    console = Console()
    with model_path(file, code, encoding) as path:
        try:
            console.checkModel(path)
        except JagsError as e:
            raise _annotate_jags_error(e, code) from None
    return True


class Model:
    """High level representation of JAGS model.

    Attributes
    ----------
    chains : int
        A number of chains in the model.

    Note
    ----
    In JAGS arrays are indexed from 1. On the other hand Python uses 0 based
    indexing. It is important to keep this in mind when providing data to JAGS
    and interpreting resulting samples. For example, what in JAGS would be
    x[4,2,7] in Python is x[3,1,6].

    Note
    ----
    The JAGS supports data sets where some of observations have no value.
    In PyJAGS those missing values are described using numpy MaskedArray.
    For example, to create a model with observations x[1] = 0.25, x[3] = 0.75,
    and observation x[2] missing, we would provide following data to Model
    constructor:

    >>> {'x': np.ma.masked_array(data=[0.25, 0, 0.75], mask=[False, True, False])}
    {'x': masked_array(data = [0.25 -- 0.75],
                 mask = [False  True False],
           fill_value = 1e+20)
    }

    From JAGS version 4.0.0 it is also possible to monitor variables that are
    not completely defined in the description of the model, e.g., if y[i] is
    defined only for y[3], then y[1], and y[2] will have missing values for
    all iterations in all chains. Those missing values are also represented
    using numpy MaskedArray.
    """

    def __init__(
        self,
        code=None,
        data=None,
        init=None,
        chains=4,
        adapt=1000,
        file=None,
        encoding="utf-8",
        generate_data=True,
        progress_bar=True,
        refresh_seconds=None,
        threads=1,
        chains_per_thread=1,
        seed=None,
    ):
        """
        Create a JAGS model and run adaptation steps.

        Parameters
        ----------
        code : str or bytes, optional
            Code of the model to load. Model may be also provided with file
            keyword argument.
        file : str or path-like, optional
            Path to the model to load. Model may be also provided with code
            keyword argument.
        init : dict or list of dicts, optional
            Specifies initial values for parameters. It can be either a
            dictionary providing initial values for parameters used as keys,
            or a list of dictionaries providing initial values separately for
            each chain. If omitted, initial values will be generated
            automatically.

            Additionally this option allows to configure random number
            generators using following special keys:

             * '.RNG.name'  str, name of random number generator
             * '.RNG.seed'  int, seed for random number generator
             * '.RNG.state' array, may be specified instead of seed, shape of
               array depends on particular generator used
        data : dict, optional
            Dictionary with observed nodes in the model. Keys are variable
            names and values should be convertible to numpy arrays with shape
            compatible with one used in the model.

            The numpy.ma.MaskedArray can be used to provide data where some of
            observations are missing.
        generate_data : bool, optional
            If true, data block in the model is used to generate data.
        chains : int, 4 by default
            A positive number specifying number of parallel chains.
        adapt : int, 1000 by default
            An integer specifying number of adaptations steps.
        encoding : str, 'utf-8' by default
            When model code is provided as a string, this specifies its encoding.
        progress_bar : bool, optional
            If true, enables the progress bar.
        threads: int, 1 by default
            A positive integer specifying number of threads used to sample from
            model. Using more than one thread is experimental functionality.
        chains_per_thread: int, 1 by default
            A positive integer specifying a maximum number of chains sampled in
            a single thread. Takes effect only when using more than one thread.
        seed : int, optional
            Random seed for reproducible sampling. Deterministically derives
            per-chain RNG names and seeds using
            ``numpy.random.SeedSequence``. Mutually exclusive with providing
            ``.RNG.name`` or ``.RNG.seed`` in *init*.
        """

        check_locale_compatibility()

        init = self._merge_seed_into_init(seed, init, chains)
        self._setup_console(
            chains, threads, chains_per_thread, progress_bar, refresh_seconds
        )

        with model_path(file, code, encoding) as path:
            try:
                self.console.checkModel(path)
            except JagsError as e:
                raise _annotate_jags_error(e, code) from None

        self._init_compile(data, generate_data)
        self._init_parameters(init)
        self.console.initialize()
        if adapt:
            self.adapt(adapt)

    @staticmethod
    def _merge_seed_into_init(seed, init, chains):
        """Merge seed-derived RNG configuration into the init dictionaries.

        Parameters
        ----------
        seed : int or None
            Master seed for reproducible sampling.
        init : dict, list of dicts, or None
            User-provided initial values.
        chains : int
            Number of chains.

        Returns
        -------
        dict, list of dicts, or None
            The init dictionaries with RNG entries merged in, or the
            original init if no seed was provided.
        """
        if seed is None:
            return init

        if init is not None:
            inits = [init] if isinstance(init, collections.abc.Mapping) else init
            for d in inits:
                if any(k in d for k in (".RNG.name", ".RNG.seed", ".RNG.state")):
                    raise ValueError(
                        "Cannot specify both 'seed' and RNG keys "
                        "('.RNG.name', '.RNG.seed', '.RNG.state') in 'init'."
                    )

        seed_inits = _seed_to_chain_inits(seed, chains)
        if init is None:
            return seed_inits
        elif isinstance(init, collections.abc.Mapping):
            return [{**init, **si} for si in seed_inits]
        else:
            return [{**d, **si} for d, si in zip(init, seed_inits, strict=True)]

    def _setup_console(
        self, chains, threads, chains_per_thread, progress_bar, refresh_seconds
    ):
        """Configure console, threading, and progress bar.

        Parameters
        ----------
        chains : int
            Number of MCMC chains.
        threads : int
            Number of threads for parallel execution.
        chains_per_thread : int
            Maximum chains per Console instance.
        progress_bar : bool
            Whether to show a progress bar.
        refresh_seconds : float or None
            Progress bar refresh interval.
        """
        load_module("basemod")
        load_module("bugs")
        load_module("lecuyer")

        self.refresh_seconds = refresh_seconds or 0.5 if sys.stdout.isatty() else 5.0
        self.progress_bar = progress_bar_factory(
            progress_bar, refresh_seconds=self.refresh_seconds
        )
        self.chains = chains
        self.threads = threads
        self.use_threads = self.threads > 1 and chains_per_thread < self.chains

        if self.use_threads:
            self.console = MultiConsole(self.chains, chains_per_thread)
        else:
            self.console = Console()

    def _init_compile(self, data, generate_data):
        """Compile the model with observed data."""
        if data is None:
            data = {}
        data = dict_to_jags(data)
        unused = set(data.keys()) - set(self.variables)
        if unused:
            raise ValueError("Unused data for variables: {}".format(",".join(unused)))
        self.console.compile(data, self.chains, generate_data)

    def _init_parameters(self, init):
        """Set parameters and configure random number generators."""
        if init is None:
            init = {}
        if isinstance(init, collections.abc.Mapping):
            init = [init] * self.chains
        elif not isinstance(init, collections.abc.Sequence):
            raise ValueError("Init should be a sequence or a dictionary.")
        if len(init) != self.chains:
            raise ValueError(
                "Length of init sequence should equal the number of chains."
            )

        if self.use_threads:
            rngs = Console.parallel_rngs("lecuyer::RngStream", self.chains)
        else:
            rngs = [{".RNG.name": None, ".RNG.seed": None}] * self.chains

        for data, rng, chain in zip(init, rngs, range(1, self.chains + 1), strict=True):
            data = dict(data)
            rng_name = data.pop(".RNG.name", None)
            if self.use_threads and rng_name is None:
                rng_name = rng[".RNG.name"]
                data[".RNG.state"] = rng[".RNG.state"]
            if rng_name is not None:
                self.console.setRNGname(rng_name, chain)
            data = dict_to_jags(data)

            unused = set(data.keys())
            unused.difference_update(self.variables)
            unused.difference_update([".RNG.seed", ".RNG.state"])
            if unused:
                raise ValueError(
                    "Unused initial values in chain {} for variables: {}".format(
                        chain, ",".join(unused)
                    )
                )
            self.console.setParameters(data, chain)

    def _update(self, iterations, header):
        """Run MCMC updates with progress bar, dispatching to sequential or parallel."""
        method = self._update_parallel if self.use_threads else self._update_sequential

        with self.progress_bar(self.chains * iterations, header=header) as pb:
            method(pb, iterations)

    def _update_sequential(self, progress, iterations):
        """Run MCMC updates sequentially on a single console."""
        for steps in const_time_partition(iterations, self.refresh_seconds):
            self.console.update(steps)
            progress.update(self.chains * steps)

    def _update_parallel(self, progress, iterations):
        """Run MCMC updates in parallel across consoles using threads."""
        from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
        from threading import Event

        with ThreadPoolExecutor(self.threads) as executor:
            # Event used to interrupt inner threads (which are
            # non-interruptable by default).
            interrupt = Event()

            def update(console, chains):
                """Run MCMC updates on a single console in a worker thread."""
                for steps in const_time_partition(iterations, self.refresh_seconds):
                    if interrupt.is_set():
                        break
                    console.update(steps)
                    progress.update(chains * steps)

            fs = [
                executor.submit(update, console, chains)
                for console, chains in zip(
                    self.console.consoles,
                    self.console.chains_per_console,
                    strict=True,
                )
            ]
            try:
                (done, not_done) = wait(fs, return_when=ALL_COMPLETED)
                for d in done:
                    d.result()
                for _d in not_done:
                    raise AssertionError("Not all futures completed")
            except KeyboardInterrupt:
                interrupt.set()
                raise

    def update(self, iterations):
        """Update the model for a given number of iterations.

        Runs the MCMC sampler without recording monitored values.
        This is typically used for burn-in.

        Parameters
        ----------
        iterations : int
            A positive integer specifying the number of iterations to run.
        """
        self._update(iterations, "updating: ")

    def adapt(self, iterations):
        """Run adaptation steps to maximize sampler efficiency.

        During adaptation, JAGS tunes its samplers to improve mixing.
        If the model does not require adaptation, this method returns
        immediately.

        Parameters
        ----------
        iterations : int
            A positive integer specifying the number of adaptation steps.

        Returns
        -------
        bool
            ``True`` if the achieved performance is close to the theoretical
            optimum for all samplers.
        """
        if not self.console.isAdapting():
            # Model does not require adaptation
            return True
        self._update(iterations, "adapting: ")
        return self.console.checkAdaptation()

    @property
    def variables(self):
        """Variable names used in the model.

        Returns
        -------
        list[str]
            Names of all variables defined in the JAGS model.
        """
        return self.console.variableNames()

    @property
    def state(self):
        """Values of model parameters and model data for each chain.

        Returns
        -------
        list[dict[str, numpy.ndarray | numpy.ma.MaskedArray]]
            A list with one dictionary per chain.  Each dictionary maps
            variable names to their current values, including both
            parameters and data.

        See Also
        --------
        parameters
        data
        """
        return [
            dict_from_jags(self.console.dumpState(DUMP_ALL, chain))
            for chain in range(1, self.chains + 1)
        ]

    @property
    def parameters(self):
        """Values of model parameters for each chain.

        Includes the name of the random number generator as
        ``'.RNG.name'`` and its state as ``'.RNG.state'``.

        Returns
        -------
        list[dict[str, numpy.ndarray | numpy.ma.MaskedArray]]
            A list with one dictionary per chain.  Each dictionary maps
            parameter names to their current values.
        """
        return [
            dict_from_jags(self.console.dumpState(DUMP_PARAMETERS, chain))
            for chain in range(1, self.chains + 1)
        ]

    @property
    def data(self):
        """Model data for the first chain.

        Includes data provided during model construction and data
        generated as part of the ``data`` block in the JAGS model.

        Returns
        -------
        dict[str, numpy.ndarray | numpy.ma.MaskedArray]
            Dictionary mapping variable names to their data values.
        """
        return dict_from_jags(self.console.dumpState(DUMP_DATA, 1))

    @property
    def samplers(self) -> list[list[str]]:
        """Information about the samplers used for each node.

        Returns
        -------
        list[list[str]]
            A list of sampler descriptions.  Each inner list contains
            the node name and the sampler method assigned to it.
        """
        return self.console.dumpSamplers()

    @property
    def is_adapted(self) -> bool:
        """Whether adaptation has achieved optimal performance.

        Returns
        -------
        bool
            ``True`` if all samplers have reached their optimal
            configuration.
        """
        return self.console.checkAdaptation()

    @property
    def iteration(self) -> int:
        """Current iteration count of the model.

        Returns
        -------
        int
            The total number of iterations completed so far, including
            adaptation, burn-in, and sampling iterations.
        """
        if self.use_threads:
            return self.console.iter()
        return self.console.iter()

    def __repr__(self) -> str:
        """Return a concise string representation of the Model."""
        n_vars = len(self.variables)
        return (
            f"Model(chains={self.chains}, variables={n_vars}, "
            f"iteration={self.iteration}, adapted={self.is_adapted})"
        )

    def iter_sample(
        self,
        iterations: int = 100000,
        chunk_size: int = 1000,
        vars: tp.Sequence[str] | None = None,
        thin: int = 1,
        monitor_type: str = "trace",
    ) -> tp.Generator[SamplingState, None, None]:
        """Yield sampling state after each chunk of iterations.

        A generator that samples *chunk_size* iterations at a time, merging
        results incrementally.  Useful for convergence monitoring, live
        diagnostics, and checkpointing.

        Parameters
        ----------
        iterations : int
            Maximum total iterations to sample.
        chunk_size : int
            Number of iterations per chunk.
        vars : list of str, optional
            Variables to monitor.  Defaults to all model variables.
        thin : int
            Thinning interval.
        monitor_type : str
            Monitor type (default ``'trace'``).

        Yields
        ------
        SamplingState
            Accumulated samples and diagnostics after each chunk.

        Examples
        --------
        >>> for state in model.iter_sample(iterations=50000, chunk_size=5000):
        ...     if max(state.rhat.values()) < 1.01:
        ...         break
        >>> final_samples = state.samples
        """
        from .chain_utilities import merge_consecutive_chains

        accumulated = None
        total = 0

        while total < iterations:
            n = min(chunk_size, iterations - total)
            new_samples = self.sample(
                iterations=n, vars=vars, thin=thin, monitor_type=monitor_type
            )
            total += n

            if accumulated is None:
                accumulated = new_samples
            else:
                accumulated = merge_consecutive_chains((accumulated, new_samples))

            yield SamplingState(samples=accumulated, iteration=total)

    def sample_more(
        self,
        iterations: int,
        previous_samples: dict[str, np.ndarray],
        vars: tp.Sequence[str] | None = None,
        thin: int = 1,
        monitor_type: str = "trace",
    ) -> dict[str, np.ndarray]:
        """Continue sampling and merge with previous results.

        Leverages the fact that JAGS retains chain state between
        :meth:`sample` calls.

        Parameters
        ----------
        iterations : int
            Number of additional iterations to sample.
        previous_samples : dict
            Samples from a prior call to :meth:`sample` or
            :meth:`sample_more`.
        vars : list of str, optional
            Variables to monitor.  Defaults to all model variables.
        thin : int
            Thinning interval.
        monitor_type : str
            Monitor type (default ``'trace'``).

        Returns
        -------
        dict
            Merged samples (previous + new) concatenated along the
            iteration axis.
        """
        from .chain_utilities import merge_consecutive_chains

        new_samples = self.sample(
            iterations=iterations, vars=vars, thin=thin, monitor_type=monitor_type
        )
        return merge_consecutive_chains((previous_samples, new_samples))

    def sample(
        self,
        iterations,
        vars=None,
        thin=1,
        monitor_type="trace",
        warn_convergence=False,
    ):
        """
        Creates monitors for given variables, runs the model for provided
        number of iterations and returns monitored samples.


        Parameters
        ----------
        iterations : int
            A positive integer specifying number of iterations.
        vars : list of str, optional
            A list of variables to monitor.
        thin : int, optional
            A positive integer specifying thinning interval.
        warn_convergence : bool, optional
            If ``True``, print warnings after sampling if any variable has
            R-hat > 1.01 or ESS < 100 per chain.
        Returns
        -------
        dict
            Sampled values of monitored variables as a dictionary where keys
            are variable names and values are numpy arrays with shape:
            (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
            shape of variable in JAGS model.
        """
        if vars is None:
            vars = self.variables
        monitored = []
        try:
            for name in vars:
                self.console.setMonitor(name, thin, monitor_type)
                monitored.append(name)
            self._update(iterations, "sampling: ")
            samples = self.console.dumpMonitors(monitor_type, False)
            samples = dict_from_jags(samples)
        finally:
            for name in monitored:
                self.console.clearMonitor(name, monitor_type)

        if warn_convergence and len(vars) > 0:
            self._check_convergence(samples)

        return samples

    @staticmethod
    def _check_convergence(samples: dict[str, np.ndarray]) -> None:
        """Emit warnings if MCMC convergence diagnostics are poor.

        Converts *samples* to an ``arviz.InferenceData`` object and
        computes R-hat and effective sample size (ESS) for each variable.
        Warnings are issued for variables with R-hat > 1.01 or ESS < 100.

        Parameters
        ----------
        samples : dict[str, numpy.ndarray]
            Dictionary mapping variable names to sample arrays with shape
            ``(*variable_dims, iterations, chains)``.

        Notes
        -----
        If ``arviz`` is not installed or any error occurs during
        computation, the method silently returns without issuing warnings.
        """
        try:
            import arviz as az

            from .arviz import from_pyjags

            idata = from_pyjags(samples)
            rhat = az.rhat(idata)
            ess = az.ess(idata)

            bad_rhat = [str(var) for var in rhat.data_vars if float(rhat[var]) > 1.01]
            bad_ess = [str(var) for var in ess.data_vars if float(ess[var]) < 100]

            if bad_rhat:
                warnings.warn(
                    f"R-hat > 1.01 for variables: {', '.join(bad_rhat)}. "
                    "Consider running more iterations.",
                    stacklevel=3,
                )
            if bad_ess:
                warnings.warn(
                    f"ESS < 100 for variables: {', '.join(bad_ess)}. "
                    "Consider running more iterations.",
                    stacklevel=3,
                )
        except Exception:
            pass
