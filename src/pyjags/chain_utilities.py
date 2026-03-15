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

"""Utilities for manipulating MCMC sample dictionaries.

Functions for discarding burn-in, merging chains from consecutive or
parallel sampling runs, and extracting final iterations for model
re-initialization.
"""

import numbers
import typing as tp

import numpy as np


def get_chain_length(samples: dict[str, np.ndarray]) -> int:
    """Determine the length of the chains in a samples dictionary.

    Parameters
    ----------
    samples : dict[str, numpy.ndarray]
        Dictionary mapping variable names to numpy arrays with shape
        ``(*variable_dims, chain_length, n_chains)``.

    Returns
    -------
    int
        The chain length (number of iterations) common to all variables.

    Raises
    ------
    ValueError
        If *samples* is empty or if chain lengths are inconsistent
        across variables.
    """
    chain_lengths = set(value.shape[1] for key, value in samples.items())

    if samples is None or len(samples) == 0:
        raise ValueError("The samples object must not be empty")

    if len(chain_lengths) > 1:
        raise ValueError("The chain lengths are not consistent across variables.")

    return next(iter(chain_lengths))


def discard_burn_in_samples(
    samples: dict[str, np.ndarray], burn_in: int
) -> dict[str, np.ndarray]:
    """Discard burn-in samples from the beginning of each chain.

    Parameters
    ----------
    samples : dict[str, numpy.ndarray]
        Dictionary mapping variable names to numpy arrays with shape
        ``(*variable_dims, chain_length, n_chains)``.
    burn_in : int
        Number of initial iterations to discard from each chain.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary with the same keys and arrays trimmed to shape
        ``(*variable_dims, chain_length - burn_in, n_chains)``.
    """
    return {
        variable_name: sample_chain[:, burn_in:, :]
        for variable_name, sample_chain in samples.items()
    }


def extract_final_iteration_from_samples_for_initialization(
    samples: dict[str, np.ndarray], variable_names: set[str]
) -> list[dict[str, numbers.Number | np.ndarray]]:
    """Extract the last iteration from each chain for re-initialization.

    This is useful for continuing sampling from the final state of a
    previous run by passing the result as the ``init`` parameter to
    :class:`~pyjags.Model`.

    Parameters
    ----------
    samples : dict[str, numpy.ndarray]
        Dictionary mapping variable names to numpy arrays with shape
        ``(*variable_dims, chain_length, n_chains)``.
    variable_names : set[str]
        Set of variable names to extract.

    Returns
    -------
    list[dict[str, numbers.Number | numpy.ndarray]]
        A list with one dictionary per chain.  Each dictionary maps
        variable names to their values at the final iteration (scalars
        are squeezed).

    Raises
    ------
    ValueError
        If the number of chains is inconsistent across the requested
        variables.
    """
    numbers_of_chains = [
        samples[variable_name].shape[2] for variable_name in variable_names
    ]

    if any(
        number_of_chains != numbers_of_chains[0]
        for number_of_chains in numbers_of_chains
    ):
        raise ValueError("The number of chains must be identical across parameters")

    number_of_chains = numbers_of_chains[0]

    result = []

    for chain in range(number_of_chains):
        init_chain = {}
        result.append(init_chain)
        for variable_name in variable_names:
            init_chain[variable_name] = samples[variable_name][:, -1, chain].squeeze()

    return result


def _check_sequence_of_chains_present(
    sequence_of_chains: tp.Sequence[dict[str, np.ndarray]],
):
    """
    This function verifies that e sequence of samples is not empty not None.

    Parameters
    ----------
    sequence_of_chains: a sequence of sample dictionaries

    Returns
    -------

    """
    if sequence_of_chains is None:
        raise ValueError("sequence_of_chains must not be none")

    if len(sequence_of_chains) == 0:
        raise ValueError("sequence_of_chains must contain at least one chain")


def _verify_and_get_variable_names_from_sequence_of_samples(
    sequence_of_samples: tp.Sequence[dict[str, np.ndarray]],
) -> set[str]:
    """
    This function verifies that all sample dictionaries in a sequence contain
    the same set of variables and returns this set of variables.

    Parameters
    ----------
    sequence_of_samples: a sequence of sample dictionaries

    Returns
    -------

    """
    sequence_of_variable_name_sets = [
        set(sample_chain.keys()) for sample_chain in sequence_of_samples
    ]

    for variable_names in sequence_of_variable_name_sets:
        if variable_names != sequence_of_variable_name_sets[0]:
            raise ValueError(
                "Each sample dictionary must contain the same set of variables."
            )

    return sequence_of_variable_name_sets[0]


def merge_consecutive_chains(
    sequence_of_samples: tp.Sequence[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenate sample dictionaries along the iteration axis.

    Merges consecutive sampling runs into a single dictionary by
    appending iterations.  This is useful when samples have been drawn
    from JAGS in successive calls where each run continues from the
    final state of the previous one.

    Parameters
    ----------
    sequence_of_samples : sequence of dict[str, numpy.ndarray]
        Two or more sample dictionaries, each mapping variable names to
        arrays with shape ``(*variable_dims, chain_length_i, n_chains)``.
        All dictionaries must share the same variable names, variable
        dimensions, and number of chains.

    Returns
    -------
    dict[str, numpy.ndarray]
        Merged sample dictionary with arrays of shape
        ``(*variable_dims, sum(chain_length_i), n_chains)``.

    Raises
    ------
    ValueError
        If variable dimensions or chain counts are inconsistent, or if
        the input sequence is empty or ``None``.
    """

    _check_sequence_of_chains_present(sequence_of_samples)

    merged_samples = {}

    variable_names = _verify_and_get_variable_names_from_sequence_of_samples(
        sequence_of_samples
    )

    for variable_name in variable_names:
        sequence_of_shapes = [
            sample_chains[variable_name].shape for sample_chains in sequence_of_samples
        ]

        sequence_of_numpy_arrays = [
            sample_chains[variable_name] for sample_chains in sequence_of_samples
        ]

        parameter_dimension, _, number_of_chains = sequence_of_shapes[0]

        if not all(shape[0] == parameter_dimension for shape in sequence_of_shapes):
            raise ValueError(
                f"The dimension of {variable_name} is inconsistent between samples."
            )

        if not all(shape[2] == number_of_chains for shape in sequence_of_shapes):
            raise ValueError("The number of chains is inconsistent across samples.")

        merged_samples[variable_name] = np.concatenate(sequence_of_numpy_arrays, axis=1)

    return merged_samples


def merge_parallel_chains(
    sequence_of_samples: tp.Sequence[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenate sample dictionaries along the chain axis.

    Merges independently started sampling runs by adding chains.
    This is useful when multiple models were run in parallel with
    different initial values.

    Parameters
    ----------
    sequence_of_samples : sequence of dict[str, numpy.ndarray]
        Two or more sample dictionaries, each mapping variable names to
        arrays with shape ``(*variable_dims, chain_length, n_chains_i)``.
        All dictionaries must share the same variable names, variable
        dimensions, and chain length.

    Returns
    -------
    dict[str, numpy.ndarray]
        Merged sample dictionary with arrays of shape
        ``(*variable_dims, chain_length, sum(n_chains_i))``.

    Raises
    ------
    ValueError
        If variable dimensions or chain lengths are inconsistent, or if
        the input sequence is empty or ``None``.
    """
    _check_sequence_of_chains_present(sequence_of_samples)

    merged_samples = {}

    variable_names = _verify_and_get_variable_names_from_sequence_of_samples(
        sequence_of_samples
    )

    for variable_name in variable_names:
        sequence_of_shapes = [
            sample_chains[variable_name].shape for sample_chains in sequence_of_samples
        ]

        sequence_of_numpy_arrays = [
            sample_chains[variable_name] for sample_chains in sequence_of_samples
        ]

        parameter_dimension, chain_length, _ = sequence_of_shapes[0]

        if not all(shape[0] == parameter_dimension for shape in sequence_of_shapes):
            raise ValueError(
                f"The dimension of {variable_name} is inconsistent between samples."
            )

        if not all(shape[1] == chain_length for shape in sequence_of_shapes):
            raise ValueError("The chain lengths are inconsistent across samples.")

        merged_samples[variable_name] = np.concatenate(sequence_of_numpy_arrays, axis=2)

    return merged_samples
