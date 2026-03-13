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
import typing as tp


def get_chain_length(samples: tp.Dict[str, np.ndarray]) -> int:
    """
    This function determines the length of the chains in the samples dictionary

    Parameters
    ----------
    samples: a dictionary mapping variable names to Numpy arrays with shape
             (parameter_dimension, chain_length, number_of_chains)

    Returns
    -------
    the chain length

    """
    chain_lengths = set(value.shape[1] for key, value in samples.items())

    if samples is None or len(samples) == 0:
        raise ValueError('The samples object must not be empty')

    if len(chain_lengths) > 1:
        raise ValueError(
            'The chain lengths are not consistent across variables.')

    return next(iter(chain_lengths))


def discard_burn_in_samples(
        samples: tp.Dict[str, np.ndarray],
        burn_in: int) -> tp.Dict[str, np.ndarray]:
    """
    This function discards a given number of samples from the beginning of each
    chain for each variable and returns the remaining samples.

    Parameters
    ----------
    samples: a dictionary mapping variable names to Numpy arrays with shape
             (parameter_dimension, chain_length, number_of_chains)

    burn_in: the number of observations to discard from the beginning

    Returns
    -------
    a dictionary with the remaining samples

    """
    return {variable_name: sample_chain[:, burn_in:, :]
            for variable_name, sample_chain
            in samples.items()}


def extract_final_iteration_from_samples_for_initialization(
        samples: tp.Dict[str, np.ndarray],
        variable_names: tp.Set[str]) \
        -> tp.List[tp.Dict[str, tp.Union[numbers.Number, np.ndarray]]]:
    """
    This function extracts the last iteration from each chain for a given set
    of variables.

    Parameters
    ----------
    samples: a dictionary mapping variable names to Numpy arrays with shape
             (parameter_dimension, chain_length, number_of_chains)

    variable_names: a set of variable names

    Returns
    -------
    a dictionary mapping variable names to a numpy array of final samples

    """
    numbers_of_chains = [samples[variable_name].shape[2]
                         for variable_name
                         in variable_names]

    if any(number_of_chains != numbers_of_chains[0] for number_of_chains in
           numbers_of_chains):
        raise ValueError(
            'The number of chains must be identical across parameters')

    number_of_chains = numbers_of_chains[0]

    result = []

    for chain in range(number_of_chains):
        init_chain = {}
        result.append(init_chain)
        for variable_name in variable_names:
            init_chain[variable_name] = \
                samples[variable_name][:, -1, chain].squeeze()

    return result


def _check_sequence_of_chains_present(
        sequence_of_chains: tp.Sequence[tp.Dict[str, np.ndarray]]):
    """
    This function verifies that e sequence of samples is not empty not None.

    Parameters
    ----------
    sequence_of_chains: a sequence of sample dictionaries

    Returns
    -------

    """
    if sequence_of_chains is None:
        raise ValueError('sequence_of_chains must not be none')

    if len(sequence_of_chains) == 0:
        raise ValueError('sequence_of_chains must contain at least one chain')


def _verify_and_get_variable_names_from_sequence_of_samples(
        sequence_of_samples: tp.Sequence[tp.Dict[str, np.ndarray]]) \
        -> tp.Set[str]:
    """
    This function verifies that all sample dictionaries in a sequence contain
    the same set of variables and returns this set of variables.

    Parameters
    ----------
    sequence_of_samples: a sequence of sample dictionaries

    Returns
    -------

    """
    sequence_of_variable_name_sets = \
        [set(sample_chain.keys())
         for sample_chain
         in sequence_of_samples]

    for variable_names in sequence_of_variable_name_sets:
        if variable_names != sequence_of_variable_name_sets[0]:
            raise ValueError('Each sample dictionary must contain the same set '
                             'of variables.')

    return sequence_of_variable_name_sets[0]


def merge_consecutive_chains(
        sequence_of_samples: tp.Sequence[tp.Dict[str, np.ndarray]]) \
        -> tp.Dict[str, np.ndarray]:
    """
    This function concatenates the chains in sample dictionaries sequentially
    (i.e. continues the chains).
    This is useful if samples have been drawn from JAGS consecutively where each
    new sample chain starts from the last iteration of the previous sample.

    Parameters
    ----------
    sequence_of_samples: a sequence of sample dictionaries

    Returns
    -------
    a single sample dictionary merged along chain iterations
    """

    _check_sequence_of_chains_present(sequence_of_samples)

    merged_samples = {}

    variable_names = _verify_and_get_variable_names_from_sequence_of_samples(sequence_of_samples)

    for variable_name in variable_names:
        sequence_of_shapes = \
            [sample_chains[variable_name].shape
             for sample_chains
             in sequence_of_samples]

        sequence_of_numpy_arrays = \
            [sample_chains[variable_name]
             for sample_chains
             in sequence_of_samples]

        parameter_dimension, _, number_of_chains = sequence_of_shapes[0]

        if not all(shape[0] == parameter_dimension
                   for shape
                   in sequence_of_shapes):
            raise ValueError(f'The dimension of {variable_name} is inconsistent'
                             f' between samples.')

        if not all(shape[2] == number_of_chains
                   for shape
                   in sequence_of_shapes):
            raise ValueError('The number of chains is inconsistent across '
                             'samples.')

        merged_samples[variable_name] = \
            np.concatenate(sequence_of_numpy_arrays, axis=1)

    return merged_samples


def merge_parallel_chains(
        sequence_of_samples: tp.Sequence[tp.Dict[str, np.ndarray]]) \
        -> tp.Dict[str, np.ndarray]:
    """
    This function concatenates sample dictionaries across chains
    (i.e. adds more chains of the same length for the same variables).
    This is useful if the starting value of each set of samples is different
    from the last iteration of the chains in the previous sample.

    Parameters
    ----------
    sequence_of_samples: a sequence of sample dictionaries

    Returns
    -------
    a single sample dictionary merged across chains
    """
    _check_sequence_of_chains_present(sequence_of_samples)

    merged_samples = {}

    variable_names = _verify_and_get_variable_names_from_sequence_of_samples(sequence_of_samples)

    for variable_name in variable_names:
        sequence_of_shapes = \
            [sample_chains[variable_name].shape
             for sample_chains
             in sequence_of_samples]

        sequence_of_numpy_arrays = \
            [sample_chains[variable_name]
             for sample_chains
             in sequence_of_samples]

        parameter_dimension, chain_length, _ = sequence_of_shapes[0]

        if not all(shape[0] == parameter_dimension
                   for shape
                   in sequence_of_shapes):
            raise ValueError(f'The dimension of {variable_name} is inconsistent'
                             f' between samples.')

        if not all(shape[1] == chain_length for shape in sequence_of_shapes):
            raise ValueError('The chain lengths are inconsistent across '
                             'samples.')

        merged_samples[variable_name] = \
            np.concatenate(sequence_of_numpy_arrays, axis=2)

    return merged_samples
