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


"""HDF5 persistence for MCMC sample dictionaries."""

import h5py
import numpy as np


def save_samples_dictionary_to_file(
    filename: str, samples: dict[str, np.ndarray], compression: bool = True
):
    """Save a sample dictionary to an HDF5 file.

    Parameters
    ----------
    filename : str
        Path where the HDF5 file should be written.
    samples : dict[str, numpy.ndarray]
        Dictionary mapping variable names to numpy arrays with shape
        ``(*variable_dims, chain_length, n_chains)`` as returned by
        :meth:`~pyjags.Model.sample`.
    compression : bool
        If ``True`` (default), apply gzip compression to each dataset.

    Raises
    ------
    OSError
        If the file cannot be created or written.
    """
    comp = "gzip" if compression else None
    with h5py.File(filename, "w") as f:
        for key, value in samples.items():
            f.create_dataset(key, data=np.asarray(value), compression=comp)


def load_samples_dictionary_from_file(filename: str) -> dict[str, np.ndarray]:
    """Load a sample dictionary from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file to read.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary mapping variable names to numpy arrays with shape
        ``(*variable_dims, chain_length, n_chains)``.

    Raises
    ------
    OSError
        If the file cannot be opened or read.
    """
    result = {}
    with h5py.File(filename, "r") as f:
        for key in f:
            result[key] = f[key][()]
    return result
