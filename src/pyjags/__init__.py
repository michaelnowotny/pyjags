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

from importlib.metadata import version as _metadata_version

from .arviz import (
    compare as compare,
    from_pyjags as from_pyjags,
    loo as loo,
    summary as summary,
)
from .chain_utilities import (
    discard_burn_in_samples as discard_burn_in_samples,
    extract_final_iteration_from_samples_for_initialization as extract_final_iteration_from_samples_for_initialization,
    merge_consecutive_chains as merge_consecutive_chains,
    merge_parallel_chains as merge_parallel_chains,
)
from .dic import dic_samples as dic_samples
from .incremental_sampling import (
    EffectiveSampleSizeAndRHatCriterion as EffectiveSampleSizeAndRHatCriterion,
    EffectiveSampleSizeCriterion as EffectiveSampleSizeCriterion,
    RHatDeviationCriterion as RHatDeviationCriterion,
    sample_until as sample_until,
)
from .io import (
    load_samples_dictionary_from_file as load_samples_dictionary_from_file,
    save_samples_dictionary_to_file as save_samples_dictionary_to_file,
)
from .model import (
    Model as Model,
    SamplingState as SamplingState,
    check_model as check_model,
)
from .modules import (
    get_modules_dir as get_modules_dir,
    list_modules as list_modules,
    load_module as load_module,
    set_modules_dir as set_modules_dir,
    unload_module as unload_module,
    version as version,
)

__version__ = _metadata_version("pyjags")


def version_info() -> dict[str, str]:
    """Report versions of pyjags and its key dependencies.

    Useful for debugging installation issues.

    Returns
    -------
    dict
        Dictionary with version strings for pyjags, JAGS, numpy, arviz,
        h5py, and Python.

    Example
    -------
    >>> pyjags.version_info()
    {'pyjags': '2.2.0', 'jags': '4.3.2', 'numpy': '2.1.0', ...}
    """
    import sys

    import numpy as np

    from .console import Console

    info = {
        "pyjags": __version__,
        "jags": Console.version(),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }

    try:
        import arviz

        info["arviz"] = arviz.__version__
    except ImportError:
        info["arviz"] = "not installed"

    try:
        import h5py

        info["h5py"] = h5py.__version__
    except ImportError:
        info["h5py"] = "not installed"

    return info
