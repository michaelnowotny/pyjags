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

from .incremental_sampling import (
    EffectiveSampleSizeCriterion,
    RHatDeviationCriterion,
    EffectiveSampleSizeAndRHatCriterion,
    sample_until
)

from .arviz import from_pyjags
from .chain_utilities import (
    discard_burn_in_samples,
    extract_final_iteration_from_samples_for_initialization,
    merge_parallel_chains,
    merge_consecutive_chains
)

from .dic import dic_samples
from .io import (
    load_samples_dictionary_from_file,
    save_samples_dictionary_to_file
)

from .model import *
from .modules import *
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
