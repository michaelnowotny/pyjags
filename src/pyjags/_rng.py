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

"""RNG initialization utilities for deterministic per-chain seeding."""

import typing as tp

import numpy as np

# JAGS built-in RNG factories, used for deterministic per-chain seeding.
JAGS_RNG_NAMES = [
    "base::Wichmann-Hill",
    "base::Marsaglia-Multicarry",
    "base::Super-Duper",
    "base::Mersenne-Twister",
]


def seed_to_chain_inits(seed: int, n_chains: int) -> list[dict[str, tp.Any]]:
    """Derive per-chain RNG names and seeds from a single integer seed.

    Uses ``numpy.random.SeedSequence`` to generate statistically independent
    child seeds, and cycles through JAGS's built-in RNG factories for
    additional structural independence.

    Parameters
    ----------
    seed : int
        Master seed for reproducible sampling.
    n_chains : int
        Number of chains to generate seeds for.

    Returns
    -------
    list[dict[str, Any]]
        One dictionary per chain with ``".RNG.name"`` and ``".RNG.seed"``
        entries suitable for passing to JAGS.
    """
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_chains)
    return [
        {
            ".RNG.name": JAGS_RNG_NAMES[i % len(JAGS_RNG_NAMES)],
            ".RNG.seed": int(cs.generate_state(1)[0] % (2**31)),
        }
        for i, cs in enumerate(child_seeds)
    ]
