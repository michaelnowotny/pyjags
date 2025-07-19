# PyJAGS: The Python Interface to JAGS

PyJAGS provides a Python interface to JAGS, a program for analysis of Bayesian
hierarchical models using Markov Chain Monte Carlo (MCMC) simulation.

PyJAGS adds the following features on top of JAGS:

* Multicore support for parallel simulation of multiple Markov chains (See Jupyter Notebook [Advanced Functionality](notebooks/Advanced%20Functionality.ipynb)
* Saving sample MCMC chains to and restoring from HDF5 files
* Functionality to merge samples along iterations or across chains so that sampling can be resumed in consecutive chunks until convergence criteria are satisfied
* Connectivity to the Bayesian analysis and visualization package Arviz

License: GPLv2

## Supported Platforms

PyJAGS works on MacOS and Linux. Windows is not currently supported.

## Installation

A working JAGS installation is required.

Using `uv` package manager:

```bash
    uv add git+https://github.com/mschulist/pyjags.git
```

### Some Notes

When installing, you must have JAGS installed. On mac, using homebrew `brew install jags` should work. On Linux, you may or may not need to update the `CMakeLists.txt` file to point to the correct header and lib files, or it may work...

If you install JAGS using conda, make sure that you are in the correct conda environment and then run `uv sync`. It _should_ find the header and lib files and build correctly then.

## Useful Links

* Package on the Python Package Index <https://pypi.python.org/pypi/pyjags>
* Project page on github <https://github.com/michaelnowotny/pyjags>
* JAGS manual and examples <http://sourceforge.net/projects/mcmc-jags/files/>

## Acknowledgements

* JAGS was created by Martyn Plummer
* PyJAGS was originally created by Tomasz Miasko
* As of May 2020, PyJAGS is developed by Michael Nowotny
* Updated in July 2025 by Mark Schulist to use uv as package manager
