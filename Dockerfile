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

FROM python:3.12-bookworm

# Install JAGS and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    jags \
    pkg-config \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Suppress "running as root" pip warning (expected in a container)
ENV PIP_ROOT_USER_ACTION=ignore

# Core build and runtime dependencies
RUN pip install --no-cache-dir numpy arviz h5py pybind11 scikit-build-core setuptools-scm

# Allow git operations on the bind-mounted source directory
RUN git config --global --add safe.directory /pyjags

# Mount point for the source code
WORKDIR /pyjags

# Keep container running
CMD ["tail", "-f", "/dev/null"]