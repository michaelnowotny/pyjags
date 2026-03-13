#!/bin/bash

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

# install-jags-manylinux.sh — Compile and install JAGS from source inside
# manylinux containers (used by cibuildwheel).

set -euo pipefail

JAGS_VERSION="${JAGS_VERSION:-4.3.2}"
JAGS_URL="https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Source/JAGS-${JAGS_VERSION}.tar.gz/download"
PREFIX="/usr/local"

echo "Installing JAGS ${JAGS_VERSION} from source..."

# Install build dependencies (handle both dnf and yum)
if command -v dnf &>/dev/null; then
    dnf install -y lapack-devel blas-devel pkgconfig
elif command -v yum &>/dev/null; then
    yum install -y lapack-devel blas-devel pkgconfig
elif command -v apt-get &>/dev/null; then
    apt-get update && apt-get install -y liblapack-dev libblas-dev pkg-config
else
    echo "ERROR: No supported package manager found"
    exit 1
fi

# Download and extract
curl -L -o /tmp/JAGS-${JAGS_VERSION}.tar.gz "${JAGS_URL}"
cd /tmp
tar xzf JAGS-${JAGS_VERSION}.tar.gz
cd JAGS-${JAGS_VERSION}

# Build and install
./configure --prefix="${PREFIX}"
make -j"$(nproc)"
make install
ldconfig

# Verify
echo "JAGS installed successfully:"
pkg-config --modversion jags