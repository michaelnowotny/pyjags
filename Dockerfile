FROM python:3.12-bookworm

# Install JAGS and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    jags \
    pkg-config \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Suppress "running as root" pip warning (expected in a container)
ENV PIP_ROOT_USER_ACTION=ignore

# Core build and runtime dependencies
RUN pip install --no-cache-dir numpy setuptools arviz h5py

# Allow git operations on the bind-mounted source directory
RUN git config --global --add safe.directory /pyjags

# Mount point for the source code
WORKDIR /pyjags

# Keep container running
CMD ["tail", "-f", "/dev/null"]