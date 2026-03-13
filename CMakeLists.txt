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

cmake_minimum_required(VERSION 3.15)
project(pyjags LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python, NumPy, and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 REQUIRED)

# ---------------------------------------------------------------------------
# Find JAGS — try pkg-config first, then fall back to direct path search
# ---------------------------------------------------------------------------
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(JAGS jags)
endif()

if(JAGS_FOUND)
    # Extract version from pkg-config for the PYJAGS_JAGS_VERSION macro
    set(JAGS_VERSION "${JAGS_VERSION}")
else()
    # Fallback: direct search (conda, Homebrew, non-standard installs)
    find_path(JAGS_INCLUDE_DIRS Console.h
        PATH_SUFFIXES JAGS
        PATHS
            /opt/homebrew/include
            /opt/homebrew/opt/jags/include
            /usr/local/include
            /usr/include
            $ENV{CONDA_PREFIX}/include
            $ENV{JAGS_HOME}/include
    )
    find_library(JAGS_LINK_LIBRARIES NAMES jags JAGS
        PATHS
            /opt/homebrew/lib
            /opt/homebrew/opt/jags/lib
            /usr/local/lib
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/lib/aarch64-linux-gnu
            $ENV{CONDA_PREFIX}/lib
            $ENV{JAGS_HOME}/lib
    )
    if(JAGS_INCLUDE_DIRS AND JAGS_LINK_LIBRARIES)
        set(JAGS_FOUND TRUE)
        set(JAGS_LIBRARIES "${JAGS_LINK_LIBRARIES}")
        # Try to determine library directory for RPATH
        get_filename_component(JAGS_LIBRARY_DIRS "${JAGS_LINK_LIBRARIES}" DIRECTORY)
    else()
        set(JAGS_FOUND FALSE)
    endif()
endif()

if(NOT JAGS_FOUND)
    message(FATAL_ERROR
        "Could not find JAGS. Install it via:\n"
        "  macOS:  brew install jags\n"
        "  Debian: sudo apt-get install jags\n"
        "  conda:  conda install -c conda-forge jags\n"
    )
endif()

message(STATUS "JAGS include: ${JAGS_INCLUDE_DIRS}")
message(STATUS "JAGS libraries: ${JAGS_LIBRARIES}")

# ---------------------------------------------------------------------------
# Build the pybind11 extension module
# ---------------------------------------------------------------------------
pybind11_add_module(console pyjags/console.cc)
target_include_directories(console PRIVATE ${JAGS_INCLUDE_DIRS})
if(JAGS_LIBRARY_DIRS)
    target_link_directories(console PRIVATE ${JAGS_LIBRARY_DIRS})
endif()
target_link_libraries(console PRIVATE ${JAGS_LIBRARIES} Python3::NumPy)

# Define PYJAGS_JAGS_VERSION macro (used in commented-out version check)
if(JAGS_VERSION)
    target_compile_definitions(console PRIVATE
        PYJAGS_JAGS_VERSION="${JAGS_VERSION}"
    )
endif()

# Ensure libjags is found at runtime via RPATH
if(JAGS_LIBRARY_DIRS)
    set_target_properties(console PROPERTIES
        BUILD_RPATH "${JAGS_LIBRARY_DIRS}"
        INSTALL_RPATH "${JAGS_LIBRARY_DIRS}"
    )
endif()

install(TARGETS console DESTINATION pyjags)