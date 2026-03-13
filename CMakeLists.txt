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

# ---------------------------------------------------------------------------
# Architecture validation (Apple Silicon with dual Homebrew)
# ---------------------------------------------------------------------------
if(APPLE)
    # Detect Python's architecture
    execute_process(
        COMMAND python3 -c "import struct; print(struct.calcsize('P') * 8)"
        OUTPUT_VARIABLE PYTHON_BITS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND python3 -c "import platform; print(platform.machine())"
        OUTPUT_VARIABLE PYTHON_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    # Detect CMake's own architecture
    execute_process(
        COMMAND file "${CMAKE_COMMAND}"
        OUTPUT_VARIABLE CMAKE_FILE_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYTHON_ARCH STREQUAL "arm64" AND CMAKE_FILE_OUTPUT MATCHES "x86_64" AND NOT CMAKE_FILE_OUTPUT MATCHES "arm64")
        message(FATAL_ERROR
            "Architecture mismatch: Python is arm64 but cmake is x86_64.\n"
            "This happens on Apple Silicon Macs with an Intel Homebrew cmake.\n"
            "Fix: install ARM-native cmake:\n"
            "  /opt/homebrew/bin/brew install cmake\n"
            "Then ensure /opt/homebrew/bin is before /usr/local/bin in PATH:\n"
            "  export PATH=\"/opt/homebrew/bin:$PATH\"\n"
        )
    endif()
endif()

# Find Python, NumPy, and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 REQUIRED)

# ---------------------------------------------------------------------------
# Helper: check if a JAGS library matches Python's architecture on macOS.
# Sets _jags_arch_ok to TRUE/FALSE in the caller's scope.
# ---------------------------------------------------------------------------
function(_check_jags_arch lib_path result_var)
    set(${result_var} TRUE PARENT_SCOPE)
    if(APPLE AND PYTHON_ARCH AND lib_path)
        execute_process(
            COMMAND file "${lib_path}"
            OUTPUT_VARIABLE _file_out
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(PYTHON_ARCH STREQUAL "arm64"
                AND _file_out MATCHES "x86_64"
                AND NOT _file_out MATCHES "arm64")
            set(${result_var} FALSE PARENT_SCOPE)
        elseif(PYTHON_ARCH STREQUAL "x86_64"
                AND _file_out MATCHES "arm64"
                AND NOT _file_out MATCHES "x86_64")
            set(${result_var} FALSE PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Find JAGS — try pkg-config first, then fall back to direct path search.
# On macOS, automatically reject wrong-architecture results and retry.
# ---------------------------------------------------------------------------
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(JAGS jags)
endif()

# Verify pkg-config result has the correct architecture
if(JAGS_FOUND AND JAGS_LINK_LIBRARIES)
    _check_jags_arch("${JAGS_LINK_LIBRARIES}" _pkg_arch_ok)
    if(NOT _pkg_arch_ok)
        message(STATUS "pkg-config found JAGS but wrong architecture — trying fallback paths")
        set(JAGS_FOUND FALSE)
    endif()
elseif(JAGS_FOUND AND JAGS_LIBRARY_DIRS)
    # pkg-config gave us a directory + library name; resolve to full path for arch check
    find_library(_JAGS_PKGCONF_LIB NAMES jags PATHS ${JAGS_LIBRARY_DIRS} NO_DEFAULT_PATH)
    if(_JAGS_PKGCONF_LIB)
        _check_jags_arch("${_JAGS_PKGCONF_LIB}" _pkg_arch_ok)
        if(NOT _pkg_arch_ok)
            message(STATUS "pkg-config found JAGS but wrong architecture — trying fallback paths")
            set(JAGS_FOUND FALSE)
        endif()
    endif()
    unset(_JAGS_PKGCONF_LIB CACHE)
endif()

if(NOT JAGS_FOUND)
    # Build platform-appropriate search paths.
    # On Apple Silicon, search /opt/homebrew first and skip /usr/local (Intel).
    if(APPLE AND PYTHON_ARCH STREQUAL "arm64")
        set(_JAGS_INCLUDE_HINTS
            /opt/homebrew/include
            /opt/homebrew/opt/jags/include
        )
        set(_JAGS_LIB_HINTS
            /opt/homebrew/lib
            /opt/homebrew/opt/jags/lib
        )
    elseif(APPLE)
        set(_JAGS_INCLUDE_HINTS
            /usr/local/include
            /opt/homebrew/include
            /opt/homebrew/opt/jags/include
        )
        set(_JAGS_LIB_HINTS
            /usr/local/lib
            /opt/homebrew/lib
            /opt/homebrew/opt/jags/lib
        )
    else()
        set(_JAGS_INCLUDE_HINTS
            /usr/local/include
            /usr/include
        )
        set(_JAGS_LIB_HINTS
            /usr/local/lib
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/lib/aarch64-linux-gnu
        )
    endif()

    # Always include conda and JAGS_HOME paths
    list(APPEND _JAGS_INCLUDE_HINTS $ENV{CONDA_PREFIX}/include $ENV{JAGS_HOME}/include)
    list(APPEND _JAGS_LIB_HINTS $ENV{CONDA_PREFIX}/lib $ENV{JAGS_HOME}/lib)

    # Use unique variable names to avoid CMake find_* cache collisions
    find_path(_JAGS_FB_INCLUDE Console.h
        PATH_SUFFIXES JAGS
        PATHS ${_JAGS_INCLUDE_HINTS}
        NO_DEFAULT_PATH
    )
    find_library(_JAGS_FB_LIB NAMES jags JAGS
        PATHS ${_JAGS_LIB_HINTS}
        NO_DEFAULT_PATH
    )
    if(_JAGS_FB_INCLUDE AND _JAGS_FB_LIB)
        # Final architecture check on the fallback result
        _check_jags_arch("${_JAGS_FB_LIB}" _fb_arch_ok)
        if(_fb_arch_ok)
            set(JAGS_FOUND TRUE)
            set(JAGS_INCLUDE_DIRS "${_JAGS_FB_INCLUDE}")
            set(JAGS_LIBRARIES "${_JAGS_FB_LIB}")
            get_filename_component(JAGS_LIBRARY_DIRS "${_JAGS_FB_LIB}" DIRECTORY)
        else()
            set(JAGS_FOUND FALSE)
        endif()
    else()
        set(JAGS_FOUND FALSE)
    endif()
    unset(_JAGS_FB_INCLUDE CACHE)
    unset(_JAGS_FB_LIB CACHE)
endif()

if(NOT JAGS_FOUND)
    if(APPLE AND PYTHON_ARCH STREQUAL "arm64")
        message(FATAL_ERROR
            "Could not find an arm64 JAGS installation.\n"
            "Install JAGS via ARM Homebrew:\n"
            "  /opt/homebrew/bin/brew install jags\n"
        )
    else()
        message(FATAL_ERROR
            "Could not find JAGS. Install it via:\n"
            "  macOS:  brew install jags\n"
            "  Debian: sudo apt-get install jags\n"
            "  conda:  conda install -c conda-forge jags\n"
        )
    endif()
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