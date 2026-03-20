# Copyright (C) 2016 Tomasz Miasko
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""Discovery and loading of JAGS extension modules."""

__all__ = [
    "get_modules_dir",
    "list_modules",
    "load_module",
    "set_modules_dir",
    "unload_module",
    "version",
]

import ctypes
import ctypes.util
import logging
import os
import shutil
import subprocess
import sys

from .console import Console

logger = logging.getLogger("pyjags")
modules_dir = None


def version():
    """JAGS version as a tuple of ints.

    >>> pyjags.version()
    (3, 4, 0)
    """
    v = Console.version()
    return tuple(map(int, v.split(".")))


if sys.platform.startswith("darwin"):

    def list_shared_objects():
        """Return paths of all currently loaded shared objects."""

        libc = ctypes.util.find_library("c")
        libc = ctypes.cdll.LoadLibrary(libc)

        dyld_image_count = libc._dyld_image_count
        dyld_image_count.argtypes = []
        dyld_image_count.restype = ctypes.c_uint32

        dyld_image_name = libc._dyld_get_image_name
        dyld_image_name.argtypes = [ctypes.c_uint32]
        dyld_image_name.restype = ctypes.c_char_p

        libraries = []

        for index in range(dyld_image_count()):
            libraries.append(dyld_image_name(index))

        return list(map(getattr(os, "fsdecode", lambda x: x), libraries))

elif sys.platform.startswith("linux"):

    def list_shared_objects():
        """Return paths of all currently loaded shared objects."""

        class dl_phdr_info(ctypes.Structure):
            """C structure for dl_iterate_phdr callback info."""

            _fields_ = [
                ("addr", ctypes.c_void_p),
                ("name", ctypes.c_char_p),
                ("phdr", ctypes.c_void_p),
                ("phnum", ctypes.c_uint16),
            ]

        dl_iterate_phdr_callback = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(dl_phdr_info),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        )

        libc = ctypes.util.find_library("c")
        libc = ctypes.cdll.LoadLibrary(libc)
        dl_iterate_phdr = libc.dl_iterate_phdr
        dl_iterate_phdr.argtypes = [dl_iterate_phdr_callback, ctypes.c_void_p]
        dl_iterate_phdr.restype = ctypes.c_int

        libraries = []

        def callback(info, size, data):
            """Collect shared object paths from dl_iterate_phdr."""
            path = info.contents.name
            if path:
                libraries.append(path)
            return 0

        dl_iterate_phdr(dl_iterate_phdr_callback(callback), None)

        return list(map(getattr(os, "fsdecode", lambda x: x), libraries))

else:

    def list_shared_objects():
        """Return paths of all currently loaded shared objects."""
        return []


def locate_modules_dir_using_shared_objects():
    """Infer the JAGS modules directory from loaded shared libraries.

    Inspects all shared objects in the current process, finds the JAGS
    library, and derives the modules path from its location.

    Returns
    -------
    str or None
        Path to the modules directory, or ``None`` if JAGS is not found.
    """
    for path in list_shared_objects():
        name = os.path.basename(path)
        if name.startswith("jags") or name.startswith("libjags"):
            dir = os.path.dirname(path)
            logger.info("Using JAGS library located in %s.", path)
            return os.path.join(dir, "JAGS", f"modules-{version()[0]}")
    return None


def _locate_via_pkg_config():
    """Try to locate the JAGS modules directory via pkg-config.

    Returns
    -------
    str or None
        Path to the modules directory, or ``None`` if pkg-config is
        unavailable or doesn't know about JAGS.
    """
    if not shutil.which("pkg-config"):
        return None
    try:
        result = subprocess.run(
            ["pkg-config", "--variable=moduledir", "jags"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            candidate = result.stdout.strip()
            if os.path.isdir(candidate):
                logger.info("Using JAGS modules from pkg-config: %s", candidate)
                return candidate
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _locate_via_multiarch():
    """Check Debian/Ubuntu multiarch library paths.

    Returns
    -------
    str or None
        Path to the modules directory, or ``None`` if not found.
    """
    major = version()[0]
    try:
        result = subprocess.run(
            ["dpkg-architecture", "-qDEB_HOST_MULTIARCH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            multiarch = result.stdout.strip()
            candidate = os.path.join("/usr/lib", multiarch, "JAGS", f"modules-{major}")
            if os.path.isdir(candidate):
                logger.info("Using JAGS modules from multiarch path: %s", candidate)
                return candidate
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _bundled_modules_dir():
    """Check for JAGS modules bundled inside the installed package.

    When PyJAGS is installed from a wheel (e.g., a manylinux wheel
    built by cibuildwheel), the JAGS module shared libraries are
    bundled in ``pyjags/jags_modules/`` alongside the extension.
    ``auditwheel`` patches their ``NEEDED`` entries to reference the
    vendored ``libjags``, so they share the same JAGS library instance
    as ``console.cpython-*.so``.

    Returns
    -------
    str or None
        Path to the bundled modules directory, or ``None`` if no
        bundled modules are present.
    """
    pkg_dir = os.path.dirname(__file__)
    candidate = os.path.join(pkg_dir, "jags_modules")
    if os.path.isdir(candidate):
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        if any(f.endswith(ext) for f in os.listdir(candidate)):
            return candidate
    return None


def locate_modules_dir():
    """Locate the JAGS modules directory using all available strategies.

    Tries the following strategies in order:

    1. ``JAGS_MODULE_PATH`` environment variable (explicit override)
    2. Bundled modules (wheel install with vendored libjags)
    3. Shared-object inspection (finds libjags in the current process)
    4. ``pkg-config --variable=moduledir jags``
    5. Conda prefix
    6. Debian/Ubuntu multiarch paths (``dpkg-architecture``)
    7. Common system paths (``/usr/lib``, ``/usr/local/lib``,
       ``/opt/homebrew/lib``)

    Returns
    -------
    str or None
        Path to the modules directory, or ``None`` if not found.
    """
    logger.debug("Locating JAGS module directory.")
    major = version()[0]

    # 1. Environment variable (explicit override)
    env_path = os.environ.get("JAGS_MODULE_PATH")
    if env_path and os.path.isdir(env_path):
        logger.info("Using JAGS modules from JAGS_MODULE_PATH: %s", env_path)
        return env_path

    # 2. Bundled modules (wheel with vendored libjags)
    bundled = _bundled_modules_dir()
    if bundled:
        logger.info("Using bundled JAGS modules: %s", bundled)
        return bundled

    # 3. Shared object inspection (works when libjags is loaded)
    result = locate_modules_dir_using_shared_objects()
    if result and os.path.isdir(result):
        return result

    # 4. pkg-config (the most reliable way on properly configured systems)
    pkg_result = _locate_via_pkg_config()
    if pkg_result:
        return pkg_result

    # 5. Conda prefix
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_modules = os.path.join(conda_prefix, "lib", "JAGS", f"modules-{major}")
        if os.path.isdir(conda_modules):
            logger.info("Using JAGS modules from conda: %s", conda_modules)
            return conda_modules

    # 6. Debian/Ubuntu multiarch paths
    multiarch_result = _locate_via_multiarch()
    if multiarch_result:
        return multiarch_result

    # 7. Common system paths (including multiarch patterns)
    for lib_dir in [
        "/usr/lib",
        "/usr/local/lib",
        "/opt/homebrew/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ]:
        candidate = os.path.join(lib_dir, "JAGS", f"modules-{major}")
        if os.path.isdir(candidate):
            logger.info("Using JAGS modules from %s", candidate)
            return candidate

    return result


def get_modules_dir():
    """Return the JAGS modules directory.

    Auto-detects the directory by inspecting loaded shared objects if
    it has not been set explicitly via :func:`set_modules_dir`.

    Returns
    -------
    str
        Absolute path to the JAGS modules directory.

    Raises
    ------
    RuntimeError
        If the modules directory cannot be located automatically.
    """
    global modules_dir
    if modules_dir is None:
        modules_dir = locate_modules_dir()
    if modules_dir is None:
        raise RuntimeError(
            "Could not locate JAGS module directory. "
            "Set the JAGS_MODULE_PATH environment variable to the directory "
            "containing JAGS module shared libraries (e.g., basemod.so), "
            "or call pyjags.set_modules_dir(path) before creating a model."
        )
    return modules_dir


def set_modules_dir(directory):
    """Set the JAGS modules directory.

    Parameters
    ----------
    directory : str
        Absolute path to the directory containing JAGS module shared
        libraries.
    """
    global modules_dir
    modules_dir = directory


def list_modules():
    """Return a list of currently loaded JAGS modules.

    Returns
    -------
    list[str]
        Names of all JAGS modules that have been loaded.
    """
    return Console.listModules()


def load_module(name, modules_dir=None):
    """Load a JAGS module by name.

    If the module has not been loaded before, locates and loads the
    shared library from disk, then registers it with JAGS.

    Parameters
    ----------
    name : str
        Name of the module to load (e.g., ``'basemod'``, ``'bugs'``,
        ``'dic'``).
    modules_dir : str, optional
        Directory where module shared libraries are located.  If not
        provided, uses the path from :func:`get_modules_dir`.

    Returns
    -------
    None
    """
    if name not in loaded_modules:
        dir = modules_dir or get_modules_dir()
        ext = ".so" if os.name == "posix" else ".dll"
        path = os.path.join(dir, name + ext)
        logger.info("Loading module %s from %s", name, path)
        module = ctypes.cdll.LoadLibrary(path)
        loaded_modules[name] = module
    Console.loadModule(name)


loaded_modules = {}


def unload_module(name):
    """Unload a JAGS module by name.

    Parameters
    ----------
    name : str
        Name of the module to unload.
    """
    return Console.unloadModule(name)
