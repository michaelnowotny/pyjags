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

__all__ = ['version', 'get_modules_dir', 'set_modules_dir', 'list_modules', 'load_module', 'unload_module']

import ctypes
import ctypes.util
import os
import logging
import sys

from .console import Console

logger = logging.getLogger('pyjags')
modules_dir = None


def version():
    """JAGS version as a tuple of ints.

    >>> pyjags.version()
    (3, 4, 0)
    """
    v = Console.version()
    return tuple(map(int, v.split('.')))


if sys.platform.startswith('darwin'):
    def list_shared_objects():
        """Return paths of all currently loaded shared objects."""

        libc = ctypes.util.find_library('c')
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

        return list(map(getattr(os, 'fsdecode', lambda x: x), libraries))

elif sys.platform.startswith('linux'):
    def list_shared_objects():
        """Return paths of all currently loaded shared objects."""

        class dl_phdr_info(ctypes.Structure):
            _fields_ = [
                ('addr', ctypes.c_void_p),
                ('name', ctypes.c_char_p),
                ('phdr', ctypes.c_void_p),
                ('phnum', ctypes.c_uint16),
            ]

        dl_iterate_phdr_callback = ctypes.CFUNCTYPE(
                ctypes.c_int,
                ctypes.POINTER(dl_phdr_info),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p)

        libc = ctypes.util.find_library('c')
        libc = ctypes.cdll.LoadLibrary(libc)
        dl_iterate_phdr = libc.dl_iterate_phdr
        dl_iterate_phdr.argtypes = [dl_iterate_phdr_callback, ctypes.c_void_p]
        dl_iterate_phdr.restype = ctypes.c_int

        libraries = []

        def callback(info, size, data):
            path = info.contents.name
            if path:
                libraries.append(path)
            return 0

        dl_iterate_phdr(dl_iterate_phdr_callback(callback), None)

        return list(map(getattr(os, 'fsdecode', lambda x: x), libraries))

else:
    def list_shared_objects():
        """Return paths of all currently loaded shared objects."""
        return []


def locate_modules_dir_using_shared_objects():
    for path in list_shared_objects():
        name = os.path.basename(path)
        if name.startswith('jags') or name.startswith('libjags'):
            dir = os.path.dirname(path)
            logger.info('Using JAGS library located in %s.', path)
            return os.path.join(dir, 'JAGS', 'modules-{}'.format(version()[0]))
    return None


def locate_modules_dir():
    logger.debug('Locating JAGS module directory.')
    return locate_modules_dir_using_shared_objects()


def get_modules_dir():
    """Return modules directory."""
    global modules_dir
    if modules_dir is None:
        modules_dir = locate_modules_dir()
    if modules_dir is None:
        raise RuntimeError(
            'Could not locate JAGS module directory. Use pyjags.set_modules_dir(path) to configure it manually.')
    return modules_dir


def set_modules_dir(directory):
    """Set modules directory."""
    global modules_dir
    modules_dir = directory


def list_modules():
    """Return a list of loaded modules."""
    return Console.listModules()


def load_module(name, modules_dir=None):
    """Load a module.

    Parameters
    ----------
    name : str
        A name of module to load.
    modules_dir : str, optional
        Directory where modules are located.
    """
    if name not in loaded_modules:
        dir = modules_dir or get_modules_dir()
        ext = '.so' if os.name == 'posix' else '.dll'
        path = os.path.join(dir, name + ext)
        logger.info('Loading module %s from %s', name, path)
        module = ctypes.cdll.LoadLibrary(path)
        loaded_modules[name] = module
    Console.loadModule(name)

loaded_modules = {}


def unload_module(name):
    """Unload a module."""
    return Console.unloadModule(name)
