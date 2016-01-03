# Copyright (C) 2016 Tomasz Miasko
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
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

from .console import Console

logger = logging.getLogger('pyjags')


def version():
    """JAGS version as a tuple of ints.

    >>> pyjags.version()
    (3, 4, 0)
    """
    v = Console.version()
    return tuple(map(int, v.split('.')))


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
        raise RuntimeError('Could not locate JAGS module directory.')
    return modules_dir

modules_dir = None


def set_modules_dir(modules_dir):
    """Set modules directory."""
    get_modules_dir.dir = modules_dir


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