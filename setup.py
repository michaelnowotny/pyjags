# encoding: utf-8
#
# Copyright (C) 2015-2016 Tomasz Miasko
#               2020 Michael Nowotny
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import io
import os
import platform
# from distutils.core import setup
# from distutils.extension import Extension
from setuptools import dist, setup, Extension
# from Cython.Distutils import build_ext
import subprocess
import sys
import versioneer


here = os.path.abspath(os.path.dirname(__file__))
DESCRIPTION = 'PyJAGS: The Python Interface to JAGS'
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


def add_pkg_config(ext, package):
    flags_map = {
        '-I': ['include_dirs'],
        '-L': ['library_dirs', 'runtime_library_dirs'],
        '-l': ['libraries'],
    }

    try:
        args = ['pkg-config', '--libs', '--cflags', package]
        output = subprocess.check_output(args)
        output = output.decode()
        for flag in output.split():
            for attr in flags_map[flag[:2]]:
                getattr(ext, attr).append(flag[2:])

        args = ['pkg-config', '--modversion', package]
        output = subprocess.check_output(args, universal_newlines=True)
        return output.strip()
    except Exception as err:
        print("Error while executing pkg-config: {}".format(err))
        sys.exit(1)


def add_jags(ext):
    if 'windows' in platform.system().lower():
        JAGS_HOME = os.environ.get('JAGS_HOME')

        if JAGS_HOME is None:
            JAGS_HOME = r'C:\Program Files\JAGS\JAGS-4.3.0'
            print(f'environment variable JAGS_HOME not defined, defaulting to {JAGS_HOME}')
        else:
            print(f'environment variable JAGS_HOME is defined as {JAGS_HOME}')
        version = os.path.split(JAGS_HOME)[1][5:]
        version = '"{}"'.format(version)

        ext.include_dirs.append(rf'{JAGS_HOME}\include')
        if any('64' in element for element in platform.architecture()):
            print('On 64 Bit Architecture')
            lib_dir = rf'{JAGS_HOME}\x64\bin'
        else:
            print('On 32 Bit Architecture')
            lib_dir = rf'{JAGS_HOME}\i386\bin'

        ext.library_dirs.append(lib_dir)

        # the Extension property runtime_library_dirs is not used on Windows
        # ext.runtime_library_dirs.append(lib_dir)

        # ToDo: How to tell the Python build system to tell the C++ linker to look for a .dll instead of a .lib?
        # ext.libraries.append('libjags')
        ext.define_macros.append(('PYJAGS_JAGS_VERSION', version))
    else:
        version = add_pkg_config(ext, 'jags')
        version = '"{}"'.format(version)
        ext.define_macros.append(('PYJAGS_JAGS_VERSION', version))


def add_numpy(ext):
    try:
        import numpy
    except ImportError:
        sys.exit('Please install numpy first.')
    ext.include_dirs.append(numpy.get_include())


def add_pybind11(ext):
    ext.include_dirs.append('pybind11/include')
    if not 'windows' in platform.system().lower():
        ext.extra_compile_args.append('-std=c++14')


if __name__ == '__main__':
    ext = Extension('pyjags.console',
                    language='c++',
                    sources=['pyjags/console.cc'],
                    # libraries=['libjags-4'],
                    # library_dirs=['.']
                    )

    add_jags(ext)
    add_numpy(ext)
    add_pybind11(ext)
    setup(name='pyjags',
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          # cmdclass={'build_ext': build_ext},
          description='Python interface to JAGS library for Bayesian data analysis.',
          long_description=long_description,
          long_description_content_type='text/markdown',
          author=u'Michael Nowotny',
          author_email='nowotnym@gmail.com',
          url='https://github.com/michaelnowotny/pyjags',
          license='GPLv2',
          python_requires='>=3.12',
          classifiers=[
              'Development Status :: 4 - Beta',
              'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
              'Operating System :: POSIX',
              'Programming Language :: C++',
              'Programming Language :: Python :: 3.12',
              'Programming Language :: Python :: 3.13',
              'Programming Language :: Python :: 3.14',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering :: Information Analysis',
              'Topic :: Scientific/Engineering',
          ],
          packages=['pyjags'],
          ext_modules=[ext],
          install_requires=[
              'numpy',
              'arviz>=1.0',
              'h5py'
          ],
          test_suite='test',
          )
