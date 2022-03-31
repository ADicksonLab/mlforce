from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
pytorch_plugin_header_dir = '@PYTORCH_PLUGIN_HEADER_DIR@'
pytorch_plugin_library_dir = '@PYTORCH_PLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_link_args = []
extra_compile_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-std=c++11',
                           '-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++',
                        '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']
    # Hard-code CC and CXX to clang, since gcc/g++ will *not* work with1
    # Anaconda, despite the fact that distutils will try to use them.
    # System Python, homebrew, and MacPorts on Macs will always use
    # clang, so this hack should always work and fix issues with users
    # that have GCC installed from MacPorts or homebrew *and* Anaconda
    os.environ['CC'] = 'clang'
    os.environ['CXX'] = 'clang++'

extension = Extension(name='_mlforce',
                      sources=['PyTorchForcePluginWrapper.cpp'],
                      libraries=['OpenMM', 'MLForce'],
                      include_dirs=[os.path.join(
                          openmm_dir, 'include'), pytorch_plugin_header_dir],
                      library_dirs=[os.path.join(
                          openmm_dir, 'lib'), pytorch_plugin_library_dir],
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib')],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                      )

setup(name='mlforce',
      version='0.0.1',
      py_modules=['mlforce'],
      ext_modules=[extension],
      )
