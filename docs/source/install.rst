.. _install:

==============================
Installation and Configuration
==============================

.. contents:: Table of Contents
   :local:
   :depth: 1

.. _prerequisites:

----------------------------------
Prerequisites
----------------------------------

If it is your first time using UM\ :sup:`2` \ , we recommend that you install the
prerequisites using the Spack_ instructions below.

UM\ :sup:`2` \ requires the following software to be installed:

.. admonition:: Required
   :class: error

    * A C/C++ compiler (gcc_ version 12+ or clang_ version 15+)

    * CMake_ cross-platform build system

    * HDF5_ library for binary data

    * PugiXML_ library for XML data

Additional software is required for some features. Note that TBB_, OPENMP_, and GMSH_ are
enabled by default. If these features are not needed, they can be disabled by setting the
corresponding CMake variables to ``OFF``.

.. admonition:: Optional
   :class: note

    * TBB_ for shared-memory parallelism in the C++ standard library

      This is currently the primary method of parallelism in UM\ :sup:`2` \ . If you have
      to pick between TBB and OpenMP, we recommend TBB.

    * OpenMP_ for additional shared-memory parallelism

    * Gmsh_ for mesh generation

      UM\ :sup:`2` \ uses Gmsh for CAD model mesh generation. Meshes can still be imported,
      exported, and manipulated without Gmsh, but Gmsh is required for most mesh generation.

    * libpng_ for exporting PNG images

.. _gcc: https://gcc.gnu.org/
.. _clang: https://clang.llvm.org/
.. _CMake: https://cmake.org
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _XDMF: https://www.xdmf.org/index.php/XDMF_Model_and_Format
.. _PugiXML: https://pugixml.org/
.. _TBB: https://github.com/oneapi-src/oneTBB
.. _OpenMP: https://www.openmp.org/
.. _Gmsh: https://gmsh.info/
.. _libpng: http://www.libpng.org/pub/png/libpng.html

.. _installing_prerequisites_with_apt:

----------------------------------
Installing Prerequisites with apt
----------------------------------

On desktop machines running Debian-based Linux distributions, if you have admin privileges,
the prerequisites can be installed with the following commands:

.. code-block:: bash

    sudo apt -y update
    sudo apt install -y g++-12 libhdf5-dev libpugixml-dev libtbb-dev libglu1-mesa
    sudo apt install -y libpng-dev

    # In a directory outside of UM2
    mkdir um2_dependencies && cd um2_dependencies

    # Install cmake
    wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6.tar.gz
    tar -xzvf cmake-3.27.6.tar.gz && cd cmake-3.27.6
    ./bootstrap && make && sudo make install && cd ..

    # Install gmsh
    wget https://gmsh.info/bin/Linux/gmsh-4.11.1-Linux64-sdk.tgz
    tar -xzvf gmsh-4.11.1-Linux64-sdk.tgz && cd gmsh-4.11.1-Linux64-sdk

    # Add GMSH_ROOT to your bashrc so cmake can find gmsh.
    echo "export GMSH_ROOT=${PWD}" >> ~/.bashrc && source ~/.bashrc && cd ..

.. admonition:: Stop!
   :class: error

    Check that the dependencies were installed correctly by running the following commands:

    .. code-block:: bash

        g++-12 --version                # Expect 12+
        ldconfig -p | grep libhdf5      # Expect non-empty output
        ldconfig -p | grep libpugixml   # Expect non-empty output
        ldconfig -p | grep libtbb       # Expect non-empty output
        ldconfig -p | grep libGLU       # Expect non-empty output
        ldconfig -p | grep libpng       # Expect non-empty output
        cmake --version                 # Expect 3.27.6
        echo $GMSH_ROOT                 # Expect the path to the gmsh directory

If you are a developer, you will also need to install the following:

.. code-block:: bash

    sudo apt install -y clang-15 clang-format-15 clang-tidy-15 libomp-15-dev cppcheck

    # It may be necessary to symlink clang-format-15 and clang-tidy-15 to clang-format
    # and clang-tidy, respectively.
    sudo ln -s /usr/bin/clang-format-15 /usr/bin/clang-format
    sudo ln -s /usr/bin/clang-tidy-15 /usr/bin/clang-tidy


Scripts to perform these steps are available in the ``UM2/dependencies/apt`` directory of the
git repository.

.. _installing_prerequisites_with_brew:

----------------------------------
Installing Prerequisites with brew
----------------------------------

.. admonition:: MacOS
   :class: note

  Support for MacOS is new and this section may be incomplete. Please report any issues.
  Suggestions for improvement are also welcome.

On desktop machines running MacOS, the prerequisites can
be installed with the following commands:

.. code-block:: bash

    brew install gcc@12 cmake hdf5 pugixml tbb libpng

    # In a directory outside of UM2
    mkdir um2_dependencies && cd um2_dependencies
    wget https://gmsh.info/bin/macOS/gmsh-4.11.1-MacOSARM-sdk.tgz # For newer Macs
    # or for older Macs
    # wget https://gmsh.info/bin/macOS/gmsh-4.11.1-MacOSX-sdk.tgz
    tar -xzvf gmsh-4.11.1-*

    # Add GMSH_ROOT to your bashrc or zshrc so cmake can find gmsh. PWD will need to be updated
    # with the full path to the gmsh directory.
    echo "export GMSH_ROOT=${PWD}/gmsh-4.11.1-MacOSARM-sdk" >> ~/.zshrc && source ~/.zshrc

If you are a developer, you will also need to install the following:

.. code-block:: bash

    brew install clang-format@15 clang-tidy@15 cppcheck

.. _installing_prerequisites_with_spack:

----------------------------------
Installing Prerequisites with Spack
----------------------------------

Spack_ is a package management tool designed to support multiple versions and
configurations of software on a wide variety of platforms and environments.
For HPC users, Spack is a great way to install and manage software on a cluster
where you do not have admin privileges.

Prior to installing Spack, ensure that Python 3.6+ is installed.

.. code-block:: bash

    python3 --version

To install Spack:

.. code-block:: bash

    git clone --depth=100 --branch=releases/v0.20 https://github.com/spack/spack.git

    # We will add the following line to your bashrc (or zshrc) so that spack is available
    # in future sessions.
    echo "source ${PWD}/spack/share/spack/setup-env.sh" >> ~/.bashrc && source ~/.bashrc

    # Verify that spack is installed correctly
    spack --version # Expect 0.20

We will now install the prerequisites with Spack. First, we will find the compilers
available on your machine:

.. code-block:: bash

    spack compiler find

There are a number of pre-defined environments for Spack in ``UM2/dependencies/spack``,
depending on whether you are a user or a developer and depending on whether you're on 
a desktop machine or a cluster.
These environments contain the dependencies for UM2 and are defined in yaml files.
Pick the appropriate yaml file in ``UM2/dependencies/spack`` for use in the next step.
Then:

.. code-block:: bash

    spack env create um2 <path/to/yaml/file>
    spack env activate -p um2


We will now tell spack to resolve the dependencies and install them. Please read the 
potential issues below before continuing.

.. admonition:: Potential Issues
   :class: warning

    * If spack complains about being unable to fetch a package, your Python installation may 
      be missing valid SSL certificates.

    * If you're on a cluster, the ``tmp`` directory may not have enough space to build the
      dependencies. You can change the build directory by adding ``TMP=/path/to/tmp`` to the
      ``spack install`` command (``TMP=/path/to/tmp spack install``).

    * By default, spack will install using all available cores. If you're on a cluster, you
      may want to limit the number of cores used by adding ``-j <number of cores>`` to the
      ``spack install`` command (``spack install -j 4``).


.. code-block:: bash

    spack spec # This may take a minute or two
    spack install # This will take a while (15 mins to 2 hours, depending on your machine)


.. admonition:: Stop!
   :class: error

    Before you install, ther

    spack install

If you're using a yaml file that includes the fltk variant (+fltk), you may need to add:

.. code-block:: yaml

   packages:
    opengl:
      buildable: false
      externals:
      - spec: opengl@<OpenGL version on your machine>
        prefix: <path to opengl, such as /usr/x86_64-linux-gnu>

in ``~/.spack/packages.yaml``.

.. _Spack: https://spack.readthedocs.io/en/latest/

.. _installing_um2:

----------------------------------
Building
----------------------------------

If you installed dependencies with apt, you will need to have defined the ``GMSH_ROOT``
environment variable.
To build UM\ :sup:`2` \ :

.. code-block:: bash

    cd UM2
    mkdir build && cd build
    cmake ..
    make -j
    # Make sure the tests pass
    ctest
    make install


.. _configuring_um2:

----------------------------------
Configuring
----------------------------------

The following options are available for configuration. There are additional options,
but the other options are either for developer use or are under development.

UM2_USE_OPENMP
  Enable shared-memory parallelism with OpenMP. (Default: ON)

UM2_USE_GMSH
  Enable Gmsh for mesh generation. (Default: ON)

UM2_ENABLE_INT64
  Set the integer type to 64-bit. (Default: OFF)

UM2_ENABLE_FLOAT64
  Set the floating point type to 64-bit. (Default: ON)

UM2_ENABLE_FASTMATH
  Enable fast math optimizations. (Default: ON)

UM2_BUILD_TESTS
  Build tests. (Default: ON)

UM2_BUILD_EXAMPLES
  Build examples. (Default: OFF)

UM2_BUILD_BENCHMARKS
  Build benchmarks. (Default: OFF)
