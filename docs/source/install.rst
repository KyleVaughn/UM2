.. _install:

==============================
Installation and Configuration
==============================

.. contents:: Table of Contents
   :local:
   :depth: 1

.. _cloning_the_repository:

--------------------------
Cloning the Repository
--------------------------

The first step to building UM\ :sup:`2` \ is to clone the repository.
The UM\ :sup:`2` \ source code is hosted on `GitHub <https://github.com/KyleVaughn/UM2>`_.
Assuming you have git_ installed, to clone the repository run the following command:

.. code-block:: bash

    git clone https://github.com/KyleVaughn/UM2.git

.. _git: https://git-scm.com/

.. _prerequisites:

----------------------------------
Prerequisites
----------------------------------

If you are building UM\ :sup:`2` \ on a local Ubuntu  machine, the prerequisites can be installed
with apt_.
If you are building UM\ :sup:`2` \ on a cluster, do not have admin privileges, or plan to
use MPACT, we recommend using the Spack_ instructions below.

UM\ :sup:`2` \ requires the following software to be installed:

.. admonition:: Required
   :class: error

    * A C/C++ compiler (gcc_ version 12+ or clang_ version 15+)

    * CMake_ cross-platform build system

    * HDF5_ library for binary data

    * PugiXML_ library for XML data

Additional software is required for some features. Note that OpenMP_ and Gmsh_ are
enabled by default. If these features are not needed, they can be disabled by setting the
corresponding CMake variables to ``OFF``.

.. admonition:: Optional
   :class: note

    * OpenMP_ for shared-memory parallelism

    * Gmsh_ for CAD-based models and CAD-based mesh generation

    * OpenBLAS_ for linear algebra

      OpenBLAS is required for advanced CMFD grid generation features.

    * MPACT_ cross section libraries

      OpenBLAS and MPACT are required for advanced cross section based mesh generation features.

.. _apt: https://en.wikipedia.org/wiki/APT_(software)
.. _gcc: https://gcc.gnu.org/
.. _clang: https://clang.llvm.org/
.. _CMake: https://cmake.org
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _XDMF: https://www.xdmf.org/index.php/XDMF_Model_and_Format
.. _PugiXML: https://pugixml.org/
.. _OpenBLAS: https://github.com/OpenMathLib/OpenBLAS
.. _OpenMP: https://www.openmp.org/
.. _Gmsh: https://gmsh.info/
.. _MPACT: https://vera.ornl.gov/mpact/

.. _installing_prerequisites_with_apt:

----------------------------------
Installing Prerequisites with apt
----------------------------------

On desktop machines running Ubuntu Linux distributions, if you have admin privileges,
the prerequisites can be installed with the following commands:

.. code-block:: bash

    # First, check that you are running Ubuntu 20.04 or higher.
    # If you are not, proceed to the Spack instructions in the next section.
    # If you are using Ubuntu version <= 24, you will need to install cmake and gmsh
    # manually as described at the end of this block
    lsb_release -a

    # From within the UM2 directory
    ./scripts/deps/apt_install.sh

    # Verify that your cmake version is 3.25 or higher
    cmake --version

    # Verify that your gcc is version 12 or higher
    g++ --version

    # If both of the above commands return the expected versions, you are ready to
    # build UM2. If not, proceed below.

    # If your gcc is not version 12+, you may need to explicitly install version 12
    # and set it as the default compiler
    sudo apt install -y gcc-12 g++-12 gfortran-12
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
    sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-12 100

    # If you are using Ubuntu version <= 24, we will install cmake and gmsh manually
    # In a directory outside of UM2:
    mkdir um2_deps && cd um2_deps

    # Install cmake
    wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6.tar.gz
    tar -xzvf cmake-3.27.6.tar.gz && cd cmake-3.27.6
    ./bootstrap && make && sudo make install && cd ..

    # Install gmsh
    wget https://gmsh.info/bin/Linux/gmsh-4.11.1-Linux64-sdk.tgz
    tar -xzvf gmsh-4.11.1-Linux64-sdk.tgz && cd gmsh-4.11.1-Linux64-sdk

    # Add GMSH_ROOT to your bashrc so cmake can find gmsh.
    echo "export GMSH_ROOT=${PWD}" >> ~/.bashrc && source ~/.bashrc && cd ..

    # You may also need the following line for Gmsh's GUI to work correctly:
    # sudo apt install -y libglu1-mesa

.. admonition:: Stop!
   :class: error

    Check that the dependencies were installed correctly by running the following commands:

    .. code-block:: bash

        g++ --version                   # Expect 12+
        gfotran --version               # Expect 12+
        cmake --version                 # Expect 3.25+
        ldconfig -p | grep libhdf5      # Expect non-empty output
        ldconfig -p | grep libpugixml   # Expect non-empty output
        ldconfig -p | grep libGLU       # Expect non-empty output
        ldconfig -p | grep libgmsh      # Expect non-empty output
        ldconfig -p | grep libopenblas  # Expect non-empty output
        ldconfig -p | grep liblapacke   # Expect non-empty output
        echo $GMSH_ROOT                 # Expect the path to the gmsh directory

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

    # In a directory outside of UM2
    git clone --depth=100 --branch=releases/v0.22 https://github.com/spack/spack.git

    # We will add the following line to your bashrc (or zshrc) so that spack is available
    # in future sessions.
    echo "source ${PWD}/spack/share/spack/setup-env.sh" >> ~/.bashrc && source ~/.bashrc

    # Verify that spack is installed correctly
    spack --version # Expect 0.22

If you do not have C, C++, and Fortran compilers available,
install them now, or you will need to modify the compilers.yaml file created in the next step.
Assuming you're using gcc, to verify that you have the necessary compilers, run the following
commands:

.. code-block:: bash

    gcc --version
    g++ --version
    gfortran --version

We will now install the prerequisites with Spack. First, we need to make Spack aware of the
compilers available on your system. To do this, run the following command:

.. code-block:: bash

    spack compiler find


Next, we will install gcc 12. But first, please examine the potential issues below.

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

    spack install gcc@12 # This will take a while (15-90 minutes)
    spack load gcc@12
    spack compiler find
    # Verify that gcc 12 is the default gcc
    gcc --version # Expect 12


You should have previously cloned the UM2 repository. If not, do so now:

.. code-block:: bash

    git clone https://github.com/KyleVaughn/UM2.git

Now, we will create a Spack environment for UM2 and install the dependencies.
Spack sometimes has issues resolving many dependencies at once, so we will add them incrementally.

.. code-block:: bash

    # Create the um2 environment and activate it
    spack env create um2
    spack env activate -p um2

    # Add the first few dependencies
    spack add cmake%gcc@12
    spack add hdf5%gcc@12 +cxx+fortran~mpi
    spack add pugixml%gcc@12
    spack add openblas%gcc@12

    # Verify that spack is able to resolve the dependencies
    spack spec # this will likely take a few seconds

    # Omit the next line if you are on a cluster and do not need CAD-based mesh generation
    spack add gmsh@4.12%gcc@12.3 +openmp+cairo+fltk+opencascade+eigen ^scotch~mpi

We will now tell spack to resolve the dependencies and install them.
See the files in ``UM2/scripts/deps`` for more information on spack environments.

.. code-block:: bash

    spack spec # This may take a minute or two
    # This will take a while (20 mins to 2 hours, depending on your machine)
    # Remember to use TMP or -j as needed, as described above
    spack install

.. admonition:: Stop!
   :class: error

    Check that the dependencies were installed correctly by running the following commands:

    .. code-block:: bash

        g++ --version                       # Expect 12+
        gfortran --version                  # Expect 12+
        cmake --version                     # Expect 3.25+
        find $SPACK_ENV -name libhdf5*      # Expect non-empty output
        find $SPACK_ENV -name libpugixml*   # Expect non-empty output
        find $SPACK_ENV -name libGLU*       # Expect non-empty output
        find $SPACK_ENV -name libopenblas*  # Expect non-empty output
        gmsh --version                      # Expect 4.10+

.. _Spack: https://spack.readthedocs.io/en/latest/

.. _installing_um2:

----------------------------------
Building
----------------------------------

To build UM\ :sup:`2` \ with the default options, run the following commands:

.. code-block:: bash

    cd UM2
    mkdir build && cd build

    # Configure the build
    # Use -DUM2_USE_XXXX=ON or -DUM2_ENABLE_XXXX=ON to enable or disable features.
    # See CMake Options below.
    cmake ..
    make

    # Make sure the tests pass
    ctest

    # Install the library and headers
    make install

You may need to specify the compiler to use during the configuration process,
e.g. ``CXX=g++-12 cmake ..``.

.. admonition:: CMake Options
   :class: note

    If you want to change the default options, you can do so by passing the appropriate
    flags to cmake, e.g. ``cmake -DUM2_USE_OPENMP=OFF ..``. The available options are
    described below.

    Also, note that when specifying a new compiler or changing cmake options once you
    have already configured, you may need to remove the ``build`` directory and start over
    for the changes to take effect.


.. _configuring_um2:

----------------------------------
Configuring
----------------------------------

The following options are available for configuration.

UM2_BUILD_BENCHMARKS
  Build code benchmarks. These are code snippets that are used to measure the
  performance of UM2. These are not IRPhE or other nuclear reactor benchmarks.
  (Default: OFF)

UM2_BUILD_MODELS
  Build models. These are nuclear reactor models or other physics benchmarks
  (e.g. C5G7).
  (Default: OFF)

UM2_BUILD_SHARED_LIB
  Build UM2 as a shared library (ON) or static library (OFF). This option is
  overriden if CUDA is enabled, in which case a static library is always built.
  (Default: ON)

UM2_BUILD_TESTS
  Build tests. These are unit tests that are used to verify the correctness of
  UM2.
  (Default: ON)

UM2_BUILD_TUTORIAL
  Build tutorial. This is a tutorial that demonstrates how to use UM2.
  (Default: ON)

UM2_ENABLE_ASSERTS
  Enable assertions. This option enables UM2_ASSERT*(condition) macros, which
  are evaluated regardless of the build type, unlike the standard assert macro
  which is only evaluated if NDEBUG is not defined.
  (Default: OFF)

UM2_ENABLE_BMI2
  Enable BMI2 instruction set. This option enables the BMI2 instruction set,
  if it is supported by the architecture. This is primarily for fast Morton
  sorting.
  (Default: ON)

UM2_ENABLE_FASTMATH
  Enable fast math optimizations. This option enables fast math optimizations
  -ffast-math on the CPU and --use_fast_math on the GPU. Note that this may
  result in a loss of precision.
  (Default: OFF)

UM2_ENABLE_FLOAT64
  Set the Float type to 64-bit (double) instead of 32-bit (float). This option
  determines the precision of the floating point numbers used in UM2.
  (Default: ON)

UM2_ENABLE_NATIVE
  Enable native architecture. This option enables the -march=native flag, which
  optimizes the code for the architecture on which it is built.
  (Default: ON)

UM2_ENABLE_SIMD_VEC
  Enable GCC vector extensions for the Vec class. Vec<D, T> uses T[D] as the
  underlying storage type by default. When ON, if D is a power of 2 and T is
  an arithmetic type, Vec<D, T> will use GCC vector extensions instead to store
  a SIMD vector of D elements of type T. Despite aligned T[D] being functionally
  the same as the SIMD vector, the compiler tends to generate slightly better code
  with the vector extensions.
  (Default: ON)

UM2_USE_BLAS_LAPACK
  Use BLAS/LAPACK for linear algebra.
  NOTE: this is required for CMFD spectral radius calculations.
  (Default: OFF)

UM2_USE_CLANG_FORMAT
  Use clang-format for code formatting. This option enables the format-check
  and format-fix targets, which check and fix the formatting of the code.
  (Default: OFF)

UM2_USE_CLANG_TIDY
  Use clang-tidy for static analysis. Enable clang-tidy on all targets.
  (Default: OFF)

UM2_USE_COVERAGE
  Use gcov for code coverage analysis.
  (Default: OFF)

UM2_USE_CUDA
  Use CUDA for GPU acceleration.
  (Default: OFF)

UM2_USE_GMSH
  Use GMSH for CAD geometry and mesh generation from CAD geometry.
  (Default: ON)

UM2_USE_HDF5
  Use HDF5 for binary data I/O. (Used for mesh I/O)
  (Default: ON)

UM2_USE_MPACT_XSLIBS
  Use MPACT's cross section libraries. Used for CMFD and advanced mesh generation.
  (Default: ON)

UM2_USE_OPENMP
  Use OpenMP for multi-threading.
  (Default: ON)

UM2_USE_PUGIXML
  Use pugixml for XML parsing. Used for mesh I/O.
  (Default: ON)

UM2_USE_VALGRIND
  Use valgrind for memory checking. Creates a valgrind_X target for each test.
  (Default: OFF)
