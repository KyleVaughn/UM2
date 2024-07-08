.. _mpact:

==============================
Building MPACT with UM\ :sup:`2`
==============================

.. contents:: Table of Contents
   :local:
   :depth: 1

.. _mpact_installing_um2:

--------------------------
Installing UM\ :sup:`2` \
--------------------------

First, follow the :ref:`install` instructions to install UM\ :sup:`2` \ with Spack.
We will create a build of UM\ :sup:`2` \ with Gmsh disabled.
It is very important that Gmsh is disabled, otherwise the MPACT build will fail on the
final linking step.

.. code-block:: bash

    cd UM2 
    mkdir mpact_build && cd mpact_build
    cmake -DUM2_USE_GMSH=OFF ..
    make -j
    # Ensure the tests pass
    ctest
    # We must install the library so that it can be properly linked with MPACT
    make install

.. _mpact_prerequisites:

----------------------------------
Installing MPACT 
----------------------------------

Now we will create a Spack environment for MPACT.
It is assumed that you have familiarized yourself with the :ref:`installing_prerequisites_with_spack`
section of the UM\ :sup:`2` \ documentation.
MPACT requires gcc version 8 and mpich version 3.3 if MPI is to be used.

.. code-block:: bash

    # Ensure you're not in a Spack environment 
    despacktivate
    # Install gcc@8.5
    spack install gcc@8.5
    spack load gcc@8.5
    spack compiler find
    spack env create mpact
    spack env activate mpact
    spack add cmake%gcc@8.5
    spack add mpich@3.3%gcc@8.5
    spack add hdf5%gcc@8.5 +cxx+fortran~mpi
    spack spec
    spack install

Now we will begin to install MPACT. We will not cover cloning MPACT, MPACT_Extras, or Futility here.

.. code-block:: bash

   cd MPACT
   # Ensure you still have gcc 8 loaded
   spacktivate -p mpact
   gcc --version # Should be 8.5.0
   # Checkout the UM2_API branch
   git checkout UM2_API
   mkdir build && cd build
   # Copy the appropriate build scripts
   cp ../build_scripts/build_with_um2.sh ./
   cp ../build_scripts/build_with_hdf5.sh ./
   cp ../build_scripts/build_mpi_release_all.sh ./
   # We need to provide the scripts the locations of the HDF5 and UM2 libraries
   SPACK_VIEW_DIR=${SPACK_ENV}/.spack-env/view
   HDF5_ROOT=$SPACK_VIEW_DIR UM2_ROOT=/path/to/UM2/mpact_build ./build_with_hdf5.sh ./build_with_um2.sh ./build_mpi_release_all.sh ..
   make -j MPACT
