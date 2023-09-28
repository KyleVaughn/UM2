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

First, follow the instructions at :ref:`install` to install UM\ :sup:`2` \ .

.. _mpact_prerequisites:

----------------------------------
Installing MPACT 
----------------------------------

It is assumed that you have familiarized yourself with the :ref:`installing_prerequisites_with_spack`
section of the UM\ :sup:`2` \ documentation.
This section assumes that you are using a minimal MPACT build, i.e. you are only using MPACT_Extras
and Futility, not Kokkos, Trilinos, etc.

MPACT requires gcc 8 and MPICH 3.3, but UM2 requires GCC 12 due to the use of C++20 features.
Luckily, we can still build MPACT with gcc 8 as long as we perform the final linking step with GCC 12.
This is a horrible hack, but until MPACT is updated to use a newer version of gcc, it's the best we can do.

.. code-block:: bash

    # Ensure you're not in a Spack environment 
    despacktivate
    # Install gcc@8
    spack install gcc@8.5.0
    spack load gcc@8.5.0
    spack install mpich@3.3%gcc@8.5.0
    spack load mpich@3.3%gcc@8.5.0

Now we will begin to install MPACT. We will not cover cloning MPACT, MPACT_Extras, or Futility here.

.. code-block:: bash

   cd MPACT
   # Ensure you still have gcc 8 loaded
   spacktivate -p um2
   gcc --version # Should be 8.5.0
   # Checkout the UM2 branch
   git checkout UM2
   mkdir build && cd build
   # Copy the appropriate build scripts
   cp ../build_scripts/build_with_um2.sh ./
   cp ../build_scripts/build_with_hdf5.sh ./
   cp ../build_scripts/build_mpi_release_all.sh ./
   # We need to provide the scripts the locations of the HDF5 and UM2 libraries
   SPACK_VIEW_DIR=${SPACK_ENV}/.spack-env/view
   HDF5_ROOT=$SPACK_VIEW_DIR UM2_ROOT=/path/to/UM2/build ./build_with_hdf5.sh ./build_with_um2.sh ./build_mpi_release_all.sh ..
   make -j MPACT
  
This will fail on the final linking step. Now for the gcc 12 environment.

.. code-block:: bash

    # Ensure you're not in a Spack environment 
    despacktivate
    spack load gcc@12.3.0
    spack install mpich@3.3%gcc@12.3.0
    spack load mpich@3.3%gcc@12.3.0
    spacktivate -p um2 

Now we will finish the MPACT build. We need to extract the command that cmake was using to make MPACT.exe and
modify it to use the gcc 12 version of MPICH.

.. code-block:: bash

   cd MPACT/build/MPACT_exe/src
   MPIF90=$(which mpif90)
   make MPACT VERBOSE=1
   # Copy the command that was used to link MPACT.exe. It will look something like:
   mpif90 -cpp -fall-intrinsics -ffree-line-length-none -DHAVE_MPI -DMPACT_HAVE_HDF5 (etc.)
   # All that is left is to replace the old gcc 8 mpif90 with the gcc 12 mpif90
   $MPIF90 -cpp -fall-intrinsics -ffree-line-length-none -DHAVE_MPI -DMPACT_HAVE_HDF5 (etc.)

The branch needs to be cleaned up, so for now there is an input file leftover in MPACT/1a/um2_real that
can be used to test the installation. Note that in simulation can be decomposed in angle if you wish
to test the parallelization.

.. code-block:: bash

   cd MPACT/1a/um2_real
   ln -s ../../MPACT_Extras/xslibs/mpact51g_71_v4.2m5_12062016_sph.fmt mpact51g_71_v4.2m5_12062016_sph.fmt
   <modify runjob.sh to use the location of your MPACT.exe>
   ./runjob.sh
