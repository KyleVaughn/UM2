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

.. admonition:: Required
   :class: error

    * A C/C++ compiler (gcc_-12+ or clang_-15+)

    * CMake_ cross-platform build system

      The compiling and linking of source files is handled by CMake in a
      platform-independent manner.

    * HDF5_ library for binary data

      UM\ :sup:`2` \ uses HDF5 for large binary data in XDMF_ mesh files. 
      The installed version will need the C++ API.

    * PugiXML_ library for XML data

      UM\ :sup:`2` \ uses PugiXML for reading and writing XML files as a part of XDMF_. 

.. admonition:: Optional
   :class: note

    * Gmsh_ for mesh generation

      UM\ :sup:`2` \ uses Gmsh for CAD model mesh generation. Meshes can still be imported, 
      exported, and manipulated without Gmsh, but Gmsh is required for mesh generation. 

.. _gcc: https://gcc.gnu.org/
.. _clang: https://clang.llvm.org/
.. _CMake: https://cmake.org
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _XDMF: https://www.xdmf.org/index.php/XDMF_Model_and_Format
.. _PugiXML: https://pugixml.org/
.. _Gmsh: https://gmsh.info/


.. _installing_prerequisites_with_apt:

----------------------------------
Installing Prerequisites with apt
----------------------------------

On desktop machines running Debian-based Linux distributions, the prerequisites can 
be installed with the following commands:

.. code-block:: bash

    sudo apt -y update
    sudo apt install -y g++-12 cmake libhdf5-dev libpugixml-dev libglu1-mesa
    wget https://gmsh.info/bin/Linux/gmsh-4.11.1-Linux64-sdk.tgz
    tar -xzvf gmsh-4.11.1-Linux64-sdk.tgz
    # Add to your bachrc to avoid declaring this every time
    export GMSH_ROOT=${PWD}/gmsh-4.11.1-Linux64-sdk

.. _installing_prerequisites_with_spack:

----------------------------------
Installing Prerequisites with Spack
----------------------------------

Spack_ is a package management tool designed to support multiple versions and
configurations of software on a wide variety of platforms and environments.

To install Spack:

.. code-block:: bash

    git clone -c feature.manyFiles=true https://github.com/spack/spack.git 
    # Ensure python 3.6+ is installed
    . spack/share/spack/setup-env.sh

Install and load gcc-12+ or clang-15+:

.. code-block:: bash

    spack compiler find    
    spack install gcc@12
    spack load gcc@12
    # or
    spack install llvm@15
    spack load llvm@15
    
Pick the appropriate yaml file in UM2/dependencies/spack for use in the next step. Then:

.. code-block:: bash

    spack compiler find    
    spack env create um2 <choice of spack env>    
    spack env activate -p um2    
    spack spec    
    spack install

If you're using a yaml file that includes the fltk variant (+fltk), you may need to add:

.. code-block:: yaml 

   packages:
    opengl:
      buildable: false
      externals:
      - spec: opengl@<OpenGL version on your machine>
        prefix: <path to opengl, such as /usr/x86_64-linux-gnu> 

in :bash:`~/.spack/packages.yaml` .

.. _Spack: https://spack.readthedocs.io/en/latest/

.. _installing_um2:

----------------------------------
Building and nstalling UM\ :sup:`2` \
----------------------------------
