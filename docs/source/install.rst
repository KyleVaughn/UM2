.. _install:

==============================
Installation and Configuration
==============================

.. currentmodule:: um2

.. _prerequisites:

----------------------------------
Prerequisites
----------------------------------

.. admonition:: Required
   :class: error

    * A C/C++ compiler (gcc_-12+ or clang_-15+)

      If you are using a Debian-based distribution, you can install the g++ using the
      following command::

          sudo apt install g++-12

    * CMake_ cross-platform build system

      The compiling and linking of source files is handled by CMake in a
      platform-independent manner. If you are using Debian-based distribution, you 
      can install CMake using the following command::

          sudo apt install cmake

    * HDF5_ library for portable binary output format

      UM\:sup:`2`\ uses HDF5 for writing XDMF_ mesh files. 
      The installed version will need the C++ API.
      If you are using a Debian-based distribution, you can install HDF5 using
      the following command::

          sudo apt install libhdf5-dev

    * PugiXML_ library for XML files

      UM\:sup:`2`\ uses PugiXML for reading and writing XML files as a part of XDMF_. 
      If you are using a Debian-based distribution, you can install PugiXML using the following
      command::

          sudo apt install libpugixml-dev

.. admonition:: Optional
   :class: note

    * Gmsh_ for mesh generation

      UM\:sup:`2`\ uses Gmsh for CAD model mesh generation. Meshes can still be imported, 
      exported, and manipulated without Gmsh, but Gmsh is required for mesh generation. 
      Gmsh offers a development version that can be installed using the following commands::

        sudo apt install libglu1-mesa-dev
        wget https://gmsh.info/bin/Linux/gmsh-4.11.1-Linux64-sdk.tgz
        tar -xvf gmsh-4.11.1-Linux64-sdk.tgz
        GMSH_ROOT=${PWD}/gmsh-4.11.1-Linux64-sdk

.. _gcc: https://gcc.gnu.org/
.. _CMake: https://cmake.org
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _XDMF: https://www.xdmf.org/index.php/XDMF_Model_and_Format
.. _PugiXML: https://pugixml.org/
.. _Gmsh: https://gmsh.info/
