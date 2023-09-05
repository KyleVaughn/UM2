.. _tutorial_introduction:

====================
Introduction
====================

The goal of UM\ :sup:`2` \ is to enable Method of Characteristics (MOC) neutron transport
calculations on unstructured polygon meshes and CAD models, and to generate the
necessary Coarse Mesh Finite Difference (CMFD) and MOC meshes automatically.
However, the automated mesh generation capabilities of UM\ :sup:`2` \ are still
under development, so users should expect to specify mesh sizes for each material or
for the global mesh.

UM\ :sup:`2` \ achieves much of its functionality by leveraging the capabilities of
`Gmsh <http://gmsh.info/>`_. It is recommended that you have a tab open to the `Gmsh
documentation <https://gmsh.info/doc/texinfo/gmsh.html>`_ while reading this tutorial.

------------------------------
Overview of Modeling and Meshing
------------------------------

UM\ :sup:`2` \ is primarily designed to be used in a two-step process. 
First, a user creates a CAD model of the geometry of interest in a CAD program such as
`FreeCAD <https://www.freecadweb.org/>`_, or using the built-in geometry creation capabilities 
of UM\ :sup:`2` \/Gmsh. 
Second, the user imports the CAD model, specifies meshing and CMFD grid parameters (a step which will
hopefully be automated soon), and generates the MOC and CMFD meshes.

Known Issues/Pain Points
------------------------

1. Due to the limitations of Gmsh and the failure of many CAD programs to export
  materials in accordance with the `STEP file standard <https://en.wikipedia.org/wiki/ISO_10303-21>`_, 
  the primary hurdle to overcome in the modeling process is to ensure that each entity has a material
  when it is time to generate the mesh.
  The easiest way to do this is to use the UM\ :sup:`2` \ C++ API to create and export your model, which 
  will ensure materials data is transferred correctly.
  However, if you are using a CAD program to create your model, the most reliable way to ensure that
  material data is transferred to Gmsh is to assign each material a unique color.
  Although many mainstream CAD programs do not export material data properly, they do export color data,
  which we can use to assign materials to entities in Gmsh.
  Ask Kyle (kcvaughn@umich.edu) if you need help with this so he finally gets around to writing
  a tutorial on it.

2. Gmsh's quadrilateral mesh algorithms are not perfect, and will sometimes silently
   create overlapping elements if the geometry is small relative to the mesh size, such as in the
   helium gap of a fuel rod. This will lead to inaccurate results or even a crash. Work is underway
   to detect and fix this issue, but for now, I would recommend sticking to triangular meshes.
   
3. Creating models for use with `MPACT <https://vera.ornl.gov/mpact/>`_ is a bit more complicated
   than it need be. Work is underway to bundle function calls and hide some of the complexity to
   create a more user-friendly interface. If you encounter any issues, or have ideas for how to
   make modeling for MPACT easier, please email Kyle (kcvaughn@umich.edu) or open an issue on
   GitHub.
