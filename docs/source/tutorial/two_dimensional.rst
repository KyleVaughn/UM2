.. _tutorial_2d:

=======================================
Two-dimensional Modeling and Meshing
=======================================

This tutorial will demonstrate how to create and mesh 2D CAD geometries of increasing complexity.
The models we will create are:

  1. `VERA benchmark problem 1A <https://corephysics.com/docs/CASL-U-2012-0131-004.pdf>`_
  
  2. `VERA benchmark problem 2A <https://corephysics.com/docs/CASL-U-2012-0131-004.pdf>`_
  
  3. `C5G7 2D <https://doi.org/10.1016/j.pnueene.2004.09.003>`_

  4. `A 2D CROCUS <https://doi.org/10.1016/j.anucene.2005.09.012>`_ model


.. _tutorial_2d_1a:

VERA Problem 1A
===============

First, we create the CAD model for VERA problem 1A. This is a simple 2D model of a fuel pin.

.. literalinclude:: ../../../tutorial/2d/1a_model.cpp
    :language: cpp

If the FLTK line is commented out and visibility options changed as indicated, the model should
look like this:

.. image:: ../_images/1a_model.png
    :width: 300px
    :align: center

Then, we create the mesh for the model.

.. literalinclude:: ../../../tutorial/2d/1a_mesh.cpp
    :language: cpp

The mesh should look like this:

.. image:: ../_images/1a_mesh.png
    :width: 400px
    :align: center

.. note::

    `ParaView <https://www.paraview.org/>`_ is a useful tool for visualizing XDMF files of the 
    final mesh. You can use the following command to visualize a .xdmf file.
    
    .. code-block:: bash
        
        paraview 1a.xdmf
 

    In the paraview gui, you will be recieve a prompt "Open data with...". Select the 
    "XDMF Reader" option and hit okay. 
    After opening, hit the Apply button which should appear on the left hand side. 
    Next, at the top select the drop down menu which has "vtkBlockColors" and swicth this 
    field to "Materials". Finally, at the top, select the drop down menu which has "Surface" 
    and switch this field to "Surface with Edges".
    You should now be able to visualize the generated mesh by material in each region.

.... _tutorial_2d_2a_nogap:
..
..VERA Problem 2A (No Gap)
..===============
..
..Now, we will move onto creating the CAD model for VERA problem 2A with no gaps. This is a simple 2D model of a
..an array of fuel rods (a fuel lattice). In this model, we will assume that all coarse cells are of identicle size.
..
.... literalinclude:: ../../../tutorial/2d/2a_nogap_model.cpp
..    :language: cpp
..
.... If the FLTK line is commented out and visibility options changed as indicated, the model should
.... look like this:
..
.... .. image:: ../_images/1a_model.png
....     :width: 300px
....     :align: center
..
..Then, we create the mesh for the model.
..
.... literalinclude:: ../../../tutorial/2d/2a_nogap_mesh.cpp
..    :language: cpp
..
.... The mesh should look like this:
..
.... .. image:: ../_images/1a_mesh.png
....     :width: 400px
....     :align: center
..
.... _tutorial_2d_2a_nogap:
..VERA Problem 2A
..===============
..Next, we will create the VERA problem 2A while including the gaps at the edges. In this model, 
..the coarse cells WILL NOT be of identicle size.
..
.... literalinclude:: ../../../tutorial/2d/2a_model.cpp
..    :language: cpp
..
.... If the FLTK line is commented out and visibility options changed as indicated, the model should
.... look like this:
..
.... .. image:: ../_images/1a_model.png
....     :width: 300px
....     :align: center
..
..Then, we create the mesh for the model.
..
.... literalinclude:: ../../../tutorial/2d/2a_mesh.cpp
..    :language: cpp
..
.... note::
..
..    All RTM (Ray Tracing Modules) must be identical in size. Since different coarse cells have 
..    different sizes in this problem, we create a single RTM for all of our course cells.
..
.... _tutorial_2d_c5g7:
..
.... The mesh should look like this:
..
.... .. image:: ../_images/1a_mesh.png
....     :width: 400px
....     :align: center
..
..C5G7 2D
..=======
..
..Coming soon!
..
.... _tutorial_2d_crocus:
..
..CROCUS 2D
..=========
..
..Coming soon!
