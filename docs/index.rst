.. UM2 documentation master file, created by sphinx-quickstart.

Dependencies are installed via spack.
-------------------------------------
In ``spack.yaml``:

- Change ``+fltk`` to ``~fltk`` if a mesh viewer is not needed
- Delete the ``cuda`` spec if cuda is not desired supported

To install spack:
-----------------
.. code-block:: bash

   git clone -c feature.manyFiles=true https://github.com/spack/spack.git
   . spack/share/spack/setup-env.sh

Then, to install UM2 dependencies:
----------------------------------

.. code-block:: bash

   spack compiler find
   spack env create um2 spack.yaml
   spack env activate -p um2
   spack spec
   spack install

You may need to place:

.. code-block:: yaml

   packages:
     opengl:
       buildable: False
       externals:
       - spec: opengl@4.6.0
         prefix: /usr/x86_64-linux-gnu

in ``~/.spack/packages.yaml``.


Welcome to UM2's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

:ref:`genindex`

Docs
====
.. TODO: add docs here.

.. toctree::
   :maxdepth: 2

   usage


.. doxygenfile:: Log.hpp

