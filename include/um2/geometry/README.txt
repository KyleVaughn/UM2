===============================================================================
OVERVIEW
===============================================================================
This directory contains various geometry related classes and functions. Most of
the classes and functions are straight forward, but some deserve a bit of 
explanation.

Point
------
The Point class is simply an alias to Vec (math/vec.hpp).

Polytope
--------
Polyhedrons, polygons, etc. can be generalized as polytopes. We wish to model
not only typical polygons with straight edges, but also polygons with curved
polynomial edges. Hence, the polytope is a useful abstraction which is
specialized via template parameters to triangles, line segments, etc. A
K-dimensional polytope, of polynomial order P, represented by the connectivity
of its vertices. These N vertices are D-dimensional points.

See https://en.wikipedia.org/wiki/Polytope for help with terminology.
See "The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th
     Edition, Chapter 8, Advanced Data Representation" for more info.

- Dion        (K = 1)
- Polygon     (K = 2)
- Polyhedron  (K = 3)

- LineSegment             (K = 1, P = 1, N = 2)
- QuadraticSegment        (K = 1, P = 2, N = 3)

- Triangle                (K = 2, P = 1, N = 3)
- Quadrilateral           (K = 2, P = 1, N = 4)
- QuadraticTriangle       (K = 2, P = 2, N = 6)
- QuadraticQuadrilateral  (K = 2, P = 2, N = 8)

- Tetrahedron             (K = 3, P = 1, N = 4)
... etc.
