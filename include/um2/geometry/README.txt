===============================================================================
OVERVIEW
===============================================================================
This directory contains various geometry related classes and functions. Most of
the classes and functions are straight forward, but some deserve a bit of 
explanation.

Point
------
The Point class is simply an alias to Vec (math/vec.hpp). This is not mathematically 
correct, but it is easier to use this way.

Polytope
--------
Polyhedrons, polygons, etc. can be generalized as polytopes. We wish to model
not only typical polygons with straight edges, but also polygons with curved
polynomial edges. Hence, we use the polytope as a useful abstraction which is
specialized via template parameters to triangles, line segments, etc. A
K-dimensional polytope, of polynomial order P, represented by the connectivity
of its vertices. These N vertices are D-dimensional points of type F.

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

intersect(Ray)
--------------
Any geometric objects which have a function Object.intersect(Ray) return the parametric
coordinate/coordinates such that Ray(r) = origin + r*direction gives the point/points
of intersection. r must be in range [0, inf). If r is negative, or there is no intersection,
then the function returns -1 as the coordinate.
