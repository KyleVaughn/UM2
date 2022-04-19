export Polytope, Edge, LineSegment, QuadraticSegment, Face, Polygon, QuadraticPolygon,
       Triangle, Quadrilateral, QuadraticTriangle, QuadraticQuadrilateral, Cell,
       Polyhedron, QuadraticPolyhedron, Tetrahedron, Hexahedron, QuadraticTetrahedron,
       QuadraticHexahedron

"""

A `K`-polytope of order `P` with `N` vertices, where the vertices are points
in `Dim`-dimensional space with coordinates of type `T`.

## Aliases

```julia
# parametric dimension 1
const Edge                      = Polytope{1}
const LineSegment               = Edge{1,2}
const QuadraticSegment          = Edge{2,3}
# parametric dimension 2
const Face                      = Polytope{2}
const Polygon                   = Face{1}
const QuadraticPolygon          = Face{2}
const Triangle                  = Polygon{3}
const Quadrilateral             = Polygon{4}
const QuadraticTriangle         = QuadraticPolygon{6}
const QuadraticQuadrilateral    = QuadraticPolygon{8}
# parametric dimension 3
const Cell                      = Polytope{3}
const Polyhedron                = Cell{1}
const QuadraticPolyhedron       = Cell{2}
const Tetrahedron               = Polyhedron{4}
const Hexahedron                = Polyhedron{8}
const QuadraticTetrahedron      = QuadraticPolyhedron{10}
const QuadraticHexahedron       = QuadraticPolyhedron{20}
```
### Notes
- These are Lagrangian finite elements.
- This struct only supports the shapes found in "The Visualization Toolkit:
  An Object-Oriented Approach to 3D Graphics, 4th Edition, Chapter 8, Advanced
  Data Representation".
- See the VTK book for specific vertex ordering info, but generally vertices are
  ordered in a counterclockwise fashion, with vertices of the linear shape given
  first.
- See https://en.wikipedia.org/wiki/Polytope for help with terminology.
"""
struct Polytope{K,P,N,Dim,T}
    vertices::Vec{N, Point{Dim,T}}
end

# type aliases
# parametric dimension 1
const Edge                      = Polytope{1}
const LineSegment               = Edge{1,2}
const QuadraticSegment          = Edge{2,3}
# parametric dimension 2
const Face                      = Polytope{2}
const Polygon                   = Face{1}
const QuadraticPolygon          = Face{2}
const Triangle                  = Polygon{3}
const Quadrilateral             = Polygon{4}
const QuadraticTriangle         = QuadraticPolygon{6}
const QuadraticQuadrilateral    = QuadraticPolygon{8}
# parametric dimension 3
const Cell                      = Polytope{3}
const Polyhedron                = Cell{1}
const QuadraticPolyhedron       = Cell{2}
const Tetrahedron               = Polyhedron{4}
const Hexahedron                = Polyhedron{8}
const QuadraticTetrahedron      = QuadraticPolyhedron{10}
const QuadraticHexahedron       = QuadraticPolyhedron{20}

# constructors
function Polytope{K,P,N}(vertices::Vec{N, Point{Dim,T}}) where {K,P,N,Dim,T}
    return Polytope{K,P,N,Dim,T}(vertices)
end
Polytope{K,P,N}(vertices...) where {K,P,N} = Polytope{K,P,N}(Vec(vertices))

Base.getindex(poly::Polytope, i::Int) = Base.getindex(poly.vertices, i)
