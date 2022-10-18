export Polytope, 
       Edge, 
       LineSegment, 
       QuadraticSegment, 
       Face, 
       Polygon, 
       QuadraticPolygon,
       Triangle, 
       Quadrilateral, 
       QuadraticTriangle, 
       QuadraticQuadrilateral,

export vertices

"""

    Polytope{K,P,N,D,T}

A `K`-polytope of order `P` with `N` vertices, which are `D`-dimensional points of type `T`.

## Aliases

```julia
# 1-polytope
const Edge                      = Polytope{1}
const LineSegment               = Edge{1,2}
const QuadraticSegment          = Edge{2,3}
# 2-polytope
const Face                      = Polytope{2}
const Polygon                   = Face{1}
const QuadraticPolygon          = Face{2}
const Triangle                  = Polygon{3}
const Quadrilateral             = Polygon{4}
const QuadraticTriangle         = QuadraticPolygon{6}
const QuadraticQuadrilateral    = QuadraticPolygon{8}
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
struct Polytope{K,P,N,D,T}
    vertices::NTuple{N, Point{D, T}}
end

#  -- Type aliases --

# 1-polytope
const Edge                      = Polytope{1}
const LineSegment               = Edge{1, 2}
const LineSegment2              = LineSegment{2}
const QuadraticSegment          = Edge{2, 3}
const QuadraticSegment2         = QuadraticSegment{2}
# 2-polytope
const Face                      = Polytope{2}
const Polygon                   = Face{1}
const Triangle                  = Polygon{3}
const Triangle2                 = Triangle{2}
const Quadrilateral             = Polygon{4}
const Quadrilateral2            = Quadrilateral{2}
const QuadraticPolygon          = Face{2}
const QuadraticTriangle         = QuadraticPolygon{6}
const QuadraticTriangle2        = QuadraticTriangle{2}
const QuadraticQuadrilateral    = QuadraticPolygon{8}
const QuadraticQuadrilateral2   = QuadraticQuadrilateral{2}

# -- Constructors --

function Polytope{K,P,N,D}(vertices::NTuple{N, Point{D, T}}) where {K,P,N,D,T}
    return Polytope{K,P,N,D,T}(vertices)
end
function Polytope{K,P,N}(vertices::NTuple{N, Point{D, T}}) where {K,P,N,D,T}
    return Polytope{K,P,N,D,T}(vertices)
end

# -- Base --

Base.getindex(poly::Polytope, i::Integer) = Base.getindex(poly.vertices, i)

# -- Accessors --

vertices(p::Polytope) = p.vertices
