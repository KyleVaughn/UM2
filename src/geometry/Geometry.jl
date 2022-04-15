module Geometry

using LinearAlgebra
using StaticArrays

import Base: -, +
## LineSegment
#export LineSegment, LineSegment2D, LineSegment3D
## QuadraticSegment
#export QuadraticSegment, QuadraticSegment2D, QuadraticSegment3D, isstraight
## Hyperplane
#export Hyperplane, Hyperplane2D, Hyperplane3D
## AABox
#export AABox, AABox2D, AABox3D, Δx, Δy, Δz
## Polygon
#export Polygon, Polygon2D, Polygon3D, Triangle, Triangle2D, Triangle3D, Quadrilateral,
#       Quadrilateral2D, Quadrilateral3D
## QuadraticPolygon
#export QuadraticPolygon, QuadraticTriangle, QuadraticTriangle2D, QuadraticTriangle3D,
#       QuadraticQuadrilateral, QuadraticQuadrilateral2D, QuadraticQuadrilateral3D
## Polyhedron
#export Polyhedron, Tetrahedron, Hexahedron
## QuadraticPolyhedron
#export QuadraticPolyhedron, QuadraticTetrahedron, QuadraticHexahedron
## triangulate
#export triangulate
## measure
#export measure
#
include("vector.jl")
include("point.jl")
include("linesegment.jl")
include("quadraticsegment.jl")
#include("geometry/Hyperplane.jl")
#include("geometry/AABox.jl")
#include("geometry/Polygon.jl")
#include("geometry/QuadraticPolygon.jl")
#include("geometry/Polyhedron.jl")
#include("geometry/QuadraticPolyhedron.jl")
#include("geometry/interpolation.jl")
#include("geometry/triangulate.jl")
#include("geometry/measure.jl")
end
