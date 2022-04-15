module Geometry

using LinearAlgebra
using StaticArrays

import Base: -, +

#export # Exported by parent
#      Point, 
#
#export # Not exported by parent 
#    Edge, Edge2D, Edge3D,
#    Face, Face2D, Face3D,
#    Cell
#
## Point
#export  +, -, *, /, ⋅, ×, ⊙, ⊘, ==, ≈, distance,
#       distance², isCCW, midpoint, nan, norm, norm²
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
#include("geometry/Edge.jl")
#include("geometry/Face.jl")
#include("geometry/Cell.jl")
include("vectors.jl")
include("points.jl")

#include("geometry/LineSegment.jl")
#include("geometry/QuadraticSegment.jl")
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
