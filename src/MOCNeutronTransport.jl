module MOCNeutronTransport
using StaticArrays

import Base: +, -, *, /, ≈, ==, intersect, in
import GLMakie: convert_arguments, LineSegments, Mesh, Scatter

include("Point_2D.jl")
include("Point_3D.jl")
include("LineSegment_2D.jl")
include("LineSegment_3D.jl")
include("QuadraticSegment_2D.jl")
include("QuadraticSegment_3D.jl")
include("Triangle_2D.jl")
include("Triangle_3D.jl")
include("Quadrilateral_2D.jl")
include("Quadrilateral_3D.jl")
include("gauss_legendre_quadrature.jl")
#include("Triangle6.jl")
#export Triangle6,
#       triangulate,
#       area,
#       intersect,
#       in
#
#include("Quadrilateral8.jl")
#export Quadrilateral8
#
#
#include("UnstructuredMesh.jl")
#export UnstructuredMesh,
#       edges,
#       AABB
#
#include("vtk.jl")
#export read_vtk,
#       write_vtk
#
#include("AngularQuadrature.jl")
#export AngularQuadrature,
#       GeneralAngularQuadrature,
#       ProductAngularQuadrature
#

export  Point_2D,
        Point_3D,
        LineSegment_2D,
        LineSegment_3D,
        QuadraticSegment_2D,
        QuadraticSegment_3D,
        Triangle_2D,
        Triangle_3D,
        Quadrilateral_2D,
        Quadrilateral_3D,
        ×,
        ⋅,
        arc_length,
        area,
        distance,
        intersect,
        gauss_legendre_quadrature,
        norm,
        triangulate






end # module
