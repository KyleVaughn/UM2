module MOCNeutronTransport
using StaticArrays
using LinearAlgebra

import Base: +, -, *, /, ≈, ==, intersect, in

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
include("Triangle6_2D.jl")
include("Triangle6_3D.jl")
include("Quadrilateral8_2D.jl")
include("Quadrilateral8_3D.jl")
include("gauss_legendre_quadrature.jl")
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
        Triangle6_2D,
        Triangle6_3D,
        Quadrilateral8_2D,
        Quadrilateral8_3D,
        ×,
        ⋅,
        arc_length,
        area,
        derivatives,
        distance,
        gauss_legendre_quadrature,
        intersect,
        intersect_iterative,
        jacobian,
        norm,
        real_to_parametric,
        triangulate


end # module
