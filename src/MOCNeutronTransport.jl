module MOCNeutronTransport
using StaticArrays
using LinearAlgebra

import Base: +, -, *, /, ≈, ==, intersect, in
import LinearAlgebra: norm
import GLMakie: convert_arguments, LineSegments, Mesh, Scatter

include("Point_2D.jl")
include("Point_3D.jl")
include("LineSegment_2D.jl")
include("LineSegment_3D.jl")
#export  LineSegment,


#
#include("QuadraticSegment.jl")
#export QuadraticSegment,
#       intersect,
#       arc_length
#
#include("Triangle.jl")
#export Triangle,
#       area,
#       intersect,
#       in
#
#include("Quadrilateral.jl")
#export Quadrilateral,
#       triangulate,
#       area,
#       intersect,
#       in
#
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
#include("gauss_legendre_quadrature.jl")
#export gauss_legendre_quadrature
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
        ×,
        ⋅,
        distance,
        norm,
        intersect,
        arc_length






end # module
