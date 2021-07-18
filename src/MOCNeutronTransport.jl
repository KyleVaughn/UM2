module MOCNeutronTransport
using StaticArrays
using LinearAlgebra
try
    # Use local gmsh install
    using gmsh
catch e
    # Fall back on Gmsh package
    using Gmsh: gmsh
end

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
include("UnstructuredMesh.jl")
include("vtk.jl")
include("gmsh_rectangular_grid.jl")
include("gmsh_group_preserving_fragment.jl")

#
#export UnstructuredMesh,
#       edges,
#       AABB
#
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
        edges,
        gauss_legendre_quadrature,
        intersect,
        intersect_iterative,
        jacobian,
        norm,
        read_vtk_2d,
        real_to_parametric,
        triangulate,
        write_vtk_2d

export gmsh,
       gmsh_rectangular_grid,
       gmsh_group_preserving_fragment


end # module
