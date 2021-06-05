module MOCNeutronTransport
using StaticArrays
using LinearAlgebra
using RecipesBase

include("Edge.jl")
export Edge
include("Face.jl")
export Face
include("Cell.jl")
export Cell

include("Point.jl")
export  Point,
        ×,
        ⋅,
        distance,
        norm

include("LineSegment.jl")
export  LineSegment,
        intersect,
        arc_length

include("QuadraticSegment.jl")
export QuadraticSegment,
       intersect,
       arc_length

include("Triangle.jl")
export Triangle,
       area,
       intersect,
       in

include("Quadrilateral.jl")
export Quadrilateral,
       triangulate,
       area,
       intersect,
       in

include("Triangle6.jl")
export Triangle6,
       area




include("UnstructuredMesh.jl")
export UnstructuredMesh,
       edges,
       AABB

include("vtk.jl")
export read_vtk,
       write_vtk

include("AngularQuadrature.jl")
export AngularQuadrature,
       GeneralAngularQuadrature,
       ProductAngularQuadrature

end # module
