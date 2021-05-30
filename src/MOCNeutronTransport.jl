module MOCNeutronTransport
using StaticArrays
using LinearAlgebra

include("Point.jl")
export  Point,
        ×,
        ⋅,
        distance,
        norm

include("Edge.jl")
export Edge

include("Face.jl")
export Face

include("Cell.jl")
export Cell

include("LineSegment.jl")
export  LineSegment,
        arc_length,
        midpoint,
        intersect,
        is_left

include("QuadraticSegment.jl")
export QuadraticSegment,
       in_area


include("Triangle.jl")
export Triangle,
       area,
       intersect

include("Quadrilateral.jl")
export Quadrilateral,
       triangulate,
       area,
       intersect

include("UnstructuredMesh.jl")
export UnstructuredMesh

include("vtk.jl")
export read_vtk

include("AngularQuadrature.jl")
export AngularQuadrature,
       GeneralAngularQuadrature,
       ProductAngularQuadrature

end # module
