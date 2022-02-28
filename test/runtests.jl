using MOCNeutronTransport
using Test

# TODO: centroid.jl, boundingbox.jl, cartesian_to_parametric.jl

include("setup/setup_primitives.jl")
tests = ["SVector",
         "primitives/Point",
         "primitives/LineSegment",
         "primitives/QuadraticSegment",
         "primitives/Hyperplane",
         "primitives/AABox",
         "primitives/ConvexPolygon",
         "primitives/QuadraticPolygon",
         "primitives/ConvexPolyhedron",
         "primitives/QuadraticPolyhedron",
         "interpolation",
         "jacobian",
         "triangulate",
         "measure",
#        # "./mesh/UnstructuredMesh_2D",
         #"./mesh/IO_abaqus",
#         "./mesh/IO_vtk",
#         "./mesh/IO_xdmf",
#         "AngularQuadrature", 
        ]
for t in tests
  include("$(t).jl")
end
