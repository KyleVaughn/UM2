using MOCNeutronTransport
using Test
tests = ["./SVector",
         "./primitives/Point",
         "./primitives/LineSegment",
         "./primitives/QuadraticSegment",
         "./primitives/Hyperplane",
         "./primitives/AABox",
         "./primitives/ConvexPolygon",
         "./primitives/QuadraticPolygon",
         "./primitives/ConvexPolyhedron",
         "./primitives/QuadraticPolyhedron",
         "./interpolation",
         "./jacobian",
#        # "./mesh/UnstructuredMesh_2D",
         #"./mesh/IO_abaqus",
#         "./mesh/IO_vtk",
#         "./mesh/IO_xdmf",
#         "AngularQuadrature", 
        ]
for t in tests
  include("$(t).jl")
end
