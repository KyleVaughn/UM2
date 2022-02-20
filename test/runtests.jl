using MOCNeutronTransport
using Test
using Logging
Logging.disable_logging(Logging.Error)
tests = ["./trees/AnyTree",
         "./primitives/Point",
         "./primitives/LineSegment",
         "./primitives/Hyperplane",
         "./primitives/AABox",
         "./primitives/QuadraticSegment",
         "./primitives/Triangle",
         "./primitives/Quadrilateral",
         "./primitives/QuadraticTriangle",
         "./primitives/QuadraticQuadrilateral",
#         "./mesh/UnstructuredMesh_2D",
         "./mesh/IO_abaqus",
#         "./mesh/IO_vtk",
#         "./mesh/IO_xdmf",
#         "AngularQuadrature", 
        ]
for t in tests
  include("$(t).jl")
end
