using MOCNeutronTransport
using Test
using Logging
Logging.disable_logging(Logging.Error)
tests = ["Tree",
         "./primitives/Point",
         "./primitives/LineSegment",
         "./primitives/AABB",
         "./primitives/QuadraticSegment",
         "./primitives/Triangle",
         "./primitives/Quadrilateral",
         "./primitives/Triangle6",
#         "./primitives/Quadrilateral8_2D",
#         "./mesh/UnstructuredMesh_2D",
#         "./mesh/IO_abaqus",
#         "./mesh/IO_vtk",
#         "./mesh/IO_xdmf",
#         "AngularQuadrature", 
        ]
for t in tests
  include("$(t).jl")
end
