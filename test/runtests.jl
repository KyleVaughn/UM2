using MOCNeutronTransport
using Test
using Logging
Logging.disable_logging(Logging.Error)
tests = ["./primitives/Point_2D",
         "./primitives/LineSegment_2D",
         "./primitives/QuadraticSegment_2D",
         "./primitives/Triangle_2D",
         "./primitives/Quadrilateral_2D",
         "./primitives/Triangle6_2D",
         "./primitives/Quadrilateral8_2D",
         "./mesh/UnstructuredMesh_2D",
         "./mesh/IO_abaqus",
#         "Tree",
#         "AngularQuadrature", 
#         "vtk",
#         "xdmf"
        ]
for t in tests
  include("$(t).jl")
end
