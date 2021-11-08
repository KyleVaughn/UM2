using MOCNeutronTransport
using Test
using Logging
Logging.disable_logging(Logging.Error)
tests = ["Tree",
         "Point_2D",
         "LineSegment_2D",
         "QuadraticSegment_2D",
         "Triangle_2D",
         "Quadrilateral_2D",
         "Triangle6_2D",
         "Quadrilateral8_2D",
         "AngularQuadrature", 
         "vtk",
         "abaqus",
         "xdmf"
#         "UnstructuredMesh",
        ]
for t in tests
  include("$(t).jl")
end
