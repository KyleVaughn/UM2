using MOCNeutronTransport
using Test
tests = ["Point_2D",
         "Point_3D",
         "LineSegment_2D",
         "LineSegment_3D",
         "QuadraticSegment_2D",
         "QuadraticSegment_3D"
#         "Triangle",
#         "Quadrilateral",
#         "Triangle6",
#         "Quadrilateral8",
#         "UnstructuredMesh",
#         "vtk",
#         "AngularQuadrature", 
        ]
for t in tests
  include("$(t).jl")
end
