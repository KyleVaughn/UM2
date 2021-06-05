using MOCNeutronTransport
using Test
tests = ["Point",
         "LineSegment",
         "QuadraticSegment",
         "Triangle",
         "Quadrilateral",
         "Triangle6",
         "UnstructuredMesh",
         "vtk",
         "AngularQuadrature", 
        ]
for t in tests
  include("$(t).jl")
end
