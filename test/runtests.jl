using MOCNeutronTransport
using Test
tests = ["Point",
         "LineSegment",
         "QuadraticSegment",
         "Triangle",
         "Quadrilateral",
         "UnstructuredMesh",
         "vtk",
         "AngularQuadrature", 
        ]
for t in tests
  include("$(t).jl")
end
