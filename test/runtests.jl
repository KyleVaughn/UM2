using MOCNeutronTransport
using Test
tests = ["Point",
         "LineSegment",
         "QuadraticSegment",
         "AngularQuadrature", 
         "Triangle",
         "UnstructuredMesh",
         "vtk"
        ]
for t in tests
  include("$(t).jl")
end
