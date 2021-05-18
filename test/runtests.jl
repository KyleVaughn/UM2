using MOCNeutronTransport
using Test
tests = ["Point", 
         "LineSegment", 
         "QuadraticSegment",
         "AngularQuadrature", 
         "Triangle",
        ]
for t in tests
  include("$(t).jl")
end
