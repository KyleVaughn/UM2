using MOCNeutronTransport
using Test
tests = ["Point", "LineSegment", "AngularQuadrature", "QuadraticSegment"]
for t in tests
  include("$(t).jl")
end
