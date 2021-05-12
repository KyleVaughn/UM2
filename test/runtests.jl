using MOCNeutronTransport
using Test
tests = ["Point", "LineSegment", "AngularQuadrature"]
for t in tests
  include("$(t).jl")
end
