using MOCNeutronTransport
using Test
tests = ["Point", "LineSegment"]
for t in tests
  include("$(t).jl")
end
