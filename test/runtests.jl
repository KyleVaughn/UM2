using MOCNeutronTransport
using Test
tests = ["Point"]
for t in tests
  include("$(t).jl")
end
