using MOCNeutronTransport
using Test
tests = ["Point", "Line"]
for t in tests
  include("$(t).jl")
end
