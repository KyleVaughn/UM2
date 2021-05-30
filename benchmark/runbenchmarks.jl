using MOCNeutronTransport
benchmarks = ["Point",
         "LineSegment",
         "QuadraticSegment",
         "Triangle"
        ]
for b in benchmarks
  include("$(b).jl")
end
