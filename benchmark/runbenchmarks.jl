using MOCNeutronTransport
benchmarks = ["Point",
         "LineSegment",
         "QuadraticSegment",
         "Triangle",
         "Quadrilateral"
        ]
for b in benchmarks
  include("$(b).jl")
end
