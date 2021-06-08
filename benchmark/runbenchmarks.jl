using MOCNeutronTransport
benchmarks = ["Point",
         "LineSegment",
         "QuadraticSegment",
         "Triangle",
         "Quadrilateral",
         "Triangle6"
        ]
for b in benchmarks
  include("$(b).jl")
end
