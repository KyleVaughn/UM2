using MOCNeutronTransport
benchmarks = ["Point_2D",
            "Point_3D",
            "LineSegment_2D",
            "LineSegment_3D",
            "QuadraticSegment_2D",
            "QuadraticSegment_3D"
#         "Triangle",
#         "Quadrilateral",
#         "Triangle6"
        ]
for b in benchmarks
  include("$(b).jl")
end
