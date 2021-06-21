using MOCNeutronTransport
benchmarks = ["Point_2D",
            "Point_3D",
            "LineSegment_2D",
#            "LineSegment_3D",
            "QuadraticSegment_2D",
#            "QuadraticSegment_3D",
            "Triangle_2D",
            "Triangle_3D",
            "Quadrilateral_2D",
            "Quadrilateral_3D",
            "Triangle6_2D",
            "Triangle6_3D",
            "Quadrilateral8_3D"
        ]
for b in benchmarks
  include("$(b).jl")
end
