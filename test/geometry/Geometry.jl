@testset "Geometry" begin
    tests = ["./vector",
             "./point",
             "./plane",
             "./axisalignedbox",
#             "./QuadraticSegment",
#             "./Hyperplane",
#             "./AABox",
#             "./Polygon",
#             "./QuadraticPolygon",
#             "./Polyhedron",
#             "./QuadraticPolyhedron"]
            ]
    for t in tests
      include("$(t).jl")
    end
end
