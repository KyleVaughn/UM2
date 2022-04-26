@testset "Geometry" begin
    tests = ["vector",
             "point",
             "plane",
             "axisalignedbox",
             "polytope",
             "polytopes/interpolate",
             "polytopes/edges"
#             "polytopes/measure"
#             "triangulate",
            ]
    for t in tests
      include("$(t).jl")
    end
end
