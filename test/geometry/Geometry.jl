@testset "Geometry" begin
    tests = ["./vector",
             "./point",
             "./plane",
             "./axisalignedbox",
             "./interpolation",
             "./triangulate",
            ]
    for t in tests
      include("$(t).jl")
    end
end
