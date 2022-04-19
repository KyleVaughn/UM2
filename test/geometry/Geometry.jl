@testset "Geometry" begin
    tests = ["./vector",
             "./point",
             "./plane",
             "./axisalignedbox",
             "./interpolation",
            ]
    for t in tests
      include("$(t).jl")
    end
end
