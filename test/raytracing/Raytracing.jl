@testset "Ray Tracing" begin
    tests = ["intersect/linesegment-linesegment",
             "intersect/linesegment-quadraticsegment"
            ]
    for t in tests
      include("$(t).jl")
    end
end
