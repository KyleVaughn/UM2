@testset "Geometry" begin
    tests = ["vector",
             "point",
             "plane",
             "axisalignedbox",
             "polytope",
             "polytopes/interpolate",
             "polytopes/jacobian",
             "polytopes/faces",
             "polytopes/edges",
             "polytopes/measure",
             "polytopes/triangulate",
            ]
    for t in tests
      include("$(t).jl")
    end
end
