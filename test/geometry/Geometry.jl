@testset "Geometry" begin
    tests = ["vector",
        "point",
        "axisalignedbox",
        "polytopes/polytope",
        "polytopes/interpolate",
        "polytopes/jacobian",
        "polytopes/faces",
        "polytopes/edges",
        "polytopes/measure",
        "polytopes/triangulate",
        "polytopes/centroid",
        "polytopes/in",
    ]
    for t in tests
        include("$(t).jl")
    end
end
