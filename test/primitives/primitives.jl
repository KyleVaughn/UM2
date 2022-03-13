@testset "primitives" begin
    tests = ["./Point",
             "./LineSegment",
             "./QuadraticSegment",
             "./Hyperplane",
             "./AABox",
             "./Polygon",
             "./QuadraticPolygon",
             "./Polyhedron",
             "./QuadraticPolyhedron"]
    for t in tests
      include("$(t).jl")
    end
end
