using MOCNeutronTransport
include("../src/UnstructuredMesh.jl")
@testset "Unstructured Mesh" begin
    @testset "Construct Edges" begin
        # Three triangles
        points = [Point(0.0), Point(2.0), Point(4.0), Point(1.0, 1.0), Point(3.0, 1.0)]
    end
end
