using MOCNeutronTransport
include("../../src/constants.jl")
include("../../src/mesh/UnstructuredMesh_2D_low_level.jl")
include("../../src/mesh/UnstructuredMesh_2D.jl")
@testset "UnstructuredMesh" begin
    @testset "Construct Edges" begin
        # Three triangles
        #      4---------5
        #     / \       / \
        #    /   \     /   \
        #   /     \   /     \
        #  /       \ /       \
        # 1---------2---------3
        points = [Point_2D(0.0), Point_2D(2.0), Point_2D(4.0), Point_2D(1.0, 1.0), Point_2D(3.0, 1.0)]
        faces = [
                 SVector{4, UInt64}(UInt64[5, 1, 2, 4]),
                 SVector{4, UInt64}(UInt64[5, 2, 5, 4]),
                 SVector{4, UInt64}(UInt64[5, 2, 3, 5])
                ]

        # edges
        cell_edges = edges(faces[1])
        @test cell_edges == [[1, 2], [2, 4], [4, 1]]
        cell_edges = edges(faces[2])
        @test cell_edges == [[2, 5], [5, 4], [4, 2]]
        cell_edges = edges(faces)
        @test cell_edges == [ [1, 2], [1, 4], [2, 3], [2, 4], [2, 5], [3, 5], [4, 5] ]
    end
end
