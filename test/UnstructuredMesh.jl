using MOCNeutronTransport
include("../src/UnstructuredMesh.jl")
@testset "Unstructured Mesh" begin
    @testset "Construct Edges" begin
        # Three triangles
        #      4---------5
        #     / \       / \
        #    /   \     /   \
        #   /     \   /     \
        #  /       \ /       \
        # 1---------2---------3
        points = [Point(0.0), Point(2.0), Point(4.0), Point(1.0, 1.0), Point(3.0, 1.0)]
        cells = [
                 [5, 1, 2, 4],
                 [5, 2, 5, 4],
                 [5, 2, 3, 5]
                ]
        edges = construct_edges(cells[1])
        @test edges == [[1, 2], [2, 4], [4, 1]]
        edges = construct_edges(cells[2])
        @test edges == [[2, 5], [5, 4], [4, 2]]
        edges = construct_edges(cells)
        @test edges == [ [1, 2], [1, 4], [2, 3], [2, 4], [2, 5], [3, 5], [4, 5] ]
    end
end
