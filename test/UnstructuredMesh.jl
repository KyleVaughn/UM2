using MOCNeutronTransport
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
        cell_edges = edges(cells[1])
        @test cell_edges == [[1, 2], [2, 4], [4, 1]]
        cell_edges = edges(cells[2])
        @test cell_edges == [[2, 5], [5, 4], [4, 2]]
        cell_edges = edges(cells)
        @test cell_edges == [ [1, 2], [1, 4], [2, 3], [2, 4], [2, 5], [3, 5], [4, 5] ]
    end
end
