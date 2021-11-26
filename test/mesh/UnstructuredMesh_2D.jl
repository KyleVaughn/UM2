using MOCNeutronTransport
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
        @test cell_edges == SVector(MVector(1, 2), MVector(2, 4), MVector(1, 4))
        cell_edges = edges(faces[2])
        @test cell_edges == SVector(MVector(2, 5), MVector(4, 5), MVector(2, 4))
        cell_edges = edges(faces)
        @test cell_edges == SVector( MVector(1, 2), MVector(1, 4), MVector(2, 3), 
                                     MVector(2, 4), MVector(2, 5), MVector(3, 5), 
                                     MVector(4, 5) )
    end
end
