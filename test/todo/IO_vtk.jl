using MOCNeutronTransport
@testset "VTK" begin @testset "Triangles" begin
    # Three triangles
    #      4---------5
    #     / \       / \
    #    /   \     /   \
    #   /     \   /     \
    #  /       \ /       \
    # 1---------2---------3
    filepath = "./mesh/mesh_files/three_triangles.vtk"
    ref_points = [
        Point_2D(0.0),
        Point_2D(2.0),
        Point_2D(4.0),
        Point_2D(1.0, 1.0),
        Point_2D(3.0, 1.0),
    ]
    ref_faces = [
        SVector(5, 1, 2, 4),
        SVector(5, 2, 5, 4),
        SVector(5, 2, 3, 5),
    ]
    ref_edges = [SVector(1, 2), SVector(1, 4), SVector(2, 3),
        SVector(2, 4), SVector(2, 5), SVector(3, 5), SVector(4, 5)]

    # read_vtk
    mesh = read_vtk_2d(filepath)
    @test mesh.points == ref_points
    @test mesh.faces == ref_faces
    @test mesh.name == "three_triangles"

    # write_vtk
    write_vtk_2d("write_three_triangle.vtk", mesh)
    ref_file = open(filepath, "r")
    test_file = open("./write_three_triangle.vtk", "r")
    while !eof(ref_file)
        ref_line = readline(ref_file)
        test_line = readline(test_file)
        @test ref_line == test_line
    end
    close(ref_file)
    close(test_file)
    rm("./write_three_triangle.vtk")
end end
