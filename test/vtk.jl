using MOCNeutronTransport
@testset "VTK" begin
    @testset "Triangles" begin
        # Three triangles
        #      4---------5
        #     / \       / \
        #    /   \     /   \
        #   /     \   /     \
        #  /       \ /       \
        # 1---------2---------3
        filepath = "./mesh_files/three_triangles.vtk"
        ref_points = (Point_2D(0.0), Point_2D(2.0), Point_2D(4.0), Point_2D(1.0, 1.0), Point_2D(3.0, 1.0))
        ref_faces = (
                        (5, 1, 2, 4),
                        (5, 2, 5, 4),
                        (5, 2, 3, 5)
                    )
        ref_edges = ((1, 2), (1, 4), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5))
        ref_cell_sets = Dict{String, Tuple{Vararg{Int64}}}()

        # Test of non-public functions.
        # Need to find a way to do the includes correctly for this, since
        # just including vtk messes up the use of UnstructuresMesh_<x>_cell_types
        #
#        file = open(filepath, "r")
#        # points
#        readuntil(file, "POINTS")
#        npoints_string, datatype_string = split(readline(file))
#        points = read_vtk_points(file, npoints_string, datatype_string)
#        @test points == ref_points
#
#        # faces
#        seekstart(file)
#        readuntil(file, "faces")
#        nfaces_string = split(readline(file))[1]
#        faces = read_vtk_faces(file, nfaces_string)
#        for (i, cell) in enumerate(faces)
#            @test cell == ref_faces[i][2:4]
#        end
#
#        # cell_types
#        seekstart(file)
#        readuntil(file, "CELL_TYPES")
#        nfaces_string = split(readline(file))[1]
#        cell_types = read_vtk_cell_types(file, nfaces_string)
#        for (i, type) in enumerate(cell_types)
#            @test type == ref_faces[i][1]
#        end
#
#        close(file)
        # read_vtk
        mesh = read_vtk_2d(filepath)
        @test mesh.points == ref_points
        @test mesh.faces == ref_faces
#        @test mesh.edges == ref_edges
        @test mesh.name == "three_triangles"
        @test mesh.cell_sets == ref_cell_sets

        # write_vtk
        write_vtk_2d("write_three_triangle.vtk", mesh)
        ref_file = open("./mesh_files/three_triangles.vtk", "r")
        test_file = open("./write_three_triangle.vtk", "r")
        while !eof(ref_file)
            ref_line = readline(ref_file)
            test_line = readline(test_file)
            @test ref_line == test_line
        end
        rm("./write_three_triangle.vtk")
    end
end
