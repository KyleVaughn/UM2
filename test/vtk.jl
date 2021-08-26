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
        ref_cells = (
                        (5, 1, 2, 4),
                        (5, 2, 5, 4),
                        (5, 2, 3, 5)
                    )
#        ref_edges = [[1, 2], [1, 4], [2, 3], [2, 4], [2, 5], [3, 5], [4, 5]]
        file = open(filepath, "r")

        # Test of non-public functions.
        # Need to find a way to do the includes correctly for this, since
        # just including vtk messes up the use of UnstructuresMesh_<x>_cell_types
#        # points
#        readuntil(file, "POINTS")
#        npoints_string, datatype_string = split(readline(file))
#        points = read_vtk_points(file, npoints_string, datatype_string)
#        @test points == ref_points
#
#        # cells
#        seekstart(file)
#        readuntil(file, "CELLS")
#        ncells_string = split(readline(file))[1]
#        cells = read_vtk_cells(file, ncells_string)
#        for (i, cell) in enumerate(cells)
#            @test cell == ref_cells[i][2:4]
#        end
#
#        # cell_types
#        seekstart(file)
#        readuntil(file, "CELL_TYPES")
#        ncells_string = split(readline(file))[1]
#        cell_types = read_vtk_cell_types(file, ncells_string)
#        for (i, type) in enumerate(cell_types)
#            @test type == ref_cells[i][1]
#        end
#
#        close(file)
        # read_vtk
        mesh = read_vtk_2d(filepath)
        @test mesh.points == ref_points
        @test mesh.faces == ref_cells
#        @test mesh.edges == ref_edges
#        @test mesh.cells == Vector{Int64}[]
#        @test mesh.dim == 2
        @test mesh.name == "three_triangles"

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
