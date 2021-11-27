using MOCNeutronTransport
using HDF5
@testset "XDMF" begin
    @testset "c5g7 pin - triangles" begin
        vtk_to_xdmf_type = Dict(
            # triangle
            5  => 4,
            # triangle6
            22 => 36,
            # quadrilateral
            9 => 5,
            # quad8
            23 => 37
           )
        ref_points = [Point_2D(0.0, 0.0),
                      Point_2D(1.26, 0.0),
                      Point_2D(0.0, 1.26),
                      Point_2D(1.26, 1.26),
                      Point_2D(1.17, 0.63),
                      Point_2D(0.63, 0.0),
                      Point_2D(0.0, 0.63),
                      Point_2D(1.26, 0.63),
                      Point_2D(0.63, 1.26),
                      Point_2D(0.96668449300372, 1.0521890005327),
                      Point_2D(0.50983869566359, 1.1564610725782),
                      Point_2D(0.14347681133269, 0.86429721912348),
                      Point_2D(0.14347681133269, 0.39570278087652),
                      Point_2D(0.50983869566359, 0.10353892742181),
                      Point_2D(0.96668449300372, 0.20781099946726),
                      Point_2D(0.63, 0.63)]
        ref_faces = [
                     SVector{4, UInt64}(UInt64[5, 12, 16, 11]),
                     SVector{4, UInt64}(UInt64[5, 13, 16, 12]),
                     SVector{4, UInt64}(UInt64[5, 14, 16, 13]),
                     SVector{4, UInt64}(UInt64[5, 11, 16, 10]),
                     SVector{4, UInt64}(UInt64[5, 10, 16, 5]), 
                     SVector{4, UInt64}(UInt64[5, 15, 16, 14]),
                     SVector{4, UInt64}(UInt64[5, 5, 16, 15]),
                     SVector{4, UInt64}(UInt64[5, 7, 1, 13]),
                     SVector{4, UInt64}(UInt64[5, 7, 12, 3]), 
                     SVector{4, UInt64}(UInt64[5, 1, 6, 14]),
                     SVector{4, UInt64}(UInt64[5, 1, 14, 13]),
                     SVector{4, UInt64}(UInt64[5, 6, 2, 15]),
                     SVector{4, UInt64}(UInt64[5, 15, 2, 8]), 
                     SVector{4, UInt64}(UInt64[5, 3, 11, 9]), 
                     SVector{4, UInt64}(UInt64[5, 3, 12, 11]),
                     SVector{4, UInt64}(UInt64[5, 10, 8, 4]), 
                     SVector{4, UInt64}(UInt64[5, 9, 10, 4]), 
                     SVector{4, UInt64}(UInt64[5, 10, 5, 8]), 
                     SVector{4, UInt64}(UInt64[5, 15, 8, 5]), 
                     SVector{4, UInt64}(UInt64[5, 14, 6, 15]),
                     SVector{4, UInt64}(UInt64[5, 7, 13, 12]),
                     SVector{4, UInt64}(UInt64[5, 11, 10, 9])
                   ]
        ref_face_sets = Dict{String, Set{UInt64}}()
        ref_face_sets["MATERIAL_UO2-3.3"] = Set([1, 2, 3, 4, 5, 6, 7])
        ref_face_sets["GRID_L1_1_1"] = Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
        ref_face_sets["MATERIAL_WATER"] = Set([8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                               18, 19, 20, 21, 22])
        ref_face_sets["test_set"] = Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
        mesh = UnstructuredMesh_2D{Float64, UInt64}(
                                   points = ref_points, 
                                   faces = ref_faces, 
                                   face_sets = ref_face_sets, 
                                   name = "c5g7_UO2_pin")
        # write_xdmf
        write_xdmf_2d("c5g7_UO2_pin.xdmf", mesh)
        # check xdmf
        ref_file = open("./mesh/mesh_files/c5g7_UO2_pin.xdmf", "r")
        test_file = open("./c5g7_UO2_pin.xdmf", "r")
        while !eof(ref_file)
            ref_line = readline(ref_file)
            test_line = readline(test_file)
            @test ref_line == test_line
        end
        close(ref_file)
        close(test_file)
        # check h5
        h5_file = h5open("c5g7_UO2_pin.h5","r")

        # points
        point_array = zeros(Float64, length(mesh.points), 3)
        npoints = length(mesh.points)
        for i = 1:npoints
            point_array[i, 1] = mesh.points[i][1]
            point_array[i, 2] = mesh.points[i][2]
        end
        point_array = copy(transpose(point_array))
        test_points = read(h5_file["c5g7_UO2_pin"]["points"])
        @test test_points == point_array

        # cells
        topo_array = UInt64[]
        for face in mesh.faces
            # convert face to vector for mutability
            face_xdmf = [x for x in face]
            # adjust vtk to xdmf type
            face_xdmf[1] = vtk_to_xdmf_type[face_xdmf[1]]
            # adjust 1-based to 0-based indexing
            face_xdmf[2:length(face_xdmf)] = face_xdmf[2:length(face_xdmf)] .- 1
            for value in face_xdmf
                push!(topo_array, value)
            end
        end
        test_cells = read(h5_file["c5g7_UO2_pin"]["cells"])
        @test test_cells == topo_array 

        # Cell sets
        test_grid_l1_1_1 = read(h5_file["c5g7_UO2_pin"]["GRID_L1_1_1"])
        ref_grid_l1_1_1 = [ x - 1 for x in ref_face_sets["GRID_L1_1_1"]]
        @test test_grid_l1_1_1 == ref_grid_l1_1_1
        test_test_set = read(h5_file["c5g7_UO2_pin"]["test_set"])
        ref_test_set = [ x - 1 for x in ref_face_sets["test_set"]]
        @test test_test_set == ref_test_set

        # Materials
        test_material_IDs = read(h5_file["c5g7_UO2_pin"]["material_id"])
        ref_material_IDs = [ i <= 7 ? 0 : 1 for i = 1:22 ]
        @test test_material_IDs == ref_material_IDs

        close(h5_file)
        rm("./c5g7_UO2_pin.xdmf")
        rm("./c5g7_UO2_pin.h5")
    end
end
