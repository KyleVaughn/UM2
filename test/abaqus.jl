using MOCNeutronTransport
@testset "Abaqus" begin
    @testset "c5g7 pin - triangles" begin
        filepath = "./mesh_files/c5g7_UO2_pin.inp"
        ref_points = (Point_2D(0.0, 0.0),
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
                      Point_2D(0.63, 0.63))
        ref_faces = (
                    (5, 12, 16, 11),
                    (5, 13, 16, 12),
                    (5, 14, 16, 13),
                    (5, 11, 16, 10),
                    (5, 10, 16, 5),
                    (5, 15, 16, 14),
                    (5, 5, 16, 15),
                    (5, 7, 1, 13),
                    (5, 7, 12, 3),
                    (5, 1, 6, 14),
                    (5, 1, 14, 13),
                    (5, 6, 2, 15),
                    (5, 15, 2, 8),
                    (5, 3, 11, 9),
                    (5, 3, 12, 11),
                    (5, 10, 8, 4),
                    (5, 9, 10, 4),
                    (5, 10, 5, 8),
                    (5, 15, 8, 5),
                    (5, 14, 6, 15),
                    (5, 7, 13, 12),
                    (5, 11, 10, 9)
                    )
        ref_face_sets = Dict{String, Tuple{Vararg{Int64}}}()
        ref_face_sets["MATERIAL_UO2-3.3"] = (1, 2, 3, 4, 5, 6, 7)
        ref_face_sets["GRID_L1_1_1"] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
        ref_face_sets["MATERIAL_WATER"] = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                                           18, 19, 20, 21, 22)

        # read_abaqus
        mesh = read_abaqus_2d(filepath)
        @test mesh.points == ref_points
        @test mesh.faces == ref_faces
        @test mesh.name == "c5g7_UO2_pin"
        @test mesh.face_sets == ref_face_sets
    end
end
