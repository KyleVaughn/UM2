# Copy the mesh files to the built test directory
file(COPY ${PROJECT_SOURCE_DIR}/tests/mesh/mesh_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

um2_add_test(./regular_grid.cpp)
um2_add_test(./rectilinear_grid.cpp)
um2_add_test(./element_types.cpp)
um2_add_test(./polytope_soup.cpp)
#um2_add_test(./face_vertex_mesh/tri_mesh.cpp)
#um2_add_test(./face_vertex_mesh/quad_mesh.cpp)
#um2_add_test(./face_vertex_mesh/quadratic_tri_mesh.cpp)
#um2_add_test(./face_vertex_mesh/quadratic_quad_mesh.cpp)
#um2_add_test(./binned_face_vertex_mesh/binned_tri_mesh.cpp)
