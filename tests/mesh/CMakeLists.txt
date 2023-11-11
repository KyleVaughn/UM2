# Copy the mesh files to the built test directory
file(COPY ${PROJECT_SOURCE_DIR}/tests/mesh/mesh_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_um2_test(./regular_grid.cpp)
add_um2_test(./regular_partition.cpp)
add_um2_test(./rectilinear_grid.cpp)
add_um2_test(./rectilinear_partition.cpp)
add_um2_test(./polytope_soup.cpp)
add_um2_test(./tri_mesh.cpp)
add_um2_test(./quad_mesh.cpp)
add_um2_test(./quadratic_tri_mesh.cpp)
add_um2_test(./quadratic_quad_mesh.cpp)
add_um2_test(./io_abaqus.cpp)
add_um2_test(./io_xdmf.cpp)
