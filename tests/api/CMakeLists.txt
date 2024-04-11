# Copy the mesh files to the built test directory
file(COPY ${PROJECT_SOURCE_DIR}/tests/api/api_mesh_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_um2_test(./um2_cpp_api.cpp)
add_um2_test(./um2_c_api.cpp)
