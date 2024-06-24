# Copy the mesh files to the built test directory
file(COPY ${PROJECT_SOURCE_DIR}/tests/mpact/mpact_mesh_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

um2_add_test(./mpact_model.cpp)
