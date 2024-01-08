file(COPY ${PROJECT_SOURCE_DIR}/benchmarks/mesh/mesh_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})    
add_um2_benchmark(./tri6mesh_intersect.cpp)
