include(${PROJECT_SOURCE_DIR}/cmake/add_um2_benchmark.cmake)

add_subdirectory(./stdlib)
add_subdirectory(./geometry)
add_subdirectory(./mesh)
#
#add_um2_benchmark(./mesh/tri6_faceContaining.cpp)
#add_um2_benchmark(./mesh/MeshFile_getSubmesh.cpp)
#
##===============================================================================
## visualization
##===============================================================================
#
#add_um2_benchmark(./visualization/rasterize_Point.cpp)
#add_um2_benchmark(./visualization/rasterize_Line.cpp)
