include(${PROJECT_SOURCE_DIR}/cmake/add_um2_benchmark.cmake)

#===============================================================================
# stdlib 
#===============================================================================

add_um2_benchmark(./stdlib/exp.cpp)
add_um2_benchmark(./stdlib/sqrt.cpp)
add_um2_benchmark(./stdlib/sort.cpp)

#===============================================================================
# math 
#===============================================================================

add_um2_benchmark(./math/Vec2.cpp)

#===============================================================================
# geometry
#===============================================================================

add_um2_benchmark(./geometry/morton_sort_points.cpp)
add_um2_benchmark(./geometry/AxisAlignedBox.cpp)
add_um2_benchmark(./geometry/Triangle.cpp)
add_um2_benchmark(./geometry/Quadrilateral.cpp)
add_um2_benchmark(./geometry/QuadraticSegment.cpp)

#===============================================================================
# mesh 
#===============================================================================

# Copy the mesh files
file(COPY ${PROJECT_SOURCE_DIR}/benchmarks/mesh/mesh_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})    

add_um2_benchmark(./mesh/tri6_faceContaining.cpp)
add_um2_benchmark(./mesh/MeshFile_getSubmesh.cpp)

#===============================================================================
# visualization
#===============================================================================

add_um2_benchmark(./visualization/rasterize_Point.cpp)
add_um2_benchmark(./visualization/rasterize_Line.cpp)
