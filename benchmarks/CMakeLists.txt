include(${PROJECT_SOURCE_DIR}/cmake/add_um2_benchmark.cmake)

#===============================================================================
# math 
#===============================================================================

add_um2_benchmark(./math/exp.cpp)
add_um2_benchmark(./math/sqrt.cpp)
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
