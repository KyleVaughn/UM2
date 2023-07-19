include(${PROJECT_SOURCE_DIR}/cmake/add_um2_benchmark.cmake)

# ------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------
add_um2_benchmark(./geometry/AxisAlignedBox.cpp)
add_um2_benchmark(./geometry/Triangle.cpp)
add_um2_benchmark(./geometry/Quadrilateral.cpp)
