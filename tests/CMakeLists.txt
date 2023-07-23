include(${PROJECT_SOURCE_DIR}/cmake/add_um2_test.cmake)

# ------------------------------------------------------------------------------
# common 
# ------------------------------------------------------------------------------
add_um2_test(./common/memory/addressof.cpp)
add_um2_test(./common/memory/construct_at.cpp)
add_um2_test(./common/Vector.cpp)
add_um2_test(./common/sto.cpp) 
add_um2_test(./common/String.cpp)
add_um2_test(./common/ShortString.cpp)
add_um2_test(./common/Color.cpp)

# ------------------------------------------------------------------------------
# math
# ------------------------------------------------------------------------------
add_um2_test(./math/morton.cpp)
add_um2_test(./math/Vec.cpp)
add_um2_test(./math/Mat.cpp)

# ------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------
add_um2_test(./geometry/Point.cpp)
add_um2_test(./geometry/morton_sort_points.cpp)
add_um2_test(./geometry/AxisAlignedBox.cpp)
add_um2_test(./geometry/LineSegment.cpp)
add_um2_test(./geometry/Triangle.cpp)
add_um2_test(./geometry/Quadrilateral.cpp)
add_um2_test(./geometry/QuadraticSegment.cpp)
add_um2_test(./geometry/QuadraticTriangle.cpp)
add_um2_test(./geometry/QuadraticQuadrilateral.cpp)

# ------------------------------------------------------------------------------
# mesh
# ------------------------------------------------------------------------------
add_um2_test(./mesh/RegularGrid.cpp)
add_um2_test(./mesh/RegularPartition.cpp)
add_um2_test(./mesh/TriMesh.cpp)
#
## ------------------------------------------------------------------------------
## ray_casting
## ------------------------------------------------------------------------------
#add_um2_test(./ray_casting/intersect/Ray-LineSegment.cpp)

# ------------------------------------------------------------------------------
# visualization 
# ------------------------------------------------------------------------------
add_um2_test(./visualization/Image2D.cpp)

#
#if (UM2_ENABLE_VIS)
#  add_um2_test(shader ./visualization/shader.cpp)
#endif()
