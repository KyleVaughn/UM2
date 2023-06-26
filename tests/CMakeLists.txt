include(${PROJECT_SOURCE_DIR}/cmake/add_um2_test.cmake)

# ------------------------------------------------------------------------------
# common 
# ------------------------------------------------------------------------------
add_um2_test(./common/memory/addressof.cpp)
add_um2_test(./common/memory/construct_at.cpp)
add_um2_test(./common/Vector.cpp)
add_um2_test(./common/sto.cpp) 
#add_um2_test(string ./common/string.cpp)
#add_um2_test(color  ./common/color.cpp)

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
add_um2_test(./geometry/AxisAlignedBox.cpp)
add_um2_test(./geometry/LineSegment.cpp)
add_um2_test(./geometry/Triangle.cpp)
add_um2_test(./geometry/Quadrilateral.cpp)
#
#if (UM2_ENABLE_VIS)
#  add_um2_test(shader ./visualization/shader.cpp)
#endif()
