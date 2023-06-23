include(${PROJECT_SOURCE_DIR}/cmake/add_um2_test.cmake)

# ------------------------------------------------------------------------------
# common 
# ------------------------------------------------------------------------------
# memory
add_um2_test(addressof ./common/memory/addressof.cpp)
add_um2_test(construct_at ./common/memory/construct_at.cpp)
add_um2_test(Vector ./common/Vector.cpp)
#add_um2_test(sto    ./common/sto.cpp) 
#add_um2_test(string ./common/string.cpp)
#add_um2_test(color  ./common/color.cpp)
#
## math
#add_um2_test(morton ./math/morton.cpp)
#add_um2_test(vec    ./math/vec.cpp)
#add_um2_test(mat    ./math/mat.cpp)
#
## geometry
#add_um2_test(point            ./geometry/point.cpp)
#add_um2_test(axis_aligned_box ./geometry/axis_aligned_box.cpp)
#add_um2_test(line_segment     ./geometry/line_segment.cpp)
#add_um2_test(triangle         ./geometry/triangle.cpp)
#
#if (UM2_ENABLE_VIS)
#  add_um2_test(shader ./visualization/shader.cpp)
#endif()
