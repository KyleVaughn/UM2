include(${PROJECT_SOURCE_DIR}/cmake/add_um2_test.cmake)

# Test the framework itself
add_um2_test(test_framework test_framework.cpp)
add_um2_test(bit_ceil ./common/bit_ceil.cpp)
add_um2_test(sto    ./common/sto.cpp) 
add_um2_test(vector ./common/vector.cpp)
add_um2_test(string ./common/string.cpp)
add_um2_test(color ./common/color.cpp)
add_um2_test(morton ./math/morton.cpp)
add_um2_test(vec2 ./math/vec2.cpp)
add_um2_test(mat2x2 ./math/mat2x2.cpp)
