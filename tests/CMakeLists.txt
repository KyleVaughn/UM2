include(${PROJECT_SOURCE_DIR}/cmake/add_um2_test.cmake)

# Test the framework itself
add_um2_test(test_framework test_framework.cpp)
add_um2_test(test_sto    ./common/sto.cpp) 
add_um2_test(test_vector ./common/vector.cpp)
add_um2_test(test_string ./common/string.cpp)
add_um2_test(test_color ./common/color.cpp)
