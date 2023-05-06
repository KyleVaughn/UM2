macro(add_um2_test TESTNAME)
  # Add test_framework.h to the list of sources
  add_executable(${TESTNAME} ${ARGN} ${PROJECT_SOURCE_DIR}/tests/test_framework.hpp)
  target_link_libraries(${TESTNAME} um2)
  add_test(${TESTNAME} ${TESTNAME})
endmacro()

# Test the framework itself
add_um2_test(test_framework test_framework.cpp)
