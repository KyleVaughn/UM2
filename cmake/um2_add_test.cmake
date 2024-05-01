set(TEST_FRAMEWORK_HEADER "${PROJECT_SOURCE_DIR}/tests/test_macros.hpp")

macro(um2_add_test FILENAME)

  # Strip the path and extension from the filename to get the test name
  set(TESTNAME ${FILENAME})
  get_filename_component(TESTNAME ${TESTNAME} NAME_WE)
  get_filename_component(TESTNAME ${TESTNAME} NAME_WLE)

  # Prepend "test_" to the test name
  set(TESTNAME "test_${TESTNAME}")

  # Always include the test framework header with the test
  add_executable(${TESTNAME} ${FILENAME} ${TEST_FRAMEWORK_HEADER})
  target_link_libraries(${TESTNAME} PRIVATE um2)

  add_test(${TESTNAME} ${TESTNAME})
  set_target_properties(${TESTNAME} PROPERTIES CXX_STANDARD 20) 

  if (UM2_USE_CLANG_TIDY)    
    set_clang_tidy_properties(${TESTNAME})    
  endif()
endmacro()
