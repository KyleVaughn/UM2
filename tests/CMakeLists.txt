set(TEST_FRAMEWORK_HEADER "${PROJECT_SOURCE_DIR}/tests/test_framework.hpp")

macro(add_um2_test TESTNAME)
  # Add the test framework header to the test
  add_executable(${TESTNAME} ${ARGN} ${TEST_FRAMEWORK_HEADER}) 
  target_link_libraries(${TESTNAME} um2)
  add_test(${TESTNAME} ${TESTNAME})
  # clang-tidy/cppcheck
  if (UM2_ENABLE_CLANG_TIDY)
    if (UM2_CLANG_TIDY_FIX)
      set_target_properties(${TESTNAME} PROPERTIES
                            CXX_CLANG_TIDY
                            "clang-tidy;--header-filter=tests/*;--fix")
    else()
      set_target_properties(${TESTNAME} PROPERTIES
                            CXX_CLANG_TIDY
                            "clang-tidy;--header-filter=tests/*")
    endif()
  endif()
  if (UM2_ENABLE_CPPCHECK)    
    set_target_properties(${TESTNAME} PROPERTIES    
                          CXX_CPPCHECK    
                          "${CPPCHECK_ARGS}")    
  endif()
  # If compiling with CUDA, compile the cpp files as cuda
  if (UM2_ENABLE_CUDA)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
    set_property(TARGET ${TESTNAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${TESTNAME} PROPERTY CUDA_ARCHITECTURES native)
  endif()
endmacro()

# Test the framework itself
add_um2_test(test_framework test_framework.cpp)
