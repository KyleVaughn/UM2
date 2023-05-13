set(TEST_FRAMEWORK_HEADER ${PROJECT_SOURCE_DIR}/tests/test_framework.hpp)

macro(add_um2_test TESTNAME)
  # Add test_framework.h to the list of sources
  add_executable(${TESTNAME} ${ARGN} ${TEST_FRAMEWORK_HEADER}) 
  target_link_libraries(${TESTNAME} um2)
  add_test(${TESTNAME} ${TESTNAME})
  if (UM2_ENABLE_CLANG_TIDY)
    set_target_properties(${TESTNAME} PROPERTIES
                          CXX_CLANG_TIDY
                          "clang-tidy;--header-filter=tests/*")
  endif()
  # If compiling with CUDA, compile the cpp files as cuda
  if (UM2_ENABLE_CUDA)
    message(STATUS "Compiling ${TESTNAME} with CUDA")
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
    set_property(TARGET ${TESTNAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${TESTNAME} PROPERTY CUDA_ARCHITECTURES native)
  endif()
endmacro()

# Test the framework itself
add_um2_test(test_framework test_framework.cpp)
