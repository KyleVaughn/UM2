set(TEST_FRAMEWORK_HEADER "${PROJECT_SOURCE_DIR}/tests/test_framework.hpp")    
    
macro(add_um2_test TESTNAME)    
  # Add the test framework header to the test    
  add_executable(${TESTNAME} ${ARGN} ${TEST_FRAMEWORK_HEADER})    
  target_link_libraries(${TESTNAME} um2 Thrust)    
  add_test(${TESTNAME} ${TESTNAME})    
  set_target_properties(${TESTNAME} PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD}) 
  set_target_properties(${TESTNAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
  # clang-tidy/cppcheck    
  if (UM2_ENABLE_CLANG_TIDY)    
    if (UM2_CLANG_TIDY_FIX)    
      set_target_properties(${TESTNAME} PROPERTIES    
                            CXX_CLANG_TIDY    
                            "clang-tidy;--fix")    
    else()    
      set_target_properties(${TESTNAME} PROPERTIES    
                            CXX_CLANG_TIDY    
                            "clang-tidy")    
    endif()    
  endif()    
  if (UM2_ENABLE_CPPCHECK)    
    set_target_properties(${TESTNAME} PROPERTIES CXX_CPPCHECK "${CPPCHECK_ARGS}")       
  endif()                                        
  # If compiling with CUDA, compile the cpp files as cuda    
  if (UM2_ENABLE_CUDA)    
    set_target_properties(${TESTNAME} PROPERTIES CUDA_STANDARD ${UM2_CUDA_STANDARD})
    set_target_properties(${TESTNAME} PROPERTIES CUDA_STANDARD_REQUIRED ON)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)    
    set_property(TARGET ${TESTNAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)    
    set_property(TARGET ${TESTNAME} PROPERTY CUDA_ARCHITECTURES native)        
  endif()                                                                      
  if (UM2_ENABLE_VIS)    
    target_link_libraries(${TESTNAME} ${UM2_VIS_LIBRARIES})
  endif()
endmacro()
