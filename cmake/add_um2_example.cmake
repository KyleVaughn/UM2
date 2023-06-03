macro(add_um2_example EXAMPLE)    
  add_executable(${EXAMPLE} ${ARGN})    
  target_link_libraries(${EXAMPLE} um2)    
  set_target_properties(${EXAMPLE} PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD}) 
  set_target_properties(${EXAMPLE} PROPERTIES CXX_STANDARD_REQUIRED ON)
  # clang-tidy/cppcheck    
  if (UM2_ENABLE_CLANG_TIDY)    
    if (UM2_CLANG_TIDY_FIX)    
      set_target_properties(${EXAMPLE} PROPERTIES    
                            CXX_CLANG_TIDY    
                            "clang-tidy;--fix;--extra-arg=-Wno-unknown-warning-option")
    else()    
      set_target_properties(${EXAMPLE} PROPERTIES    
                            CXX_CLANG_TIDY    
                            "clang-tidy;--extra-arg=-Wno-unknown-warning-option")
    endif()    
  endif()    
  if (UM2_ENABLE_CPPCHECK)    
    set_target_properties(${EXAMPLE} PROPERTIES CXX_CPPCHECK "${CPPCHECK_ARGS}")       
  endif()                                        
  # If compiling with CUDA, compile the cpp files as cuda    
  if (UM2_ENABLE_CUDA)    
    set_target_properties(${EXAMPLE} PROPERTIES CUDA_STANDARD ${UM2_CUDA_STANDARD})
    set_target_properties(${EXAMPLE} PROPERTIES CUDA_STANDARD_REQUIRED ON)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)    
    set_property(TARGET ${EXAMPLE} PROPERTY CUDA_SEPARABLE_COMPILATION ON)    
    set_property(TARGET ${EXAMPLE} PROPERTY CUDA_ARCHITECTURES native)        
  endif()                                                                      
  if (UM2_ENABLE_VIS)    
    target_link_libraries(${EXAMPLE} ${UM2_VIS_LIBRARIES})
  endif()
endmacro()
