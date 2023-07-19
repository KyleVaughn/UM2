macro(add_um2_example EXAMPLE)    
  add_executable(${EXAMPLE} ${ARGN})    
  
  target_link_libraries(${EXAMPLE} um2)    
  
  set_target_properties(${EXAMPLE} PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD}) 
  set_target_properties(${EXAMPLE} PROPERTIES CXX_STANDARD_REQUIRED ON)
  
  # clang-tidy
  if (UM2_ENABLE_CLANG_TIDY)    
    set_clang_tidy_properties(${EXAMPLE})
  endif()    

  # cppcheck
  if (UM2_ENABLE_CPPCHECK)    
    set_target_properties(${EXAMPLE} PROPERTIES CXX_CPPCHECK "${CPPCHECK_ARGS}")       
  endif()                                        

  # If compiling with CUDA, compile the cpp files as cuda    
  if (UM2_ENABLE_CUDA)    
    set_cuda_properties(${EXAMPLE} ${ARGN})
  endif()                                                                      

  # If vis is enabled, link the necessary libraries
  if (UM2_ENABLE_VIS)    
    target_link_libraries(${EXAMPLE} ${UM2_VIS_LIBRARIES})
  endif()
endmacro()
