macro(add_um2_example FILENAME)    
  # Strip the path and extension from the filename to get the example name
  set(EXAMPLE ${FILENAME})
  get_filename_component(EXAMPLE ${EXAMPLE} NAME_WE)
  get_filename_component(EXAMPLE ${EXAMPLE} NAME_WLE)
  # Prepend "ex_" to the test name
  set(EXAMPLE "ex_${EXAMPLE}")

  add_executable(${EXAMPLE} ${FILENAME})
  
  target_link_libraries(${EXAMPLE} um2)    
  
  set_target_properties(${EXAMPLE} PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD}) 
  
  # clang-tidy
  if (UM2_USE_CLANG_TIDY)    
    set_clang_tidy_properties(${EXAMPLE})
  endif()    

  # cppcheck
  if (UM2_USE_CPPCHECK)    
    set_cppcheck_properties(${EXAMPLE})
  endif()                                        

  # If compiling with CUDA, compile the cpp files as cuda    
  if (UM2_USE_CUDA)    
    set_cuda_properties(${EXAMPLE} ${ARGN})
  endif()                                                                      

  ## If vis is enabled, link the necessary libraries
  #if (UM2_USE_VIS)    
  #  target_link_libraries(${EXAMPLE} ${UM2_VIS_LIBRARIES})
  #endif()
endmacro()
