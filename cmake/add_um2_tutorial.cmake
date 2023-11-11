macro(add_um2_tutorial FILENAME)
  # Strip the path and extension from the filename to get the tutorial name
  set(TUTORIAL ${FILENAME})
  get_filename_component(TUTORIAL ${TUTORIAL} NAME_WE)
  get_filename_component(TUTORIAL ${TUTORIAL} NAME_WLE)
  # Prepend "ex_" to the test name
  set(TUTORIAL "tutorial_${TUTORIAL}")

  add_executable(${TUTORIAL} ${FILENAME})

  target_link_libraries(${TUTORIAL} um2)

  set_target_properties(${TUTORIAL} PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD})

  # clang-tidy
  if (UM2_USE_CLANG_TIDY)
    set_clang_tidy_properties(${TUTORIAL})
  endif()

  # cppcheck
  if (UM2_USE_CPPCHECK)
    set_cppcheck_properties(${TUTORIAL})
  endif()

  # If compiling with CUDA, compile the cpp files as cuda
  if (UM2_USE_CUDA)
    set_cuda_properties(${TUTORIAL} ${ARGN})
  endif()

endmacro()
