macro(add_um2_executable FILENAME)
  # Strip the path and extension from the filename to get the model name
  set(EXEC ${FILENAME})
  get_filename_component(EXEC ${EXEC} NAME_WE)
  get_filename_component(EXEC ${EXEC} NAME_WLE)

  add_executable(${EXEC} ${FILENAME})

  target_link_libraries(${EXEC} um2)

  set_target_properties(${EXEC} PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD})

  # clang-tidy
  if (UM2_USE_CLANG_TIDY)
    set_clang_tidy_properties(${EXEC})
  endif()

endmacro()
