macro(add_um2_model FILENAME)
  # Strip the path and extension from the filename to get the model name
  set(MODEL ${FILENAME})
  get_filename_component(MODEL ${MODEL} NAME_WE)
  get_filename_component(MODEL ${MODEL} NAME_WLE)

  add_executable(${MODEL} ${FILENAME})

  target_link_libraries(${MODEL} um2)

  set_target_properties(${MODEL} PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD})

  # clang-tidy
  if (UM2_USE_CLANG_TIDY)
    set_clang_tidy_properties(${MODEL})
  endif()

endmacro()
