macro(um2_add_benchmark FILENAME)
  # If this is not a release build, warn the user
  if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "Benchmarking is only recommended in release mode. " 
                    "You are building in ${CMAKE_BUILD_TYPE} mode.")
  endif()

  # Strip the path and extension from the filename to get the benchmark name
  set(BENCHNAME ${FILENAME})
  get_filename_component(BENCHNAME ${BENCHNAME} NAME_WE)
  get_filename_component(BENCHNAME ${BENCHNAME} NAME_WLE)

  # Prepend "bench_" to the test name
  set(BENCHNAME "bench_${BENCHNAME}")

  add_executable(${BENCHNAME} ${FILENAME})
  set_target_properties(${BENCHNAME} PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD})
  target_link_libraries(${BENCHNAME} 
    um2 
    PFM::libpfm
    benchmark::benchmark 
    benchmark::benchmark_main
    )

  # clang-tidy
  if (UM2_USE_CLANG_TIDY)
    set_clang_tidy_properties(${BENCHNAME})
  endif()

  if (UM2_USE_CUDA)
    set_cuda_properties(${BENCHNAME} ${FILENAME})
  endif()
endmacro()
