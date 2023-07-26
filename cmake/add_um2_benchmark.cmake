macro(add_um2_benchmark FILENAME)
  # If this is not a release build, warn the user
  if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "Benchmarking is only recommended in release mode. You are building in ${CMAKE_BUILD_TYPE} mode.")
  endif()

  # Strip the path and extension from the filename to get the test name
  set(BENCHNAME ${FILENAME})
  get_filename_component(BENCHNAME ${BENCHNAME} NAME_WE)
  get_filename_component(BENCHNAME ${BENCHNAME} NAME_WLE)
  # Prepend "benchmark_" to the test name    
  set(BENCHNAME "benchmark_${BENCHNAME}")    

  add_executable(${BENCHNAME} ${FILENAME})

  find_package(benchmark REQUIRED)

  target_link_libraries(${BENCHNAME} um2 benchmark::benchmark benchmark::benchmark_main)
  set_target_properties(${BENCHNAME} PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD})
  set_target_properties(${BENCHNAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

  # clang-tidy
  if (UM2_ENABLE_CLANG_TIDY)
    set_clang_tidy_properties(${BENCHNAME})
  endif()

  # cppcheck
  if (UM2_ENABLE_CPPCHECK)
    set_target_properties(${BENCHNAME} PROPERTIES CXX_CPPCHECK "${CPPCHECK_ARGS}")
  endif()

  # If compiling with CUDA, compile the cpp files as cuda
  if (UM2_ENABLE_CUDA)
    set_cuda_properties(${BENCHNAME} ${FILENAME})
  endif()
endmacro()
