cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(
  UM2
  VERSION 0.1.0
  DESCRIPTION "An unstructured mesh library for automated method of characteristic mesh generation"
  HOMEPAGE_URL "https://github.com/KyleVaughn/UM2"
  LANGUAGES CXX
)

# Update git submodules before doing anything else
include(cmake/update-git-submodules.cmake)

# User options
option(UM2_ENABLE_OPENMP  "Enable OpenMP"   ON)
option(UM2_ENABLE_CUDA    "Enable CUDA"     ON)
option(UM2_ENABLE_TESTS   "Enable tests"    ON)
set(UM2_LOG_LEVEL "info" CACHE STRING "Log level")
set_property(CACHE UM2_LOG_LEVEL PROPERTY STRINGS
  "trace" "debug" "info" "warn" "error" "critical" "off")

# Advanced user options
option(UM2_ENABLE_INT64 "Enable 64-bit integer types" OFF)

# Dev options
option(UM2_DEV_MODE "Enable development mode" ON)
option(UM2_ENABLE_CLANG_FORMAT "Enable clang-format targets" ${UM2_DEV_MODE})
option(UM2_ENABLE_CLANG_TIDY "Enable clang-tidy" ${UM2_DEV_MODE})
option(UM2_CLANG_TIDY_FIX "Enable clang-tidy to fix errors" OFF)
option(UM2_ENABLE_CPPCHECK "Enable cppcheck" ${UM2_DEV_MODE})

# Configure using the options above 
include(cmake/um2_config.cmake)

# Header and source files
set(UM2_HEADERS
    "include/um2/common/config.hpp"
    "include/um2c.h"
    )
set(UM2_SOURCES
    "src/um2c.cpp"
    )

# If CUDA is enabled, use CUDA for the source files
if (UM2_ENABLE_CUDA)
  set_source_files_properties(${UM2_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

# libum2
add_library(um2 ${UM2_HEADERS} ${UM2_SOURCES})
# include directories 
# -- Include the public headers
target_include_directories(um2 PUBLIC include)
# -- Treat the Thrust includes as system includes
target_include_directories(um2 SYSTEM PUBLIC
  "${PROJECT_SOURCE_DIR}/tpls/thrust/thrust/cmake/../.."
  "${PROJECT_SOURCE_DIR}/tpls/thrust/dependencies/libcudacxx/include"
  "${PROJECT_SOURCE_DIR}/tpls/thrust/dependencies/cub"
  )
# -- If compiling with CUDA, include the CUDA headers
if (UM2_ENABLE_CUDA)
  target_include_directories(um2 SYSTEM PUBLIC
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    )
  set_target_properties(um2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(um2 PROPERTIES CUDA_ARCHITECTURES native)
endif()

# link libraries
target_link_libraries(um2 PRIVATE Thrust spdlog::spdlog)

# clang-tidy/cppcheck
if (UM2_ENABLE_CLANG_TIDY)
  if (UM2_CLANG_TIDY_FIX)
    set_target_properties(um2 PROPERTIES
                          CXX_CLANG_TIDY
                          "clang-tidy;--header-filter=include;--fix")
  else()
    set_target_properties(um2 PROPERTIES
                          CXX_CLANG_TIDY
                          "clang-tidy;--header-filter=include")
  endif()
endif()
if (UM2_ENABLE_CPPCHECK)
  # Concatenate the cppcheck arguments into a single string
  string(CONCAT CPPCHECK_ARGS
    "cppcheck"
    ";--enable=warning,style,information,missingInclude"
    ";--std=c++20"
    ";--language=c++"
    ";--suppress=missingIncludeSystem"
    ";--inconclusive"
    ";--inline-suppr"
    ";--error-exitcode=10"
  )
  set_target_properties(um2 PROPERTIES
                        CXX_CPPCHECK
                        "${CPPCHECK_ARGS}")
endif()

# Tests
if (UM2_ENABLE_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()
