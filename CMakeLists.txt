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

# Configure using options
include(cmake/um2_config.cmake)

# Header and source files
set(UM2_HEADERS
    "include/um2/common/config.hpp"
    "include/um2c.h"
    )
set(UM2_SOURCES
    "src/um2c.cpp"
    )

# If compiling with CUDA, compile the cpp files with nvcc
if (UM2_ENABLE_CUDA)
  set_source_files_properties(${UM2_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

# libum2
add_library(um2 ${UM2_HEADERS} ${UM2_SOURCES})
target_include_directories(um2 PUBLIC include)
# Treat the Thrust includes as system includes
target_include_directories(um2 SYSTEM PUBLIC
  "${PROJECT_SOURCE_DIR}/tpls/thrust/thrust/cmake/../.."
  "${PROJECT_SOURCE_DIR}/tpls/thrust/dependencies/libcudacxx/include"
  "${PROJECT_SOURCE_DIR}/tpls/thrust/dependencies/cub"
  )
target_link_libraries(um2 PRIVATE Thrust spdlog::spdlog)
if (UM2_ENABLE_CLANG_TIDY)
  set_target_properties(um2 PROPERTIES
                        CXX_CLANG_TIDY
                        "clang-tidy;--header-filter=include")
endif()

# Tests
if (UM2_ENABLE_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

## Install
#include(GNUInstallDirs)
#install(
#  TARGETS UM2
#  EXPORT UM2Targets
#  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#)
#
#install(
#  DIRECTORY include/
#  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#)
#
#include(CMakePackageConfigHelpers)
#write_basic_package_version_file(
#  "${CMAKE_CURRENT_BINARY_DIR}/UM2/UM2ConfigVersion.cmake"
#  VERSION ${PROJECT_VERSION}
#  COMPATIBILITY SameMajorVersion
#)
#
#export(
#  EXPORT UM2Targets
#  FILE "${CMAKE_CURRENT_BINARY_DIR}/UM2/UM2Targets.cmake"
#  NAMESPACE UM2::
#)
#
#configure_file(cmake/UM2Config.cmake
#  "${CMAKE_CURRENT_BINARY_DIR}/UM2/UM2Config.cmake"
#  COPYONLY
#)
#
#set(ConfigPackageLocation lib/cmake/UM2)
#install(
#  EXPORT UM2Targets
#  FILE UM2Targets.cmake
#  NAMESPACE UM2::
#  DESTINATION ${ConfigPackageLocation}
#)
#
#install(
#  FILES
#    "${CMAKE_CURRENT_BINARY_DIR}/UM2/UM2Config.cmake"
#    "${CMAKE_CURRENT_BINARY_DIR}/UM2/UM2ConfigVersion.cmake"
#  DESTINATION ${ConfigPackageLocation}
#)
#
## Export from the build tree
#export(
#  PACKAGE UM2
#)
#
## Doxygen
#include(cmake/doxygen.cmake)
#
## CTest
#include(CTest)
#
## Code coverage
#include(cmake/coverage.cmake)
#
## Sanitizers
#include(cmake/sanitizers.cmake)
#
## Clang tidy
#include(cmake/clang-tidy.cmake)
#
## Clang format
#include(cmake/clang-format.cmake)
#
## Cppcheck
#include(cmake/cppcheck.cmake)
#
## Cppcheck
#include(cmake/clang-analyzer.cmake)
#
## Cppcheck
#include(cmake/clang-static-analyzer.cmake)
#
## C






## CMake compile options
#if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#  include(cmake/cmake-compile-options.cmake)
#endif()
#
## CMake compile definitions
#if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#  include(cmake/cmake-compile-definitions.cmake)
#endif()
#
## CMake compile features
#if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#  include(cmake/cmake-compile-features.cmake)
#endif()
#
## CMake compile options
#if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#  include(cmake/cmake-compile-options.cmake)
#endif()
#
## CMake compile definitions
#if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#  include(cmake/cmake-compile-definitions.cmake)
#endif()


# CUDA is currently only supported via Clang
#if(UM2_ENABLE_CUDA)
#  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#    message(
#        FATAL_ERROR
#        "CUDA supported requires Clang as the C++ compiler."
#    )
#  endif()
#  #  include(Compiler/
#  #
#  #  set(CMAKE_CUDA_COMPILER ${CMAKE_CXX_COMPILER})
#  #  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
#  enable_language(CUDA)
#  set_source_files_properties(test.cpp PROPERTIES LANGUAGE CUDA)
#endif()

#add_library(
#    um2
#    source/UM2.cpp
#)

#if(NOT BUILD_SHARED_LIBS)
#  target_compile_definitions(UM2_UM2 PUBLIC UM2_STATIC_DEFINE)
#endif()
#
#set_target_properties(
#    UM2_UM2 PROPERTIES
#    CXX_VISIBILITY_PRESET hidden
#    VISIBILITY_INLINES_HIDDEN YES
#    VERSION "${PROJECT_VERSION}"
#    SOVERSION "${PROJECT_VERSION_MAJOR}"
#    EXPORT_NAME UM2
#    OUTPUT_NAME UM2
#)
#
#target_include_directories(
#    UM2_UM2 ${warning_guard}
#    PUBLIC
#    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
#)
#
#target_include_directories(
#    UM2_UM2 SYSTEM
#    PUBLIC
#    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
#)
##
##target_compile_features(UM2_UM2 PUBLIC cxx_std_23)
##
### ---- Install rules ----
##
##if(NOT CMAKE_SKIP_INSTALL_RULES)
##  include(cmake/install-rules.cmake)
##endif()
##
### ---- Examples ----
##
##if(PROJECT_IS_TOP_LEVEL)
##  option(BUILD_EXAMPLES "Build examples tree." "${UM2_DEVELOPER_MODE}")
##  if(BUILD_EXAMPLES)
##    add_subdirectory(example)
##  endif()
#endif()
#
#
#include(cmake/dev-mode.cmake)
