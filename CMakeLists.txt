cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(
  UM2
  VERSION 0.1.0
  DESCRIPTION "An unstructured mesh library for automated method of characteristic mesh generation"
  HOMEPAGE_URL "https://github.com/KyleVaughn/UM2"
  LANGUAGES CXX
)


#if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#
#  set(CMAKE_CXX_STANDARD 23)
#  set(CMAKE_CXX_STANDARD_REQUIRED ON)
#
#  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#
#  #    include(CTest)
#
#endif()

option(UM2_ENABLE_OPENMP "Enable OpenMP" OFF)
option(UM2_ENABLE_CUDA "Enable CUDA" OFF)
option(UM2_ENABLE_TESTS "Enable tests" ON)

include(cmake/update-git-submodules.cmake)
# OpenMP
#if (UM2_ENABLE_OPENMP)
#  find_package(OpenMP REQUIRED)
#endif()

# CUDA
if (UM2_ENABLE_CUDA)
  find_package(CUDA REQUIRED)
endif()









# Thrust
include(cmake/thrust.cmake)

add_subdirectory(src)

# Tests
if (UM2_ENABLE_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory("${PROJECT_SOURCE_DIR}/tpls/googletest")
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
## Packaging
#include(cmake/packaging.cmake)
#
## CPack
#include(CPack)
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
