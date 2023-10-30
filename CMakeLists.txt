cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(
  UM2
  VERSION 0.1.0
  HOMEPAGE_URL "https://github.com/KyleVaughn/UM2"
  LANGUAGES CXX)

# Allow user to specify <project>_ROOT variables
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.27)
  cmake_policy(SET CMP0144 NEW)
else()
  cmake_policy(SET CMP0074 NEW)
endif()

# Set module path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For IDE users

#===============================================================================
# User options
#===============================================================================

option(UM2_USE_TBB          "Enable Intel TBB"                  ON)
option(UM2_USE_OPENMP       "Enable OpenMP"                     ON)
option(UM2_USE_GMSH         "Enable Gmsh"                       ON)
option(UM2_USE_PNG          "Enable PNG"                        OFF)
option(UM2_USE_CUDA         "Enable CUDA"                       OFF)

option(UM2_ENABLE_INT64     "Enable 64-bit integers"            OFF) # For large meshes
option(UM2_ENABLE_FLOAT64   "Enable 64-bit floating point"      ON)
option(UM2_ENABLE_FASTMATH  "Enable fast math"                  ON) # CPU and GPU

option(UM2_BUILD_TESTS      "Build tests"                       ON)
option(UM2_BUILD_TUTORIAL   "Build tutorial"                    OFF)
option(UM2_BUILD_EXAMPLES   "Build examples"                    OFF)
option(UM2_BUILD_BENCHMARKS "Build benchmarks"                  OFF)

#===============================================================================
# Developer options
#===============================================================================

option(UM2_DEV_MODE          "Enable development mode"         OFF)
option(UM2_USE_CLANG_FORMAT  "Enable clang-format targets"     ${UM2_DEV_MODE})
option(UM2_USE_CLANG_TIDY    "Enable clang-tidy"               ${UM2_DEV_MODE})
option(UM2_USE_CPPCHECK      "Enable cppcheck"                 ${UM2_DEV_MODE})
option(UM2_USE_COVERAGE      "Enable coverage"                 OFF)

#===============================================================================
# Set a default build configuration if not explicitly specified
#===============================================================================

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to RelWithDebInfo")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build" FORCE)
endif()

#===============================================================================
# Set minimum logging level
#===============================================================================

# minimum log level for compile-time filtering of log messages
# Off = 0,        // no messages
# Error = 1,      // only errors
# Warn = 2,       // errors and warnings
# Info = 3,       // errors, warnings and info
# Debug = 4,      // errors, warnings, info and debug
# Trace = 5,      // errors, warnings, info, debug and trace
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(UM2_MIN_LOG_LEVEL 5)
else()
  set(UM2_MIN_LOG_LEVEL 3)
endif()

#===============================================================================
# Check compiler version
#===============================================================================

set(UM2_MIN_GCC_VERSION 12.0)
set(UM2_MIN_CLANG_VERSION 15.0)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${UM2_MIN_GCC_VERSION})
    message(FATAL_ERROR "GCC version must be at least ${UM2_MIN_GCC_VERSION}")
  endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${UM2_MIN_CLANG_VERSION})
    message(FATAL_ERROR "Clang version must be at least ${UM2_MIN_CLANG_VERSION}")
  endif()
else()
  message(WARNING
          "You are using an unsupported compiler! "
          "Please use GCC >= 12.0 or Clang >= 15.0")
endif()

#===============================================================================
# RPATH information
#===============================================================================

# Provide install directory variables as defined by GNU coding standards
include(GNUInstallDirs)

# This block of code ensures that dynamic libraries can be found via the RPATH
# whether the executable is the original one from the build directory or the
# installed one in CMAKE_INSTALL_PREFIX. Ref:
# https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_FULL_LIBDIR}" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")
endif()

#===============================================================================
# Create libum2 target
#===============================================================================

# Sources
set(UM2_SOURCES
    # common
    "src/common/Log.cpp"
    # mpact
    "src/mpact/SpatialPartition.cpp"
    #    "src/mpact/io.cpp"
    #    # gmsh
    #    "src/gmsh/base_gmsh_api.cpp"
    #    "src/gmsh/io.cpp"
    #    "src/gmsh/model.cpp"
    #    "src/gmsh/mesh.cpp"
    #    # visualization
    #    "src/visualization/Image2D.cpp"
    #    # api
    #    "src/um2.cpp"
    #    "src/um2c.cpp"
  )

add_library(um2 SHARED ${UM2_SOURCES})
add_library(UM2::um2 ALIAS um2)

target_include_directories(um2
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Set C++ standard
set(UM2_CXX_STANDARD 20)
set_target_properties(um2 PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD})

# config.hpp
configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/config.hpp.in"
        "${PROJECT_SOURCE_DIR}/include/um2/config.hpp")

#===============================================================================
# Mandatory dependencies
#===============================================================================

# HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)
target_link_libraries(um2 PUBLIC "${HDF5_CXX_LIBRARIES}")
target_include_directories(um2 SYSTEM PUBLIC "${HDF5_INCLUDE_DIRS}")

# pugixml
find_package(PugiXML REQUIRED)
target_link_libraries(um2 PUBLIC "${PUGIXML_LIB}")
target_include_directories(um2 SYSTEM PUBLIC "${PUGIXML_INC}")

#===============================================================================
# Optional dependencies
#===============================================================================

# OpenMP
if (UM2_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(um2 PUBLIC OpenMP::OpenMP_CXX)
  target_include_directories(um2 SYSTEM PUBLIC "${OpenMP_CXX_INCLUDE_DIRS}")
endif()

# TBB
if (UM2_USE_TBB)
  find_package(TBB REQUIRED)
  # Tell the user where TBB was found
  message(STATUS "Found TBB: ${TBB_DIR}")
  target_link_libraries(um2 PUBLIC TBB::tbb)
  #  target_include_directories(um2 SYSTEM PUBLIC "${TBB_INC}")
endif()

# Gmsh
if (UM2_USE_GMSH)
  find_package(Gmsh REQUIRED)
  target_link_libraries(um2 PUBLIC "${GMSH_LIB}")
  target_include_directories(um2 SYSTEM PUBLIC "${GMSH_INC}")
endif()

# libpng
if (UM2_USE_LIBPNG)
  find_package(PNG REQUIRED)
  target_link_libraries(um2 PUBLIC "${PNG_LIBRARIES}")
  target_include_directories(um2 SYSTEM PUBLIC "${PNG_INCLUDE_DIRS}")
endif()

# CUDA
if (UM2_USE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  # nvcc will default to gcc and g++ if the host compiler is not set using CUDAHOSTCXX.
  # To prevent unintentional version/compiler mismatches, we set the host compiler to the
  # same compiler used to build the project.
  if (NOT DEFINED ENV{CUDAHOSTCXX})
    message(STATUS "Setting CMAKE_CUDA_HOST_COMPILER to ${CMAKE_CXX_COMPILER}."
            " Consider setting the CUDAHOSTCXX environment variable if this is not desired.")
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()

  find_package(CUDA REQUIRED)
  enable_language(CUDA)

  # Set CUDA standard to the same as the C++ standard
  set(UM2_CUDA_STANDARD ${UM2_CXX_STANDARD})

  # Macro for treating the given target as CUDA code
  macro(set_cuda_properties CUDA_TARGET)
    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_STANDARD ${UM2_CUDA_STANDARD})
    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_STANDARD_REQUIRED ON)
    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_ARCHITECTURES native)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
  endmacro()

  set_cuda_properties(um2 ${UM2_SOURCES})

  target_include_directories(um2 SYSTEM PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

#===============================================================================
# Flags
#===============================================================================

# Set fast math flags if enabled
if (UM2_ENABLE_FASTMATH)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
  if (UM2_USE_CUDA)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
  endif()
endif()

# Set release flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Set debug flags
if (!UM2_USE_CUDA)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
endif()

# Set compiler-specific warning flags
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  include(cmake/clang-cxx-flags.cmake)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UM2_CLANG_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  include(cmake/gnu-cxx-flags.cmake)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UM2_GNU_FLAGS}")
endif()

# If CUDA is enabled, pass the CXX flags via -Xcompiler
if (UM2_USE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CMAKE_CXX_FLAGS}\"")
  set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler \"${CMAKE_CXX_FLAGS_RELEASE}\"")
  set(CMAKE_CUDA_FLAGS_DEBUG
    "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler \"${CMAKE_CXX_FLAGS_DEBUG}\"")
  # If OpenMP is enabled, -fopenmp is passed via -Xcompiler
  if (UM2_USE_OPENMP)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
  endif()
endif()

#===============================================================================
# Developer options
#===============================================================================

# clang-format
if (UM2_USE_CLANG_FORMAT)
  include(cmake/clang-format.cmake)
endif()

# clang-tidy
if (UM2_USE_CLANG_TIDY)
  # Macro for running clang-tidy on the given target
  macro(set_clang_tidy_properties TIDY_TARGET)
    set_target_properties(${TIDY_TARGET} PROPERTIES
            CXX_CLANG_TIDY
            "clang-tidy;--extra-arg=-Wno-unknown-warning-option")
  endmacro()
  set_clang_tidy_properties(um2)
endif()

# cppcheck
if (UM2_USE_CPPCHECK)
  find_program(CMAKE_CXX_CPPCHECK NAMES cppcheck)
  # Copy the cppcheck-suppressions file to the build directory
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cppcheck-suppressions
                 ${CMAKE_CURRENT_BINARY_DIR}/cppcheck-suppressions COPYONLY)
  # Search and replace UM2_DIR with the absolute path to the project root
  file(READ ${CMAKE_CURRENT_BINARY_DIR}/cppcheck-suppressions CPPCHECK_SUPPRESS)
  string(REPLACE "UM2_DIR" "${CMAKE_CURRENT_SOURCE_DIR}" CPPCHECK_SUPPRESS
         "${CPPCHECK_SUPPRESS}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cppcheck-suppressions "${CPPCHECK_SUPPRESS}")
  # Concatenate the cppcheck arguments into a single string
  string(CONCAT CPPCHECK_ARGS
          ${CMAKE_CXX_CPPCHECK}
          ";--enable=all"
          ";--std=c++20"
          ";--language=c++"
          ";--suppressions-list=${CMAKE_CURRENT_BINARY_DIR}/cppcheck-suppressions"
          ";--inconclusive"
          ";--inline-suppr"
          ";--error-exitcode=10")
  macro(set_cppcheck_properties CPPCHECK_TARGET)
    set_target_properties(${CPPCHECK_TARGET} PROPERTIES CXX_CPPCHECK "${CPPCHECK_ARGS}")
  endmacro()
  set_cppcheck_properties(um2)
endif()

# coverage
if (UM2_USE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  target_link_libraries(um2 PRIVATE gcov)
endif()

#===============================================================================
# Optional subdirectories
#===============================================================================

# Tests
if (UM2_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

# Tutorial
if (UM2_BUILD_TUTORIAL)
  add_subdirectory(tutorial)
endif()

# Examples
if (UM2_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Benchmarks
if (UM2_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

#===============================================================================
# Install
#===============================================================================

# Install the library
install(TARGETS um2
        EXPORT um2-targets
        RUNTIME DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin
        LIBRARY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib
        ARCHIVE DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib)
install(DIRECTORY include/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include)
