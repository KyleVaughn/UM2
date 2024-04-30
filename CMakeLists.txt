cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(
  UM2
  VERSION 0.1.0
  HOMEPAGE_URL "https://github.com/KyleVaughn/UM2"
  LANGUAGES CXX)

#===============================================================================
# Options
#===============================================================================

# Build
option(UM2_BUILD_BENCHMARKS   "Build benchmarks"                  OFF)
option(UM2_BUILD_MODELS       "Build models"                      OFF)
option(UM2_BUILD_SHARED_LIB   "Build UM2 as a shared library"     ON)
option(UM2_BUILD_TESTS        "Build tests"                       ON)
option(UM2_BUILD_TUTORIAL     "Build tutorial"                    OFF)

# Enable/disable features
option(UM2_ENABLE_ASSERTS   "Enable assertions"                              OFF)
option(UM2_ENABLE_BMI2      "Enable BMI2 instruction set"                    ON)  # Morton sorting, etc.
option(UM2_ENABLE_FASTMATH  "Enable fast math optimizations on CPU and GPU"  ON)  # e.g. -ffast-math
option(UM2_ENABLE_FLOAT64   "Enable 64-bit Float"                            ON)  # x,y,z = double
option(UM2_ENABLE_INT64     "Enable 64-bit Int"                              OFF) # index type
option(UM2_ENABLE_NATIVE    "Enable -march=native"                           ON)
option(UM2_ENABLE_SIMD_VEC  "Enable GCC vector extensions for the Vec class" ON)

# External tools/dependencies
option(UM2_USE_CLANG_FORMAT  "Use clang-format" OFF) # Code formatting
option(UM2_USE_CLANG_TIDY    "Use clang-tidy"   OFF) # Static analysis
option(UM2_USE_COVERAGE      "Use coverage"     OFF) # Code coverage
option(UM2_USE_CUDA          "Use CUDA"         OFF) # GPU acceleration
option(UM2_USE_GMSH          "Use Gmsh"         ON)  # CAD geometry and meshing
option(UM2_USE_HDF5          "Use HDF5"         ON)  # Binary data I/O (Mesh I/O)
option(UM2_USE_OPENMP        "Use OpenMP"       OFF) # Multi-threading
option(UM2_USE_PUGIXML       "Use pugixml"      ON)  # XML parsing (Mesh I/O)
option(UM2_USE_TBB           "Use TBB"          OFF) # Parallel STL (std::execution::par, etc.)
option(UM2_USE_VALGRIND      "Use valgrind"     OFF) # Memory checking

#===============================================================================
# Basic CMake configuration
#===============================================================================

# Disable in-source builds
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR
    "In-source builds are not allowed."
    " Create a separate directory for build files and delete CMakeCache.txt.")
endif()

# If no build type is specified, default to Release
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Set the C++ standard
if (PROJECT_IS_TOP_LEVEL)
  set(CMAKE_CXX_STANDARD 20 CACHE STRING "Default C++ standard")
  set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard")
  set(CMAKE_CXX_EXTENSIONS ON CACHE BOOL "Allow C++ extensions")
endif()

## Allow user to specify <project>_ROOT variables to find dependencies
#if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.27)
#  cmake_policy(SET CMP0144 NEW)
#else()
#  cmake_policy(SET CMP0074 NEW)
#endif()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# RPATH handling
# https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_FULL_LIBDIR}" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")
endif()

# Minimum version for GCC or Clang
# This is to ensure that the compiler supports the parts of C++20 that we use
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
          "Please use GCC >= ${UM2_MIN_GCC_VERSION} or Clang >= ${UM2_MIN_CLANG_VERSION}")
endif()

# Check for BMI2 support now that we know the compiler is supported
if (UM2_ENABLE_BMI2)
  # Check to see if the CPU is one of the AMD architectures that have issues with BMI2
  execute_process(COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_bmi2_support.sh"
                  "${CMAKE_CXX_COMPILER}"
                  OUTPUT_VARIABLE BMI2_WARNING)
  if (BMI2_WARNING)
    message(WARNING "${BMI2_WARNING}")
  endif()
endif()


#===============================================================================
# Flags
#===============================================================================

macro(um2_add_cxx_compiler_flag FLAG)
  string(REGEX REPLACE "-" "" SFLAG ${FLAG})
  check_cxx_compiler_flag(${FLAG} COMPILER_SUPPORT_${SFLAG})
  if(COMPILER_SUPPORT_${SFLAG})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
  endif()
endmacro()

# Fast math optimizations
if (UM2_ENABLE_FASTMATH)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
  if (UM2_USE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
  endif()
endif()

# Native architecture
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# Sanitizers (debug only)
if (!UM2_USE_CUDA)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
endif()

# Disable math-errno
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")

# Set general warning flags
um2_add_cxx_compiler_flag("-Wall")
um2_add_cxx_compiler_flag("-Wextra")
um2_add_cxx_compiler_flag("-Werror")
um2_add_cxx_compiler_flag("-Wcast-align")
um2_add_cxx_compiler_flag("-Wcast-qual")
um2_add_cxx_compiler_flag("-Wconversion")
um2_add_cxx_compiler_flag("-Wdouble-promotion")
um2_add_cxx_compiler_flag("-Wfloat-equal")
um2_add_cxx_compiler_flag("-Wimplicit-fallthrough")
um2_add_cxx_compiler_flag("-Wmissing-noreturn")
um2_add_cxx_compiler_flag("-Wnon-virtual-dtor")
um2_add_cxx_compiler_flag("-Wold-style-cast")
um2_add_cxx_compiler_flag("-Wshadow")
um2_add_cxx_compiler_flag("-Wunused")
um2_add_cxx_compiler_flag("-Wvla")
um2_add_cxx_compiler_flag("-Wzero-as-null-pointer-constant")

# Set potentially compiler-specific warning flags
um2_add_cxx_compiler_flag("-Wsuggest-attribute=const")
um2_add_cxx_compiler_flag("-Wsuggest-attribute=noreturn")
um2_add_cxx_compiler_flag("-Wsuggest-attribute=pure")
am2_add_cxx_compiler_flag("-Wunreachable-code")
am2_add_cxx_compiler_flag("-Wconditional-uninitialized")
am2_add_cxx_compiler_flag("-Wheader-hygiene")
am2_add_cxx_compiler_flag("-Wmissing-noreturn")
am2_add_cxx_compiler_flag("-Woverriding-method-mismatch")
am2_add_cxx_compiler_flag("-Wrange-loop-analysis")
am2_add_cxx_compiler_flag("-Wreserved-identifier")
um2_add_cxx_compiler_flag("-Wshadow-all")
um2_add_cxx_compiler_flag("-Wshift-sign-overflow")
um2_add_cxx_compiler_flag("-Wtautological-compare")
um2_add_cxx_compiler_flag("-Wunaligned-access")
um2_add_cxx_compiler_flag("-Wunneeded-member-function")
um2_add_cxx_compiler_flag("-Wundef")
um2_add_cxx_compiler_flag("-Wvector-conversion")
um2_add_cxx_compiler_flag("-pedantic")

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

### Minimum log level for compile-time filtering of log messages
### Off = 0,        // no messages
### Error = 1,      // only errors
### Warn = 2,       // errors and warnings
### Info = 3,       // errors, warnings and info
### Debug = 4,      // errors, warnings, info and debug
##if (CMAKE_BUILD_TYPE STREQUAL "Release")
##  set(UM2_MIN_LOG_LEVEL 3)
##else()
##  set(UM2_MIN_LOG_LEVEL 4)
##endif()
##
### Default path to look for MPACT cross section libraries
##if (NOT MPACT_DATA)
##  set(MPACT_DATA "${CMAKE_CURRENT_SOURCE_DIR}/MPACT_Extras/xslibs")
##endif()
##

###===============================================================================
### Create libum2 target
###===============================================================================
##
### Sources
##set(UM2_SOURCES
##    "src/common/settings.cpp"
##    "src/common/logger.cpp"
##    "src/mesh/polytope_soup.cpp"
##    "src/mesh/face_vertex_mesh.cpp"
##    "src/physics/cross_section.cpp"
##    "src/physics/nuclide.cpp"
##    "src/physics/cross_section_library.cpp"
##    "src/physics/material.cpp"
##    "src/mpact/model.cpp"
##    "src/gmsh/base_gmsh_api.cpp"
##    "src/gmsh/io.cpp"
##    "src/gmsh/model.cpp"
##    "src/gmsh/mesh.cpp"
##    "src/um2.cpp"
##    "src/um2c.cpp"
##  )
##
##
##if (UM2_BUILD_SHARED_LIB)
##  if (UM2_USE_CUDA)
##    message(WARNING "Shared library is not supported with CUDA."
##      " Building static library instead.")
##    add_library(um2 STATIC ${UM2_SOURCES})
##  else()
##    add_library(um2 SHARED ${UM2_SOURCES})
##  endif()
##else()
##  add_library(um2 STATIC ${UM2_SOURCES})
##endif()
##
##target_include_directories(um2
##  PUBLIC
##  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
##  $<INSTALL_INTERFACE:include>)
##
### Set C++ standard
##set(UM2_CXX_STANDARD 20)
##set_target_properties(um2 PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD})
##
### config.hpp
##configure_file(
##  "${PROJECT_SOURCE_DIR}/cmake/config.hpp.in"
##  "${PROJECT_SOURCE_DIR}/include/um2/config.hpp")
##
###===============================================================================
### Mandatory dependencies
###===============================================================================
##
### HDF5
##find_package(HDF5 REQUIRED COMPONENTS CXX)
##target_link_libraries(um2 PUBLIC "${HDF5_CXX_LIBRARIES}")
##target_include_directories(um2 SYSTEM PUBLIC "${HDF5_INCLUDE_DIRS}")
##
### pugixml
##find_package(PugiXML REQUIRED)
##target_link_libraries(um2 PUBLIC "${PUGIXML_LIB}")
##target_include_directories(um2 SYSTEM PUBLIC "${PUGIXML_INC}")
##
###===============================================================================
### Optional dependencies
###===============================================================================
##
### Gmsh
##if (UM2_USE_GMSH)
##  find_package(Gmsh REQUIRED)
##  target_link_libraries(um2 PUBLIC "${GMSH_LIB}")
##  target_include_directories(um2 SYSTEM PUBLIC "${GMSH_INC}")
##endif()
##
### OpenMP
##if (UM2_USE_OPENMP)
##  find_package(OpenMP)
##  target_link_libraries(um2 PUBLIC OpenMP::OpenMP_CXX)
##  target_include_directories(um2 SYSTEM PUBLIC "${OpenMP_CXX_INCLUDE_DIRS}")
##endif()
##
### TBB
##if (UM2_USE_TBB)
##  find_package(TBB REQUIRED)
##  target_link_libraries(um2 PUBLIC TBB::tbb)
##endif()
##
### CUDA
##if (UM2_USE_CUDA)
##  include(CheckLanguage)
##  check_language(CUDA)
##
##  # nvcc will default to gcc and g++ if the host compiler is not set using CUDAHOSTCXX.
##  # To prevent unintentional version/compiler mismatches, we set the host compiler to the
##  # same compiler used to build the project.
##  if (NOT DEFINED ENV{CUDAHOSTCXX})
##    message(STATUS "Setting CMAKE_CUDA_HOST_COMPILER to ${CMAKE_CXX_COMPILER}."
##            " Consider setting the CUDAHOSTCXX environment variable if this is not desired.")
##    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
##  endif()
##
##  find_package(CUDAToolkit REQUIRED)
##  enable_language(CUDA)
##
##  # Set CUDA standard to the same as the C++ standard
##  set(UM2_CUDA_STANDARD ${UM2_CXX_STANDARD})
##
##  # Macro for treating the given target as CUDA code
##  macro(set_cuda_properties CUDA_TARGET)
##    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_STANDARD ${UM2_CUDA_STANDARD})
##    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_STANDARD_REQUIRED ON)
##    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
##    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_ARCHITECTURES native)
##    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
##  endmacro()
##
##  set_cuda_properties(um2 ${UM2_SOURCES})
##
##  target_include_directories(um2 SYSTEM PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
##endif()
##

##===============================================================================
## Developer options
##===============================================================================
#
## clang-format
#if (UM2_USE_CLANG_FORMAT)
#  include(cmake/clang-format.cmake)
#endif()
#
## clang-tidy
#if (UM2_USE_CLANG_TIDY)
#  # Macro for running clang-tidy on the given target
#  macro(set_clang_tidy_properties TIDY_TARGET)
#    set_target_properties(${TIDY_TARGET} PROPERTIES
#            CXX_CLANG_TIDY
#            "clang-tidy;--extra-arg=-Wno-unknown-warning-option")
#  endmacro()
#  set_clang_tidy_properties(um2)
#endif()
#
## coverage
#if (UM2_USE_COVERAGE)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
#  target_link_libraries(um2 PRIVATE gcov)
#endif()
#
#===============================================================================
# Optional subdirectories
#===============================================================================

if (UM2_BUILD_BENCHMARKS)
  find_package(benchmark REQUIRED)
  add_subdirectory(benchmarks)
endif()

if (UM2_BUILD_MODELS)
  add_subdirectory(models)
endif()

if (UM2_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

if (UM2_BUILD_TUTORIAL)
  add_subdirectory(tutorial)
endif()

#===============================================================================
# Install
#===============================================================================

install(TARGETS um2
        EXPORT um2-targets
        RUNTIME DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin
        LIBRARY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib
        ARCHIVE DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib)
install(DIRECTORY include/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include)
