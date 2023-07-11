cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(
  UM2
  VERSION 0.1.0
  DESCRIPTION "An unstructured mesh library for automated method of characteristic mesh generation"
  HOMEPAGE_URL "https://github.com/KyleVaughn/UM2"
  LANGUAGES CXX)

## User options ###############################################################
###############################################################################
option(UM2_BUILD_TESTS      "Build tests"          ON)
option(UM2_BUILD_EXAMPLES   "Build examples"       ON)

option(UM2_ENABLE_CUDA      "Enable CUDA"          OFF)
option(UM2_ENABLE_OPENMP    "Enable OpenMP"        ON)
option(UM2_ENABLE_FASTMATH  "Enable fast math"     ON)
option(UM2_ENABLE_VIS       "Enable visualization" OFF)

## Advanced options ###########################################################
###############################################################################
option(UM2_ENABLE_INT64 "Enable 64-bit integer types" OFF) # For large meshes
set(UM2_LOG_LEVEL "info" CACHE STRING "Log level")
set_property(CACHE UM2_LOG_LEVEL PROPERTY STRINGS
  "trace" "debug" "info" "warn" "error" "critical" "off")

## Dev options #################################################################
###############################################################################
option(UM2_DEV_MODE             "Enable development mode"         ON             )
option(UM2_ENABLE_CLANG_FORMAT  "Enable clang-format targets"     ${UM2_DEV_MODE})
option(UM2_ENABLE_CLANG_TIDY    "Enable clang-tidy"               ${UM2_DEV_MODE})
option(UM2_CLANG_TIDY_FIX       "Enable clang-tidy to fix errors" OFF            )  
option(UM2_ENABLE_CPPCHECK      "Enable cppcheck"                 ${UM2_DEV_MODE})

## um2 library ################################################################
###############################################################################

# Headers
set(UM2_HEADERS
    "include/um2/config.hpp"
    # common
    "include/um2/common/memory.hpp"
    "include/um2/common/algorithm.hpp"
    "include/um2/common/Vector.hpp"
    "include/um2/common/sto.hpp"
    "include/um2/common/ShortString.hpp"
    "include/um2/common/Color.hpp"
    # math
    "include/um2/math/morton.hpp"
    "include/um2/math/Vec.hpp"
    "include/um2/math/Mat.hpp"
    # geometry
    "include/um2/geometry/Point.hpp"
    "include/um2/geometry/AxisAlignedBox.hpp"
    "include/um2/geometry/LineSegment.hpp"
    # "include/um2/geometry/Triangle.hpp"
    # "include/um2/geometry/Quadrilateral.hpp"
    #    # mesh
    #    "include/um2/mesh/RegularGrid.hpp"
    #    "include/um2/mesh/RegularPartition.hpp"
    #    # ray_casting
    #    "include/um2/ray_casting/Ray.hpp"
    #    # api
    #    "include/um2c.h"
    )

# Sources
set(UM2_SOURCES        
  #"src/common/string.cpp"
    #    "src/visualization/shader.cpp"
    # api
    "src/um2c.cpp"
  ) 

# Visualization headers and sources
if (UM2_ENABLE_VIS)
  list(APPEND UM2_HEADERS
    "include/um2/visualization/shader.hpp"
    "include/um2/visualization/vertex_buffer.hpp"
    "include/um2/visualization/vertex_array.hpp"
    "include/um2/visualization/element_buffer.hpp")
  list(APPEND UM2_SOURCES
    "src/visualization/shader.cpp")
endif()

# Library
add_library(um2 ${UM2_HEADERS} ${UM2_SOURCES})
target_include_directories(um2 PUBLIC include)

# Configure
include(cmake/um2_config.cmake)
