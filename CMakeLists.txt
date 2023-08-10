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
option(UM2_BUILD_EXAMPLES   "Build examples"       OFF)
option(UM2_BUILD_BENCHMARKS "Build benchmarks"     OFF)

option(UM2_ENABLE_CUDA      "Enable CUDA"          OFF)
option(UM2_ENABLE_OPENMP    "Enable OpenMP"        ON)
option(UM2_ENABLE_FASTMATH  "Enable fast math"     ON)
option(UM2_ENABLE_OPENGL    "Enable OpenGL visualization" OFF)
option(UM2_ENABLE_GMSH      "Enable Gmsh"          ON)
option(UM2_ENABLE_PAR_STL   "Enable parallel STL"  OFF) 

## Advanced options ###########################################################
###############################################################################
option(UM2_ENABLE_INT64 "Enable 64-bit integer types" OFF) # For large meshes

# minimum log level for compile time filtering of log messages
# Off = 0,        // no messages    
# Error = 1,      // only errors    
# Warn = 2,       // errors and warnings    
# Info = 3,       // errors, warnings and info    
# Debug = 4,       // errors, warnings, info and debug    
# Trace = 5,       // errors, warnings, info, debug and trace    
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(UM2_MIN_LOG_LEVEL 5) 
else()
  set(UM2_MIN_LOG_LEVEL 3)
endif()

## Dev options #################################################################
###############################################################################
option(UM2_DEV_MODE             "Enable development mode"         ON             )
option(UM2_ENABLE_CLANG_FORMAT  "Enable clang-format targets"     ${UM2_DEV_MODE})
option(UM2_ENABLE_CLANG_TIDY    "Enable clang-tidy"               ${UM2_DEV_MODE})
option(UM2_ENABLE_CPPCHECK      "Enable cppcheck"                 ${UM2_DEV_MODE})

## um2 library ################################################################
###############################################################################

# Headers
set(UM2_HEADERS
    "include/um2/config.hpp"
    # stdlib
    "include/um2/stdlib/utility.hpp"
    "include/um2/stdlib/memory.hpp"
    "include/um2/stdlib/math.hpp"
    "include/um2/stdlib/algorithm.hpp"
    "include/um2/stdlib/Vector.hpp"
    "include/um2/stdlib/sto.hpp"
    "include/um2/stdlib/String.hpp"
     # common
    "include/um2/common/Log.hpp"
    "include/um2/common/ShortString.hpp"
    "include/um2/common/Color.hpp"
    "include/um2/common/to_vecvec.hpp"
    # math
    "include/um2/math/morton.hpp"
    "include/um2/math/Vec.hpp"
    "include/um2/math/Mat.hpp"
    # geometry
    "include/um2/geometry/Point.hpp"
    "include/um2/geometry/morton_sort_points.hpp"
    "include/um2/geometry/AxisAlignedBox.hpp"
    "include/um2/geometry/LineSegment.hpp"
    "include/um2/geometry/Triangle.hpp"
    "include/um2/geometry/Quadrilateral.hpp"
    "include/um2/geometry/QuadraticSegment.hpp"
    "include/um2/geometry/QuadraticTriangle.hpp"
    "include/um2/geometry/QuadraticQuadrilateral.hpp"
    # physics
    "include/um2/physics/Material.hpp"
    # mesh
    "include/um2/mesh/RegularGrid.hpp"
    "include/um2/mesh/RegularPartition.hpp"
    "include/um2/mesh/RectilinearGrid.hpp"
    "include/um2/mesh/RectilinearPartition.hpp"
    "include/um2/mesh/CellType.hpp"
    "include/um2/mesh/MeshType.hpp"
    "include/um2/mesh/MeshFile.hpp"
    "include/um2/mesh/io_abaqus.hpp"
    "include/um2/mesh/io_xdmf.hpp"
    "include/um2/mesh/io.hpp"
    "include/um2/mesh/TriMesh.hpp"
    "include/um2/mesh/QuadMesh.hpp"
    "include/um2/mesh/QuadraticTriMesh.hpp"
    "include/um2/mesh/QuadraticQuadMesh.hpp"
    # mpact
    "include/um2/mpact/SpatialPartition.hpp"
    "include/um2/mpact/io.hpp"
    # gmsh
    "include/um2/gmsh/base_gmsh_api.hpp"
    "include/um2/gmsh/io.hpp"
    "include/um2/gmsh/model.hpp"
    "include/um2/gmsh/mesh.hpp"
    #    #    # ray_casting
    #    #    "include/um2/ray_casting/Ray.hpp"
    #    # visualization
    #    "include/um2/visualization/Image2D.hpp"
    # api
    "include/um2.hpp"
    #"include/um2c.h"
    )

# Sources
set(UM2_SOURCES        
    # common
    "src/common/Log.cpp"
    # gmsh
    "src/gmsh/base_gmsh_api.cpp"
    "src/gmsh/io.cpp"
    "src/gmsh/model.cpp"
    "src/gmsh/mesh.cpp"
  #    # visualization
  #    "src/visualization/Image2D.cpp"
    # api
    "src/um2.cpp"
    #"src/um2c.cpp"
  ) 

# Visualization headers and sources
#if (UM2_ENABLE_OPENGL)
#  list(APPEND UM2_HEADERS
#    "include/um2/visualization/shader.hpp"
#    "include/um2/visualization/vertex_buffer.hpp"
#    "include/um2/visualization/vertex_array.hpp"
#    "include/um2/visualization/element_buffer.hpp")
#  list(APPEND UM2_SOURCES
#    "src/visualization/shader.cpp")
#endif()

# Library
add_library(um2 ${UM2_HEADERS} ${UM2_SOURCES})
target_include_directories(um2 PUBLIC include)

# Configure
include(cmake/um2_config.cmake)

add_subdirectory(docs)
