cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(
  UM2
  VERSION 0.1.0
  DESCRIPTION "An unstructured mesh library for automated method of characteristic mesh generation"
  HOMEPAGE_URL "https://github.com/KyleVaughn/UM2"
  LANGUAGES CXX)

## User options ###############################################################
###############################################################################
option(UM2_ENABLE_CUDA      "Enable CUDA"       OFF)
option(UM2_ENABLE_OPENMP    "Enable OpenMP"     ON)
option(UM2_ENABLE_TESTS     "Enable tests"      ON)
option(UM2_ENABLE_FASTMATH  "Enable fast math"  ON)

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    "include/um2/common/config.hpp"
    "include/um2/common/vector.hpp"
    "include/um2c.h")    
    
# Sources
set(UM2_SOURCES        
    "src/um2c.cpp")    
    
# Library
add_library(um2 ${UM2_HEADERS} ${UM2_SOURCES})
target_include_directories(um2 PUBLIC include)

# Configure
include(cmake/um2_config.cmake)