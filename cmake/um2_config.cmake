## System ########################################
##################################################
# No Windows support
if (WIN32)
  message(FATAL_ERROR "Windows is not supported")
endif()

## Compiler ######################################
##################################################
# Check for gcc and clang
if (NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
         CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()
# Some features are only available in clang, such as CUDA support
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(UM2_HAS_CLANG 1)
else()
  set(UM2_HAS_CLANG 0)
endif()

## Standard ######################################
##################################################
# Don't use -std=c++23, use -std=c++2b instead.
# Change this for future compiler versions when -std=c++23 is available.
set(CMAKE_CXX23_STANDARD_COMPILE_OPTION "-std=c++2b")
set(CMAKE_CXX23_EXTENSION_COMPILE_OPTION "-std=gnu++2b")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## OpenMP ########################################
##################################################
if (UM2_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(UM2_HAS_OPENMP 1)
else()
  set(UM2_HAS_OPENMP 0)
endif()

## CUDA ##########################################
##################################################
if (UM2_ENABLE_CUDA AND UM2_HAS_CLANG)
  find_package(CUDA REQUIRED)
  set(UM2_HAS_CUDA 1)
else()
  set(UM2_HAS_CUDA 0)
endif()

## Thrust ########################################
##################################################
set(Thrust_DIR "${PROJECT_SOURCE_DIR}/tpls/thrust/thrust/cmake")
find_package(Thrust REQUIRED CONFIG)
# Host backend (OpenMP > Sequential)
if (UM2_HAS_OPENMP)
  set(UM2_THRUST_HOST "OMP" CACHE STRING "Thrust host backend")
else()
  set(UM2_THRUST_HOST "CPP" CACHE STRING "Thrust host backend")
endif()
set_property(CACHE UM2_THRUST_HOST PROPERTY STRINGS "OMP" "CPP")
# Device backend (CUDA > OpenMP > Sequential)
if (UM2_HAS_CUDA)
  set(UM2_THRUST_DEVICE "CUDA" CACHE STRING "Thrust device backend")
elseif (UM2_HAS_OPENMP)
  set(UM2_THRUST_DEVICE "OMP" CACHE STRING "Thrust device backend")
else()
  set(UM2_THRUST_DEVICE "CPP" CACHE STRING "Thrust device backend")
endif()
set_property(CACHE UM2_THRUST_DEVICE PROPERTY STRINGS "CUDA" "OMP" "CPP")
MESSAGE(STATUS "Thrust host backend: ${UM2_THRUST_HOST}")
MESSAGE(STATUS "Thrust device backend: ${UM2_THRUST_DEVICE}")
thrust_create_target(Thrust HOST ${UM2_THRUST_HOST} DEVICE ${UM2_THRUST_DEVICE})

# Configure config.hpp
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/config.hpp.in"
  "${PROJECT_SOURCE_DIR}/include/um2/common/config.hpp"
)

## clang-format ##################################
##################################################
if (UM2_ENABLE_CLANG_FORMAT)
  include(cmake/clang-format.cmake)
endif()
