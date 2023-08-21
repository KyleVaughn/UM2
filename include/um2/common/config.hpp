#pragma once

#include <cassert> // assert
#include <cstdint> // int32_t, int64_t

// OpenMP
#define UM2_ENABLE_OPENMP 1

// CUDA
#define UM2_ENABLE_CUDA 0
#if UM2_ENABLE_CUDA
#  include <cuda_runtime.h>
#  define UM2_HOST    __host__
#  define UM2_DEVICE  __device__
#  define UM2_HOSTDEV __host__ __device__
#  define UM2_GLOBAL  __global__
#else
#  define UM2_HOST
#  define UM2_DEVICE
#  define UM2_HOSTDEV
#  define UM2_GLOBAL
#endif

// spdlog
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO

// Visualization
#define UM2_ENABLE_VIS 0

// Attributes
// -----------------------------------------------------------------------------
// These are supported by GCC and Clang, but cause issues when using nvcc
#if defined(__NVCC__)
#  define UM2_PURE
#  define UM2_CONST
#  define UM2_HOT
#else
#  define UM2_PURE  [[gnu::pure]]
#  define UM2_CONST [[gnu::const]]
#  define UM2_HOT   [[gnu::hot]]
#endif

// Define a PURE/CONST attribute that depends on a release build to be enabled
#if defined(NDEBUG)
#  define UM2_NDEBUG_PURE  UM2_PURE
#  define UM2_NDEBUG_CONST UM2_CONST
#else
#  define UM2_NDEBUG_PURE
#  define UM2_NDEBUG_CONST
#endif

// Typedefs
// -----------------------------------------------------------------------------
#define UM2_ENABLE_INT64 0
#if UM2_ENABLE_INT64
// We want this alias to be lower case to match int32_t, int64_t, etc.
// NOLINTBEGIN(readability-identifier-naming)
using len_t = int64_t;
using ulen_t = uint64_t;
#else
using len_t = int32_t;
using ulen_t = uint32_t;
// NOLINTEND(readability-identifier-naming)
#endif
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE len_t
