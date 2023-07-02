#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

// --------------------------------------------------------------------------
// sqrt
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__ 

template <typename T>
inline auto sqrt(T x) -> T
{
    return std::sqrt(x);
}

#else

__device__ inline auto sqrt(float x) -> float
{
    return sqrtf(x);
}

__device__ inline auto sqrt(double x) -> double 
{
    return sqrt(x);
}

#endif

} // namespace um2
