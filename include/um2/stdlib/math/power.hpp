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
PURE HOST constexpr auto
sqrt(T x) noexcept -> T
{
  return std::sqrt(x);
}

#else

PURE DEVICE constexpr auto
sqrt(float x) noexcept -> float
{
  return ::sqrtf(x);
}

PURE DEVICE constexpr auto
sqrt(double x) noexcept -> double
{
  return ::sqrt(x);
}

#endif

} // namespace um2
