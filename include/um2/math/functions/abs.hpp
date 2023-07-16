#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

// --------------------------------------------------------------------------
// abs
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
abs(T x) noexcept -> T
{
  return std::abs(x);
}

#else

template <typename T>
PURE DEVICE constexpr auto
abs(T x) noexcept -> T
{
  static_assert(false, "abs not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
abs(float x) noexcept -> float
{
  return ::fabsf(x);
}

template <>
PURE DEVICE constexpr auto
abs(double x) noexcept -> double
{
  return ::fabs(x);
}

#endif

}  // namespace um2
