#pragma once

#include <um2/config.hpp>

namespace um2
{

// --------------------------------------------------------------------------
// sqrt
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
sqrt(T x) -> T
{
  return std::sqrt(x);
}

#else

template <typename T>
PURE DEVICE constexpr auto
sqrt(T x) -> T
{
  static_assert(false, "sqrt not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
sqrt(float x) -> float
{
  return ::sqrtf(x);
}

template <>
PURE DEVICE constexpr auto
sqrt(double x) -> double
{
  return ::sqrt(x);
}

#endif

} // namespace um2
