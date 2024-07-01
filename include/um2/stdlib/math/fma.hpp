#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// fma
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] inline auto
fma(T x, T y, T z) noexcept -> T
{
  return std::fma(x, y, z);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] inline auto
fma(T x, T y, T z) noexcept -> T
{
  static_assert(always_false<T>, "fma is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] inline auto
fma(float x, float y, float z) noexcept -> float
{
  return ::fmaf(x, y, z);
}

template <>
CONST DEVICE [[nodiscard]] inline auto
fma(double x, double y, double z) noexcept -> double
{
  return ::fma(x, y, z);
}

#endif

} // namespace um2
