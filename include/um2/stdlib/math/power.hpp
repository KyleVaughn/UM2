#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// sqrt
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
sqrt(T x) noexcept -> T
{
  return std::sqrt(x);
}

#else

template <typename T>
PURE HOST constexpr auto
sqrt(T x) noexcept -> T
{
  static_assert(!sizeof(T), "sqrt is not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
sqrt(float x) noexcept -> float
{
  return ::sqrtf(x);
}

template <>
PURE DEVICE constexpr auto
sqrt(double x) noexcept -> double
{
  return ::sqrt(x);
}

#endif

//==============================================================================
// cbrt
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
cbrt(T x) noexcept -> T
{
  return std::cbrt(x);
}

#else

template <typename T>
PURE HOST constexpr auto
cbrt(T x) noexcept -> T
{
  static_assert(!sizeof(T), "cbrt is not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
cbrt(float x) noexcept -> float
{
  return ::cbrtf(x);
}

template <>
PURE DEVICE constexpr auto
cbrt(double x) noexcept -> double
{
  return ::cbrt(x);
}

#endif

} // namespace um2
