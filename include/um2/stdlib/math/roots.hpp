#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// cbrt
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] inline auto
cbrt(T x) noexcept -> T
{
  return std::cbrt(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] inline auto
cbrt(T x) noexcept -> T
{
  static_assert(always_false<T>, "cbrt is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] inline auto
cbrt(float x) noexcept -> float
{
  return ::cbrtf(x);
}

template <>
CONST DEVICE [[nodiscard]] inline auto
cbrt(double x) noexcept -> double
{
  return ::cbrt(x);
}

#endif

//==============================================================================
// sqrt
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] inline auto
sqrt(T x) noexcept -> T
{
  return std::sqrt(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] inline auto
sqrt(T x) noexcept -> T
{
  static_assert(always_false<T>, "sqrt is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] inline auto
sqrt(float x) noexcept -> float
{
  return ::sqrtf(x);
}

template <>
CONST DEVICE [[nodiscard]] inline auto
sqrt(double x) noexcept -> double
{
  return ::sqrt(x);
}

#endif

} // namespace um2
