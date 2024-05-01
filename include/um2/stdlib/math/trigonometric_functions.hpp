#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// cos
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] inline auto
cos(T x) noexcept -> T
{
  return std::cos(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] inline auto
cos(T x) noexcept -> T
{
  static_assert(always_false<T>, "cos is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] inline auto
cos(float x) noexcept -> float
{
  return ::cosf(x);
}

template <>
CONST DEVICE [[nodiscard]] inline auto
cos(double x) noexcept -> double
{
  return ::cos(x);
}

#endif

//==============================================================================
// sin
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] inline auto
sin(T x) noexcept -> T
{
  return std::sin(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] inline auto
sin(T x) noexcept -> T
{
  static_assert(always_false<T>, "sin is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] inline auto
sin(float x) noexcept -> float
{
  return ::sinf(x);
}

template <>
CONST DEVICE [[nodiscard]] inline auto
sin(double x) noexcept -> double
{
  return ::sin(x);
}

#endif

} // namespace um2
