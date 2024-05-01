#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// ceil
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] constexpr auto
ceil(T x) noexcept -> T
{
  return std::ceil(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] constexpr auto
ceil(T x) noexcept -> T
{
  static_assert(always_false<T>, "ceil is not implemented for this type");
  return T();
}

CONST DEVICE [[nodiscard]] constexpr auto
ceil(float x) noexcept -> float
{
  return ::ceilf(x);
}

CONST DEVICE [[nodiscard]] constexpr auto
ceil(double x) noexcept -> double
{
  return ::ceil(x);
}

#endif

//==============================================================================
// floor
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] constexpr auto
floor(T x) noexcept -> T
{
  return std::floor(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] constexpr auto
floor(T x) noexcept -> T
{
  static_assert(always_false<T>, "floor is not implemented for this type");
  return T();
}

CONST DEVICE [[nodiscard]] constexpr auto
floor(float x) noexcept -> float
{
  return ::floorf(x);
}

CONST DEVICE [[nodiscard]] constexpr auto
floor(double x) noexcept -> double
{
  return ::floor(x);
}

#endif

} // namespace um2
