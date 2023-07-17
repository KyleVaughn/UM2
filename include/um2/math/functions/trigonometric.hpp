#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

template <std::floating_point T>
PURE HOSTDEV constexpr auto
pi() noexcept -> T
{
  return static_cast<T>(3.14159265358979323846);
}

// --------------------------------------------------------------------------
// sin
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
sin(T x) noexcept -> T
{
  return std::sin(x);
}

#else

template <typename T>
PURE DEVICE constexpr auto
sin(T x) noexcept -> T
{
  static_assert(false, "sin not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
sin(float x) noexcept -> float
{
  return ::sinf(x);
}

template <>
PURE DEVICE constexpr auto
sin(double x) noexcept -> double
{
  return ::sin(x);
}

#endif

} // namespace um2
