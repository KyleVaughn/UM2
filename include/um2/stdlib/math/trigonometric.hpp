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

//==============================================================================
// sin
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
sin(T x) noexcept -> T
{
  return std::sin(x);
}

#else

PURE DEVICE constexpr auto
sin(float x) noexcept -> float
{
  return ::sinf(x);
}

PURE DEVICE constexpr auto
sin(double x) noexcept -> double
{
  return ::sin(x);
}

#endif

} // namespace um2
