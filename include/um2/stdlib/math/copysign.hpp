#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// copysign 
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] inline constexpr auto
copysign(T x, T y) noexcept -> T
{
  return std::copysign(x, y);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] inline constexpr auto
copysign(T x, T y) noexcept -> T
{
  static_assert(always_false<T>, "copysign is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] inline constexpr auto
copysign(float x, float y) noexcept -> float
{
  return ::copysignf(x, y);
}

template <>
CONST DEVICE [[nodiscard]] inline constexpr auto
copysign(double x, double y) noexcept -> double 
{
  return ::copysign(x, y);
}

#endif

} // namespace um2
