#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// exp
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] constexpr auto
exp(T x) noexcept -> T
{
  return std::exp(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] constexpr auto
exp(T x) noexcept -> T
{
  static_assert(always_false<T>, "exp is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] constexpr auto
exp(float x) noexcept -> float
{
  return ::expf(x);
}

template <>
CONST DEVICE [[nodiscard]] constexpr auto
exp(double x) noexcept -> double
{
  return ::exp(x);
}

#endif

} // namespace um2
