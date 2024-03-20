#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// log 
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] constexpr auto
log(T x) noexcept -> T
{
  return std::log(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] constexpr auto
log(T x) noexcept -> T
{
  static_assert(always_false<T>, "log is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] constexpr auto
log(float x) noexcept -> float
{
  return ::logf(x);
}

template <>
CONST DEVICE [[nodiscard]] constexpr auto
log(double x) noexcept -> double
{
  return ::log(x);
}

#endif

} // namespace um2
