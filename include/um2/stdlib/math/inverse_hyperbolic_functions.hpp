#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// atanh
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] constexpr auto
atanh(T x) noexcept -> T
{
  return std::atanh(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] constexpr auto
atanh(T x) noexcept -> T
{
  static_assert(always_false<T>, "atanh is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] constexpr auto
atanh(float x) noexcept -> float
{
  return ::atanhf(x);
}

template <>
CONST DEVICE [[nodiscard]] constexpr auto
atanh(double x) noexcept -> double
{
  return ::atanh(x);
}

#endif

} // namespace um2
