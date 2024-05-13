#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// tanh
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] constexpr auto
tanh(T x) noexcept -> T
{
  return std::tanh(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] constexpr auto
tanh(T x) noexcept -> T
{
  static_assert(always_false<T>, "tanh is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] constexpr auto
tanh(float x) noexcept -> float
{
  return ::tanhf(x);
}

template <>
CONST DEVICE [[nodiscard]] constexpr auto
tanh(double x) noexcept -> double
{
  return ::tanh(x);
}

#endif

} // namespace um2
