#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

// --------------------------------------------------------------------------
// atanh
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
atanh(T x) noexcept -> T
{
  return std::atanh(x);
}

#else

template <typename T>
PURE DEVICE constexpr auto
atanh(T x) noexcept -> T
{
  static_assert(false, "atanh not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
atanh(float x) noexcept -> float
{
  return ::atanhf(x);
}

template <>
PURE DEVICE constexpr auto
atanh(double x) noexcept -> double
{
  return ::atanh(x);
}

#endif

} // namespace um2
