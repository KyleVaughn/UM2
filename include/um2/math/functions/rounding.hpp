#pragma once

#include <um2/config.hpp>

#include <algorithm> // clamp
#include <cmath>

namespace um2
{

// --------------------------------------------------------------------------
// abs
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
abs(T x) noexcept -> T
{
  return std::abs(x);
}

#else

template <typename T>
PURE DEVICE constexpr auto
abs(T x) noexcept -> T
{
  static_assert(false, "abs not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
abs(float x) noexcept -> float
{
  return ::fabsf(x);
}

template <>
PURE DEVICE constexpr auto
abs(double x) noexcept -> double
{
  return ::fabs(x);
}

#endif

// --------------------------------------------------------------------------
// ceil
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
ceil(T x) noexcept -> T
{
  return std::ceil(x);
}

#else

template <typename T>
PURE DEVICE constexpr auto
ceil(T x) noexcept -> T
{
  static_assert(false, "ceil not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
ceil(float x) noexcept -> float
{
  return ::ceilf(x);
}

template <>
PURE DEVICE constexpr auto
ceil(double x) noexcept -> double
{
  return ::ceil(x);
}

#endif

// --------------------------------------------------------------------------
// clamp
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
clamp(T const & v, T const & lo, T const & hi) noexcept -> T
{
  return std::clamp(v, lo, hi);
}

#else

template <typename T>
PURE DEVICE constexpr auto
clamp(T const & v, T const & lo, T const & hi) noexcept -> T
{
  return v < lo ? lo : hi < v ? hi : v;
} 

#endif

// --------------------------------------------------------------------------
// floor
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
floor(T x) noexcept -> T
{
  return std::floor(x);
}

#else

template <typename T>
PURE DEVICE constexpr auto
floor(T x) noexcept -> T
{
  static_assert(false, "floor not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
floor(float x) noexcept -> float
{
  return ::floorf(x);
}

template <>
PURE DEVICE constexpr auto
floor(double x) noexcept -> double
{
  return ::floor(x);
}

#endif

} // namespace um2
