#pragma once

#include <um2/config.hpp>

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
// max
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOSTDEV constexpr auto
max(T x, T y) noexcept -> T
{
  return std::max(x, y);
}

#else

template <typename T>
PURE DEVICE constexpr auto
max(T x, T y) noexcept -> T
{
  static_assert(false, "max not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
max(float x, float y) noexcept -> float
{
  return ::fmaxf(x, y);
}

template <>
PURE DEVICE constexpr auto
max(double x, double y) noexcept -> double
{
  return ::fmax(x, y);
}

template <std::integral T>
PURE DEVICE constexpr auto
max(T x, T y) noexcept -> T
{
  return ::max(x, y);
}

#endif

// --------------------------------------------------------------------------
// min
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
min(T x, T y) noexcept -> T
{
  return std::min(x, y);
}

#else

template <typename T>
PURE DEVICE constexpr auto
min(T x, T y) noexcept -> T
{
  static_assert(false, "min not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
min(float x, float y) noexcept -> float
{
  return ::fminf(x, y);
}

template <>
PURE DEVICE constexpr auto
min(double x, double y) noexcept -> double
{
  return ::fmin(x, y);
}

template <std::integral T>
PURE DEVICE constexpr auto
min(T x, T y) noexcept -> T
{
  return ::min(x, y);
}

#endif

// --------------------------------------------------------------------------
// sqrt
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
sqrt(T x) noexcept -> T
{
  return std::sqrt(x);
}

#else

template <typename T>
PURE DEVICE constexpr auto
sqrt(T x) noexcept -> T
{
  static_assert(false, "sqrt not implemented for this type");
  return T();
}

template <>
PURE DEVICE constexpr auto
sqrt(float x) noexcept -> float
{
  return ::sqrtf(x);
}

template <>
PURE DEVICE constexpr auto
sqrt(double x) noexcept -> double
{
  return ::sqrt(x);
}

#endif

} // namespace um2
