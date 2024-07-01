#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// acos
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] inline auto
acos(T x) noexcept -> T
{
  return std::acos(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] inline auto
acos(T x) noexcept -> T
{
  static_assert(always_false<T>, "cos is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] inline auto
acos(float x) noexcept -> float
{
  return ::acosf(x);
}

template <>
CONST DEVICE [[nodiscard]] inline auto
acos(double x) noexcept -> double
{
  return ::acos(x);
}

#endif

//==============================================================================
// asin
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] inline auto
asin(T x) noexcept -> T
{
  return std::asin(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] inline auto
asin(T x) noexcept -> T
{
  static_assert(always_false<T>, "asin is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] inline auto
asin(float x) noexcept -> float
{
  return ::asinf(x);
}

template <>
CONST DEVICE [[nodiscard]] inline auto
asin(double x) noexcept -> double
{
  return ::asin(x);
}

#endif

//==============================================================================
// atan
//==============================================================================

#ifndef __CUDA_ARCH__

template <class T>
CONST HOST [[nodiscard]] inline auto
atan(T x) noexcept -> T
{
  return std::atan(x);
}

#else

template <class T>
CONST DEVICE [[nodiscard]] inline auto
atan(T x) noexcept -> T
{
  static_assert(always_false<T>, "atan is not implemented for this type");
  return T();
}

template <>
CONST DEVICE [[nodiscard]] inline auto
atan(float x) noexcept -> float
{
  return ::atanf(x);
}

template <>
CONST DEVICE [[nodiscard]] inline auto
atan(double x) noexcept -> double
{
  return ::atan(x);
}

#endif

} // namespace um2
