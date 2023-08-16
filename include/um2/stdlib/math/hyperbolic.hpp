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

PURE DEVICE constexpr auto
atanh(float x) noexcept -> float
{
  return ::atanhf(x);
}

PURE DEVICE constexpr auto
atanh(double x) noexcept -> double
{
  return ::atanh(x);
}

#endif

// --------------------------------------------------------------------------
// exp
// --------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
exp(T x) noexcept -> T
{
  return std::exp(x);
}

#else

PURE DEVICE constexpr auto
exp(float x) noexcept -> float
{
  return ::expf(x);
}

PURE DEVICE constexpr auto
exp(double x) noexcept -> double
{
  return ::exp(x);
}

#endif

} // namespace um2
