#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/math/roots.hpp>

namespace um2
{

template <class T>
CONST HOSTDEV [[nodiscard]] constexpr auto
abs(T x) noexcept -> T
{
  return x < 0 ? -x : x;
}

template <class T>
CONST HOSTDEV [[nodiscard]] auto
abs(Complex<T> x) noexcept -> T
{
  return um2::sqrt(x.real() * x.real() + x.imag() * x.imag());
}

} // namespace um2
