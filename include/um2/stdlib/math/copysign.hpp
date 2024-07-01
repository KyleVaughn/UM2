#pragma once

#include <um2/config.hpp>

#include <cmath>

namespace um2
{

//==============================================================================
// copysign
//==============================================================================

CONST HOSTDEV [[nodiscard]] constexpr auto
copysign(float x, float y) noexcept -> float
{
  return __builtin_copysignf(x, y);
}

CONST HOSTDEV [[nodiscard]] constexpr auto
copysign(double x, double y) noexcept -> double
{
  return __builtin_copysign(x, y);
}

} // namespace um2
