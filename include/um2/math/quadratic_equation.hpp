#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/assert.hpp>
#include <um2/math/vec.hpp>
#include <um2/stdlib/math/abs.hpp>
#include <um2/stdlib/math/roots.hpp>
#include <um2/stdlib/numbers.hpp>
#include <um2/common/cast_if_not.hpp>

namespace um2
{

//=============================================================================
// solveQuadratic
//=============================================================================
// Solves the quadratic equation a*x^2 + b*x + c = 0. 
// Returns 1e16 for roots that are not real. 
//
// References:
// The Ins and Outs of Solving Quadratic Equations with Floating-Point Arithmetic
// by Frédéric Goualard

PURE HOSTDEV inline auto
solveQuadratic(Float a, Float b, Float c) -> Vec2F 
{
  // Ensure that a is not zero.

  auto constexpr invalid = castIfNot<Float>(1e16);
  Vec2F roots;
  roots[0] = invalid;
  roots[1] = invalid;

  auto constexpr eps = castIfNot<Float>(1e-8);
  if (um2::abs(a) < eps) {
    roots[0] = -c / b;
    return roots;
  }

  auto const disc = b * b - 4 * a * c;

  if (disc < 0) {
    return roots;
  }

  auto const sqrt_disc = um2::sqrt(disc);

  // Compute one root with high precision, avoiding catastrophic cancellation.
  // Then use the fact that r1 * r2 = c / a to compute the other root, without
  // loss of precision.
  if (b >= 0) {
    roots[0] = (-b - sqrt_disc) / (2 * a);
  } else {
    roots[0] = (-b + sqrt_disc) / (2 * a);
  }
  roots[1] = (c / a) / roots[0];
  return roots; 
}

} // namespace um2
