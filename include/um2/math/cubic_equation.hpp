#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/assert.hpp>
#include <um2/math/vec.hpp>
#include <um2/stdlib/math.hpp>
#include <um2/stdlib/numbers.hpp>
#include <um2/common/cast_if_not.hpp>

namespace um2
{

//=============================================================================
// solveCubic
//=============================================================================
// Solves the cubic equation a*x^3 + b*x^2 + c*x + d = 0. 
// Returns 1e16 for roots that are not real. 

PURE HOSTDEV inline auto
solveCubic(Float a, Float b, Float c, Float d) -> Vec3F 
{
  // https://en.wikipedia.org/wiki/Cubic_equation
  // https://stackoverflow.com/questions/27176423/function-to-solve-cubic-equation-analytically

  // Check that a is not zero.
  auto constexpr eps = castIfNot<Float>(5e-7);
  auto constexpr invalid = castIfNot<Float>(1e16);
  ASSERT(um2::abs(a) > eps);
  
  // Convert to depressed cubic t^3 + p*t + q = 0 (t = x - b / (3 * a))
  Float const p = (3 * a * c - b * b) / (3 * a * a);
  Float const q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a);

  Vec3F roots = um2::Vec3F::zero() + invalid;
  if (um2::abs(p) < eps) {
    // p = 0 -> t^3 = -q -> t = -q^(1/3)
    ASSERT(um2::abs(q) > eps);
    roots[0] = um2::cbrt(-q);
  } else if (um2::abs(q) < eps) {
    // q = 0 -> t^3 + p*t = 0 -> t(t^2 + p) = 0 -> t = 0, -sqrt(-p), sqrt(-p)
    roots[0] = 0;
    if (p < 0) {
      roots[1] = -um2::sqrt(-p);
      roots[2] = um2::sqrt(-p);
    }
  } else {
    Float const disc = q * q / 4 + p * p * p / 27;
    // After shrinking the tolerance on the discriminant 4 times, I finally just
    // made the single root case return the double root case in the other 2 roots.
    // Therefore, it should be verified that the roots returned are correct.
//    if (um2::abs(disc) < eps / 1000) {
//      // Two real roots.
//      Float const qp3 = 3 * q / p;
//      roots[0] = -qp3 / 2;
//      roots[1] = qp3;
//    } else 
    if (disc > 0) {
      // One real root.
      Float const sqrt_disc = um2::sqrt(disc);
      Float const u = um2::cbrt(-q / 2 + sqrt_disc);
      // v = -p/(3*u) 
      roots[0] = u - p / (3 * u);

      Float const qp3 = 3 * q / p;
      roots[1] = -qp3 / 2;
      roots[2] = qp3;
    } else {
      ASSERT(p < 0);
      // Three real roots.
      Float const sqrt_p3 = um2::sqrt(-p / 3);
      ASSERT(3 * q / (2 * p * sqrt_p3) <= 1);
      ASSERT(3 * q / (2 * p * sqrt_p3) >= -1);
      Float const theta = um2::acos(3 * q / (2 * p * sqrt_p3)) / 3;
      Float const shift = 2 * um2::pi<Float> / 3;
      for (Int i = 0; i < 3; ++i) {
        roots[i] = 2 * sqrt_p3 * um2::cos(theta - i * shift);
      }
    }
  }

  // Convert back from depressed cubic to original cubic.
  for (Int i = 0; i < 3; ++i) {
    roots[i] -= b / (3 * a);
  }
  return roots; 
}

} // namespace um2
