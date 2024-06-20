#pragma once

#include <um2/config.hpp>

#include <um2/common/cast_if_not.hpp>
#include <um2/math/vec.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/math/copysign.hpp>
#include <um2/stdlib/math/fma.hpp>
#include <um2/stdlib/math/roots.hpp>
#include <um2/stdlib/numbers.hpp>
#include <um2/stdlib/utility/swap.hpp>

namespace um2
{

//=============================================================================
// solveQuadratic
//=============================================================================
// Solves the quadratic equation a*x^2 + b*x + c = 0.
// Returns 1e16 for roots that are not real.
// We use 1e16 instead of NaN, since we want consistent behavior using -ffast-math.
//
// References:
// The Ins and Outs of Solving Quadratic Equations with Ting-Point Arithmetic
// by Frédéric Goualard
//
// Cases:
// ---------------------------------------------
//     a   |  b   |  c   |  roots
// ---------------------------------------------
// Cases where a = 0:
// 1.  0   |  0   |  0   |  invalid, invalid (infinite solutions)
// 2.  0   |  0   |  !=0 |  invalid, invalid (no solution)
// 3.  0   |  !=0 |  0   |  0, invalid
// 4.  0   |  !=0 |  !=0 |  -c/b, invalid
//
// Cases where b = 0, a != 0:
// 5.  !=0 |  0   |  0   |  0, 0
// 6.  ac > 0, b = 0     |  invalid, invalid (complex roots)
// 7.  ac < 0, b = 0     |  -sqrt(-c/a), sqrt(-c/a)
//
// Cases where c = 0, a != 0, b != 0:
// 8.  !=0 |  !=0 |  0   |  -b/a, 0
//
//
// We do not handle cases: 1, 2

// Handle discriminant with Kahan's algorithm, with fma.
template <class T>
CONST HOSTDEV inline auto
quadraticDiscriminant(T const a, T const b, T const c) noexcept -> T
{
  // b * b - 4 * a * c
  T const a4 = 4 * a;
  T const w = a4 * c;
  T const e = um2::fma(-c, a4, w);
  T const f = um2::fma(b, b, -w);
  return f + e;
}

// We want exact float comparison here.
// NOLINTBEGIN(clang-diagnostic-float-equal)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
template <class T>
PURE HOSTDEV inline auto
solveQuadratic(T const a, T const b, T const c) noexcept -> Vec2<T>
{
  auto constexpr invalid = castIfNot<T>(1e16);
  Vec2<T> roots;
  roots[0] = invalid;
  roots[1] = invalid;

  if (a == 0) {
    // Case 3 or 4. Assert that b != 0, so not case 1 or 2.
    if (b == 0) {
      return roots;
    }
    roots[0] = -c / b;
    return roots;
  }

  if (b == 0) {
    T const x0_sq = -c / a;
    if (x0_sq < 0) {
      return roots;
    }
    T const x0 = um2::sqrt(x0_sq);
    roots[0] = -x0;
    roots[1] = x0;
    return roots;
  }

  auto const disc = quadraticDiscriminant(a, b, c);

  if (disc < 0) {
    return roots;
  }

  T const q = -(b + um2::copysign(um2::sqrt(disc), b)) / 2;
  roots[0] = q / a;
  roots[1] = c / q;
  if (roots[0] > roots[1]) {
    um2::swap(roots[0], roots[1]);
  }
  return roots;
}
#pragma GCC diagnostic pop
// NOLINTEND(clang-diagnostic-float-equal)

} // namespace um2
