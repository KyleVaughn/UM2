#pragma once

#include <um2/common/branchless_sort.hpp>
#include <um2/math/quadratic_equation.hpp>
#include <um2/stdlib/math/abs.hpp>
#include <um2/stdlib/math/inverse_trigonometric_functions.hpp>
#include <um2/stdlib/math/trigonometric_functions.hpp>

namespace um2
{

//=============================================================================
// solveCubic
//=============================================================================
// Solves the cubic equation a*x^3 + b*x^2 + c*x + d = 0.
// Returns 1e16 for roots that are not real.
//
// References:
// To solve a real cubic equation by W. Kahan (Nov. 10, 1986)
//
// Overview:
// The closed-form solution for the cubic equation is prone to numerical issues.
// Instead, we use Newton-Raphson iteration to find the smallest real root, then
// deflate the polynomial by dividing by (x - root). Then, we solve the quadratic
// equation that results from the deflation.

// We want exact float comparison here.
// NOLINTBEGIN(clang-diagnostic-float-equal, cppcoreguidelines-*)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
template <class T>
CONST HOSTDEV inline auto
solveCubic(T a, T const b, T const c, T const d) noexcept -> Vec3<T>
{
  auto constexpr invalid = castIfNot<T>(1e16);
  Vec3<T> roots;
  roots[0] = invalid;
  roots[1] = invalid;
  roots[2] = invalid;

  // Variable initialization.
  auto constexpr lambda = castIfNot<T>(1.32471795724474602596); // lambda^3 = lambda + 1
  // Compiler complains if we do the same with a ternary operator. Why?
  T one_plus_eps;
  if constexpr (std::same_as<float, T>) {
    one_plus_eps = 1 + 1.1920928955078125e-07F;
  } else {
    one_plus_eps = 1 + 2.2204460492503131e-16;
  }
  T & x = roots[2];
  T b1; // b in ax^2 + bx + c for the quadratic equation.
  T c2; // c in ax^2 + bx + c for the quadratic equation.
  T q0; // Used to evaluate q and q'.
  T dq; // q' = 3ax^2 + 2bx + c
  T q;  // q = ax^3 + bx^2 + cx + d
  T t;  // tmp variable
  T r;  // tmp variable
  T s;  // tmp variable
  T x0; // initial guess for Newton-Raphson iteration.

  // Equation is quadratic.
  if (a == 0) {
    a = b;
    b1 = c;
    c2 = d;
    goto fin;
  }

  // x = 0 is a root.
  if (d == 0) {
    roots[2] = 0;
    b1 = b;
    c2 = c;
    goto fin;
  }

  x = -(b / a) / 3;
  // Evaluate q and q' (dq), matching Kahan's naming.
  // eval
  q0 = a * x;
  b1 = q0 + b;             // ax + b
  c2 = b1 * x + c;         // (ax + b)x + c = ax^2 + bx + c
  dq = (q0 + b1) * x + c2; // 3ax^2 + 2bx + c
  q = c2 * x + d;          // ax^3 + bx^2 + cx + d

  t = q / a;
  r = um2::cbrt(um2::abs(t));
  s = t < 0 ? -1 : 1;
  t = -dq / a;
  if (t > 0) {
    r = lambda * um2::max(r, um2::sqrt(t));
  }
  x0 = x - s * r;
  if (x0 == x) {
    goto fin;
  }

  do {
    x = x0;
    // eval
    q0 = a * x;
    b1 = q0 + b;
    c2 = b1 * x + c;
    dq = (q0 + b1) * x + c2;
    q = c2 * x + d;
    if (dq == 0) {
      x0 = x;
    } else {
      x0 = x - (q / dq) / one_plus_eps;
    }
  } while (s * x0 > s * x);

  if (um2::abs(a) * x * x > um2::abs(d / x)) {
    c2 = -d / x;
    b1 = (c2 - c) / x;
  }

fin:
  auto const quadratic_roots = solveQuadratic(a, b1, c2);
  roots[0] = quadratic_roots[0];
  roots[1] = quadratic_roots[1];
  um2::sort3(&roots[0], &roots[1], &roots[2]);
  return roots;
}
#pragma GCC diagnostic pop
// NOLINTEND(clang-diagnostic-float-equal, cppcoreguidelines-*)

} // namespace um2
