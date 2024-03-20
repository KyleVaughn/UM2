#include <um2/math/quadratic_equation.hpp>
#include <um2/stdlib/math/abs.hpp>
#include <um2/stdlib/utility/swap.hpp>

#include "../test_macros.hpp"

// References:    
// The Ins and Outs of Solving Quadratic Equations with Floating-Point Arithmetic    
// by Frédéric Goualard
//
// Test using:
// M F_n x^2 - 2 M F_n-1 x + M F_n-2 = 0
// where F_n is the nth Fibonacci number
// M = floor(R / F_n) where R is a random integer drawn from [2^(p-1), 2^p - 1].
// Here, p is the precision of the floating-point type.
//
// The solutions are
// x = (F_(n-1) pm sqrt(-1^n)) / F_n
//
// To get real solutions, we consider even n.

auto
fib(Int n) -> Int
{
  // Arbitray cutoff for max number, since at some point they will overflow
  ASSERT(n <= 30);
  ASSERT(n >= 0);
  if (n == 0) {
    return 0;
  }
  if (n == 1) {
    return 1;
  }
  // n >= 2
  Int fn_2 = 0;
  Int fn_1 = 1;
  for (Int i = 2; i < n; ++i) {
    Int const fn0 = fn_1 + fn_2;
    fn_2 = fn_1;
    fn_1 = fn0;
  }
  return fn_1 + fn_2;
}

TEST_CASE(quadratic)
{
  // First check that fib is working
  ASSERT(fib(0) == 0);
  ASSERT(fib(1) == 1);
  ASSERT(fib(2) == 1);
  ASSERT(fib(3) == 2);
  ASSERT(fib(4) == 3);
  ASSERT(fib(5) == 5);
  ASSERT(fib(6) == 8);
  ASSERT(fib(7) == 13);
  auto constexpr eps = castIfNot<Float>(1e-6);
  Int const max_fib = 30;
  Int const max_r = 10000;
  for (Int n = 2; n < max_fib; n += 2) { 
    Int const fn_1 = fib(n - 1);
    Int const fn_2 = fib(n - 2);
    Int const fn = fn_1 + fn_2;  // by definition
    Float x1 = static_cast<Float>(fn_1 + 1) / fn; 
    Float x2 = static_cast<Float>(fn_1 - 1) / fn;
    if (x1 < x2) {
      um2::swap(x1, x2);
    }
    for (Int r = 1; r < max_r; ++r) {
      Int m = r / fn;
      if (m == 0) {
        m = 1;
      }
      Float const a = static_cast<Float>(m) * fn;
      Float const b = static_cast<Float>(m) * fn_1 * -2;
      Float const c = static_cast<Float>(m) * fn_2;
      auto roots = um2::solveQuadratic(a, b, c);
      if (roots[0] < roots[1]) {
        um2::swap(roots[0], roots[1]);
      }
      ASSERT_NEAR(x1, roots[0], eps);
      ASSERT_NEAR(x2, roots[1], eps);
    }
  }

  // Check for a = 0
  {
    Float const a = 0;
    Float const b = 1;
    Float const c = 1;
    auto roots = um2::solveQuadratic(a, b, c);
    ASSERT_NEAR(roots[0], -1, eps);
  }

  // Check for discriminant < 0
  {
    Float const a = 2;
    Float const b = 2;
    Float const c = 1;
    auto roots = um2::solveQuadratic(a, b, c);
    ASSERT(roots[0] > 1000000);
    ASSERT(roots[1] > 1000000);
  }

  // Check for numericla issues
  {
    Float const a = 1;
    Float const b = -1634;
    Float const c = 2;
    // x = 817 +- sqrt(667487)
    auto roots = um2::solveQuadratic(a, b, c);
    if (roots[0] > roots[1]) {
      um2::swap(roots[0], roots[1]);
    }
    Float const x1 = 1.223991125e-3;
    Float const x2 = 1.633998776e3;
    ASSERT_NEAR(roots[0], x1, eps); 
    ASSERT_NEAR(roots[1], x2, eps); 
  }
}

TEST_SUITE(quadratic_equation)
{
  TEST_HOSTDEV(quadratic);
}

auto
main() -> int
{
  RUN_SUITE(quadratic_equation);
  return 0;
}
