#include <um2/common/cast_if_not.hpp>
#include <um2/config.hpp>
#include <um2/math/quadratic_equation.hpp>
#include <um2/stdlib/math/abs.hpp>
#include <um2/stdlib/utility/swap.hpp>

#include "../test_macros.hpp"

// NOLINTNEXTLINE(misc-include-cleaner)
#include <concepts>

#include <cstdint>
#include <random>

// References:
// The Ins and Outs of Solving Quadratic Equations with floating-Point Arithmetic
// by Frédéric Goualard

// Check the degenerate cases
// Table 2, pg 15 in the reference
template <class T>
HOSTDEV
TEST_CASE(degenerate_cases)
{
  auto constexpr invalid = castIfNot<T>(1e15);
  auto constexpr eps = castIfNot<T>(1e-6);
  // Not handled:
  // a = 0, b = 0, c = 0
  // a = 0, b = 0, c != 0

  // a = 0, b != 0, c = 0
  {
    T const a = 0;
    T const b = 1;
    T const c = 0;
    auto roots = um2::solveQuadratic(a, b, c);
    ASSERT_NEAR(roots[0], 0, eps);
    ASSERT(roots[1] > invalid);
  }

  // a = 0, b != 0, c != 0
  {
    T const a = 0;
    T const b = 1;
    T const c = 1;
    auto roots = um2::solveQuadratic(a, b, c);
    ASSERT_NEAR(roots[0], -1, eps);
    ASSERT(roots[1] > invalid);
  }

  // a != 0, b = 0, c = 0
  {
    T const a = 1;
    T const b = 0;
    T const c = 0;
    auto roots = um2::solveQuadratic(a, b, c);
    ASSERT_NEAR(roots[0], 0, eps);
    ASSERT_NEAR(roots[1], 0, eps);
  }

  // ac > 0, b = 0
  {
    T const a = 1;
    T const b = 0;
    T const c = 1;
    auto roots = um2::solveQuadratic(a, b, c);
    ASSERT(roots[0] > invalid);
    ASSERT(roots[1] > invalid);
  }

  // ac < 0, b = 0
  {
    T const a = 1;
    T const b = 0;
    T const c = -1;
    auto roots = um2::solveQuadratic(a, b, c);
    ASSERT_NEAR(roots[0], -1, eps);
    ASSERT_NEAR(roots[1], 1, eps);
  }

  // a != 0, b != 0, c = 0
  {
    T const a = 1;
    T const b = 4;
    T const c = 0;
    auto roots = um2::solveQuadratic(a, b, c);
    ASSERT_NEAR(roots[0], -4, eps);
    ASSERT_NEAR(roots[1], 0, eps);
  }
}

// As suggested by Kahan, test using:
// M F_n x^2 - 2 M F_n-1 x + M F_n-2 = 0
// where F_n is the nth Fibonacci number
// M = floor(R / F_n) where R is a random integer drawn from [2^(p-1), 2^p - 1].
// Here, p is the precision of the floating-point type.
//
// The solutions are
// x = (F_(n-1) pm sqrt(-1^n)) / F_n
//
// To get real solutions, we consider even n.
HOSTDEV CONST auto
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

template <class T>
HOSTDEV
TEST_CASE(fibonacci)
{
  // Check the 8 degenerate cases
  // First check that fib is working
  ASSERT(fib(0) == 0);
  ASSERT(fib(1) == 1);
  ASSERT(fib(2) == 1);
  ASSERT(fib(3) == 2);
  ASSERT(fib(4) == 3);
  ASSERT(fib(5) == 5);
  ASSERT(fib(6) == 8);
  ASSERT(fib(7) == 13);
  auto constexpr eps = castIfNot<T>(1e-6);
  Int const max_fib = 30;
  Int const max_r = 10000;
  for (Int n = 2; n < max_fib; n += 2) {
    Int const fn_1 = fib(n - 1);
    Int const fn_2 = fib(n - 2);
    Int const fn = fn_1 + fn_2; // by definition
    T x1 = static_cast<T>(fn_1 + 1) / static_cast<T>(fn);
    T x2 = static_cast<T>(fn_1 - 1) / static_cast<T>(fn);
    if (x1 < x2) {
      um2::swap(x1, x2);
    }
    for (Int r = 1; r < max_r; ++r) {
      Int m = r / fn;
      if (m == 0) {
        m = 1;
      }
      T const a = static_cast<T>(m) * static_cast<T>(fn);
      T const b = static_cast<T>(m) * static_cast<T>(fn_1) * -2;
      T const c = static_cast<T>(m) * static_cast<T>(fn_2);
      auto roots = um2::solveQuadratic(a, b, c);
      if (roots[0] < roots[1]) {
        um2::swap(roots[0], roots[1]);
      }
      ASSERT_NEAR(x1, roots[0], eps);
      ASSERT_NEAR(x2, roots[1], eps);
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(numerical)
{
  // Check for numerical issues
  auto constexpr eps = castIfNot<T>(1e-6);
  T const a = 1;
  T const b = -1634;
  T const c = 2;
  // x = 817 +- sqrt(667487)
  auto roots = um2::solveQuadratic(a, b, c);
  if (roots[0] > roots[1]) {
    um2::swap(roots[0], roots[1]);
  }
  auto const x1 = castIfNot<T>(1.223991125e-3);
  auto const x2 = castIfNot<T>(1.633998776e3);
  ASSERT_NEAR(roots[0], x1, eps);
  ASSERT_NEAR(roots[1], x2, eps);
}

template <class T>
TEST_CASE(random_coeff)
{
  auto constexpr invalid = castIfNot<T>(1e15);
  T eps = 0;
  if constexpr (std::same_as<T, double>) {
    eps = 1e-6;
  } else {
    eps = 1e-3F;
  }
  Int constexpr num_random_tests = 100000;
  // Check for random values
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dis(-10000, 10000);
  for (Int i = 0; i < num_random_tests; ++i) {
    T const a = dis(gen);
    T const b = dis(gen);
    T const c = dis(gen);
    auto roots = um2::solveQuadratic(a, b, c);
    T const discriminant = b * b - 4 * a * c;
    // No real roots
    if (discriminant < 0) {
      ASSERT(roots[0] > invalid);
      ASSERT(roots[1] > invalid);
    } else {
      // Compute residuals
      // Get the largest coefficient and largest root
      // to compute the relative error
      auto const largest_coeff =
          um2::max(um2::abs(a), um2::max(um2::abs(b), um2::abs(c)));
      if (roots[0] < invalid) {
        auto const rel_eps = eps * um2::max(um2::abs(roots[0]), largest_coeff);
        T const res1 = c + roots[0] * (b + roots[0] * a);
        ASSERT_NEAR(res1, 0, rel_eps);
      }
      if (roots[1] < invalid) {
        auto const rel_eps = eps * um2::max(um2::abs(roots[1]), largest_coeff);
        T const res2 = c + roots[1] * (b + roots[1] * a);
        ASSERT_NEAR(res2, 0, rel_eps);
      }
    }
  }
}

#if UM2_USE_CUDA
template <class T>
MAKE_CUDA_KERNEL(degenerate_cases, T);

template <class T>
MAKE_CUDA_KERNEL(fibonacci, T);

template <class T>
MAKE_CUDA_KERNEL(numerical, T);

#endif

template <class T>
TEST_SUITE(quadratic_equation)
{
  TEST_HOSTDEV(degenerate_cases, T);
  TEST_HOSTDEV(fibonacci, T);
  TEST_HOSTDEV(numerical, T);
  TEST(random_coeff<T>);
}

auto
main() -> int
{
  RUN_SUITE(quadratic_equation<float>);
  RUN_SUITE(quadratic_equation<double>);
  return 0;
}
