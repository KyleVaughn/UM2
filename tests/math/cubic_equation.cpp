#include <um2/config.hpp>
#include <um2/math/cubic_equation.hpp>

#include "../test_macros.hpp"

#include <cstdint>
#include <random>

template <class T>
HOSTDEV
TEST_CASE(degenerate_cases)
{
  T eps = 0;
  if constexpr (std::is_same_v<T, float>) {
    eps = 1e-3F;
  } else {
    eps = 1e-6;
  }
  auto constexpr invalid = castIfNot<T>(1e15);

  // a = 0
  {
    T const a = 0;
    T const b = 4;
    T const c = -2;
    T const d = -10;
    auto const roots = um2::solveCubic(a, b, c, d);
    auto const r0 = castIfNot<T>(-1.35078105935821);
    auto const r1 = castIfNot<T>(1.85078105935821);
    ASSERT_NEAR(roots[0], r0, eps);
    ASSERT_NEAR(roots[1], r1, eps);
    ASSERT(roots[2] > invalid);
  }

  // d == 0
  {
    T const a = 4;
    T const b = -2;
    T const c = -10;
    T const d = 0;
    auto const roots = um2::solveCubic(a, b, c, d);
    auto const r0 = castIfNot<T>(-1.35078105935821);
    auto const r2 = castIfNot<T>(1.85078105935821);
    ASSERT_NEAR(roots[0], r0, eps);
    ASSERT_NEAR(roots[1], 0, eps);
    ASSERT_NEAR(roots[2], r2, eps);
  }

  // q = 0
  // a: 32 b: -48 c: 44 d: -14
  T a = 32;
  T b = -48;
  T c = 44;
  T d = -14;
  auto roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<T>(0.5), eps);

  // disc > 0
  // a: 32 b: -48 c: 44 d: 10
  d = 10;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<T>(-0.185224222599), eps);

  // disc < 0
  // a: 576 b: -936 c: 402.4 d: -35.4
  a = 576;
  b = -936;
  c = castIfNot<T>(402.4);
  d = castIfNot<T>(-35.4);
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<T>(0.11801594), eps);
  ASSERT_NEAR(roots[1], castIfNot<T>(0.53672655), eps);
  ASSERT_NEAR(roots[2], castIfNot<T>(0.97025750), eps);

  // p = 0
  // a: 1, b: 3, c: 3, d: -10
  a = 1;
  b = 3;
  c = 3;
  d = -10;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<T>(1.2239800), eps);

  // p approx 0
  // a: 1, b: 3.000001, c: 3, d: -10
  b = castIfNot<T>(3.0001);
  c = 3;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<T>(1.22396999), eps);

  // disc = 0
  // a: 64, b: -120, c: 72, d: -14
  a = 64;
  b = -120;
  c = 72;
  d = -14;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<T>(0.5), eps);
  ASSERT_NEAR(roots[1], castIfNot<T>(0.5), eps);
  ASSERT_NEAR(roots[2], castIfNot<T>(0.875), eps);
}

template <class T>
TEST_CASE(random_coeff)
{
  T eps = 0;
  if constexpr (std::is_same_v<T, float>) {
    eps = 1e-3F;
  } else {
    eps = 1e-6;
  }
  auto constexpr invalid = castIfNot<T>(1e15);
  Int constexpr num_random_tests = 100000;
  // Check for random values
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dis(-1000, 1000);
  for (Int i = 0; i < num_random_tests; ++i) {
    T const a = dis(gen);
    T const b = dis(gen);
    T const c = dis(gen);
    T const d = dis(gen);
    auto const max_ab = um2::max(um2::abs(a), um2::abs(b));
    auto const max_cd = um2::max(um2::abs(c), um2::abs(d));
    auto const largest_coeff = um2::max(max_ab, max_cd);
    auto roots = um2::solveCubic(a, b, c, d);
    for (Int j = 0; j < 3; ++j) {
      // Compute residuals
      if (roots[j] < invalid) {
        T const x3 = um2::abs(roots[j] * roots[j] * roots[j]);
        T const rel_eps = eps * um2::max(x3, largest_coeff);
        T const res = d + roots[j] * (c + roots[j] * (b + roots[j] * a));
        // Ensure that the residual is small relative to the roots
        ASSERT_NEAR(res, 0, rel_eps);
      }
    }
  }
}

#if UM2_USE_CUDA
template <class T>
MAKE_CUDA_KERNEL(degenerate_cases, T);
#endif

template <class T>
TEST_SUITE(cubic_equation)
{
  TEST_HOSTDEV(degenerate_cases, T);
  TEST(random_coeff<T>);
}

auto
main() -> int
{
  RUN_SUITE(cubic_equation<float>);
  RUN_SUITE(cubic_equation<double>);
  return 0;
}
