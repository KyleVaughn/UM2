#include <um2/math/cubic_equation.hpp>

#include "../test_macros.hpp"

#include <random>

HOSTDEV
TEST_CASE(degenerate_cases)
{
#if UM2_ENABLE_FLOAT64
  auto constexpr eps = 1e-6;
#else
  auto constexpr eps = 1e-3F;
#endif
  auto constexpr invalid = castIfNot<Float>(1e15);

  // a = 0
  {
    Float const a = 0;
    Float const b = 4;
    Float const c = -2;
    Float const d = -10;
    auto const roots = um2::solveCubic(a, b, c, d);
    auto const r0 = castIfNot<Float>(-1.35078105935821);
    auto const r1 = castIfNot<Float>(1.85078105935821);
    ASSERT_NEAR(roots[0], r0, eps);
    ASSERT_NEAR(roots[1], r1, eps);
    ASSERT(roots[2] > invalid);
  }

  // d == 0
  {
    Float const a = 4;
    Float const b = -2;
    Float const c = -10;
    Float const d = 0;
    auto const roots = um2::solveCubic(a, b, c, d);
    auto const r0 = castIfNot<Float>(-1.35078105935821);
    auto const r2 = castIfNot<Float>(1.85078105935821);
    ASSERT_NEAR(roots[0], r0, eps);
    ASSERT_NEAR(roots[1], 0, eps);
    ASSERT_NEAR(roots[2], r2, eps);
  }

  // q = 0
  // a: 32 b: -48 c: 44 d: -14
  Float a = 32;
  Float b = -48;
  Float c = 44;
  Float d = -14;
  auto roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<Float>(0.5), eps);

  // disc > 0
  // a: 32 b: -48 c: 44 d: 10
  d = 10;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<Float>(-0.185224222599), eps);

  // disc < 0
  // a: 576 b: -936 c: 402.4 d: -35.4
  a = 576;
  b = -936;
  c = castIfNot<Float>(402.4);
  d = castIfNot<Float>(-35.4);
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<Float>(0.11801594), eps);
  ASSERT_NEAR(roots[1], castIfNot<Float>(0.53672655), eps);
  ASSERT_NEAR(roots[2], castIfNot<Float>(0.97025750), eps);

  // p = 0
  // a: 1, b: 3, c: 3, d: -10
  a = 1;
  b = 3;
  c = 3;
  d = -10;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<Float>(1.2239800), eps);

  // p approx 0
  // a: 1, b: 3.000001, c: 3, d: -10
  b = castIfNot<Float>(3.0001);
  c = 3;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<Float>(1.22396999), eps);

  // disc = 0
  // a: 64, b: -120, c: 72, d: -14
  a = 64;
  b = -120;
  c = 72;
  d = -14;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<Float>(0.5), eps);
  ASSERT_NEAR(roots[1], castIfNot<Float>(0.5), eps);
  ASSERT_NEAR(roots[2], castIfNot<Float>(0.875), eps);
}

HOSTDEV
TEST_CASE(random_coeff)
{
#if UM2_ENABLE_FLOAT64
  auto constexpr eps = 1e-6; 
#else
  auto constexpr eps = 1e-4F;
#endif
  auto constexpr invalid = castIfNot<Float>(1e15);
  Int constexpr num_random_tests = 100000;
  // Check for random values
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Float> dis(-1000, 1000);
  for (Int i = 0; i < num_random_tests; ++i) {
    Float const a = dis(gen);
    Float const b = dis(gen);
    Float const c = dis(gen);
    Float const d = dis(gen);
    auto const max_ab = um2::max(um2::abs(a), um2::abs(b));
    auto const max_cd = um2::max(um2::abs(c), um2::abs(d));
    auto const largest_coeff = um2::max(max_ab, max_cd);
    auto roots = um2::solveCubic(a, b, c, d);
    for (Int j = 0; j < 3; ++j) {
      // Compute residuals
      if (roots[j] < invalid) {
        Float const x3 = um2::abs(roots[j] * roots[j] * roots[j]);
        Float const rel_eps = eps * um2::max(x3, largest_coeff);
        Float const res = d + roots[j] * (c + roots[j] * (b + roots[j] * a));
        // Ensure that the residual is small relative to the roots
        ASSERT_NEAR(res, 0, rel_eps);
      }
    }
  }
}

TEST_SUITE(cubic_equation)
{
  TEST_HOSTDEV(degenerate_cases);
  TEST_HOSTDEV(random_coeff);
}

auto
main() -> int
{
  RUN_SUITE(cubic_equation);
  return 0;
}
