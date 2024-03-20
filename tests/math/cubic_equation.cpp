#include <um2/math/cubic_equation.hpp>

#include "../test_macros.hpp"

HOSTDEV
TEST_CASE(cubic)
{
  // Reference solutions from Wolfram Alpha
  auto constexpr eps = castIfNot<Float>(1e-4);

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
  ASSERT_NEAR(roots[0], castIfNot<Float>(-0.18522), eps);

  // disc < 0
  // a: 576 b: -936 c: 402.4 d: -35.4
  a = 576;
  b = -936;
  c = castIfNot<Float>(402.4);
  d = castIfNot<Float>(-35.4);
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<Float>(0.97026), eps);
  ASSERT_NEAR(roots[1], castIfNot<Float>(0.53673), eps);
  ASSERT_NEAR(roots[2], castIfNot<Float>(0.118016), eps);

  // p = 0
  // a: 1, b: 3, c: 3, d: -10
  a = 1; 
  b = 3;
  c = 3; 
  d = -10;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<Float>(1.2240), eps);

  // p approx 0
  // a: 1, b: 3.000001, c: 3, d: -10
  b = castIfNot<Float>(3.0001);
  c = 3;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<Float>(1.22398), eps);

  // disc = 0
  // a: 64, b: -120, c: 72, d: -14
  a = 64;
  b = -120;
  c = 72;
  d = -14;
  roots = um2::solveCubic(a, b, c, d);
  ASSERT_NEAR(roots[0], castIfNot<Float>(0.875), eps);
  ASSERT_NEAR(roots[1], castIfNot<Float>(0.5), eps);
}

TEST_SUITE(cubic_equation)
{
  TEST_HOSTDEV(cubic);
}

auto
main() -> int
{
  RUN_SUITE(cubic_equation);
  return 0;
}
