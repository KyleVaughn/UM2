#include <um2/stdlib/math.hpp>

#include "../test_macros.hpp"

// NOLINTBEGIN(cert-dcl03-c,misc-static-assert); justification: compiler-dependent

//=============================================================================
// abs
//=============================================================================

HOSTDEV
TEST_CASE(abs_int)
{
  ASSERT(um2::abs(-1) == 1);
  ASSERT(um2::abs(0) == 0);
  ASSERT(um2::abs(1) == 1);
}

HOSTDEV
TEST_CASE(abs_float)
{
  ASSERT_NEAR(um2::abs(-1.0F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::abs(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::abs(1.0F), 1.0F, 1e-6F);
}

//=============================================================================
// atan
//=============================================================================

HOSTDEV
TEST_CASE(atan_float)
{
  ASSERT_NEAR(um2::atan(-1.0F), -um2::pi_4<float>, 1e-6F);
  ASSERT_NEAR(um2::atan(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::atan(1.0F), um2::pi_4<float>, 1e-6F);
}

//=============================================================================
// atanh
//=============================================================================

HOSTDEV
TEST_CASE(atanh_float)
{
  ASSERT_NEAR(um2::atanh(0.5F), 0.549306F, 1e-3F);
  ASSERT_NEAR(um2::atanh(0.0F), 0.0F, 1e-3F);
}

//=============================================================================
// cbrt
//=============================================================================

HOSTDEV
TEST_CASE(cbrt_float)
{
  ASSERT_NEAR(um2::cbrt(8.0F), 2.0F, 1e-6F);
  ASSERT_NEAR(um2::cbrt(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::cbrt(1.0F), 1.0F, 1e-6F);
}

//=============================================================================
// ceil/floor
//=============================================================================

HOSTDEV
TEST_CASE(ceilfloor_float)
{
  ASSERT_NEAR(um2::ceil(1.1F), 2.0F, 1e-6F);
  ASSERT_NEAR(um2::ceil(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::ceil(-1.1F), -1.0F, 1e-6F);

  ASSERT_NEAR(um2::floor(1.1F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::floor(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::floor(-1.1F), -2.0F, 1e-6F);
}

//=============================================================================
// cos
//=============================================================================

HOSTDEV
TEST_CASE(cos_float)
{
  ASSERT_NEAR(um2::cos(0.0F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::cos(um2::pi_2<float>), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::cos(um2::pi<float>), -1.0F, 1e-6F);
}

//=============================================================================
// exp
//=============================================================================

HOSTDEV
TEST_CASE(exp_float)
{
  ASSERT_NEAR(um2::exp(0.0F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::exp(1.0F), 2.718281828459045F, 1e-6F);
}

//=============================================================================
// sin
//=============================================================================

HOSTDEV
TEST_CASE(sin_float)
{
  ASSERT_NEAR(um2::sin(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::sin(um2::pi_2<float>), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::sin(um2::pi<float>), 0.0F, 1e-6F);
}

//=============================================================================
// sqrt
//=============================================================================

HOSTDEV
TEST_CASE(sqrt_float)
{
  ASSERT_NEAR(um2::sqrt(4.0F), 2.0F, 1e-6F);
  ASSERT_NEAR(um2::sqrt(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::sqrt(1.0F), 1.0F, 1e-6F);
}

#if UM2_USE_CUDA
MAKE_CUDA_KERNEL(abs_int);
MAKE_CUDA_KERNEL(abs_float);
MAKE_CUDA_KERNEL(atan_float);
MAKE_CUDA_KERNEL(atanh_float);
MAKE_CUDA_KERNEL(cbrt_float);
MAKE_CUDA_KERNEL(ceilfloor_float);
MAKE_CUDA_KERNEL(cos_float);
MAKE_CUDA_KERNEL(exp_float);
MAKE_CUDA_KERNEL(sin_float);
MAKE_CUDA_KERNEL(sqrt_float);
#endif

TEST_SUITE(abs)
{
  TEST_HOSTDEV(abs_int);
  TEST_HOSTDEV(abs_float);
}

TEST_SUITE(atan) { TEST_HOSTDEV(atan_float); }

TEST_SUITE(atanh) { TEST_HOSTDEV(atanh_float); }

TEST_SUITE(cbrt) { TEST_HOSTDEV(cbrt_float); }

TEST_SUITE(ceilfloor) { TEST_HOSTDEV(ceilfloor_float); }

TEST_SUITE(cos) { TEST_HOSTDEV(cos_float); }

TEST_SUITE(exp) { TEST_HOSTDEV(exp_float); }

TEST_SUITE(sin) { TEST_HOSTDEV(sin_float); }

TEST_SUITE(sqrt) { TEST_HOSTDEV(sqrt_float); }

auto
main() -> int
{
  RUN_SUITE(abs);
  RUN_SUITE(atan);
  RUN_SUITE(atanh);
  RUN_SUITE(cbrt);
  RUN_SUITE(ceilfloor);
  RUN_SUITE(cos);
  RUN_SUITE(exp);
  RUN_SUITE(sin);
  RUN_SUITE(sqrt);
  return 0;
}

// NOLINTEND(cert-dcl03-c,misc-static-assert)
