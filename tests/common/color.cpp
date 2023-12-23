#include <um2/common/color.hpp>

#include "../test_macros.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunreachable-code"

// clang-tidy complains that these can be static asserts, but gcc complains when they are
// NOLINTBEGIN(cert-dcl03-c,misc-static-assert) justified
HOSTDEV
TEST_CASE(color_default_constructor)
{
  um2::Color const black;
  static_assert(sizeof(um2::Color) == 4, "Color should be 4 bytes");
  ASSERT(black.r() == 0);
  ASSERT(black.g() == 0);
  ASSERT(black.b() == 0);
  ASSERT(black.a() == 255);
}
MAKE_CUDA_KERNEL(color_default_constructor)

HOSTDEV
TEST_CASE(color_int_constructor)
{
  um2::Color const black(0, 0, 0, 255);
  um2::Color const white(255, 255, 255, 255);
  um2::Color const transparent_red(255, 0, 0, 0);
  ASSERT(black.r() == 0);
  ASSERT(black.g() == 0);
  ASSERT(black.b() == 0);
  ASSERT(black.a() == 255);
  ASSERT(white.r() == 255);
  ASSERT(white.g() == 255);
  ASSERT(white.b() == 255);
  ASSERT(white.a() == 255);
  ASSERT(transparent_red.r() == 255);
  ASSERT(transparent_red.g() == 0);
  ASSERT(transparent_red.b() == 0);
  ASSERT(transparent_red.a() == 0);
}
MAKE_CUDA_KERNEL(color_int_constructor)

HOSTDEV
TEST_CASE(color_float_constructor)
{
  um2::Color const black(0.0, 0.0, 0.0, 1.0);
  um2::Color const white(1.0, 1.0, 1.0, 1.0);
  um2::Color const transparent_red(1.0, 0.0, 0.0, 0.0);
  ASSERT(black.r() == 0);
  ASSERT(black.g() == 0);
  ASSERT(black.b() == 0);
  ASSERT(black.a() == 255);
  ASSERT(white.r() == 255);
  ASSERT(white.g() == 255);
  ASSERT(white.b() == 255);
  ASSERT(white.a() == 255);
  ASSERT(transparent_red.r() == 255);
  ASSERT(transparent_red.g() == 0);
  ASSERT(transparent_red.b() == 0);
  ASSERT(transparent_red.a() == 0);
}
MAKE_CUDA_KERNEL(color_float_constructor)

HOSTDEV
TEST_CASE(color_double_constructor)
{
  um2::Color const black(0.0, 0.0, 0.0, 1.0);
  um2::Color const white(1.0, 1.0, 1.0, 1.0);
  um2::Color const transparent_red(1.0, 0.0, 0.0, 0.0);
  ASSERT(black.r() == 0);
  ASSERT(black.g() == 0);
  ASSERT(black.b() == 0);
  ASSERT(black.a() == 255);
  ASSERT(white.r() == 255);
  ASSERT(white.g() == 255);
  ASSERT(white.b() == 255);
  ASSERT(white.a() == 255);
  ASSERT(transparent_red.r() == 255);
  ASSERT(transparent_red.g() == 0);
  ASSERT(transparent_red.b() == 0);
  ASSERT(transparent_red.a() == 0);
}
MAKE_CUDA_KERNEL(color_double_constructor)

HOSTDEV
TEST_CASE(color_comparison)
{
  um2::Color const black(0, 0, 0, 255);
  um2::Color const white(255, 255, 255, 255);
  um2::Color const transparent_red(255, 0, 0, 0);
  ASSERT(black == um2::black);
  ASSERT(white == um2::white);
  ASSERT(transparent_red == transparent_red);
  ASSERT(black != white);
  ASSERT(black != transparent_red);
  ASSERT(white != transparent_red);
}
MAKE_CUDA_KERNEL(color_comparison)

// NOLINTEND(cert-dcl03-c,misc-static-assert)

TEST_SUITE(color)
{
  TEST_HOSTDEV(color_default_constructor);
  TEST_HOSTDEV(color_int_constructor);
  TEST_HOSTDEV(color_float_constructor);
  TEST_HOSTDEV(color_double_constructor);
  TEST_HOSTDEV(color_comparison);
}

auto
main() -> int
{
  RUN_SUITE(color);
  return 0;
}

#pragma GCC diagnostic pop
