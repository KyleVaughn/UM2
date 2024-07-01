#include <um2/common/color.hpp>

#include <um2/config.hpp>

#include "../test_macros.hpp"

HOSTDEV
TEST_CASE(color_default_constructor)
{
  um2::Color constexpr black;
  static_assert(sizeof(um2::Color) == 4, "Color should be 4 bytes");
  static_assert(black.r() == 0);
  static_assert(black.g() == 0);
  static_assert(black.b() == 0);
  static_assert(black.a() == 255);
}
MAKE_CUDA_KERNEL(color_default_constructor)

HOSTDEV
TEST_CASE(color_int_constructor)
{
  um2::Color constexpr black(0, 0, 0, 255);
  um2::Color constexpr white(255, 255, 255, 255);
  um2::Color constexpr transparent_red(255, 0, 0, 0);
  static_assert(black.r() == 0);
  static_assert(black.g() == 0);
  static_assert(black.b() == 0);
  static_assert(black.a() == 255);
  static_assert(white.r() == 255);
  static_assert(white.g() == 255);
  static_assert(white.b() == 255);
  static_assert(white.a() == 255);
  static_assert(transparent_red.r() == 255);
  static_assert(transparent_red.g() == 0);
  static_assert(transparent_red.b() == 0);
  static_assert(transparent_red.a() == 0);
}
MAKE_CUDA_KERNEL(color_int_constructor)

HOSTDEV
TEST_CASE(color_float_constructor)
{
  um2::Color constexpr black(0.0F, 0.0F, 0.0F, 1.0F);
  um2::Color constexpr white(1.0F, 1.0F, 1.0F, 1.0F);
  um2::Color constexpr transparent_red(1.0F, 0.0F, 0.0F, 0.0F);
  static_assert(black.r() == 0);
  static_assert(black.g() == 0);
  static_assert(black.b() == 0);
  static_assert(black.a() == 255);
  static_assert(white.r() == 255);
  static_assert(white.g() == 255);
  static_assert(white.b() == 255);
  static_assert(white.a() == 255);
  static_assert(transparent_red.r() == 255);
  static_assert(transparent_red.g() == 0);
  static_assert(transparent_red.b() == 0);
  static_assert(transparent_red.a() == 0);
}
MAKE_CUDA_KERNEL(color_float_constructor)

HOSTDEV
TEST_CASE(color_double_constructor)
{
  um2::Color constexpr black(0.0, 0.0, 0.0, 1.0);
  um2::Color constexpr white(1.0, 1.0, 1.0, 1.0);
  um2::Color constexpr transparent_red(1.0, 0.0, 0.0, 0.0);
  static_assert(black.r() == 0);
  static_assert(black.g() == 0);
  static_assert(black.b() == 0);
  static_assert(black.a() == 255);
  static_assert(white.r() == 255);
  static_assert(white.g() == 255);
  static_assert(white.b() == 255);
  static_assert(white.a() == 255);
  static_assert(transparent_red.r() == 255);
  static_assert(transparent_red.g() == 0);
  static_assert(transparent_red.b() == 0);
  static_assert(transparent_red.a() == 0);
}
MAKE_CUDA_KERNEL(color_double_constructor)

HOSTDEV
TEST_CASE(color_comparison)
{
  um2::Color constexpr black(0, 0, 0, 255);
  um2::Color constexpr white(255, 255, 255, 255);
  um2::Color constexpr transparent_red(255, 0, 0, 0);
  static_assert(black == um2::black);
  static_assert(white == um2::white);
  static_assert(transparent_red == transparent_red);
  static_assert(black != white);
  static_assert(black != transparent_red);
  static_assert(white != transparent_red);
}
MAKE_CUDA_KERNEL(color_comparison)

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
