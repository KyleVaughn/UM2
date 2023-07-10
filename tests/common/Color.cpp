#include <um2/common/Color.hpp>

#include "../test_macros.hpp"

HOSTDEV
TEST_CASE(color_default_constructor)
{
  um2::Color black;
  ASSERT(black.r() == 0);
  ASSERT(black.g() == 0);
  ASSERT(black.b() == 0);
  ASSERT(black.a() == 255);
}
MAKE_CUDA_KERNEL(color_default_constructor)

HOSTDEV
TEST_CASE(color_int_constructor)
{
  um2::Color black(0, 0, 0, 255);
  um2::Color white(255, 255, 255, 255);
  um2::Color transparent_red(255, 0, 0, 0);
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
  um2::Color black(0.0, 0.0, 0.0, 1.0);
  um2::Color white(1.0, 1.0, 1.0, 1.0);
  um2::Color transparent_red(1.0, 0.0, 0.0, 0.0);
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
  um2::Color black(0.0, 0.0, 0.0, 1.0);
  um2::Color white(1.0, 1.0, 1.0, 1.0);
  um2::Color transparent_red(1.0, 0.0, 0.0, 0.0);
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
TEST_CASE(toColor)
{
  um2::Color aliceblue = um2::toColor(um2::String("aliceblue"));
  um2::Color aliceblue_ref(240, 248, 255, 255);
  ASSERT(aliceblue.r() == aliceblue_ref.r());
  ASSERT(aliceblue.g() == aliceblue_ref.g());
  ASSERT(aliceblue.b() == aliceblue_ref.b());
  ASSERT(aliceblue.a() == aliceblue_ref.a());
  um2::Color yellow = um2::toColor(um2::String("yellow"));
  um2::Color yellow_ref(255, 255, 0, 255);
  ASSERT(yellow.r() == yellow_ref.r());
  ASSERT(yellow.g() == yellow_ref.g());
  ASSERT(yellow.b() == yellow_ref.b());
  ASSERT(yellow.a() == yellow_ref.a());
}
MAKE_CUDA_KERNEL(toColor)

HOSTDEV
TEST_CASE(color_string_constructor)
{
  um2::Color aliceblue(um2::String("aliceblue"));
  um2::Color aliceblue_ref(240, 248, 255, 255);
  ASSERT(aliceblue.r() == aliceblue_ref.r());
  ASSERT(aliceblue.g() == aliceblue_ref.g());
  ASSERT(aliceblue.b() == aliceblue_ref.b());
  ASSERT(aliceblue.a() == aliceblue_ref.a());
  um2::Color yellow("yellow");
  um2::Color yellow_ref(255, 255, 0, 255);
  ASSERT(yellow.r() == yellow_ref.r());
  ASSERT(yellow.g() == yellow_ref.g());
  ASSERT(yellow.b() == yellow_ref.b());
  ASSERT(yellow.a() == yellow_ref.a());
}
MAKE_CUDA_KERNEL(color_string_constructor)

HOSTDEV
TEST_CASE(color_comparison)
{
  um2::Color black(0, 0, 0, 255);
  um2::Color white(255, 255, 255, 255);
  um2::Color transparent_red(255, 0, 0, 0);
  ASSERT(black == black);
  ASSERT(white == white);
  ASSERT(transparent_red == transparent_red);
  ASSERT(black != white);
  ASSERT(black != transparent_red);
  ASSERT(white != transparent_red);
}
MAKE_CUDA_KERNEL(color_comparison)

TEST_SUITE(color)
{
  TEST_HOSTDEV(color_default_constructor);
  TEST_HOSTDEV(color_int_constructor);
  TEST_HOSTDEV(color_float_constructor);
  TEST_HOSTDEV(color_double_constructor);
  TEST_HOSTDEV(toColor);
  TEST_HOSTDEV(color_string_constructor);
  TEST_HOSTDEV(color_comparison);
}

auto
main() -> int
{
  RUN_SUITE(color);
  return 0;
}
