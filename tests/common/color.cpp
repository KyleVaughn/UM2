#include "../test_framework.hpp"
#include <um2/common/color.hpp>

UM2_HOSTDEV
TEST_CASE(color_default_constructor)
{
  um2::Color black;
  EXPECT_EQ(black.r, 0);
  EXPECT_EQ(black.g, 0);
  EXPECT_EQ(black.b, 0);
  EXPECT_EQ(black.a, 255);
}
MAKE_CUDA_KERNEL(color_default_constructor)

UM2_HOSTDEV
TEST_CASE(color_int_constructor)
{
  um2::Color black(0, 0, 0, 255);
  um2::Color white(255, 255, 255, 255);
  um2::Color transparent_red(255, 0, 0, 0);
  EXPECT_EQ(black.r, 0);
  EXPECT_EQ(black.g, 0);
  EXPECT_EQ(black.b, 0);
  EXPECT_EQ(black.a, 255);
  EXPECT_EQ(white.r, 255);
  EXPECT_EQ(white.g, 255);
  EXPECT_EQ(white.b, 255);
  EXPECT_EQ(white.a, 255);
  EXPECT_EQ(transparent_red.r, 255);
  EXPECT_EQ(transparent_red.g, 0);
  EXPECT_EQ(transparent_red.b, 0);
  EXPECT_EQ(transparent_red.a, 0);
}
MAKE_CUDA_KERNEL(color_int_constructor)

UM2_HOSTDEV
TEST_CASE(color_float_constructor)
{
  um2::Color black(0.0, 0.0, 0.0, 1.0);
  um2::Color white(1.0, 1.0, 1.0, 1.0);
  um2::Color transparent_red(1.0, 0.0, 0.0, 0.0);
  EXPECT_EQ(black.r, 0);
  EXPECT_EQ(black.g, 0);
  EXPECT_EQ(black.b, 0);
  EXPECT_EQ(black.a, 255);
  EXPECT_EQ(white.r, 255);
  EXPECT_EQ(white.g, 255);
  EXPECT_EQ(white.b, 255);
  EXPECT_EQ(white.a, 255);
  EXPECT_EQ(transparent_red.r, 255);
  EXPECT_EQ(transparent_red.g, 0);
  EXPECT_EQ(transparent_red.b, 0);
  EXPECT_EQ(transparent_red.a, 0);
}
MAKE_CUDA_KERNEL(color_float_constructor)

UM2_HOSTDEV
TEST_CASE(color_double_constructor)
{
  um2::Color black(0.0, 0.0, 0.0, 1.0);
  um2::Color white(1.0, 1.0, 1.0, 1.0);
  um2::Color transparent_red(1.0, 0.0, 0.0, 0.0);
  EXPECT_EQ(black.r, 0);
  EXPECT_EQ(black.g, 0);
  EXPECT_EQ(black.b, 0);
  EXPECT_EQ(black.a, 255);
  EXPECT_EQ(white.r, 255);
  EXPECT_EQ(white.g, 255);
  EXPECT_EQ(white.b, 255);
  EXPECT_EQ(white.a, 255);
  EXPECT_EQ(transparent_red.r, 255);
  EXPECT_EQ(transparent_red.g, 0);
  EXPECT_EQ(transparent_red.b, 0);
  EXPECT_EQ(transparent_red.a, 0);
}
MAKE_CUDA_KERNEL(color_double_constructor)

UM2_HOSTDEV
TEST_CASE(toColor)
{
  um2::Color aliceblue = um2::toColor(um2::String("aliceblue"));
  um2::Color aliceblue_ref(240, 248, 255, 255);
  um2::Color yellow = um2::toColor(um2::String("yellow"));
  um2::Color yellow_ref(255, 255, 0, 255);
  EXPECT_EQ(aliceblue.r, aliceblue_ref.r);
  EXPECT_EQ(aliceblue.g, aliceblue_ref.g);
  EXPECT_EQ(aliceblue.b, aliceblue_ref.b);
  EXPECT_EQ(aliceblue.a, aliceblue_ref.a);
  EXPECT_EQ(yellow.r, yellow_ref.r);
  EXPECT_EQ(yellow.g, yellow_ref.g);
  EXPECT_EQ(yellow.b, yellow_ref.b);
  EXPECT_EQ(yellow.a, yellow_ref.a);
}
MAKE_CUDA_KERNEL(toColor)

UM2_HOSTDEV
TEST_CASE(color_string_constructor)
{
  um2::Color aliceblue(um2::String("aliceblue"));
  um2::Color aliceblue_ref(240, 248, 255, 255);
  um2::Color yellow("yellow");
  um2::Color yellow_ref(255, 255, 0, 255);
  EXPECT_EQ(aliceblue.r, aliceblue_ref.r);
  EXPECT_EQ(aliceblue.g, aliceblue_ref.g);
  EXPECT_EQ(aliceblue.b, aliceblue_ref.b);
  EXPECT_EQ(aliceblue.a, aliceblue_ref.a);
  EXPECT_EQ(yellow.r, yellow_ref.r);
  EXPECT_EQ(yellow.g, yellow_ref.g);
  EXPECT_EQ(yellow.b, yellow_ref.b);
  EXPECT_EQ(yellow.a, yellow_ref.a);
}
MAKE_CUDA_KERNEL(color_string_constructor)

UM2_HOSTDEV
TEST_CASE(color_comparison)
{
  um2::Color black(0, 0, 0, 255);
  um2::Color white(255, 255, 255, 255);
  um2::Color transparent_red(255, 0, 0, 0);
  EXPECT_EQ(black, black);
  EXPECT_EQ(white, white);
  EXPECT_EQ(transparent_red, transparent_red);
  EXPECT_NE(black, white);
  EXPECT_NE(black, transparent_red);
  EXPECT_NE(white, transparent_red);
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
  RUN_TESTS(color);
  return 0;
}
