#include <um2/common/Color.hpp>

#include "../test_macros.hpp"

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
TEST_CASE(toColor)
{
  um2::Color const aliceblue = um2::toColor(um2::ShortString("aliceblue"));
  um2::Color const aliceblue_ref(240, 248, 255, 255);
  ASSERT(aliceblue.r() == aliceblue_ref.r());
  ASSERT(aliceblue.g() == aliceblue_ref.g());
  ASSERT(aliceblue.b() == aliceblue_ref.b());
  ASSERT(aliceblue.a() == aliceblue_ref.a());
  um2::Color const yellow = um2::toColor(um2::ShortString("yellow"));
  um2::Color const yellow_ref(255, 255, 0, 255);
  ASSERT(yellow.r() == yellow_ref.r());
  ASSERT(yellow.g() == yellow_ref.g());
  ASSERT(yellow.b() == yellow_ref.b());
  ASSERT(yellow.a() == yellow_ref.a());
}
MAKE_CUDA_KERNEL(toColor)

HOSTDEV
TEST_CASE(color_string_constructor)
{
  um2::Color const aliceblue(um2::ShortString("aliceblue"));
  um2::Color const aliceblue_ref(240, 248, 255, 255);
  ASSERT(aliceblue.r() == aliceblue_ref.r());
  ASSERT(aliceblue.g() == aliceblue_ref.g());
  ASSERT(aliceblue.b() == aliceblue_ref.b());
  ASSERT(aliceblue.a() == aliceblue_ref.a());
  um2::Color const yellow("yellow");
  um2::Color const yellow_ref(255, 255, 0, 255);
  ASSERT(yellow.r() == yellow_ref.r());
  ASSERT(yellow.g() == yellow_ref.g());
  ASSERT(yellow.b() == yellow_ref.b());
  ASSERT(yellow.a() == yellow_ref.a());
}
MAKE_CUDA_KERNEL(color_string_constructor)

HOSTDEV
TEST_CASE(color_comparison)
{
  um2::Color const black(0, 0, 0, 255);
  um2::Color const white(255, 255, 255, 255);
  um2::Color const transparent_red(255, 0, 0, 0);
  ASSERT(black == black);
  ASSERT(white == white);
  ASSERT(transparent_red == transparent_red);
  ASSERT(black != white);
  ASSERT(black != transparent_red);
  ASSERT(white != transparent_red);
}
MAKE_CUDA_KERNEL(color_comparison)

TEST_CASE(colors_enum)
{
  um2::Color const black(0, 0, 0);
  ASSERT(black.rep.u32 == static_cast<uint32_t>(um2::Colors::Black));
  um2::Color const white(255, 255, 255);
  ASSERT(white.rep.u32 == static_cast<uint32_t>(um2::Colors::White));
  um2::Color const red(255, 0, 0);
  ASSERT(red.rep.u32 == static_cast<uint32_t>(um2::Colors::Red));
  um2::Color const green(0, 255, 0);
  ASSERT(green.rep.u32 == static_cast<uint32_t>(um2::Colors::Green));
  um2::Color const blue(0, 0, 255);
  ASSERT(blue.rep.u32 == static_cast<uint32_t>(um2::Colors::Blue));
}

// NOLINTEND(cert-dcl03-c,misc-static-assert)

TEST_SUITE(color)
{
  TEST_HOSTDEV(color_default_constructor);
  TEST_HOSTDEV(color_int_constructor);
  TEST_HOSTDEV(color_float_constructor);
  TEST_HOSTDEV(color_double_constructor);
  TEST_HOSTDEV(toColor);
  TEST_HOSTDEV(color_string_constructor);
  TEST_HOSTDEV(color_comparison);
  TEST(colors_enum);
}

auto
main() -> int
{
  RUN_SUITE(color);
  return 0;
}
