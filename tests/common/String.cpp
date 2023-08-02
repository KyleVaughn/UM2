#include <um2/common/String.hpp>

#include "../test_macros.hpp"

// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------

HOSTDEV
TEST_CASE(default_constructor)
{
  um2::String s;
  static_assert(sizeof(s) == 24);
  assert(s.size() == 0);
  assert(s.capacity() == 22);
  for (int i = 0; i < 22; ++i) {
    // cppcheck-suppress assertWithSideEffect
    assert(s.data()[i] == '\0');
  }
}
MAKE_CUDA_KERNEL(default_constructor)

HOSTDEV
TEST_CASE(const_char_array_constructor)
{
  um2::String s("hello");
  assert(s.size() == 5);
  assert(s.capacity() == 22);
  assert(!s.isLong());
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[0] == 'h');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[1] == 'e');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[2] == 'l');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[3] == 'l');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[4] == 'o');

  um2::String s2("This string will be too long to fit in the small string optimization");
  assert(s2.size() == 68);
  assert(s2.capacity() == 68);
  assert(s2.isLong());
  // cppcheck-suppress assertWithSideEffect
  assert(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(const_char_array_constructor);

HOSTDEV
TEST_CASE(copy_constructor)
{
  um2::String s0("hello");
  assert(!s0.isLong());
  um2::String s(s0);
  assert(s.size() == 5);
  assert(s.capacity() == 22);
  assert(!s.isLong());
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[0] == 'h');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[1] == 'e');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[2] == 'l');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[3] == 'l');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[4] == 'o');
  // Ensure that s0 is not modified
  s0.data()[0] = 'a';
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[0] == 'h');

  um2::String s1("This string will be too long to fit in the small string optimization");
  assert(s1.isLong());
  um2::String s2(s1);
  assert(s2.size() == 68);
  assert(s2.capacity() == 68);
  assert(s2.isLong());
  // cppcheck-suppress assertWithSideEffect
  assert(s2.data()[0] == 'T');
  // Check that s1 is not modified
  s1.data()[0] = 'a';
  // cppcheck-suppress assertWithSideEffect
  assert(s2.data()[0] == 'T');
  // NOLINTNEXTLINE
}
MAKE_CUDA_KERNEL(copy_constructor);

HOSTDEV
TEST_CASE(move_constructor)
{
  um2::String s1("This string will be too long to fit in the small string optimization");
  assert(s1.isLong());
  um2::String s2(um2::move(s1));
  assert(s2.size() == 68);
  assert(s2.capacity() == 68);
  assert(s2.isLong());
  // cppcheck-suppress assertWithSideEffect
  assert(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(move_constructor);

HOSTDEV
TEST_CASE(const_char_constructor)
{
  const char * input = "Short String";
  um2::String s(input);
  ASSERT(s.size() == 12);
  ASSERT(s.capacity() == 22);
  ASSERT(!s.isLong());
  // cppcheck-suppress assertWithSideEffect
  ASSERT(s.data()[0] == 'S');
  const char * input2 =
      "This string will be too long to fit in the small string optimization";
  um2::String s2(input2);
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.isLong());
  // cppcheck-suppress assertWithSideEffect
  ASSERT(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(const_char_constructor);

// -----------------------------------------------------------------------------
// Operators
// -----------------------------------------------------------------------------

HOSTDEV
TEST_CASE(assign_operator)
{
  um2::String s0("hello");
  assert(!s0.isLong());
  um2::String s("This string will be too long to fit in the small string optimization");
  s = s0;
  assert(s.size() == 5);
  assert(s.capacity() == 22);
  assert(!s.isLong());
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[0] == 'h');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[1] == 'e');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[2] == 'l');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[3] == 'l');
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[4] == 'o');
  // Ensure that s0 is not modified
  // cppcheck-suppress unreadVariable
  s0.data()[0] = 'a';
  // cppcheck-suppress assertWithSideEffect
  assert(s.data()[0] == 'h');

  um2::String s1("This string will be too long to fit in the small string optimization");
  assert(s1.isLong());
  um2::String s2;
  s2 = s1;
  assert(s2.size() == 68);
  assert(s2.capacity() == 68);
  assert(s2.isLong());
  // cppcheck-suppress assertWithSideEffect
  assert(s2.data()[0] == 'T');
  // Check that s1 is not modified
  // cppcheck-suppress unreadVariable
  s1.data()[0] = 'a';
  // cppcheck-suppress assertWithSideEffect
  assert(s2.data()[0] == 'T');
  // NOLINTNEXTLINE
}
MAKE_CUDA_KERNEL(assign_operator);

HOSTDEV
TEST_CASE(equals_operator)
{
  um2::String const s0("hello");
  um2::String const s1("helo");
  um2::String const s2("hello");
  ASSERT(s0 == s0);
  ASSERT(s0 == s2);
  ASSERT(s0 != s1);
}
MAKE_CUDA_KERNEL(equals_operator);

HOSTDEV
TEST_CASE(comparison)
{
  ASSERT(um2::String("Ant") < um2::String("Zebra"));
  ASSERT(um2::String("Zebra") > um2::String("Ant"));
  ASSERT(um2::String("Zebra") <= um2::String("ant"));
  ASSERT(um2::String("ant") >= um2::String("Zebra"));
  ASSERT(um2::String("Zebra") <= um2::String("Zebra"));
  ASSERT(um2::String("Zebra") >= um2::String("Zebra"));
}
MAKE_CUDA_KERNEL(comparison);
//
//// -----------------------------------------------------------------------------
//// Methods
//// -----------------------------------------------------------------------------
//
// HOSTDEV TEST_CASE(contains)
//{
//  um2::String s("hello");
//  EXPECT_TRUE(s.contains('h'));
//  EXPECT_TRUE(s.contains('e'));
//  EXPECT_TRUE(s.contains('l'));
//  EXPECT_TRUE(s.contains('o'));
//  EXPECT_FALSE(s.contains('a'));
//  EXPECT_FALSE(s.contains('b'));
//}
// MAKE_CUDA_KERNEL(contains);
//
HOSTDEV
TEST_CASE(starts_ends_with)
{
  um2::String const s("hello");
  ASSERT(s.starts_with(um2::String("he")));
  ASSERT(!s.starts_with(um2::String("eh")));
  ASSERT(s.ends_with("lo"));
  ASSERT(!s.ends_with("ol"));
}
MAKE_CUDA_KERNEL(starts_ends_with);
//
// TEST_CASE(toString)
//{
//  um2::String s("hello");
//  EXPECT_EQ(toString(s), std::string("hello"));
//}

TEST_SUITE(String)
{
  // Constructors
  TEST_HOSTDEV(default_constructor)
  TEST_HOSTDEV(const_char_array_constructor)
  TEST_HOSTDEV(copy_constructor)
  TEST_HOSTDEV(move_constructor)
  TEST_HOSTDEV(const_char_constructor)
  // Operators
  TEST_HOSTDEV(assign_operator)
  TEST_HOSTDEV(equals_operator)
  TEST_HOSTDEV(comparison)
  //  // Methods
  //  TEST_HOSTDEV(contains)
  TEST(starts_ends_with)
  //  TEST(toString)
}

auto
main() -> int
{
  RUN_SUITE(String)
  return 0;
}
