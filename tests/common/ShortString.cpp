#include <um2/common/ShortString.hpp>

#include "../test_macros.hpp"

//==============================================================================
// Constructors
//==============================================================================

HOSTDEV
TEST_CASE(default_constructor)
{
  um2::ShortString s;
  static_assert(sizeof(s) == 32);
  assert(s.size() == 0);
  static_assert(s.capacity() == 31);
  for (int i = 0; i < 31; ++i) {
    // cppcheck-suppress assertWithSideEffect justification: false positive
    assert(s.data()[i] == '\0');
  }
}
MAKE_CUDA_KERNEL(default_constructor)

HOSTDEV
TEST_CASE(const_char_array_constructor)
{
  um2::ShortString s("hello");
  assert(s.size() == 5);
  static_assert(s.capacity() == 31);
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[0] == 'h');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[1] == 'e');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[2] == 'l');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[3] == 'l');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[4] == 'o');
}
MAKE_CUDA_KERNEL(const_char_array_constructor);

HOSTDEV
TEST_CASE(copy_constructor)
{
  um2::ShortString s0("hello");
  um2::ShortString s(s0);
  assert(s.size() == 5);
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[0] == 'h');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[1] == 'e');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[2] == 'l');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[3] == 'l');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[4] == 'o');
  // Ensure that s0 is not modified
  s0.data()[0] = 'a';
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[0] == 'h');
}
MAKE_CUDA_KERNEL(copy_constructor);

HOSTDEV
TEST_CASE(const_char_pointer_constructor)
{
  um2::ShortString s("hello");
  assert(s.size() == 5);
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[0] == 'h');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[1] == 'e');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[2] == 'l');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[3] == 'l');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[4] == 'o');
}
MAKE_CUDA_KERNEL(const_char_pointer_constructor);

HOSTDEV
TEST_CASE(move_constructor)
{
  um2::ShortString s1("Garbage");
  um2::ShortString s2(um2::move(s1));
  assert(s2.size() == 7);
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s2.data()[0] == 'G');
}
MAKE_CUDA_KERNEL(move_constructor);

//==============================================================================
// Operators
//==============================================================================

HOSTDEV
TEST_CASE(assign_operator)
{
  um2::ShortString s0("hello");
  um2::ShortString s("Garbage");
  s = s0;
  assert(s.size() == 5);
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[0] == 'h');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[1] == 'e');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[2] == 'l');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[3] == 'l');
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[4] == 'o');
  // Ensure that s0 is not modified
  // cppcheck-suppress unreadVariable justification: checking that s is not modified
  s0.data()[0] = 'a';
  // cppcheck-suppress assertWithSideEffect justification: false positive
  assert(s.data()[0] == 'h');
}
MAKE_CUDA_KERNEL(assign_operator);

HOSTDEV
TEST_CASE(access_operator)
{
  um2::ShortString s0("hello");
  um2::ShortString const s("Garbage");
  s0[0] = 'G';
  assert(s0[0] == 'G');
  assert(s[1] == 'a');
  assert(s[2] == 'r');
  assert(s[3] == 'b');
  assert(s[4] == 'a');
  ASSERT(s0[0] == 'G');
  ASSERT(s0[1] == 'e');
}
MAKE_CUDA_KERNEL(access_operator);

HOSTDEV
TEST_CASE(equals_operator)
{
  um2::ShortString const s0("hello");
  um2::ShortString const s1("helo");
  um2::ShortString const s2("hello");
  ASSERT(s0 == s0);
  ASSERT(s0 == s2);
  ASSERT(s0 != s1);
}
MAKE_CUDA_KERNEL(equals_operator);

TEST_CASE(equals_operator_std_string)
{
  um2::ShortString const s0("hello");
  ASSERT(s0 == std::string("hello"));
  ASSERT(!(s0 == std::string("helo")));
}

HOSTDEV
TEST_CASE(comparison)
{
// nvcc, gcc, or clang is bound to complain about the following static_asserts or lack thereof
// NOLINTBEGIN(cert-dcl03-c,misc-static-assert) justified above
#ifdef _clang_
  static_assert(um2::ShortString("Ant") < um2::ShortString("Zebra"));
  static_assert(um2::ShortString("Zebra") > um2::ShortString("Ant"));
  static_assert(um2::ShortString("Zebra") <= um2::ShortString("ant"));
  static_assert(um2::ShortString("ant") >= um2::ShortString("Zebra"));
  static_assert(um2::ShortString("Zebra") <= um2::ShortString("Zebra"));
  static_assert(um2::ShortString("Zebra") >= um2::ShortString("Zebra"));
#else
  ASSERT(um2::ShortString("Ant") < um2::ShortString("Zebra"));
  ASSERT(um2::ShortString("Zebra") > um2::ShortString("Ant"));
  ASSERT(um2::ShortString("Zebra") <= um2::ShortString("ant"));
  ASSERT(um2::ShortString("ant") >= um2::ShortString("Zebra"));
  ASSERT(um2::ShortString("Zebra") <= um2::ShortString("Zebra"));
  ASSERT(um2::ShortString("Zebra") >= um2::ShortString("Zebra"));
#endif
// NOLINTEND(cert-dcl03-c,misc-static-assert)
}
MAKE_CUDA_KERNEL(comparison);

TEST_SUITE(ShortString)
{
  // Constructors
  TEST_HOSTDEV(default_constructor)
  TEST_HOSTDEV(const_char_array_constructor)
  TEST_HOSTDEV(copy_constructor)
  TEST_HOSTDEV(move_constructor)
  TEST_HOSTDEV(access_operator)
  TEST_HOSTDEV(const_char_pointer_constructor)

  // Operators
  TEST_HOSTDEV(assign_operator)
  TEST_HOSTDEV(equals_operator)
  TEST(equals_operator_std_string)
  TEST_HOSTDEV(comparison)
}

auto
main() -> int
{
  RUN_SUITE(ShortString)
  return 0;
}
