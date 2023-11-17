#include <um2/stdlib/string.hpp>

#include "../test_macros.hpp"

// clang-tidy does poorly with branched code like string. Valgrind and gcc's
// address sanitizer say that there are no leaks or undefined behavior, so
// we disable the warnings for this file.
// NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks) justified above
// NOLINTBEGIN(clang-analyzer-core.UndefinedBinaryOperatorResult) justified above

//==============================================================================
// Constructors
//==============================================================================

HOSTDEV
TEST_CASE(default_constructor)
{
  um2::String s;
  static_assert(sizeof(s) == 24);
  ASSERT(s.size() == 0);
  ASSERT(s.capacity() == 22);
  for (int i = 0; i < 22; ++i) {
    ASSERT(s.data()[i] == '\0');
  }
}
MAKE_CUDA_KERNEL(default_constructor)

HOSTDEV
TEST_CASE(const_char_array_constructor)
{
  um2::String s("hello");
  ASSERT(s.size() == 5);
  ASSERT(s.capacity() == 22);
  ASSERT(!s.isLong());
  ASSERT(s.data()[0] == 'h');
  ASSERT(s.data()[1] == 'e');
  ASSERT(s.data()[2] == 'l');
  ASSERT(s.data()[3] == 'l');
  ASSERT(s.data()[4] == 'o');

  um2::String s2("This string will be too long to fit in the small string optimization");
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.isLong());
  ASSERT(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(const_char_array_constructor);

TEST_CASE(int_float_constructors)
{
  {
    um2::String const s(5);
    ASSERT(s.size() == 1);
    ASSERT(s[0] == '5');
  }
  {
    um2::String const s(-5);
    ASSERT(s.size() == 2);
    ASSERT(s[0] == '-');
    ASSERT(s[1] == '5');
  }
  {
    um2::String const s(15);
    ASSERT(s.size() == 2);
    ASSERT(s[0] == '1');
    ASSERT(s[1] == '5');
  }
  {
    um2::String const s(-15);
    ASSERT(s.size() == 3);
    ASSERT(s[0] == '-');
    ASSERT(s[1] == '1');
    ASSERT(s[2] == '5');
  }
  {
    um2::String const s(1.5F);
    ASSERT(s[0] == '1');
    ASSERT(s[1] == '.');
    ASSERT(s[2] == '5');
  }
  {
    um2::String const s(-1.5F);
    ASSERT(s[0] == '-');
    ASSERT(s[1] == '1');
    ASSERT(s[2] == '.');
    ASSERT(s[3] == '5');
  }
}

HOSTDEV
TEST_CASE(copy_constructor)
{
  um2::String s0("hello");
  ASSERT(!s0.isLong());
  um2::String s(s0);
  ASSERT(s.size() == 5);
  ASSERT(s.capacity() == 22);
  ASSERT(!s.isLong());
  ASSERT(s.data()[0] == 'h');
  ASSERT(s.data()[1] == 'e');
  ASSERT(s.data()[2] == 'l');
  ASSERT(s.data()[3] == 'l');
  ASSERT(s.data()[4] == 'o');
  // Ensure that s0 is not modified
  s0.data()[0] = 'a';
  ASSERT(s.data()[0] == 'h');

  um2::String s1("This string will be too long to fit in the small string optimization");
  ASSERT(s1.isLong());
  um2::String s2(s1);
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.isLong());
  char * data = s2.data();
  ASSERT(data[0] == 'T');
  // Check that s1 is not modified
  s1.data()[0] = 'a';
  ASSERT(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(copy_constructor);

HOSTDEV
TEST_CASE(move_constructor)
{
  um2::String s1("This string will be too long to fit in the small string optimization");
  ASSERT(s1.isLong());
  um2::String s2(um2::move(s1));
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.isLong());
  ASSERT(s2.data()[0] == 'T');
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
  ASSERT(s.data()[0] == 'S');
  const char * input2 =
      "This string will be too long to fit in the small string optimization";
  um2::String s2(input2);
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.isLong());
  ASSERT(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(const_char_constructor);

//==============================================================================
// Operators
//==============================================================================

HOSTDEV
TEST_CASE(index_operator)
{
  um2::String s("hello");
  ASSERT(s[0] == 'h');
  ASSERT(s[1] == 'e');
  ASSERT(s[2] == 'l');
  ASSERT(s[3] == 'l');
  ASSERT(s[4] == 'o');
  s[0] = 'a';
  ASSERT(s[0] == 'a');
}
MAKE_CUDA_KERNEL(index_operator);

HOSTDEV
TEST_CASE(addition_operator)
{
  um2::String s0("hi");
  um2::String const s1(" there");
  s0 += s1;
  ASSERT(s0.size() == 8);
  ASSERT(s0[0] == 'h');
  ASSERT(s0[1] == 'i');
  ASSERT(s0[2] == ' ');
  ASSERT(s0[3] == 't');
  ASSERT(s0[4] == 'h');
  ASSERT(s0[5] == 'e');
  ASSERT(s0[6] == 'r');
  ASSERT(s0[7] == 'e');
  s0 = "hi";
  ASSERT(s0.size() == 2);
  s0 += " there";
  ASSERT(s0.size() == 8);
  ASSERT(s0[0] == 'h');
  ASSERT(s0[1] == 'i');
  ASSERT(s0[2] == ' ');
  ASSERT(s0[3] == 't');
  ASSERT(s0[4] == 'h');
  ASSERT(s0[5] == 'e');
  ASSERT(s0[6] == 'r');
  ASSERT(s0[7] == 'e');
  s0 = "hi";
  ASSERT(s0.size() == 2);
  um2::String s2 = s0 + s1;
  ASSERT(s2.size() == 8);
  ASSERT(s2[0] == 'h');
  ASSERT(s2[1] == 'i');
  ASSERT(s2[2] == ' ');
  ASSERT(s2[3] == 't');
}
MAKE_CUDA_KERNEL(addition_operator);

HOSTDEV
TEST_CASE(assign_operator)
{
  um2::String s0("hello");
  ASSERT(!s0.isLong());
  um2::String s("This string will be too long to fit in the small string optimization");
  s = s0;
  ASSERT(s.size() == 5);
  ASSERT(s.capacity() == 22);
  ASSERT(!s.isLong());
  ASSERT(s.data()[0] == 'h');
  ASSERT(s.data()[1] == 'e');
  ASSERT(s.data()[2] == 'l');
  ASSERT(s.data()[3] == 'l');
  ASSERT(s.data()[4] == 'o');
  // Ensure that s0 is not modified
  // cppcheck-suppress unreadVariable; justification: We are checking that s is not
  // modified
  s0.data()[0] = 'a';
  ASSERT(s.data()[0] == 'h');

  um2::String s1("This string will be too long to fit in the small string optimization");
  ASSERT(s1.isLong());
  um2::String s2;
  s2 = s1;
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.isLong());
  ASSERT(s2.data()[0] == 'T');
  // Check that s1 is not modified
  // cppcheck-suppress unreadVariable; justification: We are checking that s2 is not
  // modified
  s1.data()[0] = 'a';
  ASSERT(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(assign_operator);

TEST_CASE(std_string_assign_operator)
{
  std::string s0("hello");
  um2::String s("This string will be too long to fit in the small string optimization");
  s = s0;
  ASSERT(s.size() == 5);
  ASSERT(s.capacity() == 22);
  ASSERT(!s.isLong());
  ASSERT(s.data()[0] == 'h');
  ASSERT(s.data()[1] == 'e');
  ASSERT(s.data()[2] == 'l');
  ASSERT(s.data()[3] == 'l');
  ASSERT(s.data()[4] == 'o');

  std::string const s1(
      "This string will be too long to fit in the small string optimization");
  um2::String s2;
  s2 = s1;
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.isLong());
  ASSERT(s2.data()[0] == 'T');

  // Move assignment
  um2::String s3;
  s3 = um2::move(s0);
  ASSERT(s3.size() == 5);
  ASSERT(s3.capacity() == 22);
  ASSERT(!s3.isLong());
  ASSERT(s3.data()[0] == 'h');
  ASSERT(s3.data()[1] == 'e');
  ASSERT(s3.data()[2] == 'l');

  um2::String s4;
  s4 = um2::move(s1);
  ASSERT(s4.size() == 68);
  ASSERT(s4.capacity() == 68);
  ASSERT(s4.isLong());
  ASSERT(s4.data()[0] == 'T');
}

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

HOSTDEV
TEST_CASE(starts_ends_with)
{
  um2::String const s = "hello";
  ASSERT(s.starts_with("he"));
  ASSERT(!s.starts_with("eh"));
  ASSERT(s.ends_with("lo"));
  ASSERT(!s.ends_with("ol"));
}
MAKE_CUDA_KERNEL(starts_ends_with);

HOSTDEV
TEST_CASE(substr)
{
  um2::String const s("hello");
  um2::String const s0 = s.substr(1);
  ASSERT(s0.size() == 4);
  ASSERT(s0[0] == 'e');
  ASSERT(s0[1] == 'l');
  ASSERT(s0[2] == 'l');
  ASSERT(s0[3] == 'o');
  um2::String const s1 = s.substr(1, 2);
  ASSERT(s1.size() == 2);
  ASSERT(s1[0] == 'e');
  ASSERT(s1[1] == 'l');
}
MAKE_CUDA_KERNEL(substr);

HOSTDEV
TEST_CASE(find_last_of)
{
  um2::String const s("hello");
  ASSERT(s.find_last_of('l') == 3);
  ASSERT(s.find_last_of('o') == 4);
  ASSERT(s.find_last_of('h') == 0);
  ASSERT(s.find_last_of('a') == um2::String::npos);
}

TEST_SUITE(String)
{
  // Constructors
  TEST_HOSTDEV(default_constructor)
  TEST_HOSTDEV(const_char_array_constructor)
  TEST_HOSTDEV(copy_constructor)
  TEST_HOSTDEV(move_constructor)
  TEST_HOSTDEV(const_char_constructor)
  TEST(int_float_constructors)

  // Operators
  TEST_HOSTDEV(assign_operator)
  TEST(std_string_assign_operator)
  TEST_HOSTDEV(equals_operator)
  TEST_HOSTDEV(comparison)
  TEST_HOSTDEV(index_operator)
  TEST_HOSTDEV(addition_operator)

  // Methods
  TEST(starts_ends_with)
  TEST_HOSTDEV(substr)
}

auto
main() -> int
{
  RUN_SUITE(String)
  return 0;
}

// NOLINTEND(clang-analyzer-core.UndefinedBinaryOperatorResult)
// NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)
