#include <um2/stdlib/string_view.hpp>

#include "../test_macros.hpp"

#include <iostream>

//==============================================================================
// Constructors and assignment
//==============================================================================

// Compiler complains if it's static or not static...
// NOLINTBEGIN(cert-dcl03-c,misc-static-assert)

HOSTDEV
TEST_CASE(constructor_default)
{
  um2::StringView const s;
  ASSERT(s.empty());
  ASSERT(s.data() == nullptr);
}
MAKE_CUDA_KERNEL(constructor_default)

HOSTDEV
TEST_CASE(constructor_ptr_size)
{
  char const * const data = "Hello, World!";
  um2::StringView const s(data, 13);
  ASSERT(s.size() == 13);
  ASSERT(s.data() == data);
}
MAKE_CUDA_KERNEL(constructor_ptr_size)

HOSTDEV
TEST_CASE(constructor_ptr)
{
  char const * const data = "Hello, World!";
  um2::StringView const s(data);
  ASSERT(s.size() == 13);
  ASSERT(s.data() == data);
}
MAKE_CUDA_KERNEL(constructor_ptr)

HOSTDEV
TEST_CASE(constructor_copy)
{
  char const * const data = "Hello, World!";
  um2::StringView const s1(data);
  um2::StringView const s2(s1);
  ASSERT(s1.size() == s2.size());
  ASSERT(s1.data() == s2.data());
}
MAKE_CUDA_KERNEL(constructor_copy)

HOSTDEV
TEST_CASE(constructor_ptr_ptr)
{
  char const * const data = "Hello, World!";
  um2::StringView const s(data, data + 13);
  ASSERT(s.size() == 13);
  ASSERT(s.data() == data);
}
MAKE_CUDA_KERNEL(constructor_ptr_ptr)

HOSTDEV
TEST_CASE(element_access)
{
  char const * const data = "Hello, World!";
  um2::StringView const s(data, 13);
  ASSERT(s[0] == 'H');
  ASSERT(s[1] == 'e');
  ASSERT(s.front() == 'H');
  ASSERT(s.back() == '!');
  ASSERT(s.data() == data);
}
MAKE_CUDA_KERNEL(element_access)

HOSTDEV
TEST_CASE(compare)
{
  char const * const data1 = "Hello, World!";
  char const * const data2 = "Hello, Worl!";
  char const * const data3 = "Hekko, World!";
  um2::StringView const s1(data1, 13);
  um2::StringView const s2(data2, 12);
  um2::StringView const s3(data3, 13);
  ASSERT(s1.compare(s1) == 0);
  ASSERT(s1.compare(s2) > 0);
  ASSERT(s2.compare(s1) < 0);
  ASSERT(s1.compare(s3) > 0);
}
MAKE_CUDA_KERNEL(compare)

HOSTDEV
TEST_CASE(substr)
{
  char const * const data = "Hello, World!";
  um2::StringView const s(data, 13);
  um2::StringView const s1 = s.substr(0, 5);
  um2::StringView const s2 = s.substr(7, 5);
  ASSERT(s1.size() == 5);
  ASSERT(s1.data() == data);
  ASSERT(s2.size() == 5);
  ASSERT(s2.data() == data + 7);
}
MAKE_CUDA_KERNEL(substr)

HOSTDEV
TEST_CASE(starts_with)
{
  char const * const data = "Hello, World!";
  um2::StringView const s(data, 13);
  ASSERT(s.starts_with("Hello"));
  ASSERT(s.starts_with("Hello, World!"));
  ASSERT(!s.starts_with("World"));
  ASSERT(!s.starts_with("Hello, World! "));
}
MAKE_CUDA_KERNEL(starts_with)


HOSTDEV
TEST_CASE(end_with)
{
  char const * const data = "StringView";
  um2::StringView const s(data, 10);
  ASSERT(s.ends_with("View"));
  ASSERT(s.ends_with("StringView"));
  ASSERT(!s.ends_with("String"));
  ASSERT(!s.ends_with("StringView "));
}
MAKE_CUDA_KERNEL(end_with)

HOSTDEV
TEST_CASE(removeLeadingSpaces)
{
  char const * const data = "   Hello, World!";
  um2::StringView s1(data, 16);
  s1.removeLeadingSpaces();
  ASSERT(s1.size() == 13);
  ASSERT(s1.data() == data + 3);

  char const * const data2 = "Hello, World!";
  um2::StringView s2(data2, 13);
  s2.removeLeadingSpaces();
  ASSERT(s2.size() == 13);
  ASSERT(s2.data() == data2);

  char const * const data3 = "   ";
  um2::StringView s3(data3, 3);
  s3.removeLeadingSpaces();
  ASSERT(s3.empty());
}
MAKE_CUDA_KERNEL(removeLeadingSpaces)

HOSTDEV
TEST_CASE(getTokenAndShrink)
{
  char const * const data = "A BB  C";
  um2::StringView s(data, 7); 
  um2::StringView token = s.getTokenAndShrink();
  ASSERT(token.size() == 1);
  ASSERT(*token.data() == 'A'); 
  ASSERT(s.size() == 5);
  ASSERT(*s.data() == 'B');
  token = s.getTokenAndShrink();
  ASSERT(token.size() == 2);
  ASSERT(*token.data() == 'B');
  ASSERT(s.size() == 2);
  ASSERT(*s.data() == ' ');
  token = s.getTokenAndShrink();
  ASSERT(token.size() == 1);
  ASSERT(*token.data() == 'C');
  ASSERT(s.empty());

  char const * const data2 = "  A BB  C";
  um2::StringView s2(data2, 9);
  token = s2.getTokenAndShrink();
  ASSERT(token.size() == 1);
  ASSERT(*token.data() == 'A');
  ASSERT(s2.size() == 5);
  ASSERT(*s2.data() == 'B');

  char const * const data3 = "AABBBC";
  um2::StringView s3(data3, 6);
  token = s3.getTokenAndShrink();
  ASSERT(token.size() == 6);
  ASSERT(*token.data() == 'A');
  ASSERT(s3.empty());
}
MAKE_CUDA_KERNEL(getTokenAndShrink)

// NOLINTEND(cert-dcl03-c,misc-static-assert)

TEST_SUITE(StringView)
{
  // Constructors and assignment
  TEST_HOSTDEV(constructor_default)
  TEST_HOSTDEV(constructor_ptr_size)
  TEST_HOSTDEV(constructor_ptr)
  TEST_HOSTDEV(constructor_copy)
  TEST_HOSTDEV(constructor_ptr_ptr)

  // Element access
  TEST_HOSTDEV(element_access)

  // Operations
  TEST_HOSTDEV(compare)
  TEST_HOSTDEV(substr)
  TEST_HOSTDEV(starts_with)
  TEST_HOSTDEV(end_with)

  // Non-standard modifiers
  TEST_HOSTDEV(removeLeadingSpaces)
  TEST_HOSTDEV(getTokenAndShrink)
}

auto
main() -> int
{
  RUN_SUITE(StringView)
  return 0;
}
