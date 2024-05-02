#include <um2/stdlib/string_view.hpp>

#include "../test_macros.hpp"

//==============================================================================
// Constructors and assignment
//==============================================================================

constexpr const char * data = "Hello, World!";

HOSTDEV
TEST_CASE(constructor_default)
{
  um2::StringView constexpr s;
  STATIC_ASSERT(s.empty());
  STATIC_ASSERT(s.data() == nullptr);
}
MAKE_CUDA_KERNEL(constructor_default)

HOSTDEV
TEST_CASE(constructor_ptr_size)
{
  um2::StringView constexpr s(data, 13);
  STATIC_ASSERT(s.size() == 13);
  STATIC_ASSERT(s.data() == data);
}
MAKE_CUDA_KERNEL(constructor_ptr_size)

HOSTDEV
TEST_CASE(constructor_ptr)
{
  um2::StringView constexpr s(data);
  STATIC_ASSERT(s.size() == 13);
  STATIC_ASSERT(s.data() == data);
}
MAKE_CUDA_KERNEL(constructor_ptr)

HOSTDEV
TEST_CASE(constructor_copy)
{
  um2::StringView constexpr s1(data);
  um2::StringView constexpr s2(s1);
  STATIC_ASSERT(s1.size() == s2.size());
  STATIC_ASSERT(s1.data() == s2.data());
}
MAKE_CUDA_KERNEL(constructor_copy)

HOSTDEV
TEST_CASE(constructor_ptr_ptr)
{
  um2::StringView constexpr s(data, data + 13);
  STATIC_ASSERT(s.size() == 13);
  STATIC_ASSERT(s.data() == data);
}
MAKE_CUDA_KERNEL(constructor_ptr_ptr)

HOSTDEV
TEST_CASE(element_access)
{
  um2::StringView constexpr s(data, 13);
  STATIC_ASSERT(s[0] == 'H');
  STATIC_ASSERT(s[1] == 'e');
  STATIC_ASSERT(s.front() == 'H');
  STATIC_ASSERT(s.back() == '!');
  STATIC_ASSERT(s.data() == data);
}
MAKE_CUDA_KERNEL(element_access)

HOSTDEV
TEST_CASE(compare)
{
  constexpr const char * data1 = "Hello, World!";
  constexpr const char * data2 = "Hello, Worl!";
  constexpr const char * data3 = "Hekko, World!";
  um2::StringView constexpr s1(data1, 13);
  um2::StringView constexpr s2(data2, 12);
  um2::StringView constexpr s3(data3, 13);
  STATIC_ASSERT(s1.compare(s1) == 0);
  STATIC_ASSERT(s1.compare(s2) > 0);
  STATIC_ASSERT(s2.compare(s1) < 0);
  STATIC_ASSERT(s1.compare(s3) > 0);
}
MAKE_CUDA_KERNEL(compare)

HOSTDEV
TEST_CASE(substr)
{
  um2::StringView constexpr s(data, 13);
  um2::StringView constexpr s1 = s.substr(0, 5);
  um2::StringView constexpr s2 = s.substr(7, 5);
  STATIC_ASSERT(s1.size() == 5);
  STATIC_ASSERT(s1.data() == data);
  STATIC_ASSERT(s2.size() == 5);
  STATIC_ASSERT(s2.data() == data + 7);
}
MAKE_CUDA_KERNEL(substr)

HOSTDEV
TEST_CASE(starts_with)
{
  um2::StringView constexpr s(data, 13);
  STATIC_ASSERT(s.starts_with("Hello"));
  STATIC_ASSERT(s.starts_with("Hello, World!"));
  STATIC_ASSERT(!s.starts_with("World"));
  STATIC_ASSERT(!s.starts_with("Hello, World! "));
}
MAKE_CUDA_KERNEL(starts_with)

HOSTDEV
TEST_CASE(find_first_of)
{
  um2::StringView constexpr s(data, 13);
  STATIC_ASSERT(s.find_first_of('H') == 0);
  STATIC_ASSERT(s.find_first_of('W') == 7);
  STATIC_ASSERT(s.find_first_of('!') == 12);
  STATIC_ASSERT(s.find_first_of('z') == um2::StringView::npos);
  STATIC_ASSERT(s.find_first_of('l', 4) == 10);
  STATIC_ASSERT(s.find_first_of('l', 20) == um2::StringView::npos);

  um2::StringView constexpr s2(data + 4, 9);
  STATIC_ASSERT(s.find_first_of(s2) == 4);
  STATIC_ASSERT(s.find_first_of(s2, 5) == um2::StringView::npos);
  STATIC_ASSERT(s.find_first_of("ello") == 1);
  STATIC_ASSERT(s.find_first_of("ello", 6) == um2::StringView::npos);
}
MAKE_CUDA_KERNEL(find_first_of)

HOSTDEV
TEST_CASE(find_first_not_of)
{
  um2::StringView constexpr s(data, 13);
  STATIC_ASSERT(s.find_first_not_of('H') == 1);
  STATIC_ASSERT(s.find_first_not_of('l', 2) == 4);
  STATIC_ASSERT(s.find_first_not_of('l', 20) == um2::StringView::npos);
}

HOSTDEV
TEST_CASE(find_last_of)
{
  um2::StringView constexpr s(data, 13);
  STATIC_ASSERT(s.find_last_of('H') == 0);
  STATIC_ASSERT(s.find_last_of('W') == 7);
  STATIC_ASSERT(s.find_last_of('!') == 12);
  STATIC_ASSERT(s.find_last_of('z') == um2::StringView::npos);
  STATIC_ASSERT(s.find_last_of('l') == 10);
  STATIC_ASSERT(s.find_last_of('o', 4) == 4);
  STATIC_ASSERT(s.find_last_of('o', 3) == um2::StringView::npos);
}

HOSTDEV
TEST_CASE(ends_with)
{
  constexpr const char * data2 = "StringView";
  um2::StringView constexpr s(data2, 10);
  STATIC_ASSERT(s.ends_with("View"));
  STATIC_ASSERT(s.ends_with("StringView"));
  STATIC_ASSERT(!s.ends_with("String"));
  STATIC_ASSERT(!s.ends_with("StringView "));
}
MAKE_CUDA_KERNEL(ends_with)

HOSTDEV
TEST_CASE(removeLeadingSpaces)
{
  char const * const data1 = "   Hello, World!";
  um2::StringView s1(data1, 16);
  s1.removeLeadingSpaces();
  ASSERT(s1.size() == 13);
  ASSERT(s1.data() == data1 + 3);

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
  char const * const data1 = "A BB  C";
  um2::StringView s(data1, 7);
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
  TEST_HOSTDEV(find_first_of)
  TEST_HOSTDEV(find_first_not_of)
  TEST_HOSTDEV(find_last_of)
  TEST_HOSTDEV(ends_with)

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
