#include <um2/stdlib/string_view.hpp>

#include "../test_macros.hpp"

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
}

auto
main() -> int
{
  RUN_SUITE(StringView)
  return 0;
}
