#include "../test_framework.hpp"
#include <um2/common/string.hpp>

// -----------------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------------

UM2_HOSTDEV TEST_CASE(begin_end)
{
  um2::String s;
  EXPECT_EQ(s.begin(), s.end());
  EXPECT_EQ(s.cbegin(), s.cend());

  s = "hello";
  EXPECT_NE(s.begin(), s.end());
  EXPECT_NE(s.cbegin(), s.cend());
  EXPECT_EQ(*s.begin(), 'h');
  EXPECT_EQ(*s.cbegin(), 'h');
  EXPECT_EQ(*(s.end() - 1), 'o');
  EXPECT_EQ(*(s.cend() - 1), 'o');
}
MAKE_CUDA_KERNEL(begin_end);

// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------

UM2_HOSTDEV TEST_CASE(const_char_array_constructor)
{
  um2::String s("hello");
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.capacity(), 6);
  EXPECT_EQ(s.data()[0], 'h');
  EXPECT_EQ(s.data()[1], 'e');
  EXPECT_EQ(s.data()[2], 'l');
  EXPECT_EQ(s.data()[3], 'l');
  EXPECT_EQ(s.data()[4], 'o');
  EXPECT_EQ(s.data()[5], '\0');
}
MAKE_CUDA_KERNEL(const_char_array_constructor);

UM2_HOSTDEV TEST_CASE(um2_string_constructor)
{
  um2::String s0("hello");
  um2::String s(s0);
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.capacity(), 6);
  EXPECT_EQ(s.data()[0], 'h');
  EXPECT_EQ(s.data()[1], 'e');
  EXPECT_EQ(s.data()[2], 'l');
  EXPECT_EQ(s.data()[3], 'l');
  EXPECT_EQ(s.data()[4], 'o');
  EXPECT_EQ(s.data()[5], '\0');
}
MAKE_CUDA_KERNEL(um2_string_constructor);

TEST_CASE(std_string_constructor)
{
  std::string s0("hello");
  um2::String s(s0);
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.capacity(), 8);
  EXPECT_EQ(s.data()[0], 'h');
  EXPECT_EQ(s.data()[1], 'e');
  EXPECT_EQ(s.data()[2], 'l');
  EXPECT_EQ(s.data()[3], 'l');
  EXPECT_EQ(s.data()[4], 'o');
  EXPECT_EQ(s.data()[5], '\0');
}

// -----------------------------------------------------------------------------
// Operators
// -----------------------------------------------------------------------------

UM2_HOSTDEV TEST_CASE(assign_um2_string)
{
  um2::String s0("hello");
  um2::String s = s0;
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.capacity(), 6);
  EXPECT_EQ(s.data()[0], 'h');
  EXPECT_EQ(s.data()[1], 'e');
  EXPECT_EQ(s.data()[2], 'l');
  EXPECT_EQ(s.data()[3], 'l');
  EXPECT_EQ(s.data()[4], 'o');
  EXPECT_EQ(s.data()[5], '\0');
  EXPECT_NE(s.data(), s0.data());
}
MAKE_CUDA_KERNEL(assign_um2_string);

UM2_HOSTDEV TEST_CASE(assign_const_char_array)
{
  um2::String s;
  s = "hello";
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.capacity(), 8);
  EXPECT_EQ(s.data()[0], 'h');
  EXPECT_EQ(s.data()[1], 'e');
  EXPECT_EQ(s.data()[2], 'l');
  EXPECT_EQ(s.data()[3], 'l');
  EXPECT_EQ(s.data()[4], 'o');
  EXPECT_EQ(s.data()[5], '\0');
}
MAKE_CUDA_KERNEL(assign_const_char_array);

TEST_CASE(assign_std_string)
{
  um2::String s;
  s = "hello";
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.capacity(), 8);
  EXPECT_EQ(s.data()[0], 'h');
  EXPECT_EQ(s.data()[1], 'e');
  EXPECT_EQ(s.data()[2], 'l');
  EXPECT_EQ(s.data()[3], 'l');
  EXPECT_EQ(s.data()[4], 'o');
  EXPECT_EQ(s.data()[5], '\0');
}

UM2_HOSTDEV TEST_CASE(equals_um2_string)
{
  um2::String s0("hello");
  um2::String s1("helo");
  um2::String s2("hello");
  EXPECT_EQ(s0, s0);
  EXPECT_EQ(s0, s2);
  EXPECT_NE(s0, s1);
}
MAKE_CUDA_KERNEL(equals_um2_string);

UM2_HOSTDEV TEST_CASE(equals_const_char_array)
{
  um2::String s("hello");
  EXPECT_EQ(s, "hello");
  EXPECT_NE(s, "helo");
}
MAKE_CUDA_KERNEL(equals_const_char_array);

TEST_CASE(equals_std_string)
{
  um2::String s("hello");
  EXPECT_EQ(s, std::string("hello"));
  EXPECT_NE(s, std::string("helo"));
}

UM2_HOSTDEV TEST_CASE(comparison)
{
  EXPECT_LT(um2::String("Ant"), um2::String("Zebra"));
  EXPECT_GT(um2::String("Zebra"), um2::String("Ant"));
  EXPECT_LE(um2::String("Zebra"), um2::String("ant"));
  EXPECT_GE(um2::String("ant"), um2::String("Zebra"));
  EXPECT_LE(um2::String("Zebra"), um2::String("Zebra"));
  EXPECT_GE(um2::String("Zebra"), um2::String("Zebra"));
}
MAKE_CUDA_KERNEL(comparison);

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------

UM2_HOSTDEV TEST_CASE(compare)
{
  EXPECT_GT(um2::String("Zebra").compare(um2::String("Ant")), 0);
  EXPECT_LT(um2::String("Ant").compare(um2::String("Zebra")), 0);
  EXPECT_EQ(um2::String("Zebra").compare(um2::String("Zebra")), 0);
  EXPECT_LT(um2::String("Zebra").compare(um2::String("ant")), 0);
}
MAKE_CUDA_KERNEL(compare);

UM2_HOSTDEV TEST_CASE(contains)
{
  um2::String s("hello");
  EXPECT_TRUE(s.contains('h'));
  EXPECT_TRUE(s.contains('e'));
  EXPECT_TRUE(s.contains('l'));
  EXPECT_TRUE(s.contains('o'));
  EXPECT_FALSE(s.contains('a'));
  EXPECT_FALSE(s.contains('b'));
}
MAKE_CUDA_KERNEL(contains);

TEST_CASE(starts_ends_with)
{
  um2::String s("hello");
  EXPECT_TRUE(s.starts_with("he"));
  EXPECT_FALSE(s.starts_with("eh"));
  EXPECT_TRUE(s.ends_with("lo"));
  EXPECT_FALSE(s.ends_with("ol"));
}

TEST_CASE(to_string)
{
  um2::String s("hello");
  EXPECT_EQ(to_string(s), std::string("hello"));
}

TEST_SUITE(string)
{
  // Accessors
  TEST_HOSTDEV(begin_end)
  // Constructors
  TEST_HOSTDEV(const_char_array_constructor)
  TEST_HOSTDEV(um2_string_constructor)
  TEST(std_string_constructor)
  // Operators
  TEST_HOSTDEV(assign_um2_string)
  TEST_HOSTDEV(assign_const_char_array)
  TEST(assign_std_string)
  TEST_HOSTDEV(equals_um2_string)
  TEST_HOSTDEV(equals_const_char_array)
  TEST(equals_std_string)
  TEST_HOSTDEV(comparison)
  // Methods
  TEST_HOSTDEV(compare)
  TEST_HOSTDEV(contains)
  TEST(starts_ends_with)
  TEST(to_string)
}

auto main() -> int
{
  RUN_TESTS(string)
  return 0;
}