#include <um2/stdlib/string.hpp>

#include "../test_macros.hpp"

// clang-tidy does poorly with branched code like String. Valgrind and gcc's
// address sanitizer say that there are no leaks or undefined behavior, so
// we disable the warnings for this file.
// NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks) justified above
// NOLINTBEGIN(clang-analyzer-core.UndefinedBinaryOperatorResult) justified above

//==============================================================================
// Constructors
//==============================================================================

// Compiler complains if it's a static_assert and if it's a normal assert
// NOLINTBEGIN(cert-dcl03-c, misc-static-assert)
HOSTDEV
TEST_CASE(constructor_default)
{
  um2::String const s;
  static_assert(sizeof(s) == 24);
  ASSERT(s.empty());
  ASSERT(s.capacity() == 22);
}
MAKE_CUDA_KERNEL(constructor_default);

HOSTDEV
TEST_CASE(constructor_const_char_ptr)
{
  const char * input = "Short String";
  um2::String s(input);
  ASSERT(s.size() == 12);
  ASSERT(s.capacity() == 22);
  ASSERT(s.data()[0] == 'S');
  const char * input2 =
      "This string will be too long to fit in the small string optimization";
  um2::String s2(input2);
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(constructor_const_char_ptr);

HOSTDEV
TEST_CASE(constructor_copy)
{
  um2::String s0("hello");
  um2::String s(s0);
  ASSERT(s.size() == 5);
  ASSERT(s.capacity() == 22);
  ASSERT(s.data()[0] == 'h');
  ASSERT(s.data()[1] == 'e');
  ASSERT(s.data()[2] == 'l');
  ASSERT(s.data()[3] == 'l');
  ASSERT(s.data()[4] == 'o');
  // Ensure that s0 is not modified
  s0.data()[0] = 'a';
  ASSERT(s.data()[0] == 'h');

  um2::String s1("This string will be too long to fit in the small string optimization");
  um2::String s2(s1);
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  char * data = s2.data();
  ASSERT(data[0] == 'T');
  // Check that s1 is not modified
  s1.data()[0] = 'a';
  ASSERT(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(constructor_copy);

HOSTDEV
TEST_CASE(constructor_move)
{
  um2::String s1("This string will be too long to fit in the small string optimization");
  um2::String s2(um2::move(s1));
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(constructor_move);

HOSTDEV
TEST_CASE(assign_copy)
{
  um2::String s0("hello");
  um2::String s("This string will be too long to fit in the small string optimization");
  s = s0;
  ASSERT(s.size() == 5);
  ASSERT(s.capacity() == 68);
  ASSERT(s.data()[0] == 'h');
  ASSERT(s.data()[1] == 'e');
  ASSERT(s.data()[2] == 'l');
  ASSERT(s.data()[3] == 'l');
  ASSERT(s.data()[4] == 'o');
  // Ensure that s0 is not modified
  s0.data()[0] = 'a';
  ASSERT(s.data()[0] == 'h');

  um2::String s1("This string will be too long to fit in the small string optimization");
  um2::String s2;
  s2 = s1;
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.data()[0] == 'T');
  // Check that s1 is not modified
  s1.data()[0] = 'a';
  ASSERT(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(assign_copy);

HOSTDEV
TEST_CASE(assign_move)
{
  um2::String s0("hello");
  um2::String s("This string will be too long to fit in the small string optimization");
  s = um2::move(s0);
  ASSERT(s.size() == 5);
  ASSERT(s.capacity() == 22);
  ASSERT(s.data()[0] == 'h');
  ASSERT(s.data()[1] == 'e');
  ASSERT(s.data()[2] == 'l');
  ASSERT(s.data()[3] == 'l');
  ASSERT(s.data()[4] == 'o');

  um2::String s1("This string will be too long to fit in the small string optimization");
  um2::String s2;
  s2 = um2::move(s1);
  ASSERT(s2.size() == 68);
  ASSERT(s2.capacity() == 68);
  ASSERT(s2.data()[0] == 'T');
}
MAKE_CUDA_KERNEL(assign_move);


////TEST_CASE(int_float_constructors)
////{
////  {
////    um2::String const s(5);
////    ASSERT(s.size() == 1);
////    ASSERT(s[0] == '5');
////  }
////  {
////    um2::String const s(-5);
////    ASSERT(s.size() == 2);
////    ASSERT(s[0] == '-');
////    ASSERT(s[1] == '5');
////  }
////  {
////    um2::String const s(15);
////    ASSERT(s.size() == 2);
////    ASSERT(s[0] == '1');
////    ASSERT(s[1] == '5');
////  }
////  {
////    um2::String const s(-15);
////    ASSERT(s.size() == 3);
////    ASSERT(s[0] == '-');
////    ASSERT(s[1] == '1');
////    ASSERT(s[2] == '5');
////  }
////  {
////    um2::String const s(1.5F);
////    ASSERT(s[0] == '1');
////    ASSERT(s[1] == '.');
////    ASSERT(s[2] == '5');
////  }
////  {
////    um2::String const s(-1.5F);
////    ASSERT(s[0] == '-');
////    ASSERT(s[1] == '1');
////    ASSERT(s[2] == '.');
////    ASSERT(s[3] == '5');
////  }
////}
//

//==============================================================================
// Operations
//==============================================================================

HOSTDEV    
TEST_CASE(compare)    
{    
  char const * const data1 = "Hello, World!";    
  char const * const data2 = "Hello, Worl!";     
  char const * const data3 = "Hekko, World!";    
  um2::String const s1(data1);    
  um2::String const s2(data2);    
  um2::String const s3(data3);    
  ASSERT(s1.compare(s1) == 0);    
  ASSERT(s1.compare(s2) > 0);    
  ASSERT(s2.compare(s1) < 0);    
  ASSERT(s1.compare(s3) > 0);    
}    
MAKE_CUDA_KERNEL(compare)

HOSTDEV
TEST_CASE(starts_with)
{
  char const * const data = "Hello, World!";
  um2::String const s(data);
  ASSERT(s.starts_with("Hello"));
  ASSERT(s.starts_with("Hello, World!"));
  ASSERT(!s.starts_with("Hello, World! "));
  ASSERT(!s.starts_with("World"));
}
MAKE_CUDA_KERNEL(starts_with)

HOSTDEV
TEST_CASE(end_with)
{
  char const * const data = "StringView";
  um2::String const s(data);
  ASSERT(s.ends_with("View"));
  ASSERT(s.ends_with("StringView"));
  ASSERT(!s.ends_with("String"));
  ASSERT(!s.ends_with("StringView "));
}
MAKE_CUDA_KERNEL(end_with)

////HOSTDEV
////TEST_CASE(index_operator)
////{
////  um2::String s("hello");
////  ASSERT(s[0] == 'h');
////  ASSERT(s[1] == 'e');
////  ASSERT(s[2] == 'l');
////  ASSERT(s[3] == 'l');
////  ASSERT(s[4] == 'o');
////  s[0] = 'a';
////  ASSERT(s[0] == 'a');
////}
////MAKE_CUDA_KERNEL(index_operator);
////

////TEST_CASE(std_string_assign_operator)
////{
////  std::string s0("hello");
////  um2::String s("This string will be too long to fit in the small string optimization");
////  s = s0;
////  ASSERT(s.size() == 5);
////  ASSERT(s.capacity() == 22);
////  ASSERT(!s.isLong());
////  ASSERT(s.data()[0] == 'h');
////  ASSERT(s.data()[1] == 'e');
////  ASSERT(s.data()[2] == 'l');
////  ASSERT(s.data()[3] == 'l');
////  ASSERT(s.data()[4] == 'o');
////
////  std::string const s1(
////      "This string will be too long to fit in the small string optimization");
////  um2::String s2;
////  s2 = s1;
////  ASSERT(s2.size() == 68);
////  ASSERT(s2.capacity() == 68);
////  ASSERT(s2.isLong());
////  ASSERT(s2.data()[0] == 'T');
////
////  // Move assignment
////  um2::String s3;
////  s3 = um2::move(s0);
////  ASSERT(s3.size() == 5);
////  ASSERT(s3.capacity() == 22);
////  ASSERT(!s3.isLong());
////  ASSERT(s3.data()[0] == 'h');
////  ASSERT(s3.data()[1] == 'e');
////  ASSERT(s3.data()[2] == 'l');
////
////  um2::String s4;
////  s4 = um2::move(s1);
////  ASSERT(s4.size() == 68);
////  ASSERT(s4.capacity() == 68);
////  ASSERT(s4.isLong());
////  ASSERT(s4.data()[0] == 'T');
////}
////
////MAKE_CUDA_KERNEL(equals_operator);
////
HOSTDEV
TEST_CASE(relational_operators)
{
  um2::String const s0("Ant");
  um2::String const s1("Zebra");
  ASSERT(s0 < s1);
  ASSERT(s1 > s0);
  ASSERT(s0 <= s1);
  ASSERT(s1 >= s0);
  ASSERT(s0 <= s0);
  ASSERT(s0 >= s0);
}
MAKE_CUDA_KERNEL(relational_operators);
////
////HOSTDEV
////TEST_CASE(starts_ends_with)
////{
////  um2::String const s = "hello";
////  ASSERT(s.starts_with("he"));
////  ASSERT(!s.starts_with("eh"));
////  ASSERT(s.ends_with("lo"));
////  ASSERT(!s.ends_with("ol"));
////}
////MAKE_CUDA_KERNEL(starts_ends_with);
////
HOSTDEV
TEST_CASE(substr)
{
  um2::String const s("hello");
  um2::String const s0 = s.substr(1);
  ASSERT(s0.size() == 4);
  ASSERT(s0.data()[0] == 'e');
  ASSERT(s0.data()[1] == 'l');
  ASSERT(s0.data()[2] == 'l');
  ASSERT(s0.data()[3] == 'o');
  um2::String const s1 = s.substr(1, 2);
  ASSERT(s1.size() == 2);
  ASSERT(s1.data()[0] == 'e');
  ASSERT(s1.data()[1] == 'l');
  um2::String const s2 = s.substr(0, 0);
  ASSERT(s2.empty());
  um2::String const s3 = s.substr(0, 100);
  ASSERT(s3.size() == 5);
  ASSERT(s3.data()[0] == 'h');
  ASSERT(s3.data()[1] == 'e');
  ASSERT(s3.data()[2] == 'l');
  ASSERT(s3.data()[3] == 'l');
  ASSERT(s3.data()[4] == 'o');
}
MAKE_CUDA_KERNEL(substr);

HOSTDEV    
TEST_CASE(find_last_of)    
{    
  char const * const data = "Hello, World!";    
  um2::String const s(data);    
  ASSERT(s.find_last_of('H') == 0);    
  ASSERT(s.find_last_of('W') == 7);    
  ASSERT(s.find_last_of('!') == 12);    
  ASSERT(s.find_last_of('z') == um2::String::npos);    
  ASSERT(s.find_last_of('l') == 10);    
  ASSERT(s.find_last_of('o', 4) == 4);    
  ASSERT(s.find_last_of('o', 3) == um2::String::npos);    
} 

HOSTDEV
TEST_CASE(compound_addition)
{
  // Start with a small string
  um2::String s("hello");
  ASSERT(s.size() == 5);
  ASSERT(s.capacity() == 22);
  // Add a small string
  // This should not change the capacity
  s += um2::String(" there");
  ASSERT(s.size() == 11);
  ASSERT(s.capacity() == 22);

  // Add a small string which will exceed the capacity
  // and force a reallocation
  s += " ... General Kenobi";
  ASSERT(s.size() == 30);
  ASSERT(s.capacity() == 44);
}

// NOLINTEND(cert-dcl03-c,misc-static-assert)

TEST_SUITE(String)
{
  // Constructors
  TEST_HOSTDEV(constructor_default)
  TEST_HOSTDEV(constructor_const_char_ptr)
  TEST_HOSTDEV(constructor_copy)
  TEST_HOSTDEV(constructor_move)
  TEST_HOSTDEV(assign_copy)
  TEST_HOSTDEV(assign_move)
//  TEST(int_float_constructors)
//
  // Operations
  TEST_HOSTDEV(compare)
  TEST_HOSTDEV(starts_with)
  TEST_HOSTDEV(end_with)
//  TEST(std_string_assign_operator)
  TEST_HOSTDEV(relational_operators)
//  TEST_HOSTDEV(index_operator)
//  TEST_HOSTDEV(addition_operator)
//
  // Methods
//  TEST(starts_ends_with)
  TEST_HOSTDEV(substr)
  TEST_HOSTDEV(find_last_of)
  TEST_HOSTDEV(compound_addition)
}

auto
main() -> int
{
  RUN_SUITE(String)
  return 0;
}

// NOLINTEND(clang-analyzer-core.UndefinedBinaryOperatorResult)
// NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)
