#include <um2/common/string_to_lattice.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

template <typename T>
TEST_CASE(one_token)
{
  um2::Vector<um2::Vector<T>> const v = um2::stringToLattice<T>(R"(1)");
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 1);
  ASSERT(v[0][0] == 1);
}

template <typename T>
TEST_CASE(one_line)
{
  um2::Vector<um2::Vector<T>> v = um2::stringToLattice<T>(R"(1 2)");
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  v.clear();

  v = um2::stringToLattice<T>(R"(1, 2)", ',');
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  v.clear();

  v = um2::stringToLattice<T>(R"(
    1 2)");
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  v.clear();

  v = um2::stringToLattice<T>(R"(
    1, 2)",
                              ',');
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  v.clear();

  v = um2::stringToLattice<T>(R"(
  1 2
    )");
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  v.clear();

  v = um2::stringToLattice<T>(R"(
  1, 2
    )",
                              ',');
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  v.clear();
}

template <typename T>
TEST_CASE(two_lines)
{
  um2::Vector<um2::Vector<T>> v = um2::stringToLattice<T>(R"(
    1 2
    3 4)");
  ASSERT(v.size() == 2);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  ASSERT(v[1].size() == 2);
  ASSERT(v[1][0] == 3);
  ASSERT(v[1][1] == 4);

  v = um2::stringToLattice<T>(R"(
    1, 2,
    3, 4)",
                              ',');
  ASSERT(v.size() == 2);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  ASSERT(v[1].size() == 2);
  ASSERT(v[1][0] == 3);
  ASSERT(v[1][1] == 4);

  v = um2::stringToLattice<T>(R"(
    1, 2
    3, 4)",
                              ',');
  ASSERT(v.size() == 2);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  ASSERT(v[1].size() == 2);
  ASSERT(v[1][0] == 3);
  ASSERT(v[1][1] == 4);

  v = um2::stringToLattice<T>(R"(
    1 2
    3 4 5
    )");
  ASSERT(v.size() == 2);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  ASSERT(v[1].size() == 3);
  ASSERT(v[1][0] == 3);
  ASSERT(v[1][1] == 4);
  ASSERT(v[1][2] == 5);
}

template <class T>
TEST_CASE(many_lines)
{
  um2::Vector<um2::Vector<T>> v = um2::stringToLattice<T>(R"(
    1 2
    3 4
    5 6
    7 8 9
    10 11)");
  ASSERT(v.size() == 5);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  ASSERT(v[1].size() == 2);
  ASSERT(v[1][0] == 3);
  ASSERT(v[1][1] == 4);
  ASSERT(v[2].size() == 2);
  ASSERT(v[2][0] == 5);
  ASSERT(v[2][1] == 6);
  ASSERT(v[3].size() == 3);
  ASSERT(v[3][0] == 7);
  ASSERT(v[3][1] == 8);
  ASSERT(v[3][2] == 9);
  ASSERT(v[4].size() == 2);
  ASSERT(v[4][0] == 10);
  ASSERT(v[4][1] == 11);

  v = um2::stringToLattice<T>(R"(
    1, 2,
    3, 4,
    5, 6,
    7, 8, 9
    10, 11)",
                              ',');
  ASSERT(v.size() == 5);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  ASSERT(v[1].size() == 2);
  ASSERT(v[1][0] == 3);
  ASSERT(v[1][1] == 4);
  ASSERT(v[2].size() == 2);
  ASSERT(v[2][0] == 5);
  ASSERT(v[2][1] == 6);
  ASSERT(v[3].size() == 3);
  ASSERT(v[3][0] == 7);
  ASSERT(v[3][1] == 8);
  ASSERT(v[3][2] == 9);
  ASSERT(v[4].size() == 2);
  ASSERT(v[4][0] == 10);
  ASSERT(v[4][1] == 11);
}

template <class T>
TEST_SUITE(stringToLattice)
{
  TEST(one_token<T>);
  TEST(one_line<T>);
  TEST(two_lines<T>);
  TEST(many_lines<T>);
}

auto
main() -> int
{
  RUN_SUITE(stringToLattice<int>);
  return 0;
}
