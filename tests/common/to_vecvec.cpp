#include <um2/common/to_vecvec.hpp>

#include "../test_macros.hpp"

template <typename T>
TEST_CASE(empty)
{
  std::vector<std::vector<T>> const v = um2::to_vecvec<T>(R"()");
  ASSERT(v.empty());
}

template <typename T>
TEST_CASE(one_token)
{
  std::vector<std::vector<T>> const v = um2::to_vecvec<T>(R"(1)");
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 1);
  ASSERT(v[0][0] == 1);
}

template <typename T>
TEST_CASE(one_line)
{
  std::vector<std::vector<T>> const v = um2::to_vecvec<T>(R"(1 2)");
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  std::vector<std::vector<T>> const x = um2::to_vecvec<T>(R"(
    1 2)");
  ASSERT(x.size() == 1);
  ASSERT(x[0].size() == 2);
  ASSERT(x[0][0] == 1);
  ASSERT(x[0][1] == 2);
  std::vector<std::vector<T>> const y = um2::to_vecvec<T>(R"(1 2
    )");
  ASSERT(y.size() == 1);
  ASSERT(y[0].size() == 2);
  ASSERT(y[0][0] == 1);
  ASSERT(y[0][1] == 2);
}

template <typename T>
TEST_CASE(two_lines)
{
  std::vector<std::vector<T>> const v = um2::to_vecvec<T>(R"(
    1 2
    3 4)");
  ASSERT(v.size() == 2);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  ASSERT(v[1].size() == 2);
  ASSERT(v[1][0] == 3);
  ASSERT(v[1][1] == 4);

  std::vector<std::vector<T>> const u = um2::to_vecvec<T>(R"(
    1 2
    3 4 5)");
  ASSERT(u.size() == 2);
  ASSERT(u[0].size() == 2);
  ASSERT(u[0][0] == 1);
  ASSERT(u[0][1] == 2);
  ASSERT(u[1].size() == 3);
  ASSERT(u[1][0] == 3);
  ASSERT(u[1][1] == 4);
  ASSERT(u[1][2] == 5);
}

template <typename T>
TEST_SUITE(to_vecvec)
{
  TEST(empty<T>);
  TEST(one_token<T>);
  TEST(one_line<T>);
  TEST(two_lines<T>);
}

auto
main() -> int
{
  RUN_SUITE(to_vecvec<int>);
  return 0;
}
