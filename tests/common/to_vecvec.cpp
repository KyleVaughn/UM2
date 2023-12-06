#include <um2/common/to_vecvec.hpp>

#include "../test_macros.hpp"

template <typename T>
TEST_CASE(empty)
{
  std::vector<std::vector<T>> const v = um2::to_vecvec<T>(R"()");
  ASSERT(v.empty());

  um2::Vector<um2::Vector<T>> const u = um2::toVecVec<T>(R"()");
  ASSERT(u.empty());
}

template <typename T>
TEST_CASE(one_token)
{
  std::vector<std::vector<T>> const v = um2::to_vecvec<T>(R"(1)");
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 1);
  ASSERT(v[0][0] == 1);

  um2::Vector<um2::Vector<T>> const u = um2::toVecVec<T>(R"(1)");
  ASSERT(u.size() == 1);
  ASSERT(u[0].size() == 1);
  ASSERT(u[0][0] == 1);
}

template <typename T>
TEST_CASE(one_line)
{
  std::vector<std::vector<T>> v = um2::to_vecvec<T>(R"(1 2)");
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  v.clear();

  um2::Vector<um2::Vector<T>> u = um2::toVecVec<T>(R"(1 2)");
  ASSERT(u.size() == 1);
  ASSERT(u[0].size() == 2);
  ASSERT(u[0][0] == 1);
  ASSERT(u[0][1] == 2);
  u.clear();

  v = um2::to_vecvec<T>(R"(
    1 2)");
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  v.clear();

  u = um2::toVecVec<T>(R"(
    1 2)");
  ASSERT(u.size() == 1);
  ASSERT(u[0].size() == 2);
  ASSERT(u[0][0] == 1);
  ASSERT(u[0][1] == 2);
  u.clear();

  v = um2::to_vecvec<T>(R"(1 2
    )");
  ASSERT(v.size() == 1);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  v.clear();

  u = um2::toVecVec<T>(R"(1 2
    )");
  ASSERT(u.size() == 1);
  ASSERT(u[0].size() == 2);
  ASSERT(u[0][0] == 1);
  ASSERT(u[0][1] == 2);
  u.clear();
}

template <typename T>
TEST_CASE(two_lines)
{
  std::vector<std::vector<T>> v = um2::to_vecvec<T>(R"(
    1 2
    3 4)");
  ASSERT(v.size() == 2);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  ASSERT(v[1].size() == 2);
  ASSERT(v[1][0] == 3);
  ASSERT(v[1][1] == 4);

  v = um2::to_vecvec<T>(R"(
    1 2
    3 4 5)");
  ASSERT(v.size() == 2);
  ASSERT(v[0].size() == 2);
  ASSERT(v[0][0] == 1);
  ASSERT(v[0][1] == 2);
  ASSERT(v[1].size() == 3);
  ASSERT(v[1][0] == 3);
  ASSERT(v[1][1] == 4);
  ASSERT(v[1][2] == 5);

  um2::Vector<um2::Vector<T>> u = um2::toVecVec<T>(R"(
    1 2
    3 4)");
  ASSERT(u.size() == 2);
  ASSERT(u[0].size() == 2);
  ASSERT(u[0][0] == 1);
  ASSERT(u[0][1] == 2);
  ASSERT(u[1].size() == 2);
  ASSERT(u[1][0] == 3);
  ASSERT(u[1][1] == 4);

  u = um2::toVecVec<T>(R"(
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
