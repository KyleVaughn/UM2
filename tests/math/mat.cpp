#include <um2/math/mat.hpp>

#include <um2/stdlib/numbers.hpp>

#include "../test_macros.hpp"

template <Int M, Int N, typename T>
HOSTDEV constexpr auto
makeMat() -> um2::Mat<M, N, T>
{
  um2::Mat<M, N, T> m;
  T tm = static_cast<T>(M);
  for (Int j = 0; j < N; ++j) {
    T tj = static_cast<T>(j);
    for (Int i = 0; i < M; ++i) {
      T ti = static_cast<T>(i);
      m(i, j) = tj * tm + ti;
    }
  }
  return m;
}

template <Int M, Int N, typename T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::Mat<M, N, T> m = makeMat<M, N, T>();
  for (Int j = 0; j < N; ++j) {
    for (Int i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        ASSERT_NEAR(m.col(j)[i], static_cast<T>(j * M + i), static_cast<T>(1e-6));
      } else {
        assert(m.col(j)[i] == static_cast<T>(j * M + i));
      }
    }
  }
  for (Int j = 0; j < N; ++j) {
    for (Int i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        ASSERT_NEAR(m(i, j), static_cast<T>(j * M + i), static_cast<T>(1e-6));
      } else {
        assert(m(i, j) == static_cast<T>(j * M + i));
      }
    }
  }
}

template <Int M, Int N, typename T>
HOSTDEV
TEST_CASE(mat_vec)
{
  um2::Mat<M, N, T> m = makeMat<M, N, T>();
  um2::Vec<N, T> v;
  for (Int i = 0; i < N; ++i) {
    v[i] = static_cast<T>(i);
  }
  um2::Vec<M, T> mv = m * v;
  for (Int i = 0; i < M; ++i) {
    T mv_i = 0;
    for (Int j = 0; j < N; ++j) {
      mv_i += m(i, j) * v[j];
    }
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(mv[i], mv_i, static_cast<T>(1e-6));
    } else {
      assert(mv[i] == mv_i);
    }
  }
}

template <typename T>
HOSTDEV
TEST_CASE(makeRotationMatrix)
{
  um2::Vec2<T> const v(1, 0);

  auto const v45 = um2::makeRotationMatrix(um2::pi_4<T>) * v;
  T const sqrt2_2 = um2::sqrt(static_cast<T>(2)) / 2;
  ASSERT_NEAR(v45[0], sqrt2_2, static_cast<T>(1e-6));
  ASSERT_NEAR(v45[1], sqrt2_2, static_cast<T>(1e-6));
  auto const v90 = um2::makeRotationMatrix(um2::pi_2<T>) * v;
  ASSERT_NEAR(v90[0], 0, static_cast<T>(1e-6));
  ASSERT_NEAR(v90[1], 1, static_cast<T>(1e-6));
  auto const v180 = um2::makeRotationMatrix(um2::pi<T>) * v;
  ASSERT_NEAR(v180[0], -1, static_cast<T>(1e-6));
  ASSERT_NEAR(v180[1], 0, static_cast<T>(1e-6));
  auto const v270 = um2::makeRotationMatrix(um2::pi<T> + um2::pi_2<T>) * v;
  ASSERT_NEAR(v270[0], 0, static_cast<T>(1e-6));
  ASSERT_NEAR(v270[1], -1, static_cast<T>(1e-6));
  auto const v360 = um2::makeRotationMatrix(um2::pi<T> * 2) * v;
  ASSERT_NEAR(v360[0], 1, static_cast<T>(1e-6));
  ASSERT_NEAR(v360[1], 0, static_cast<T>(1e-6));
  auto const mv45 = um2::makeRotationMatrix(-um2::pi_4<T>) * v;
  ASSERT_NEAR(mv45[0], sqrt2_2, static_cast<T>(1e-6));
  ASSERT_NEAR(mv45[1], -sqrt2_2, static_cast<T>(1e-6));
  auto const mv90 = um2::makeRotationMatrix(-um2::pi_2<T>) * v;
  ASSERT_NEAR(mv90[0], 0, static_cast<T>(1e-6));
  ASSERT_NEAR(mv90[1], -1, static_cast<T>(1e-6));
  auto const mv180 = um2::makeRotationMatrix(-um2::pi<T>) * v;
  ASSERT_NEAR(mv180[0], -1, static_cast<T>(1e-6));
  ASSERT_NEAR(mv180[1], 0, static_cast<T>(1e-6));
  auto const mv270 = um2::makeRotationMatrix(-um2::pi<T> - um2::pi_2<T>) * v;
  ASSERT_NEAR(mv270[0], 0, static_cast<T>(1e-6));
  ASSERT_NEAR(mv270[1], 1, static_cast<T>(1e-6));
}

template <typename T>
HOSTDEV
TEST_CASE(inv)
{
  um2::Mat2x2<T> m;
  m(0, 0) = 4;
  m(0, 1) = 3;
  m(1, 0) = 1;
  m(1, 1) = 1;
  um2::Mat2x2<T> const m_inv = um2::inv(m);
  ASSERT_NEAR(m_inv(0, 0), 1, static_cast<T>(1e-6));
  ASSERT_NEAR(m_inv(0, 1), -3, static_cast<T>(1e-6));
  ASSERT_NEAR(m_inv(1, 0), -1, static_cast<T>(1e-6));
  ASSERT_NEAR(m_inv(1, 1), 4, static_cast<T>(1e-6));
  um2::Mat2x2<T> const m_id = m * m_inv;
  ASSERT_NEAR(m_id(0, 0), 1, static_cast<T>(1e-6));
  ASSERT_NEAR(m_id(0, 1), 0, static_cast<T>(1e-6));
  ASSERT_NEAR(m_id(1, 0), 0, static_cast<T>(1e-6));
  ASSERT_NEAR(m_id(1, 1), 1, static_cast<T>(1e-6));
}

#if UM2_USE_CUDA
template <Int M, Int N, typename T>
MAKE_CUDA_KERNEL(accessors, M, N, T);

template <Int M, Int N, typename T>
MAKE_CUDA_KERNEL(mat_vec, M, N, T);

#endif

template <Int M, Int N, typename T>
TEST_SUITE(Mat)
{
  TEST_HOSTDEV(accessors, M, N, T);
  TEST_HOSTDEV(mat_vec, M, N, T);
  if constexpr (M == 2 && N == 2 && std::floating_point<T>) {
    TEST_HOSTDEV(makeRotationMatrix, T);
    TEST_HOSTDEV(inv, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((Mat<2, 2, float>));
  RUN_SUITE((Mat<2, 2, double>));
  RUN_SUITE((Mat<2, 2, int32_t>));
  RUN_SUITE((Mat<2, 2, uint32_t>));
  RUN_SUITE((Mat<2, 2, int64_t>));
  RUN_SUITE((Mat<2, 2, uint64_t>));

  RUN_SUITE((Mat<3, 3, float>));
  RUN_SUITE((Mat<3, 3, double>));
  RUN_SUITE((Mat<3, 3, int32_t>));
  RUN_SUITE((Mat<3, 3, uint32_t>));
  RUN_SUITE((Mat<3, 3, int64_t>));
  RUN_SUITE((Mat<3, 3, uint64_t>));
  return 0;
}
