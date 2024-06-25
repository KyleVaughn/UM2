#include <um2/config.hpp>
#include <um2/math/mat.hpp>
#include <um2/math/vec.hpp>
#include <um2/stdlib/numbers.hpp>

#include "../test_macros.hpp"

// NOLINTNEXTLINE(misc-include-cleaner)
#include <concepts>

#include <cstdint>

template <Int M, Int N, class T>
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

template <Int M, Int N, class T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::Mat<M, N, T> m = makeMat<M, N, T>();
  for (Int j = 0; j < N; ++j) {
    for (Int i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        ASSERT_NEAR(m.col(j)[i], static_cast<T>(j * M + i), static_cast<T>(1e-6));
      } else {
        ASSERT(m.col(j)[i] == static_cast<T>(j * M + i));
      }
    }
  }
  for (Int j = 0; j < N; ++j) {
    for (Int i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        ASSERT_NEAR(m(i, j), static_cast<T>(j * M + i), static_cast<T>(1e-6));
      } else {
        ASSERT(m(i, j) == static_cast<T>(j * M + i));
      }
    }
  }
}

template <Int M, Int N, class T>
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
      ASSERT(mv[i] == mv_i);
    }
  }
}

template <class T>
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

template <class T>
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

template <class T>
HOSTDEV
TEST_CASE(accessors_real)
{
  auto const eps = castIfNot<T>(1e-6);
  um2::Mat<3, 4, T> m;
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      m(i, j) = static_cast<T>(j * 3 + i);
    }
  }
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      ASSERT_NEAR(m(i, j), static_cast<T>(j * 3 + i), eps);
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(mat_vec_real)
{
  auto const eps = castIfNot<T>(1e-6);
  Int constexpr n = 3;
  um2::Mat3x3<T> const identity = um2::Mat3x3<T>::identity();
  um2::Vec3<T> v;
  for (Int i = 0; i < n; ++i) {
    v[i] = static_cast<T>(i);
  }
  auto mv = identity * v;
  for (Int i = 0; i < n; ++i) {
    ASSERT_NEAR(mv[i], v[i], eps);
  }

  //  5  4  3     100    1000
  //  8  9  5  x  80  =  1820
  //  6  5  3     60     1180
  // 11  9  6            2180

  um2::Mat<4, 3, T> a;
  a(0) = static_cast<T>(5);
  a(1) = static_cast<T>(8);
  a(2) = static_cast<T>(6);
  a(3) = static_cast<T>(11);

  a(4) = static_cast<T>(4);
  a(5) = static_cast<T>(9);
  a(6) = static_cast<T>(5);
  a(7) = static_cast<T>(9);

  a(8) = static_cast<T>(3);
  a(9) = static_cast<T>(5);
  a(10) = static_cast<T>(3);
  a(11) = static_cast<T>(6);

  v[0] = static_cast<T>(100);
  v[1] = static_cast<T>(80);
  v[2] = static_cast<T>(60);

  auto const b = a * v;
  ASSERT_NEAR(b[0], static_cast<T>(1000), eps);
  ASSERT_NEAR(b[1], static_cast<T>(1820), eps);
  ASSERT_NEAR(b[2], static_cast<T>(1180), eps);
  ASSERT_NEAR(b[3], static_cast<T>(2180), eps);
}

template <class T>
HOSTDEV
TEST_CASE(add_sub_real)
{
  auto const eps = castIfNot<T>(1e-6);
  um2::Mat<3, 4, T> a;
  um2::Mat<3, 4, T> b;
  um2::Mat<3, 4, T> c;
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      a(i, j) = static_cast<T>(j * 3 + i);
      b(i, j) = static_cast<T>(j * 3 + i + 1);
      c(i, j) = static_cast<T>(2 * j * 3 + 2 * i + 1);
    }
  }
  auto const d = a + b;
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      ASSERT_NEAR(d(i, j), c(i, j), eps);
    }
  }
  auto const e = b - a;
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      ASSERT_NEAR(e(i, j), static_cast<T>(1), eps);
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(mat_mul_real)
{
  auto const eps = castIfNot<T>(1e-6);
  // a = 1  0  1
  //     2  1  1
  //     0  1  1
  //     1  1  2
  //
  // b = 1  2  1
  //     2  3  1
  //     4  2  2
  //
  // a * b =  5  4  3
  //          8  9  5
  //          6  5  3
  //         11  9  6
  um2::Mat<4, 3, T> a;
  a(0) = 1;
  a(1) = 2;
  a(2) = 0;
  a(3) = 1;

  a(4) = 0;
  a(5) = 1;
  a(6) = 1;
  a(7) = 1;

  a(8) = 1;
  a(9) = 1;
  a(10) = 1;
  a(11) = 2;
  um2::Mat<3, 3, T> b;
  b(0) = 1;
  b(1) = 2;
  b(2) = 4;

  b(3) = 2;
  b(4) = 3;
  b(5) = 2;

  b(6) = 1;
  b(7) = 1;
  b(8) = 2;

  auto const ab = a * b;
  ASSERT_NEAR(ab(0), 5, eps);
  ASSERT_NEAR(ab(1), 8, eps);
  ASSERT_NEAR(ab(2), 6, eps);
  ASSERT_NEAR(ab(3), 11, eps);

  ASSERT_NEAR(ab(4), 4, eps);
  ASSERT_NEAR(ab(5), 9, eps);
  ASSERT_NEAR(ab(6), 5, eps);
  ASSERT_NEAR(ab(7), 9, eps);

  ASSERT_NEAR(ab(8), 3, eps);
  ASSERT_NEAR(ab(9), 5, eps);
  ASSERT_NEAR(ab(10), 3, eps);
  ASSERT_NEAR(ab(11), 6, eps);
}

#if UM2_USE_CUDA
template <Int M, Int N, class T>
MAKE_CUDA_KERNEL(accessors, M, N, T);

template <Int M, Int N, class T>
MAKE_CUDA_KERNEL(mat_vec, M, N, T);

template <class T>
MAKE_CUDA_KERNEL(makeRotationMatrix, T);

template <class T>
MAKE_CUDA_KERNEL(inv, T);

template <class T>
MAKE_CUDA_KERNEL(accessors_real, T);

template <class T>
MAKE_CUDA_KERNEL(mat_vec_real, T);

template <class T>
MAKE_CUDA_KERNEL(add_sub_real, T);

template <class T>
MAKE_CUDA_KERNEL(mat_mul_real, T);

#endif

template <Int M, Int N, class T>
TEST_SUITE(Mat)
{
  TEST_HOSTDEV(accessors, M, N, T);
  TEST_HOSTDEV(mat_vec, M, N, T);
  if constexpr (M == 2 && N == 2 && std::floating_point<T>) {
    TEST_HOSTDEV(makeRotationMatrix, T);
    TEST_HOSTDEV(inv, T);
  }
}

template <class T>
TEST_SUITE(Mat_real)
{
  TEST_HOSTDEV(accessors_real, T);
  TEST_HOSTDEV(mat_vec_real, T);
  TEST_HOSTDEV(add_sub_real, T);
  TEST_HOSTDEV(mat_mul_real, T);
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

  RUN_SUITE(Mat_real<float>);
  RUN_SUITE(Mat_real<double>);
  return 0;
}
