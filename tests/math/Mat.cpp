#include <um2/math/Mat.hpp>

#include "../test_macros.hpp"

template <Size M, Size N, typename T>
HOSTDEV static constexpr auto
makeMat() -> um2::Mat<M, N, T>
{
  um2::Mat<M, N, T> m;
  T tm = static_cast<T>(M);
  for (Size j = 0; j < N; ++j) {
    T tj = static_cast<T>(j);
    for (Size i = 0; i < M; ++i) {
      T ti = static_cast<T>(i);
      m(i, j) = tj * tm + ti;
    }
  }
  return m;
}

template <Size M, Size N, typename T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::Mat<M, N, T> m = makeMat<M, N, T>();
  for (Size j = 0; j < N; ++j) {
    for (Size i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        ASSERT_NEAR(m.col(j)[i], static_cast<T>(j * M + i), static_cast<T>(1e-6));
      } else {
        assert(m.col(j)[i] == static_cast<T>(j * M + i));
      }
    }
  }
  for (Size j = 0; j < N; ++j) {
    for (Size i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        ASSERT_NEAR(m(i, j), static_cast<T>(j * M + i), static_cast<T>(1e-6));
      } else {
        assert(m(i, j) == static_cast<T>(j * M + i));
      }
    }
  }
}

// template <Size M, Size N, typename T>
// HOSTDEV
// TEST_CASE(unary_minus)
//{
//   um2::Mat<M, N, T> m = makeMat<M, N, T>();
//   um2::Mat<M, N, T> neg_m = -m;
//   for (Size j = 0; j < N; ++j) {
//     for (Size i = 0; i < M; ++i) {
//       if constexpr (std::floating_point<T>) {
//         ASSERT_NEAR(neg_m(i, j), -static_cast<T>(j * M + i), static_cast<T>(1e-6));
//       } else {
//         assert(neg_m(i, j) == -static_cast<T>(j * M + i));
//       }
//     }
//   }
// }
//
// template <Size M, Size N, typename T>
// HOSTDEV
// TEST_CASE(add)
//{
//   um2::Mat<M, N, T> m = makeMat<M, N, T>();
//   um2::Mat<M, N, T> n = makeMat<M, N, T>();
//   um2::Mat<M, N, T> sum = m + n;
//   for (Size j = 0; j < N; ++j) {
//     for (Size i = 0; i < M; ++i) {
//       if constexpr (std::floating_point<T>) {
//         ASSERT_NEAR(sum(i, j), 2 * static_cast<T>(j * M + i), static_cast<T>(1e-6));
//       } else {
//         assert(sum(i, j) == 2 * static_cast<T>(j * M + i));
//       }
//     }
//   }
// }
//
// template <Size M, Size N, typename T>
// HOSTDEV
// TEST_CASE(sub)
//{
//   um2::Mat<M, N, T> m = makeMat<M, N, T>();
//   um2::Mat<M, N, T> n = makeMat<M, N, T>();
//   um2::Mat<M, N, T> diff = m - n;
//   for (Size j = 0; j < N; ++j) {
//     for (Size i = 0; i < M; ++i) {
//       if constexpr (std::floating_point<T>) {
//         ASSERT_NEAR(diff(i, j), 0, static_cast<T>(1e-6));
//       } else {
//         assert(diff(i, j) == 0);
//       }
//     }
//   }
// }
//
// template <Size M, Size N, typename T>
// HOSTDEV
// TEST_CASE(scalar_mul)
//{
//   um2::Mat<M, N, T> m = makeMat<M, N, T>();
//   um2::Mat<M, N, T> scaled = m * 2;
//   for (Size j = 0; j < N; ++j) {
//     for (Size i = 0; i < M; ++i) {
//       if constexpr (std::floating_point<T>) {
//         ASSERT_NEAR(scaled(i, j), 2 * static_cast<T>(j * M + i), static_cast<T>(1e-6));
//       } else {
//         assert(scaled(i, j) == 2 * static_cast<T>(j * M + i));
//       }
//     }
//   }
//   um2::Mat<M, N, T> scaled2 = 2 * m;
//   for (Size j = 0; j < N; ++j) {
//     for (Size i = 0; i < M; ++i) {
//       if constexpr (std::floating_point<T>) {
//         ASSERT_NEAR(scaled2(i, j), 2 * static_cast<T>(j * M + i),
//         static_cast<T>(1e-6));
//       } else {
//         assert(scaled2(i, j) == 2 * static_cast<T>(j * M + i));
//       }
//     }
//   }
// }
//
// template <Size M, Size N, typename T>
// HOSTDEV
// TEST_CASE(mat_vec)
//{
//   um2::Mat<M, N, T> m = makeMat<M, N, T>();
//   um2::Vec<N, T> v;
//   for (Size i = 0; i < N; ++i) {
//     v(i) = static_cast<T>(i);
//   }
//   um2::Vec<M, T> mv = m * v;
//   for (Size i = 0; i < M; ++i) {
//     T mv_i = 0;
//     for (Size j = 0; j < N; ++j) {
//       mv_i += m(i, j) * v(j);
//     }
//     if constexpr (std::floating_point<T>) {
//       ASSERT_NEAR(mv[i], mv_i, static_cast<T>(1e-6));
//     } else {
//       assert(mv[i] == mv_i);
//     }
//   }
// }
//
// template <Size M, Size N, typename T>
// HOSTDEV
// TEST_CASE(mat_mat)
//{
//   um2::Mat<M, N, T> m = makeMat<M, N, T>();
//   um2::Mat<M, N, T> n = makeMat<M, N, T>();
//   um2::Mat<M, N, T> prod = m * n;
//   for (Size j = 0; j < N; ++j) {
//     for (Size i = 0; i < M; ++i) {
//       T prod_ij = 0;
//       for (Size k = 0; k < N; ++k) {
//         prod_ij += m(i, k) * n(k, j);
//       }
//       if constexpr (std::floating_point<T>) {
//         ASSERT_NEAR(prod(i, j), prod_ij, static_cast<T>(1e-6));
//       } else {
//         assert(prod(i, j) == prod_ij);
//       }
//     }
//   }
// }
//
// template <Size M, Size N, typename T>
// HOSTDEV
// TEST_CASE(scalar_div)
//{
//   um2::Mat<M, N, T> m = makeMat<M, N, T>();
//   um2::Mat<M, N, T> quot = (2 * m) / 2;
//   for (Size j = 0; j < N; ++j) {
//     for (Size i = 0; i < M; ++i) {
//       if constexpr (std::floating_point<T>) {
//         ASSERT_NEAR(quot(i, j), static_cast<T>(j * M + i), static_cast<T>(1e-6));
//       } else {
//         assert(quot(i, j) == static_cast<T>(j * M + i));
//       }
//     }
//   }
// }
//
// template <Size M, Size N, typename T>
// HOSTDEV
// TEST_CASE(determinant)
//{
//   um2::Mat<M, N, T> m = um2::Mat<M, N, T>::Identity();
//   T detv = m.determinant();
//   if constexpr (std::floating_point<T>) {
//     ASSERT_NEAR(detv, 1, static_cast<T>(1e-6));
//   } else {
//     assert(detv == 1);
//   }
// }

#if UM2_ENABLE_CUDA
template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(accessors, M, N, T);

template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(unary_minus, M, N, T);

template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(add, M, N, T);

template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(sub, M, N, T);

template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(scalar_mul, M, N, T);

template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(mat_vec, M, N, T);

template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(mat_mat, M, N, T);

template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(scalar_div, M, N, T);

template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(determinant, M, N, T);
#endif

template <Size M, Size N, typename T>
TEST_SUITE(Mat)
{
  TEST_HOSTDEV(accessors, 1, 1, M, N, T);
  //  if constexpr (!std::unsigned_integral<T>) {
  //    TEST_HOSTDEV(unary_minus, 1, 1, M, N, T);
  //  }
  //  TEST_HOSTDEV(add, 1, 1, M, N, T);
  //  TEST_HOSTDEV(sub, 1, 1, M, N, T);
  //  TEST_HOSTDEV(scalar_mul, 1, 1, M, N, T);
  //  TEST_HOSTDEV(mat_vec, 1, 1, M, N, T);
  //  TEST_HOSTDEV(mat_mat, 1, 1, M, N, T);
  //  TEST_HOSTDEV(scalar_div, 1, 1, M, N, T);
  //  if constexpr (!std::unsigned_integral<T>) {
  //    TEST_HOSTDEV(determinant, 1, 1, M, N, T);
  //  }
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

  RUN_SUITE((Mat<4, 4, float>));
  RUN_SUITE((Mat<4, 4, double>));
  RUN_SUITE((Mat<4, 4, int32_t>));
  RUN_SUITE((Mat<4, 4, uint32_t>));
  RUN_SUITE((Mat<4, 4, int64_t>));
  RUN_SUITE((Mat<4, 4, uint64_t>));
  return 0;
}
