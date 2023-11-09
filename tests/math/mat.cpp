#include <um2/math/mat.hpp>

#include "../test_macros.hpp"

template <Size M, Size N, typename T>
HOSTDEV constexpr auto
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

template <Size M, Size N, typename T>
HOSTDEV
TEST_CASE(mat_vec)
{
  um2::Mat<M, N, T> m = makeMat<M, N, T>();
  um2::Vec<N, T> v;
  for (Size i = 0; i < N; ++i) {
    v[i] = static_cast<T>(i);
  }
  um2::Vec<M, T> mv = m * v;
  for (Size i = 0; i < M; ++i) {
    T mv_i = 0;
    for (Size j = 0; j < N; ++j) {
      mv_i += m(i, j) * v[j];
    }
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(mv[i], mv_i, static_cast<T>(1e-6));
    } else {
      assert(mv[i] == mv_i);
    }
  }
}

#if UM2_USE_CUDA
template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(accessors, M, N, T);

template <Size M, Size N, typename T>
MAKE_CUDA_KERNEL(mat_vec, M, N, T);

#endif

template <Size M, Size N, typename T>
TEST_SUITE(Mat)
{
  TEST_HOSTDEV(accessors, 1, 1, M, N, T);
  TEST_HOSTDEV(mat_vec, 1, 1, M, N, T);
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
