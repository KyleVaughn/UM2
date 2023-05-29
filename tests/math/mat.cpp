#include "../test_framework.hpp"
#include <um2/math/vec.hpp>
#include <um2/math/mat.hpp>

template <len_t M, len_t N, typename T>
UM2_HOSTDEV static constexpr
auto make_mat() -> um2::Mat<M, N, T>  
{
  um2::Mat<M, N, T> m;
  T tm = static_cast<T>(M);
  for (len_t j = 0; j < N; ++j) {
    T tj = static_cast<T>(j);
    for (len_t i = 0; i < M; ++i) {
      T ti = static_cast<T>(i);
      m(i, j) = tj * tm + ti;
    }
  }
  return m;
}

template <len_t M, len_t N, typename T>    
UM2_HOSTDEV TEST_CASE(accessors)    
{
  um2::Mat<M, N, T> m = make_mat<M, N, T>();    
  for (len_t j = 0; j < N; ++j) {    
    for (len_t i = 0; i < M; ++i) {    
      if constexpr (std::floating_point<T>) {    
        EXPECT_NEAR(m.col(j)[i], static_cast<T>(j * M + i), 1e-6);
      } else {
        EXPECT_EQ(m.col(j)[i], static_cast<T>(j * M + i));
      }
    }    
  }    
  for (len_t j = 0; j < N; ++j) {    
    for (len_t i = 0; i < M; ++i) {    
      if constexpr (std::floating_point<T>) {    
        EXPECT_NEAR(m(i, j), static_cast<T>(j * M + i), 1e-6); 
      } else {
        EXPECT_EQ(m(i, j), static_cast<T>(j * M + i));
      }
    }    
  }    
}

template <len_t M, len_t N, typename T>
UM2_HOSTDEV TEST_CASE(unary_minus)
{
  um2::Mat<M, N, T> m = make_mat<M, N, T>();
  um2::Mat<M, N, T> neg_m = -m;
  for (len_t j = 0; j < N; ++j) {
    for (len_t i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        EXPECT_NEAR(neg_m(i, j), -static_cast<T>(j * M + i), 1e-6);
      } else {
        EXPECT_EQ(neg_m(i, j), -static_cast<T>(j * M + i));
      }
    }
  }
}

template <len_t M, len_t N, typename T>
UM2_HOSTDEV TEST_CASE(add)
{
  um2::Mat<M, N, T> m = make_mat<M, N, T>();
  um2::Mat<M, N, T> n = make_mat<M, N, T>();
  um2::Mat<M, N, T> sum = m + n;
  for (len_t j = 0; j < N; ++j) {
    for (len_t i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        EXPECT_NEAR(sum(i, j), 2 * static_cast<T>(j * M + i), 1e-6);
      } else {
        EXPECT_EQ(sum(i, j), 2 * static_cast<T>(j * M + i));
      }
    }
  }
}

template <len_t M, len_t N, typename T>
UM2_HOSTDEV TEST_CASE(sub)
{
  um2::Mat<M, N, T> m = make_mat<M, N, T>();
  um2::Mat<M, N, T> n = make_mat<M, N, T>();
  um2::Mat<M, N, T> diff = m - n;
  for (len_t j = 0; j < N; ++j) {
    for (len_t i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        EXPECT_NEAR(diff(i, j), 0, 1e-6);
      } else {
        EXPECT_EQ(diff(i, j), 0);
      }
    }
  }
}

template <len_t M, len_t N, typename T>
UM2_HOSTDEV TEST_CASE(scalar_mul)
{
  um2::Mat<M, N, T> m = make_mat<M, N, T>();
  um2::Mat<M, N, T> scaled = m * 2;
  for (len_t j = 0; j < N; ++j) {
    for (len_t i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        EXPECT_NEAR(scaled(i, j), 2 * static_cast<T>(j * M + i), 1e-6);
      } else {
        EXPECT_EQ(scaled(i, j), 2 * static_cast<T>(j * M + i));
      }
    }
  }
  um2::Mat<M, N, T> scaled2 = 2 * m;
  for (len_t j = 0; j < N; ++j) {
    for (len_t i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        EXPECT_NEAR(scaled2(i, j), 2 * static_cast<T>(j * M + i), 1e-6);
      } else {
        EXPECT_EQ(scaled2(i, j), 2 * static_cast<T>(j * M + i));
      }
    }
  }
}

template <len_t M, len_t N, typename T>    
UM2_HOSTDEV TEST_CASE(mat_vec)
{
  um2::Mat<M, N, T> m = make_mat<M, N, T>();    
  um2::Vec<N, T> v;    
  for (len_t i = 0; i < N; ++i) {    
    v(i) = static_cast<T>(i);    
  }    
  um2::Vec<M, T> mv = m * v;    
  for (len_t i = 0; i < M; ++i) {    
    T mv_i = 0;    
    for (len_t j = 0; j < N; ++j) {    
        mv_i += m(i, j) * v(j);    
    }    
    if constexpr (std::floating_point<T>) {    
      EXPECT_NEAR(mv[i], mv_i, 1e-6);    
    } else {    
      EXPECT_EQ(mv[i], mv_i);    
    }
  }    
}

template <len_t M, len_t N, typename T>
UM2_HOSTDEV TEST_CASE(mat_mat)
{
  um2::Mat<M, N, T> m = make_mat<M, N, T>();
  um2::Mat<M, N, T> n = make_mat<M, N, T>();
  um2::Mat<M, N, T> prod = m * n;
  for (len_t j = 0; j < N; ++j) {
    for (len_t i = 0; i < M; ++i) {
      T prod_ij = 0;
      for (len_t k = 0; k < N; ++k) {
        prod_ij += m(i, k) * n(k, j);
      }
      if constexpr (std::floating_point<T>) {
        EXPECT_NEAR(prod(i, j), prod_ij, 1e-6);
      } else {
        EXPECT_EQ(prod(i, j), prod_ij);
      }
    }
  }
}

template <len_t M, len_t N, typename T>
UM2_HOSTDEV TEST_CASE(scalar_div)
{
  um2::Mat<M, N, T> m = make_mat<M, N, T>();
  um2::Mat<M, N, T> quot = (2 * m) / 2;
  for (len_t j = 0; j < N; ++j) {
    for (len_t i = 0; i < M; ++i) {
      if constexpr (std::floating_point<T>) {
        EXPECT_NEAR(quot(i, j), static_cast<T>(j * M + i), 1e-6);
      } else {
        EXPECT_EQ(quot(i, j), static_cast<T>(j * M + i));
      }
    }
  }
}

template <len_t M, len_t N, typename T>
UM2_HOSTDEV TEST_CASE(determinant)
{
  um2::Mat<M, N, T> m = um2::Mat<M, N, T>::Identity(); 
  T detv = m.determinant();
  if constexpr (std::floating_point<T>) {
    EXPECT_NEAR(detv, 1, 1e-6);
  } else {
    EXPECT_EQ(detv, 1);
  }
}

#if UM2_ENABLE_CUDA
template <len_t M, len_t N, typename T>
MAKE_CUDA_KERNEL(accessors, M, N, T);

template <len_t M, len_t N, typename T>
MAKE_CUDA_KERNEL(unary_minus, M, N, T);

template <len_t M, len_t N, typename T>
MAKE_CUDA_KERNEL(add, M, N, T);

template <len_t M, len_t N, typename T>
MAKE_CUDA_KERNEL(sub, M, N, T);

template <len_t M, len_t N, typename T>
MAKE_CUDA_KERNEL(scalar_mul, M, N, T);

template <len_t M, len_t N, typename T>
MAKE_CUDA_KERNEL(mat_vec, M, N, T);

template <len_t M, len_t N, typename T>
MAKE_CUDA_KERNEL(mat_mat, M, N, T);

template <len_t M, len_t N, typename T>
MAKE_CUDA_KERNEL(scalar_div, M, N, T);

template <len_t M, len_t N, typename T>
MAKE_CUDA_KERNEL(determinant, M, N, T);
#endif

template <len_t M, len_t N, typename T>
TEST_SUITE(mat)
{
  TEST_HOSTDEV(accessors, 1, 1, M, N, T);
  if constexpr (!std::unsigned_integral<T>) {
    TEST_HOSTDEV(unary_minus, 1, 1, M, N, T);
  }
  TEST_HOSTDEV(add, 1, 1, M, N, T);
  TEST_HOSTDEV(sub, 1, 1, M, N, T);
  TEST_HOSTDEV(scalar_mul, 1, 1, M, N, T);
  TEST_HOSTDEV(mat_vec, 1, 1, M, N, T);
  TEST_HOSTDEV(mat_mat, 1, 1, M, N, T);
  TEST_HOSTDEV(scalar_div, 1, 1, M, N, T);
  if constexpr (!std::unsigned_integral<T>) {
    TEST_HOSTDEV(determinant, 1, 1, M, N, T);
  }
}

auto main() -> int
{
  RUN_TESTS((mat<2, 2, float>));
  RUN_TESTS((mat<2, 2, double>));
  RUN_TESTS((mat<2, 2, int32_t>));
  RUN_TESTS((mat<2, 2, uint32_t>));
  RUN_TESTS((mat<2, 2, int64_t>));
  RUN_TESTS((mat<2, 2, uint64_t>));

  RUN_TESTS((mat<3, 3, float>));
  RUN_TESTS((mat<3, 3, double>));
  RUN_TESTS((mat<3, 3, int32_t>));
  RUN_TESTS((mat<3, 3, uint32_t>));
  RUN_TESTS((mat<3, 3, int64_t>));
  RUN_TESTS((mat<3, 3, uint64_t>));

  RUN_TESTS((mat<4, 4, float>));
  RUN_TESTS((mat<4, 4, double>));
  RUN_TESTS((mat<4, 4, int32_t>));
  RUN_TESTS((mat<4, 4, uint32_t>));
  RUN_TESTS((mat<4, 4, int64_t>));
  RUN_TESTS((mat<4, 4, uint64_t>));
  return 0;
}
