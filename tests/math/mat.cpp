#include "../test_framework.hpp"
#include <um2/math/mat.hpp>

//template <typename T>
//UM2_HOSTDEV static constexpr auto make_mat() -> um2::Mat2x2<T>
//{
//  um2::Mat2x2<T> mat;
//  mat.cols[0][0] = static_cast<T>(0);
//  mat.cols[0][1] = static_cast<T>(1);
//  mat.cols[1][0] = static_cast<T>(2);
//  mat.cols[1][1] = static_cast<T>(3);
//  return mat;
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(accessors)
//{
//  um2::Mat2x2<T> m = make_mat<T>();
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(m[0][0], 0, 1e-6);
//    EXPECT_NEAR(m[0][1], 1, 1e-6);
//    EXPECT_NEAR(m[1][0], 2, 1e-6);
//    EXPECT_NEAR(m[1][1], 3, 1e-6);
//  } else {
//    EXPECT_EQ(m[0][0], 0);
//    EXPECT_EQ(m[0][1], 1);
//    EXPECT_EQ(m[1][0], 2);
//    EXPECT_EQ(m[1][1], 3);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(unary_minus)
//{
//  um2::Mat2x2<T> m = make_mat<T>();
//  um2::Mat2x2<T> neg_m = -m;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(neg_m[0][0], -0, 1e-6);
//    EXPECT_NEAR(neg_m[0][1], -1, 1e-6);
//    EXPECT_NEAR(neg_m[1][0], -2, 1e-6);
//    EXPECT_NEAR(neg_m[1][1], -3, 1e-6);
//  } else {
//    EXPECT_EQ(neg_m[0][0], -0);
//    EXPECT_EQ(neg_m[0][1], -1);
//    EXPECT_EQ(neg_m[1][0], -2);
//    EXPECT_EQ(neg_m[1][1], -3);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(add)
//{
//  um2::Mat2x2<T> m = make_mat<T>();
//  um2::Mat2x2<T> n = make_mat<T>();
//  um2::Mat2x2<T> sum = m + n;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(sum[0][0], 0, 1e-6);
//    EXPECT_NEAR(sum[0][1], 2, 1e-6);
//    EXPECT_NEAR(sum[1][0], 4, 1e-6);
//    EXPECT_NEAR(sum[1][1], 6, 1e-6);
//  } else {
//    EXPECT_EQ(sum[0][0], 0);
//    EXPECT_EQ(sum[0][1], 2);
//    EXPECT_EQ(sum[1][0], 4);
//    EXPECT_EQ(sum[1][1], 6);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(sub)
//{
//  um2::Mat2x2<T> m = make_mat<T>();
//  um2::Mat2x2<T> n = make_mat<T>();
//  um2::Mat2x2<T> diff = m - n;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(diff[0][0], 0, 1e-6);
//    EXPECT_NEAR(diff[0][1], 0, 1e-6);
//    EXPECT_NEAR(diff[1][0], 0, 1e-6);
//    EXPECT_NEAR(diff[1][1], 0, 1e-6);
//  } else {
//    EXPECT_EQ(diff[0][0], 0);
//    EXPECT_EQ(diff[0][1], 0);
//    EXPECT_EQ(diff[1][0], 0);
//    EXPECT_EQ(diff[1][1], 0);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(scalar_mul)
//{
//  um2::Mat2x2<T> m = make_mat<T>();
//  um2::Mat2x2<T> scaled = m * 2;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(scaled[0][0], 0, 1e-6);
//    EXPECT_NEAR(scaled[0][1], 2, 1e-6);
//    EXPECT_NEAR(scaled[1][0], 4, 1e-6);
//    EXPECT_NEAR(scaled[1][1], 6, 1e-6);
//  } else {
//    EXPECT_EQ(scaled[0][0], 0);
//    EXPECT_EQ(scaled[0][1], 2);
//    EXPECT_EQ(scaled[1][0], 4);
//    EXPECT_EQ(scaled[1][1], 6);
//  }
//  um2::Mat2x2<T> scaled2 = 2 * m;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(scaled2[0][0], 0, 1e-6);
//    EXPECT_NEAR(scaled2[0][1], 2, 1e-6);
//    EXPECT_NEAR(scaled2[1][0], 4, 1e-6);
//    EXPECT_NEAR(scaled2[1][1], 6, 1e-6);
//  } else {
//    EXPECT_EQ(scaled2[0][0], 0);
//    EXPECT_EQ(scaled2[0][1], 2);
//    EXPECT_EQ(scaled2[1][0], 4);
//    EXPECT_EQ(scaled2[1][1], 6);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(mat_vec)
//{
//  um2::Mat2x2<T> m = make_mat<T>();
//  um2::Vec2<T> v(1, 2);
//  um2::Vec2<T> mv = m * v;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(mv[0], 4, 1e-6);
//    EXPECT_NEAR(mv[1], 7, 1e-6);
//  } else {
//    EXPECT_EQ(mv[0], 4);
//    EXPECT_EQ(mv[1], 7);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(mat_mat)
//{
//  um2::Mat2x2<T> m = make_mat<T>();
//  um2::Mat2x2<T> n = make_mat<T>();
//  um2::Mat2x2<T> prod = m * n;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(prod[0][0], 2, 1e-6);
//    EXPECT_NEAR(prod[0][1], 3, 1e-6);
//    EXPECT_NEAR(prod[1][0], 6, 1e-6);
//    EXPECT_NEAR(prod[1][1], 11, 1e-6);
//  } else {
//    EXPECT_EQ(prod[0][0], 2);
//    EXPECT_EQ(prod[0][1], 3);
//    EXPECT_EQ(prod[1][0], 6);
//    EXPECT_EQ(prod[1][1], 11);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(scalar_div)
//{
//  um2::Mat2x2<T> m = make_mat<T>();
//  um2::Mat2x2<T> quot = (2 * m) / 2;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(quot[0][0], 0, 1e-6);
//    EXPECT_NEAR(quot[0][1], 1, 1e-6);
//    EXPECT_NEAR(quot[1][0], 2, 1e-6);
//    EXPECT_NEAR(quot[1][1], 3, 1e-6);
//  } else {
//    EXPECT_EQ(quot[0][0], 0);
//    EXPECT_EQ(quot[0][1], 1);
//    EXPECT_EQ(quot[1][0], 2);
//    EXPECT_EQ(quot[1][1], 3);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(mat2x2_det)
//{
//  um2::Mat2x2<T> m = make_mat<T>();
//  T detv = det(m);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(detv, -2, 1e-6);
//  } else {
//    EXPECT_EQ(detv, -2);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(mat2x2_inv)
//{
//  um2::Mat2x2<T> m = make_mat<T>();
//  um2::Mat2x2<T> m_inv = inv(m);
//  um2::Mat2x2<T> identity = m * m_inv;
//  EXPECT_NEAR(identity[0][0], 1, 1e-6);
//  EXPECT_NEAR(identity[0][1], 0, 1e-6);
//  EXPECT_NEAR(identity[1][0], 0, 1e-6);
//  EXPECT_NEAR(identity[1][1], 1, 1e-6);
//}
//
//#if UM2_ENABLE_CUDA
//template <typename T>
//MAKE_CUDA_KERNEL(accessors, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(unary_minus, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(add, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(sub, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(scalar_mul, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(mat_vec, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(mat_mat, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(scalar_div, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(mat2x2_det, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(mat2x2_inv, T);
//#endif
//
//template <typename T>
//TEST_SUITE(mat2x2)
//{
//  TEST_HOSTDEV(accessors, 1, 1, T);
//  if constexpr (!std::unsigned_integral<T>) {
//    TEST_HOSTDEV(unary_minus, 1, 1, T);
//  }
//  TEST_HOSTDEV(add, 1, 1, T);
//  TEST_HOSTDEV(sub, 1, 1, T);
//  TEST_HOSTDEV(scalar_mul, 1, 1, T);
//  TEST_HOSTDEV(mat_vec, 1, 1, T);
//  TEST_HOSTDEV(mat_mat, 1, 1, T);
//  TEST_HOSTDEV(scalar_div, 1, 1, T);
//  if constexpr (!std::unsigned_integral<T>) {
//    TEST_HOSTDEV(mat2x2_det, 1, 1, T);
//  }
//  if constexpr (std::floating_point<T>) {
//    TEST_HOSTDEV(mat2x2_inv, 1, 1, T);
//  }
//}

auto main() -> int
{
//  RUN_TESTS(mat2x2<float>);
//  RUN_TESTS(mat2x2<double>);
//  RUN_TESTS(mat2x2<int32_t>);
//  RUN_TESTS(mat2x2<uint32_t>);
  return 0;
}
