#include "../test_framework.hpp"
#include <um2/math/vec.hpp>

template <typename T>
UM2_HOSTDEV TEST_CASE(accessor)
{
  um2::Vec2<T> v(1, 2);
  if constexpr (std::floating_point<T>) {
    EXPECT_NEAR(v[0], 1, 1e-6);
    EXPECT_NEAR(v[1], 2, 1e-6);
  } else {
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
  }
}
//template <typename T>
//UM2_HOSTDEV TEST_CASE(unary_minus)
//{
//  um2::Vec2<T> v0(1, -1);
//  um2::Vec2<T> v1 = -v0;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v1.x, -1, 1e-6);
//    EXPECT_NEAR(v1.y, 1, 1e-6);
//  } else {
//    EXPECT_EQ(v1.x, -1);
//    EXPECT_EQ(v1.y, 1);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(compound_add)
//{
//  um2::Vec2<T> v0(1, 2);
//  um2::Vec2<T> v1(3, 4);
//  v0 += v1;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v0.x, 4, 1e-6);
//    EXPECT_NEAR(v0.y, 6, 1e-6);
//  } else {
//    EXPECT_EQ(v0.x, 4);
//    EXPECT_EQ(v0.y, 6);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(compound_sub)
//{
//  um2::Vec2<T> v0(3, 4);
//  um2::Vec2<T> v1(1, 2);
//  v0 -= v1;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v0.x, 2, 1e-6);
//    EXPECT_NEAR(v0.y, 2, 1e-6);
//  } else {
//    EXPECT_EQ(v0.x, 2);
//    EXPECT_EQ(v0.y, 2);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(compound_mul)
//{
//  um2::Vec2<T> v0(1, 2);
//  um2::Vec2<T> v1(3, 4);
//  v0 *= v1;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v0.x, 3, 1e-6);
//    EXPECT_NEAR(v0.y, 8, 1e-6);
//  } else {
//    EXPECT_EQ(v0.x, 3);
//    EXPECT_EQ(v0.y, 8);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(compound_div)
//{
//  um2::Vec2<T> v0(2, 8);
//  um2::Vec2<T> v1(1, 2);
//  v0 /= v1;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v0.x, 2, 1e-6);
//    EXPECT_NEAR(v0.y, 4, 1e-6);
//  } else {
//    EXPECT_EQ(v0.x, 2);
//    EXPECT_EQ(v0.y, 4);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(add)
//{
//  um2::Vec2<T> v0(1, 2);
//  um2::Vec2<T> v1(3, 4);
//  um2::Vec2<T> v = v0 + v1;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 4, 1e-6);
//    EXPECT_NEAR(v.y, 6, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 4);
//    EXPECT_EQ(v.y, 6);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(sub)
//{
//  um2::Vec2<T> v0(3, 4);
//  um2::Vec2<T> v1(1, 2);
//  um2::Vec2<T> v = v0 - v1;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 2, 1e-6);
//    EXPECT_NEAR(v.y, 2, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 2);
//    EXPECT_EQ(v.y, 2);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(mul)
//{
//  um2::Vec2<T> v0(1, 2);
//  um2::Vec2<T> v1(3, 4);
//  um2::Vec2<T> v = v0 * v1;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 3, 1e-6);
//    EXPECT_NEAR(v.y, 8, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 3);
//    EXPECT_EQ(v.y, 8);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(div)
//{
//  um2::Vec2<T> v0(2, 8);
//  um2::Vec2<T> v1(1, 2);
//  um2::Vec2<T> v = v0 / v1;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 2, 1e-6);
//    EXPECT_NEAR(v.y, 4, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 2);
//    EXPECT_EQ(v.y, 4);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(compound_scalar_add)
//{
//  um2::Vec2<T> v(1, 2);
//  v += 2;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 3, 1e-6);
//    EXPECT_NEAR(v.y, 4, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 3);
//    EXPECT_EQ(v.y, 4);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(compound_scalar_sub)
//{
//  um2::Vec2<T> v(2, 3);
//  v -= 2;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 0, 1e-6);
//    EXPECT_NEAR(v.y, 1, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 0);
//    EXPECT_EQ(v.y, 1);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(compound_scalar_mul)
//{
//  um2::Vec2<T> v(1, 2);
//  v *= 2;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 2, 1e-6);
//    EXPECT_NEAR(v.y, 4, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 2);
//    EXPECT_EQ(v.y, 4);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(compound_scalar_div)
//{
//  um2::Vec2<T> v(2, 8);
//  v /= 2;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 1, 1e-6);
//    EXPECT_NEAR(v.y, 4, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 1);
//    EXPECT_EQ(v.y, 4);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(scalar_add)
//{
//  um2::Vec2<T> v0(1, 2);
//  um2::Vec2<T> v = v0 + 2;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 3, 1e-6);
//    EXPECT_NEAR(v.y, 4, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 3);
//    EXPECT_EQ(v.y, 4);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(scalar_sub)
//{
//  um2::Vec2<T> v0(2, 3);
//  um2::Vec2<T> v = v0 - 2;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 0, 1e-6);
//    EXPECT_NEAR(v.y, 1, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 0);
//    EXPECT_EQ(v.y, 1);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(scalar_mul)
//{
//  um2::Vec2<T> v0(1, 2);
//  um2::Vec2<T> v = v0 * 2;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 2, 1e-6);
//    EXPECT_NEAR(v.y, 4, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 2);
//    EXPECT_EQ(v.y, 4);
//  }
//  v = 3 * v0;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 3, 1e-6);
//    EXPECT_NEAR(v.y, 6, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 3);
//    EXPECT_EQ(v.y, 6);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(scalar_div)
//{
//  um2::Vec2<T> v0(2, 8);
//  um2::Vec2<T> v = v0 / 2;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 1, 1e-6);
//    EXPECT_NEAR(v.y, 4, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 1);
//    EXPECT_EQ(v.y, 4);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(min)
//{
//  um2::Vec2<T> v0(1, 2);
//  um2::Vec2<T> v1(3, 0);
//  um2::Vec2<T> v = um2::min(v0, v1);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 1, 1e-6);
//    EXPECT_NEAR(v.y, 0, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 1);
//    EXPECT_EQ(v.y, 0);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(max)
//{
//  um2::Vec2<T> v0(1, 2);
//  um2::Vec2<T> v1(3, 0);
//  um2::Vec2<T> v = um2::max(v0, v1);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.x, 3, 1e-6);
//    EXPECT_NEAR(v.y, 2, 1e-6);
//  } else {
//    EXPECT_EQ(v.x, 3);
//    EXPECT_EQ(v.y, 2);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(dot)
//{
//  um2::Vec2<T> v0(1, 2);
//  um2::Vec2<T> v1(3, 4);
//  T dot = um2::dot(v0, v1);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(dot, 11, 1e-6);
//  } else {
//    EXPECT_EQ(dot, 11);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(cross)
//{
//  um2::Vec2<T> v0(1, 2);
//  um2::Vec2<T> v1(3, 10);
//  T cross = um2::cross(v0, v1);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(cross, 4, 1e-6);
//  } else {
//    EXPECT_EQ(cross, 4);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(norm2)
//{
//  um2::Vec2<T> v(1, 2);
//  T norm2 = um2::norm2(v);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(norm2, 5, 1e-6);
//  } else {
//    EXPECT_EQ(norm2, 5);
//  }
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(norm)
//{
//  um2::Vec2<T> v(1, 2);
//  T norm = um2::norm(v);
//  EXPECT_NEAR(norm, sqrt(5.0), 1e-6);
//}
//
//template <typename T>
//UM2_HOSTDEV TEST_CASE(normalize)
//{
//  um2::Vec2<T> v(1, 2);
//  um2::Vec2<T> v2 = um2::normalize(v);
//  T norm = um2::norm(v2);
//  EXPECT_NEAR(norm, 1, 1e-6);
//}
//
//// --------------------------------------------------------------------------
//// CUDA
//// --------------------------------------------------------------------------
//#if UM2_ENABLE_CUDA
//template <typename T>
//MAKE_CUDA_KERNEL(accessor, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(unary_minus, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(compound_add, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(compound_sub, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(compound_mul, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(compound_div, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(add, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(sub, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(mul, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(div, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(compound_scalar_add, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(compound_scalar_sub, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(compound_scalar_mul, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(compound_scalar_div, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(scalar_add, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(scalar_sub, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(scalar_mul, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(scalar_div, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(min, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(max, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(dot, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(cross, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(norm2, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(norm, T);
//
//template <typename T>
//MAKE_CUDA_KERNEL(normalize, T);
//
//#endif
//
template <typename T>
TEST_SUITE(vec2)
{
  TEST_HOSTDEV(accessor, 1, 1, T);
//  if constexpr (!std::unsigned_integral<T>) {
//    TEST_HOSTDEV(unary_minus, 1, 1, T);
//  }
//  TEST_HOSTDEV(compound_add, 1, 1, T);
//  TEST_HOSTDEV(compound_sub, 1, 1, T);
//  TEST_HOSTDEV(compound_mul, 1, 1, T);
//  TEST_HOSTDEV(compound_div, 1, 1, T);
//  TEST_HOSTDEV(add, 1, 1, T);
//  TEST_HOSTDEV(sub, 1, 1, T);
//  TEST_HOSTDEV(mul, 1, 1, T);
//  TEST_HOSTDEV(div, 1, 1, T);
//  TEST_HOSTDEV(compound_scalar_add, 1, 1, T);
//  TEST_HOSTDEV(compound_scalar_sub, 1, 1, T);
//  TEST_HOSTDEV(compound_scalar_mul, 1, 1, T);
//  TEST_HOSTDEV(compound_scalar_div, 1, 1, T);
//  TEST_HOSTDEV(scalar_add, 1, 1, T);
//  TEST_HOSTDEV(scalar_sub, 1, 1, T);
//  TEST_HOSTDEV(scalar_mul, 1, 1, T);
//  TEST_HOSTDEV(scalar_div, 1, 1, T);
//  TEST_HOSTDEV(min, 1, 1, T);
//  TEST_HOSTDEV(max, 1, 1, T);
//  TEST_HOSTDEV(dot, 1, 1, T);
//  TEST_HOSTDEV(cross, 1, 1, T);
//  TEST_HOSTDEV(norm2, 1, 1, T);
//  if constexpr (std::floating_point<T>) {
//    TEST_HOSTDEV(norm, 1, 1, T);
//    TEST_HOSTDEV(normalize, 1, 1, T);
//  }
}

auto main() -> int
{
  RUN_TESTS(vec2<float>);
  RUN_TESTS(vec2<double>);
  RUN_TESTS(vec2<int32_t>);
  RUN_TESTS(vec2<uint32_t>);
  return 0;
}
