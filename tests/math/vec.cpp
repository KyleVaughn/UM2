#include "../test_framework.hpp"
#include <um2/math/vec.hpp>

template <len_t D, typename T>
UM2_HOSTDEV static constexpr
auto make_vec() -> um2::Vec<D, T> 
{
  um2::Vec<D, T> v;
  for (len_t i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 1);
  }
  return v;
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(accessor)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(i + 1), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(i + 1));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(unary_minus)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = -v;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v2[i], -static_cast<T>(i + 1), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v2[i], -static_cast<T>(i + 1));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(compound_add)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  v += v2;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(2 * (i + 1)), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(2 * (i + 1)));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(compound_sub)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  v -= v2;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], 0, 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], 0);
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(compound_mul)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  v.array() *= v2.array();
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>((i + 1) * (i + 1)), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>((i + 1) * (i + 1)));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(compound_div)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  v.array() /= v2.array();
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], 1, 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], 1);
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(add)
{
  um2::Vec<D, T> v0 = make_vec<D, T>();
  um2::Vec<D, T> v1 = make_vec<D, T>();
  um2::Vec<D, T> v = v0 + v1;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(2 * (i + 1)), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(2 * (i + 1)));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(sub)
{
  um2::Vec<D, T> v0 = make_vec<D, T>();
  um2::Vec<D, T> v1 = make_vec<D, T>();
  um2::Vec<D, T> v = v0 - v1;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], 0, 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], 0);
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(mul)
{
  um2::Vec<D, T> v0 = make_vec<D, T>();
  um2::Vec<D, T> v1 = make_vec<D, T>();
  um2::Vec<D, T> v = v0.array() * v1.array();
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>((i + 1) * (i + 1)), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>((i + 1) * (i + 1)));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(div)
{
  um2::Vec<D, T> v0 = make_vec<D, T>();
  um2::Vec<D, T> v1 = make_vec<D, T>();
  um2::Vec<D, T> v = v0.array() / v1.array(); 
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], 1, 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], 1);
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(compound_scalar_add)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  v.array() += 2;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(i + 3), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(i + 3));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(compound_scalar_sub)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  v.array() -= 2;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(i - 1), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(i - 1));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(compound_scalar_mul)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  v *= 2;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(2 * (i + 1)), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(2 * (i + 1)));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(compound_scalar_div)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  v /= 2;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(i + 1) / 2, 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(i + 1) / 2);
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(scalar_add)
{
  um2::Vec<D, T> v0 = make_vec<D, T>();
  um2::Vec<D, T> v = v0.array() + 2;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(i + 3), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(i + 3));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(scalar_sub)
{
  um2::Vec<D, T> v0 = make_vec<D, T>();
  um2::Vec<D, T> v = v0.array() - 2;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(i - 1), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(i - 1));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(scalar_mul)
{
  um2::Vec<D, T> v0 = make_vec<D, T>();
  um2::Vec<D, T> v = v0 * 2;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(2 * (i + 1)), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(2 * (i + 1)));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(scalar_div)
{
  um2::Vec<D, T> v0 = make_vec<D, T>();
  um2::Vec<D, T> v = v0 / 2;
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(i + 1) / 2, 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(i + 1) / 2);
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(min)
{
  um2::Vec<D, T> v0 = make_vec<D, T>();
  um2::Vec<D, T> v1 = make_vec<D, T>().array() + 1;
  um2::Vec<D, T> v = v0.cwiseMin(v1); 
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(i + 1), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(i + 1));
    }
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(max)
{
  um2::Vec<D, T> v0 = make_vec<D, T>();
  um2::Vec<D, T> v1 = make_vec<D, T>().array() + 1;
  um2::Vec<D, T> v = v0.cwiseMax(v1); 
  if constexpr (std::floating_point<T>) {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_NEAR(v[i], static_cast<T>(i + 2), 1e-6);
    }
  } else {
    for (len_t i = 0; i < D; ++i) {
      EXPECT_EQ(v[i], static_cast<T>(i + 2));
    }
  }
}

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
// --------------------------------------------------------------------------
// CUDA
// --------------------------------------------------------------------------
#if UM2_ENABLE_CUDA
template <len_t D, typename T>
MAKE_CUDA_KERNEL(accessor, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(unary_minus, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(compound_add, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(compound_sub, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(compound_mul, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(compound_div, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(add, D, T);

template <len_t D, typename T> 
MAKE_CUDA_KERNEL(sub, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(mul, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(div, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_add, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_sub, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_mul, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_div, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(scalar_add, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(scalar_sub, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(scalar_mul, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(scalar_div, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(min, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(max, D, T);
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

#endif

template <len_t D, typename T>
TEST_SUITE(vec)
{
  TEST_HOSTDEV(accessor, 1, 1, D, T); 
  if constexpr (!std::unsigned_integral<T>) {
    TEST_HOSTDEV(unary_minus, 1, 1, D, T);
  }
  TEST_HOSTDEV(compound_add, 1, 1, D, T);
  TEST_HOSTDEV(compound_sub, 1, 1, D, T);
  TEST_HOSTDEV(compound_mul, 1, 1, D, T);
  TEST_HOSTDEV(compound_div, 1, 1, D, T);
  TEST_HOSTDEV(add, 1, 1, D, T);
  TEST_HOSTDEV(sub, 1, 1, D, T);
  TEST_HOSTDEV(mul, 1, 1, D, T);
  TEST_HOSTDEV(div, 1, 1, D, T);
  TEST_HOSTDEV(compound_scalar_add, 1, 1, D, T);
  TEST_HOSTDEV(compound_scalar_sub, 1, 1, D, T);
  TEST_HOSTDEV(compound_scalar_mul, 1, 1, D, T);
  TEST_HOSTDEV(compound_scalar_div, 1, 1, D, T);
  TEST_HOSTDEV(scalar_add, 1, 1, D, T);
  TEST_HOSTDEV(scalar_sub, 1, 1, D, T);
  TEST_HOSTDEV(scalar_mul, 1, 1, D, T);
  TEST_HOSTDEV(scalar_div, 1, 1, D, T);
  TEST_HOSTDEV(min, 1, 1, D, T); 
  TEST_HOSTDEV(max, 1, 1, D, T);
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
  RUN_TESTS((vec<2, float>));
  RUN_TESTS((vec<2, double>));
  RUN_TESTS((vec<2, int32_t>));
  RUN_TESTS((vec<2, uint32_t>));
  RUN_TESTS((vec<3, float>));
  RUN_TESTS((vec<3, double>));
  RUN_TESTS((vec<3, int32_t>));
  RUN_TESTS((vec<3, uint32_t>));
  RUN_TESTS((vec<4, float>));
  RUN_TESTS((vec<4, double>));
  RUN_TESTS((vec<4, int32_t>));
  RUN_TESTS((vec<4, uint32_t>));
  return 0;
}
