#include "../test_framework.hpp"
#include <um2/math/vec.hpp>

#include <concepts>

static consteval auto is_power_of_2(len_t x) -> bool { return (x & (x - 1)) == 0; }

template <len_t L, typename T>
concept is_simd_vector = std::is_arithmetic_v<T> && is_power_of_2(L);

template <len_t D, typename T>
UM2_HOSTDEV static constexpr auto make_vec() -> um2::Vec<D, T>
{
  um2::Vec<D, T> v;
  for (len_t i = 0; i < D; ++i) {
    v.data[i] = static_cast<T>(i + 1);
  }
  return v;
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(array_type)
{
  um2::Vec<D, T> v;
  EXPECT_TRUE((std::same_as<decltype(v.data), um2::Array<D, T>>));
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_accessors)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  // Test const reference or value accessor
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v[i], i + 1, 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_unary_minus)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = -v;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v2[i], -(i + 1), 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_compound_add)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  v += v2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v[i], 2 * (i + 1), 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_compound_sub)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  v -= v2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v[i], 0, 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_compound_mul)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  v *= v2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v[i], (i + 1) * (i + 1), 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_compound_div)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  v /= v2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v[i], 1, 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_add)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  um2::Vec<D, T> v3 = v + v2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v3[i], 2 * (i + 1), 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_sub)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  um2::Vec<D, T> v3 = v - v2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v3[i], 0, 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_mul)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  um2::Vec<D, T> v3 = v * v2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v3[i], (i + 1) * (i + 1), 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_div)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  um2::Vec<D, T> v3 = v / v2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v3[i], 1, 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_compound_scalar_add)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  v += 2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v[i], i + 3, 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_compound_scalar_sub)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  v -= 2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v[i], int(i) - 1, 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_compound_scalar_mul)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  v *= 2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v[i], 2 * (i + 1), 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_compound_scalar_div)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  v *= 2;
  v /= 2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v[i], (i + 1), 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_scalar_mul)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = v * 2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v2[i], 2 * (i + 1), 1e-6);
  }
  um2::Vec<D, T> v3 = 2 * v;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v3[i], 2 * (i + 1), 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_scalar_div)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = 2 * v;
  v2 = v2 / 2;
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v2[i], (i + 1), 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_min)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  v -= 1;
  um2::Vec<D, T> v2 = make_vec<D, T>();
  um2::Vec<D, T> v3 = um2::min(v, v2);
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v3[i], i, 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_max)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  v -= 1;
  um2::Vec<D, T> v2 = make_vec<D, T>();
  um2::Vec<D, T> v3 = um2::max(v, v2);
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(v3[i], i + 1, 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_dot)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = make_vec<D, T>();
  T dot = um2::dot(v, v2);
  EXPECT_NEAR(dot, D * (D + 1) * (2 * D + 1) / 6, 1e-6);
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_norm2)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  T norm2 = um2::norm2(v);
  EXPECT_NEAR(norm2, D * (D + 1) * (2 * D + 1) / 6, 1e-6);
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_norm)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  T norm = um2::norm(v);
  EXPECT_NEAR(norm, sqrt(D * (D + 1) * (2 * D + 1) / 6), 1e-4);
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(vec_normalize)
{
  um2::Vec<D, T> v = make_vec<D, T>();
  um2::Vec<D, T> v2 = um2::normalize(v);
  T norm = um2::norm(v2);
  EXPECT_NEAR(norm, 1, 1e-6);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(vec2_cross)
{
  um2::vec2<T> v(1, 2);
  um2::vec2<T> v2(3, 4);
  T cross = um2::cross(v, v2);
  EXPECT_NEAR(cross, -2, 1e-6);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(vec3_cross)
{
  um2::vec3<T> v(1, 2, 3);
  um2::vec3<T> v2(4, 5, 6);
  um2::vec3<T> cross(um2::cross(v, v2));
  EXPECT_NEAR(cross[0], -3, 1e-6);
  EXPECT_NEAR(cross[1], 6, 1e-6);
  EXPECT_NEAR(cross[2], -3, 1e-6);
}

#if UM2_ENABLE_CUDA
template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_accessors, vec_accessors_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_accessors_kernel, vec_accessors_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_unary_minus, vec_unary_minus_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_unary_minus_kernel, vec_unary_minus_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_compound_add, vec_compound_add_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_compound_add_kernel, vec_compound_add_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_compound_sub, vec_compound_sub_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_compound_sub_kernel, vec_compound_sub_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_compound_mul, vec_compound_mul_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_compound_mul_kernel, vec_compound_mul_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_compound_div, vec_compound_div_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_compound_div_kernel, vec_compound_div_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_add, vec_add_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_add_kernel, vec_add_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_sub, vec_sub_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_sub_kernel, vec_sub_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_mul, vec_mul_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_mul_kernel, vec_mul_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_div, vec_div_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_div_kernel, vec_div_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_compound_scalar_add, vec_compound_scalar_add_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_compound_scalar_add_kernel, vec_compound_scalar_add_cuda, D,
                          T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_compound_scalar_sub, vec_compound_scalar_sub_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_compound_scalar_sub_kernel, vec_compound_scalar_sub_cuda, D,
                          T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_compound_scalar_mul, vec_compound_scalar_mul_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_compound_scalar_mul_kernel, vec_compound_scalar_mul_cuda, D,
                          T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_compound_scalar_div, vec_compound_scalar_div_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_compound_scalar_div_kernel, vec_compound_scalar_div_cuda, D,
                          T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_scalar_mul, vec_scalar_mul_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_scalar_mul_kernel, vec_scalar_mul_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_scalar_div, vec_scalar_div_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_scalar_div_kernel, vec_scalar_div_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_min, vec_min_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_min_kernel, vec_min_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_max, vec_max_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_max_kernel, vec_max_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_dot, vec_dot_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_dot_kernel, vec_dot_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_norm2, vec_norm2_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_norm2_kernel, vec_norm2_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_norm, vec_norm_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_norm_kernel, vec_norm_cuda, D, T)

template <len_t D, typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec_normalize, vec_normalize_kernel, D, T)
template <len_t D, typename T>
ADD_TEMPLATED_KERNEL_TEST(vec_normalize_kernel, vec_normalize_cuda, D, T)

template <typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec2_cross, vec2_cross_kernel, T)
template <typename T>
ADD_TEMPLATED_KERNEL_TEST(vec2_cross_kernel, vec2_cross_cuda, T)

template <typename T>
ADD_TEMPLATED_CUDA_KERNEL(vec3_cross, vec3_cross_kernel, T)
template <typename T>
ADD_TEMPLATED_KERNEL_TEST(vec3_cross_kernel, vec3_cross_cuda, T)
#endif

template <len_t D, typename T>
TEST_SUITE(vec)
{
  TEST((array_type<D, T>));
  TEST((vec_accessors<D, T>));
  TEST((vec_unary_minus<D, T>));
  TEST((vec_compound_add<D, T>));
  TEST((vec_compound_sub<D, T>));
  TEST((vec_compound_mul<D, T>));
  TEST((vec_compound_div<D, T>));
  TEST((vec_add<D, T>));
  TEST((vec_sub<D, T>));
  TEST((vec_mul<D, T>));
  TEST((vec_div<D, T>));
  TEST((vec_compound_scalar_add<D, T>));
  TEST((vec_compound_scalar_sub<D, T>));
  TEST((vec_compound_scalar_mul<D, T>));
  TEST((vec_compound_scalar_div<D, T>));
  TEST((vec_scalar_mul<D, T>));
  TEST((vec_scalar_div<D, T>));
  TEST((vec_min<D, T>));
  TEST((vec_max<D, T>));
  TEST((vec_dot<D, T>));
  TEST((vec_norm2<D, T>));
  if constexpr (std::floating_point<T>) {
    TEST((vec_norm<D, T>));
    TEST((vec_normalize<D, T>));
  }
  if constexpr (D == 2) {
    TEST((vec2_cross<T>));
  }
  if constexpr (D == 3) {
    TEST((vec3_cross<T>));
  }

  TEST_CUDA_KERNEL("accessors_cuda", (vec_accessors_cuda<D, T>));
  TEST_CUDA_KERNEL("unary_minus_cuda", (vec_unary_minus_cuda<D, T>));
  TEST_CUDA_KERNEL("compound_add_cuda", (vec_compound_add_cuda<D, T>));
  TEST_CUDA_KERNEL("compound_sub_cuda", (vec_compound_sub_cuda<D, T>));
  TEST_CUDA_KERNEL("compound_mul_cuda", (vec_compound_mul_cuda<D, T>));
  TEST_CUDA_KERNEL("compound_div_cuda", (vec_compound_div_cuda<D, T>));
  TEST_CUDA_KERNEL("add_cuda", (vec_add_cuda<D, T>));
  TEST_CUDA_KERNEL("sub_cuda", (vec_sub_cuda<D, T>));
  TEST_CUDA_KERNEL("mul_cuda", (vec_mul_cuda<D, T>));
  TEST_CUDA_KERNEL("div_cuda", (vec_div_cuda<D, T>));
  TEST_CUDA_KERNEL("compound_scalar_add_cuda", (vec_compound_scalar_add_cuda<D, T>));
  TEST_CUDA_KERNEL("compound_scalar_sub_cuda", (vec_compound_scalar_sub_cuda<D, T>));
  TEST_CUDA_KERNEL("compound_scalar_mul_cuda", (vec_compound_scalar_mul_cuda<D, T>));
  TEST_CUDA_KERNEL("compound_scalar_div_cuda", (vec_compound_scalar_div_cuda<D, T>));
  TEST_CUDA_KERNEL("scalar_mul_cuda", (vec_scalar_mul_cuda<D, T>));
  TEST_CUDA_KERNEL("scalar_div_cuda", (vec_scalar_div_cuda<D, T>));
  TEST_CUDA_KERNEL("min_cuda", (vec_min_cuda<D, T>));
  TEST_CUDA_KERNEL("max_cuda", (vec_max_cuda<D, T>));
  TEST_CUDA_KERNEL("dot_cuda", (vec_dot_cuda<D, T>));
  TEST_CUDA_KERNEL("norm2_cuda", (vec_norm2_cuda<D, T>));
  if constexpr (std::floating_point<T>) {
    TEST_CUDA_KERNEL("norm_cuda", (vec_norm_cuda<D, T>));
    TEST_CUDA_KERNEL("normalize_cuda", (vec_normalize_cuda<D, T>));
  }
  if constexpr (D == 2) {
    TEST_CUDA_KERNEL("cross_cuda", (vec2_cross_cuda<T>));
  }
  if constexpr (D == 3) {
    TEST_CUDA_KERNEL("cross_cuda", (vec3_cross_cuda<T>));
  }
}

int main(int argc, char ** argv)
{
  // Packed high precision
  RUN_TESTS((vec<1, float>));
  RUN_TESTS((vec<1, int>));
  RUN_TESTS((vec<1, double>));

  RUN_TESTS((vec<2, float>));
  RUN_TESTS((vec<2, int>));
  RUN_TESTS((vec<2, double>));

  RUN_TESTS((vec<3, float>));
  RUN_TESTS((vec<3, int>));
  RUN_TESTS((vec<3, double>));

  RUN_TESTS((vec<4, float>));
  RUN_TESTS((vec<4, int>));
  RUN_TESTS((vec<4, double>));

  RUN_TESTS((vec<16, float>));
  RUN_TESTS((vec<16, int>));

  // Packed medium precision
  RUN_TESTS((vec<1, float>));
  RUN_TESTS((vec<1, int>));
  RUN_TESTS((vec<1, double>));

  RUN_TESTS((vec<2, float>));
  RUN_TESTS((vec<2, int>));
  RUN_TESTS((vec<2, double>));

  RUN_TESTS((vec<3, float>));
  RUN_TESTS((vec<3, int>));
  RUN_TESTS((vec<3, double>));

  RUN_TESTS((vec<4, float>));
  RUN_TESTS((vec<4, int>));
  RUN_TESTS((vec<4, double>));

  RUN_TESTS((vec<16, float>));
  RUN_TESTS((vec<16, int>));

  // Packed low precision
  RUN_TESTS((vec<1, float>));
  RUN_TESTS((vec<1, int>));
  RUN_TESTS((vec<1, double>));

  RUN_TESTS((vec<2, float>));
  RUN_TESTS((vec<2, int>));
  RUN_TESTS((vec<2, double>));

  RUN_TESTS((vec<3, float>));
  RUN_TESTS((vec<3, int>));
  RUN_TESTS((vec<3, double>));

  RUN_TESTS((vec<4, float>));
  RUN_TESTS((vec<4, int>));
  RUN_TESTS((vec<4, double>));

  RUN_TESTS((vec<16, float>));
  RUN_TESTS((vec<16, int>));

  // Aligned high precision
  RUN_TESTS((vec<1, float>));
  RUN_TESTS((vec<1, int>));
  RUN_TESTS((vec<1, double>));

  RUN_TESTS((vec<2, float>));
  RUN_TESTS((vec<2, int>));
  RUN_TESTS((vec<2, double>));

  RUN_TESTS((vec<3, float>));
  RUN_TESTS((vec<3, int>));
  RUN_TESTS((vec<3, double>));

  RUN_TESTS((vec<4, float>));
  RUN_TESTS((vec<4, int>));
  RUN_TESTS((vec<4, double>));

  RUN_TESTS((vec<16, float>));
  RUN_TESTS((vec<16, int>));

  // Aligned medium precision
  RUN_TESTS((vec<1, float>));
  RUN_TESTS((vec<1, int>));
  RUN_TESTS((vec<1, double>));

  RUN_TESTS((vec<2, float>));
  RUN_TESTS((vec<2, int>));
  RUN_TESTS((vec<2, double>));

  RUN_TESTS((vec<3, float>));
  RUN_TESTS((vec<3, int>));
  RUN_TESTS((vec<3, double>));

  RUN_TESTS((vec<4, float>));
  RUN_TESTS((vec<4, int>));
  RUN_TESTS((vec<4, double>));

  RUN_TESTS((vec<16, float>));
  RUN_TESTS((vec<16, int>));

  // Aligned low precision
  RUN_TESTS((vec<1, float>));
  RUN_TESTS((vec<1, int>));
  RUN_TESTS((vec<1, double>));

  RUN_TESTS((vec<2, float>));
  RUN_TESTS((vec<2, int>));
  RUN_TESTS((vec<2, double>));

  RUN_TESTS((vec<3, float>));
  RUN_TESTS((vec<3, int>));
  RUN_TESTS((vec<3, double>));

  RUN_TESTS((vec<4, float>));
  RUN_TESTS((vec<4, int>));
  RUN_TESTS((vec<4, double>));

  RUN_TESTS((vec<16, float>));
  RUN_TESTS((vec<16, int>));

  return 0;
}