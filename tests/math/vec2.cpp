#include "../test_framework.hpp"
#include <um2/math/vec2.hpp>

template <typename T>
UM2_HOSTDEV TEST_CASE(unary_minus) {
  um2::Vec2<T> v0(1, -1);
  um2::Vec2<T> v1 = -v0;
  EXPECT_NEAR(v1.x, -1, 1e-6);
  EXPECT_NEAR(v1.y, 1, 1e-6);
}
MAKE_CUDA_KERNEL(unary_minus);

//template <typename T>
//UM2_HOSTDEV TEST(vec_compound_add) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//v += v2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v[i], 2 * (i + 1), 1e-6, "Vec::operator+=");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_compound_sub) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//v -= v2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v[i], 0, 1e-6, "Vec::operator-=");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_compound_mul) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//v *= v2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v[i], (i + 1) * (i + 1), 1e-6, "Vec::operator*=");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_compound_div) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//v /= v2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v[i], 1, 1e-6, "Vec::operator/=");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_add) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v3 = v + v2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v3[i], 2 * (i + 1), 1e-6, "Vec::operator+");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_sub) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v3 = v - v2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v3[i], 0, 1e-6, "Vec::operator-");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_mul) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v3 = v * v2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v3[i], (i + 1) * (i + 1), 1e-6, "Vec::operator*");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_div) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v3 = v / v2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v3[i], 1, 1e-6, "Vec::operator/");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_compound_scalar_add) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//v += 2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v[i], i + 3, 1e-6, "Vec::operator+= (scalar)");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_compound_scalar_sub) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//v -= 2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v[i], int(i) - 1, 1e-6, "Vec::operator-= (scalar)");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_compound_scalar_mul) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//v *= 2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v[i], 2 * (i + 1), 1e-6, "Vec::operator*= (scalar)");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_compound_scalar_div) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//v *= 2;
//v /= 2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v[i], (i + 1), 1e-6, "Vec::operator*= (scalar)");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_scalar_mul) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = v * 2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v2[i], 2 * (i + 1), 1e-6, "Vec::operator* (scalar)");
//}
//um2::Vec<D, T, Q> v3 = 2 * v;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v3[i], 2 * (i + 1), 1e-6, "Vec::operator* (scalar)");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_scalar_div) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = 2 * v;
//v2 = v2 / 2;
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v2[i], (i + 1), 1e-6, "Vec::operator/ (scalar)");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_min) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//v -= 1;
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v3 = um2::min(v, v2);
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v3[i], i, 1e-6, "Vec::min");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_max) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//v -= 1;
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v3 = um2::max(v, v2);
//for (len_t i = 0; i < D; ++i) {
//  ASSERT_APPROX(v3[i], i + 1, 1e-6, "Vec::max");
//}
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_dot) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = make_vec<D, T, Q>();
//T dot = um2::dot(v, v2);
//ASSERT_APPROX(dot, D *(D + 1) * (2 * D + 1) / 6, 1e-6, "Vec::dot");
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_norm2) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//T norm2 = um2::norm2(v);
//ASSERT_APPROX(norm2, D *(D + 1) * (2 * D + 1) / 6, 1e-6, "Vec::norm2");
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_norm) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//T norm = um2::norm(v);
//ASSERT_APPROX(norm, sqrt(D *(D + 1) * (2 * D + 1) / 6), 1e-4, "Vec::norm");
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec_normalize) um2::Vec<D, T, Q> v = make_vec<D, T, Q>();
//um2::Vec<D, T, Q> v2 = um2::normalize(v);
//T norm = um2::norm(v2);
//ASSERT_APPROX(norm, 1, 1e-6, "Vec::normalize");
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec2_cross) um2::Vec2<T> v(1, 2);
//um2::Vec2<T> v2(3, 4);
//T cross = um2::cross(v, v2);
//ASSERT_APPROX(cross, -2, 1e-6, "Vec2::cross");
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(vec3_cross) um2::Vec3<T> v(1, 2, 3);
//um2::Vec3<T> v2(4, 5, 6);
//um2::Vec3<T> cross = um2::cross(v, v2);
//ASSERT_APPROX(cross[0], -3, 1e-6, "Vec3::cross");
//ASSERT_APPROX(cross[1], 6, 1e-6, "Vec3::cross");
//ASSERT_APPROX(cross[2], -3, 1e-6, "Vec3::cross");
//END_TEST
//
//#if UM2_HAS_CUDA
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_accessors, vec_accessors_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_accessors_kernel, vec_accessors_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_unary_minus, vec_unary_minus_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_unary_minus_kernel, vec_unary_minus_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_compound_add, vec_compound_add_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_compound_add_kernel, vec_compound_add_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_compound_sub, vec_compound_sub_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_compound_sub_kernel, vec_compound_sub_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_compound_mul, vec_compound_mul_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_compound_mul_kernel, vec_compound_mul_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_compound_div, vec_compound_div_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_compound_div_kernel, vec_compound_div_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_add, vec_add_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_add_kernel, vec_add_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_sub, vec_sub_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_sub_kernel, vec_sub_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_mul, vec_mul_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_mul_kernel, vec_mul_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_div, vec_div_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_div_kernel, vec_div_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_compound_scalar_add, vec_compound_scalar_add_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_compound_scalar_add_kernel, vec_compound_scalar_add_cuda, D,
//                          T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_compound_scalar_sub, vec_compound_scalar_sub_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_compound_scalar_sub_kernel, vec_compound_scalar_sub_cuda, D,
//                          T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_compound_scalar_mul, vec_compound_scalar_mul_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_compound_scalar_mul_kernel, vec_compound_scalar_mul_cuda, D,
//                          T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_compound_scalar_div, vec_compound_scalar_div_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_compound_scalar_div_kernel, vec_compound_scalar_div_cuda, D,
//                          T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_scalar_mul, vec_scalar_mul_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_scalar_mul_kernel, vec_scalar_mul_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_scalar_div, vec_scalar_div_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_scalar_div_kernel, vec_scalar_div_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_min, vec_min_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_min_kernel, vec_min_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_max, vec_max_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_max_kernel, vec_max_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_dot, vec_dot_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_dot_kernel, vec_dot_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_norm2, vec_norm2_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_norm2_kernel, vec_norm2_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_norm, vec_norm_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_norm_kernel, vec_norm_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec_normalize, vec_normalize_kernel, D, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec_normalize_kernel, vec_normalize_cuda, D, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec2_cross, vec2_cross_kernel, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec2_cross_kernel, vec2_cross_cuda, T)
//
//template <typename T>
//ADD_TEMPLATED_CUDA_KERNEL(vec3_cross, vec3_cross_kernel, T)
//template <typename T>
//ADD_TEMPLATED_KERNEL_TEST(vec3_cross_kernel, vec3_cross_cuda, T)
//#endif
//
template <typename T>
TEST_SUITE(vec2) {
  TEST_HOSTDEV(unary_minus, 1, 1, T);
//RUN_TEST("compound_add", (vec_compound_add<D, T, Q>));
//RUN_TEST("compound_sub", (vec_compound_sub<D, T, Q>));
//RUN_TEST("compound_mul", (vec_compound_mul<D, T, Q>));
//RUN_TEST("compound_div", (vec_compound_div<D, T, Q>));
//RUN_TEST("add", (vec_add<D, T, Q>));
//RUN_TEST("sub", (vec_sub<D, T, Q>));
//RUN_TEST("mul", (vec_mul<D, T, Q>));
//RUN_TEST("div", (vec_div<D, T, Q>));
//RUN_TEST("compound_scalar_add", (vec_compound_scalar_add<D, T, Q>));
//RUN_TEST("compound_scalar_sub", (vec_compound_scalar_sub<D, T, Q>));
//RUN_TEST("compound_scalar_mul", (vec_compound_scalar_mul<D, T, Q>));
//RUN_TEST("compound_scalar_div", (vec_compound_scalar_div<D, T, Q>));
//RUN_TEST("scalar_mul", (vec_scalar_mul<D, T, Q>));
//RUN_TEST("scalar_div", (vec_scalar_div<D, T, Q>));
//RUN_TEST("min", (vec_min<D, T, Q>));
//RUN_TEST("max", (vec_max<D, T, Q>));
//RUN_TEST("dot", (vec_dot<D, T, Q>));
//RUN_TEST("norm2", (vec_norm2<D, T, Q>));
//if constexpr (std::floating_point<T>) {
//  RUN_TEST("norm", (vec_norm<D, T, Q>));
//  RUN_TEST("normalize", (vec_normalize<D, T, Q>));
//}
//if constexpr (D == 2) {
//  RUN_TEST("cross", (vec2_cross<T>));
//}
//if constexpr (D == 3) {
//  RUN_TEST("cross", (vec3_cross<T>));
//}
//
//RUN_CUDA_TEST("accessors_cuda", (vec_accessors_cuda<D, T, Q>));
//RUN_CUDA_TEST("unary_minus_cuda", (vec_unary_minus_cuda<D, T, Q>));
//RUN_CUDA_TEST("compound_add_cuda", (vec_compound_add_cuda<D, T, Q>));
//RUN_CUDA_TEST("compound_sub_cuda", (vec_compound_sub_cuda<D, T, Q>));
//RUN_CUDA_TEST("compound_mul_cuda", (vec_compound_mul_cuda<D, T, Q>));
//RUN_CUDA_TEST("compound_div_cuda", (vec_compound_div_cuda<D, T, Q>));
//RUN_CUDA_TEST("add_cuda", (vec_add_cuda<D, T, Q>));
//RUN_CUDA_TEST("sub_cuda", (vec_sub_cuda<D, T, Q>));
//RUN_CUDA_TEST("mul_cuda", (vec_mul_cuda<D, T, Q>));
//RUN_CUDA_TEST("div_cuda", (vec_div_cuda<D, T, Q>));
//RUN_CUDA_TEST("compound_scalar_add_cuda", (vec_compound_scalar_add_cuda<D, T, Q>));
//RUN_CUDA_TEST("compound_scalar_sub_cuda", (vec_compound_scalar_sub_cuda<D, T, Q>));
//RUN_CUDA_TEST("compound_scalar_mul_cuda", (vec_compound_scalar_mul_cuda<D, T, Q>));
//RUN_CUDA_TEST("compound_scalar_div_cuda", (vec_compound_scalar_div_cuda<D, T, Q>));
//RUN_CUDA_TEST("scalar_mul_cuda", (vec_scalar_mul_cuda<D, T, Q>));
//RUN_CUDA_TEST("scalar_div_cuda", (vec_scalar_div_cuda<D, T, Q>));
//RUN_CUDA_TEST("min_cuda", (vec_min_cuda<D, T, Q>));
//RUN_CUDA_TEST("max_cuda", (vec_max_cuda<D, T, Q>));
//RUN_CUDA_TEST("dot_cuda", (vec_dot_cuda<D, T, Q>));
//RUN_CUDA_TEST("norm2_cuda", (vec_norm2_cuda<D, T, Q>));
//if constexpr (std::floating_point<T>) {
//  RUN_CUDA_TEST("norm_cuda", (vec_norm_cuda<D, T, Q>));
//  RUN_CUDA_TEST("normalize_cuda", (vec_normalize_cuda<D, T, Q>));
//}
//if constexpr (D == 2) {
//  RUN_CUDA_TEST("cross_cuda", (vec2_cross_cuda<T>));
//}
//if constexpr (D == 3) {
//  RUN_CUDA_TEST("cross_cuda", (vec3_cross_cuda<T>));
//}

}

auto main() -> int
{
  RUN_TESTS(vec2<float>);
  return 0;
}