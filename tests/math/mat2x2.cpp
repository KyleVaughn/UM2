#include "../test_framework.hpp" 
#include <um2/math/mat2x2.hpp>

template <typename T>
UM2_HOSTDEV static constexpr
auto make_mat() -> um2::Mat2x2<T> 
{
    um2::Mat2x2<T> mat;
    mat.cols[0][0] = static_cast<T>(0);
    mat.cols[0][1] = static_cast<T>(1);
    mat.cols[1][0] = static_cast<T>(2); 
    mat.cols[1][1] = static_cast<T>(3);
    return mat;
}

template <typename T>
UM2_HOSTDEV TEST_CASE(accessors) {
    um2::Mat2x2<T> m = make_mat<T>();
    if constexpr (std::floating_point<T>) {
        EXPECT_NEAR(m[0][0], 0, 1e-6);
        EXPECT_NEAR(m[0][1], 1, 1e-6);
        EXPECT_NEAR(m[1][0], 2, 1e-6);
        EXPECT_NEAR(m[1][1], 3, 1e-6);
    } else {
        EXPECT_EQ(m[0][0], 0);
        EXPECT_EQ(m[0][1], 1);
        EXPECT_EQ(m[1][0], 2);
        EXPECT_EQ(m[1][1], 3);
    }
}

//template <typename T>
//UM2_HOSTDEV TEST(mat_unary_minus)
//    um2::Mat2x2<T> m = make_mat<M, N, T>();
//    um2::Mat2x2<T> neg_m = -m;
//    for (len_t j = 0; j < N; ++j) {
//        for (len_t i = 0; i < M; ++i) {
//            ASSERT_APPROX((neg_m[i, j]), (-m[i, j]), 1e-6, "Mat::operator-");
//        }
//    }
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(mat_add)
//    um2::Mat2x2<T> m = make_mat<M, N, T>();
//    um2::Mat2x2<T> n = make_mat<M, N, T>();
//    um2::Mat2x2<T> sum = m + n;
//    for (len_t j = 0; j < N; ++j) {
//        for (len_t i = 0; i < M; ++i) {
//            ASSERT_APPROX((sum[i, j]),    (m[i, j] + n[i, j]), 1e-6, "Mat::operator+");
//        }
//    }
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(mat_sub)
//    um2::Mat2x2<T> m = make_mat<M, N, T>();
//    um2::Mat2x2<T> n = make_mat<M, N, T>();
//    um2::Mat2x2<T> diff = m - n;
//    for (len_t j = 0; j < N; ++j) {
//        for (len_t i = 0; i < M; ++i) {
//            ASSERT_APPROX((diff[i, j]), 0, 1e-6, "Mat::operator-");
//        }
//    }
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(mat_vec_mul)
//    um2::Mat2x2<T> m = make_mat<M, N, T>();
//    um2::Vec<N, T> v;
//    for (len_t i = 0; i < N; ++i) {
//        v.data[i] = static_cast<T>(i);
//    }
//    um2::Vec<M, T> mv = m * v;
//    for (len_t i = 0; i < M; ++i) {
//        T mv_i = 0;
//        for (len_t j = 0; j < N; ++j) {
//            mv_i += m[i, j] * v[j];
//        }
//        ASSERT_APPROX(mv[i], mv_i, 1e-6, "Mat::operator*");
//    }
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(mat_mul)
//    um2::Mat2x2<T> m = make_mat<M, N, T>();
//    um2::Mat2x2<T> n = make_mat<M, N, T>();
//    um2::Mat2x2<T> prod = m * n;
//    for (len_t j = 0; j < N; ++j) {
//        for (len_t i = 0; i < M; ++i) {
//            T expected = 0;
//            for (len_t k = 0; k < M; ++k) {
//                expected += m[k, j] * n[i, k];
//            }
//            ASSERT_APPROX((prod[i, j]), expected, 1e-6, "Mat::operator*");
//        }
//    }
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(mat_scalar_mul)
//    um2::Mat2x2<T> m = make_mat<M, N, T>();
//    um2::Mat2x2<T> prod = m * 2;
//    for (len_t j = 0; j < N; ++j) {
//        for (len_t i = 0; i < M; ++i) {
//            ASSERT_APPROX((prod[i, j]), (m[i, j]) * 2, 1e-6, "Mat::operator*");
//        }
//    }
//    um2::Mat2x2<T> prod2 = 2 * m;
//    for (len_t j = 0; j < N; ++j) {
//        for (len_t i = 0; i < M; ++i) {
//            ASSERT_APPROX((prod2[i, j]), (m[i, j]) * 2, 1e-6, "Mat::operator*");
//        }
//    }
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(mat_scalar_div)
//    um2::Mat2x2<T> m = make_mat<M, N, T>();
//    um2::Mat2x2<T> quot = m / 2;
//    for (len_t j = 0; j < N; ++j) {
//        for (len_t i = 0; i < M; ++i) {
//            ASSERT_APPROX((quot[i, j]), (m[i, j]) / 2, 1e-6, "Mat::operator/");
//        }
//    }
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(mat2x2_det)
//    um2::Mat<2, 2, T> m = make_mat<2, 2, T>();
//    T detv = det(m);
//    T expected = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0];
//    ASSERT_APPROX(detv, expected, 1e-6, "det");
//END_TEST
//
//template <typename T>
//UM2_HOSTDEV TEST(mat2x2_inv)
//    um2::Mat<2, 2, T> m = make_mat<2, 2, T>();
//    um2::Mat<2, 2, T> m_inv = inv(m);
//    um2::Mat<2, 2, T> expected;
//    T detv = det(m);
//    expected.cols[0].data[0] =  m[1, 1] / detv;
//    expected.cols[1].data[0] = -m[0, 1] / detv;
//    expected.cols[0].data[1] = -m[1, 0] / detv;
//    expected.cols[1].data[1] =  m[0, 0] / detv;
//    for (len_t j = 0; j < 2; ++j) {
//        for (len_t i = 0; i < 2; ++i) {
//            ASSERT_APPROX((m_inv[i, j]), (expected[i, j]), 1e-6, "inv");
//        }
//    }
//END_TEST
//
////#if UM2_HAS_CUDA
////#endif
//
template <typename T>
TEST_SUITE(mat2x2) {
    TEST(accessors<T>);
////    TEST((unary_minus<2, 2, T>) );
////    TEST((add<2, 2, T>) );
////    TEST((sub<2, 2, T>) );
////    TEST((vec_mul<2, 2, T>) );
////    TEST((mul<2, 2, T>) );
////    TEST((scalar_mul<2, 2, T>) );
////    TEST((scalar_div<2, 2, T>) );
////    TEST((mat2x2_det<T>) );
////    TEST((mat2x2_inv<T>) );
}

auto main() -> int
{
    RUN_TESTS(mat2x2<float>);
    RUN_TESTS(mat2x2<double>);
    return 0;
}
