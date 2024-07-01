#include <um2/common/cast_if_not.hpp>
#include <um2/config.hpp>
#include <um2/math/matrix.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

//=============================================================================
// Real
//=============================================================================

template <class T>
TEST_CASE(accessors_real)
{
  auto const eps = castIfNot<T>(1e-6);
  um2::Matrix<T> m(3, 4);
  ASSERT(m.rows() == 3);
  ASSERT(m.cols() == 4);
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
TEST_CASE(mat_vec_real)
{
  auto const eps = castIfNot<T>(1e-6);
  Int constexpr n = 3;
  um2::Matrix<T> const identity = um2::Matrix<T>::identity(n);
  um2::Vector<T> v(n);
  for (Int i = 0; i < n; ++i) {
    v[i] = static_cast<T>(i);
  }
  auto mv = identity * v;
  ASSERT(mv.size() == n);
  for (Int i = 0; i < n; ++i) {
    ASSERT_NEAR(mv[i], static_cast<T>(i), eps);
  }

  //  5  4  3     100    1000
  //  8  9  5  x  80  =  1820
  //  6  5  3     60     1180
  // 11  9  6            2180

  um2::Matrix<T> a(4, 3);
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
  ASSERT(b.size() == 4);
  ASSERT_NEAR(b[0], static_cast<T>(1000), eps);
  ASSERT_NEAR(b[1], static_cast<T>(1820), eps);
  ASSERT_NEAR(b[2], static_cast<T>(1180), eps);
  ASSERT_NEAR(b[3], static_cast<T>(2180), eps);
}

template <class T>
TEST_CASE(add_sub_real)
{
  auto const eps = castIfNot<T>(1e-6);
  um2::Matrix<T> a(3, 4);
  um2::Matrix<T> b(3, 4);
  um2::Matrix<T> c(3, 4);
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
  um2::Matrix<T> a(4, 3);
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
  um2::Matrix<T> b(3, 3);
  b(0) = 1;
  b(1) = 2;
  b(2) = 4;

  b(3) = 2;
  b(4) = 3;
  b(5) = 2;

  b(6) = 1;
  b(7) = 1;
  b(8) = 2;

  auto ab = a * b;
  ASSERT(ab.rows() == 4);
  ASSERT(ab.cols() == 3);
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

  ab.zero();
  matmul(ab, a, b);
  ASSERT(ab.rows() == 4);
  ASSERT(ab.cols() == 3);
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

#if UM2_USE_BLAS_LAPACK
template <class T>
TEST_CASE(lin_solve_real)
{

  // Solve A * X = B
  // A = 6.80  -6.05  -0.45   8.32  -9.67
  //    -2.11  -3.30   2.58   2.71  -5.14
  //     5.66   5.36  -2.70   4.35  -7.26
  //     5.97  -4.44   0.27  -7.17   6.08
  //     8.23   1.08   9.04   2.14  -6.87
  //
  // B = 4.02  -1.56   9.81
  //     6.19   4.00  -4.09
  //    -8.22  -8.67  -4.57
  //    -7.57   1.75  -8.61
  //    -3.03   2.86   8.99

  auto constexpr eps = castIfNot<T>(1e-2);
  um2::Matrix<T> a(5, 5);
  a(0) = static_cast<T>(6.80);
  a(1) = static_cast<T>(-2.11);
  a(2) = static_cast<T>(5.66);
  a(3) = static_cast<T>(5.97);
  a(4) = static_cast<T>(8.23);

  a(5) = static_cast<T>(-6.05);
  a(6) = static_cast<T>(-3.30);
  a(7) = static_cast<T>(5.36);
  a(8) = static_cast<T>(-4.44);
  a(9) = static_cast<T>(1.08);

  a(10) = static_cast<T>(-0.45);
  a(11) = static_cast<T>(2.58);
  a(12) = static_cast<T>(-2.70);
  a(13) = static_cast<T>(0.27);
  a(14) = static_cast<T>(9.04);

  a(15) = static_cast<T>(8.32);
  a(16) = static_cast<T>(2.71);
  a(17) = static_cast<T>(4.35);
  a(18) = static_cast<T>(-7.17);
  a(19) = static_cast<T>(2.14);

  a(20) = static_cast<T>(-9.67);
  a(21) = static_cast<T>(-5.14);
  a(22) = static_cast<T>(-7.26);
  a(23) = static_cast<T>(6.08);
  a(24) = static_cast<T>(-6.87);

  um2::Matrix<T> b(5, 3);
  b(0) = static_cast<T>(4.02);
  b(1) = static_cast<T>(6.19);
  b(2) = static_cast<T>(-8.22);
  b(3) = static_cast<T>(-7.57);
  b(4) = static_cast<T>(-3.03);

  b(5) = static_cast<T>(-1.56);
  b(6) = static_cast<T>(4.00);
  b(7) = static_cast<T>(-8.67);
  b(8) = static_cast<T>(1.75);
  b(9) = static_cast<T>(2.86);

  b(10) = static_cast<T>(9.81);
  b(11) = static_cast<T>(-4.09);
  b(12) = static_cast<T>(-4.57);
  b(13) = static_cast<T>(-8.61);
  b(14) = static_cast<T>(8.99);

  // X = -0.80  -0.39   0.96
  //     -0.70  -0.55   0.22
  //      0.59   0.84   1.90
  //      1.32  -0.10   5.36
  //      0.57   0.11   4.04

  auto x = linearSolve(a, b);
  ASSERT(x.rows() == 5);
  ASSERT(x.cols() == 3);

  ASSERT_NEAR(x(0, 0), castIfNot<T>(-0.80), eps);
  ASSERT_NEAR(x(1, 0), castIfNot<T>(-0.70), eps);
  ASSERT_NEAR(x(2, 0), castIfNot<T>(0.59), eps);
  ASSERT_NEAR(x(3, 0), castIfNot<T>(1.32), eps);
  ASSERT_NEAR(x(4, 0), castIfNot<T>(0.57), eps);

  ASSERT_NEAR(x(0, 1), castIfNot<T>(-0.39), eps);
  ASSERT_NEAR(x(1, 1), castIfNot<T>(-0.55), eps);
  ASSERT_NEAR(x(2, 1), castIfNot<T>(0.84), eps);
  ASSERT_NEAR(x(3, 1), castIfNot<T>(-0.10), eps);
  ASSERT_NEAR(x(4, 1), castIfNot<T>(0.11), eps);

  ASSERT_NEAR(x(0, 2), castIfNot<T>(0.96), eps);
  ASSERT_NEAR(x(1, 2), castIfNot<T>(0.22), eps);
  ASSERT_NEAR(x(2, 2), castIfNot<T>(1.90), eps);
  ASSERT_NEAR(x(3, 2), castIfNot<T>(5.36), eps);
  ASSERT_NEAR(x(4, 2), castIfNot<T>(4.04), eps);

  um2::Vector<Int> ipiv(5);
  linearSolve(a, b, ipiv);
  ASSERT(b.rows() == 5);
  ASSERT(b.cols() == 3);

  ASSERT_NEAR(b(0, 0), castIfNot<T>(-0.80), eps);
  ASSERT_NEAR(b(1, 0), castIfNot<T>(-0.70), eps);
  ASSERT_NEAR(b(2, 0), castIfNot<T>(0.59), eps);
  ASSERT_NEAR(b(3, 0), castIfNot<T>(1.32), eps);
  ASSERT_NEAR(b(4, 0), castIfNot<T>(0.57), eps);

  ASSERT_NEAR(b(0, 1), castIfNot<T>(-0.39), eps);
  ASSERT_NEAR(b(1, 1), castIfNot<T>(-0.55), eps);
  ASSERT_NEAR(b(2, 1), castIfNot<T>(0.84), eps);
  ASSERT_NEAR(b(3, 1), castIfNot<T>(-0.10), eps);
  ASSERT_NEAR(b(4, 1), castIfNot<T>(0.11), eps);

  ASSERT_NEAR(b(0, 2), castIfNot<T>(0.96), eps);
  ASSERT_NEAR(b(1, 2), castIfNot<T>(0.22), eps);
  ASSERT_NEAR(b(2, 2), castIfNot<T>(1.90), eps);
  ASSERT_NEAR(b(3, 2), castIfNot<T>(5.36), eps);
  ASSERT_NEAR(b(4, 2), castIfNot<T>(4.04), eps);
}

template <class T>
TEST_CASE(eigvals_real)
{
  // A = -1.01   0.86  -4.60   3.31  -4.81
  //      3.98   0.53  -7.04   5.29   3.55
  //      3.30   8.26  -3.89   8.20  -1.51
  //      4.43   4.96  -7.66  -7.33   6.18
  //      7.31  -6.43  -6.16   2.47   5.58
  //
  // lambda =
  //   (  2.86, 10.76)
  //   (  2.86,-10.76)
  //   ( -0.69,  4.70)
  //   ( -0.69, -4.70)
  //   (-10.46,  0.00)

  auto constexpr eps = castIfNot<T>(1e-2);
  um2::Matrix<T> a(5, 5);
  a(0) = static_cast<T>(-1.01);
  a(1) = static_cast<T>(3.98);
  a(2) = static_cast<T>(3.30);
  a(3) = static_cast<T>(4.43);
  a(4) = static_cast<T>(7.31);

  a(5) = static_cast<T>(0.86);
  a(6) = static_cast<T>(0.53);
  a(7) = static_cast<T>(8.26);
  a(8) = static_cast<T>(4.96);
  a(9) = static_cast<T>(-6.43);

  a(10) = static_cast<T>(-4.60);
  a(11) = static_cast<T>(-7.04);
  a(12) = static_cast<T>(-3.89);
  a(13) = static_cast<T>(-7.66);
  a(14) = static_cast<T>(-6.16);

  a(15) = static_cast<T>(3.31);
  a(16) = static_cast<T>(5.29);
  a(17) = static_cast<T>(8.20);
  a(18) = static_cast<T>(-7.33);
  a(19) = static_cast<T>(2.47);

  a(20) = static_cast<T>(-4.81);
  a(21) = static_cast<T>(3.55);
  a(22) = static_cast<T>(-1.51);
  a(23) = static_cast<T>(6.18);
  a(24) = static_cast<T>(5.58);

  auto const lambda = eigvals(a);

  ASSERT(lambda.size() == 5);
  ASSERT_NEAR(lambda[0].real(), static_cast<T>(2.86), eps);
  ASSERT_NEAR(lambda[0].imag(), static_cast<T>(10.76), eps);

  ASSERT_NEAR(lambda[1].real(), static_cast<T>(2.86), eps);
  ASSERT_NEAR(lambda[1].imag(), static_cast<T>(-10.76), eps);

  ASSERT_NEAR(lambda[2].real(), static_cast<T>(-0.69), eps);
  ASSERT_NEAR(lambda[2].imag(), static_cast<T>(4.70), eps);

  ASSERT_NEAR(lambda[3].real(), static_cast<T>(-0.69), eps);
  ASSERT_NEAR(lambda[3].imag(), static_cast<T>(-4.70), eps);

  ASSERT_NEAR(lambda[4].real(), static_cast<T>(-10.46), eps);
  ASSERT_NEAR(lambda[4].imag(), static_cast<T>(0.00), eps);
}
#endif // UM2_USE_BLAS_LAPACK

template <class T>
TEST_CASE(transpose_real)
{

  // A = 1  2  3
  //     4  5  6
  //     7  8  9
  um2::Matrix<T> a(3, 3);
  a(0) = 1;
  a(1) = 4;
  a(2) = 7;

  a(3) = 2;
  a(4) = 5;
  a(5) = 8;

  a(6) = 3;
  a(7) = 6;
  a(8) = 9;

  a.transpose();
  for (Int i = 0; i < 9; ++i) {
    ASSERT_NEAR(a(i), static_cast<T>(i + 1), castIfNot<T>(1e-6));
  }
}

//=============================================================================
// Complex
//=============================================================================

template <class T>
TEST_CASE(accessors_complex)
{
  using ComplexT = Complex<T>;
  auto const eps = castIfNot<T>(1e-6);
  um2::Matrix<ComplexT> m(3, 4);
  ASSERT(m.rows() == 3);
  ASSERT(m.cols() == 4);
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      ComplexT const c(static_cast<T>(j * 3 + i), static_cast<T>(j * 3 + i + 1));
      m(i, j) = c;
    }
  }
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      ComplexT const c(static_cast<T>(j * 3 + i), static_cast<T>(j * 3 + i + 1));
      ASSERT_NEAR(m(i, j).real(), c.real(), eps);
      ASSERT_NEAR(m(i, j).imag(), c.imag(), eps);
    }
  }
}

template <class T>
TEST_CASE(mat_vec_complex)
{
  using ComplexT = Complex<T>;
  auto const eps = castIfNot<T>(1e-6);
  Int constexpr n = 3;
  um2::Matrix<ComplexT> const identity = um2::Matrix<ComplexT>::identity(n);
  um2::Vector<ComplexT> v(n);
  for (Int i = 0; i < n; ++i) {
    ComplexT const c(static_cast<T>(i), static_cast<T>(i + 1));
    v[i] = c;
  }
  auto mv = identity * v;
  ASSERT(mv.size() == n);
  for (Int i = 0; i < n; ++i) {
    ComplexT const c(static_cast<T>(i), static_cast<T>(i + 1));
    ASSERT_NEAR(mv[i].real(), c.real(), eps);
    ASSERT_NEAR(mv[i].imag(), c.imag(), eps);
  }
}

template <class T>
TEST_CASE(add_sub_complex)
{
  using ComplexT = Complex<T>;
  auto const eps = castIfNot<T>(1e-6);
  um2::Matrix<ComplexT> a(3, 4);
  um2::Matrix<ComplexT> b(3, 4);
  um2::Matrix<ComplexT> c(3, 4);
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      a(i, j) = ComplexT(static_cast<T>(j * 3 + i), static_cast<T>(j * 3 + i));
      b(i, j) = ComplexT(static_cast<T>(j * 3 + i + 1), static_cast<T>(j * 3 + i + 1));
      c(i, j) = ComplexT(static_cast<T>(2 * j * 3 + 2 * i + 1),
                         static_cast<T>(2 * j * 3 + 2 * i + 1));
    }
  }
  auto const d = a + b;
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      ASSERT_NEAR(d(i, j).real(), c(i, j).real(), eps);
      ASSERT_NEAR(d(i, j).imag(), c(i, j).imag(), eps);
    }
  }
  auto const e = b - a;
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      ASSERT_NEAR(e(i, j).real(), static_cast<T>(1), eps);
      ASSERT_NEAR(e(i, j).imag(), static_cast<T>(1), eps);
    }
  }
}

template <class T>
TEST_CASE(mat_mul_complex)
{
  using ComplexT = Complex<T>;
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
  um2::Matrix<ComplexT> a(4, 3);
  a(0) = ComplexT(1, 0);
  a(1) = ComplexT(2, 0);
  a(2) = ComplexT(0, 0);
  a(3) = ComplexT(1, 0);

  a(4) = ComplexT(0, 0);
  a(5) = ComplexT(1, 0);
  a(6) = ComplexT(1, 0);
  a(7) = ComplexT(1, 0);

  a(8) = ComplexT(1, 0);
  a(9) = ComplexT(1, 0);
  a(10) = ComplexT(1, 0);
  a(11) = ComplexT(2, 0);
  um2::Matrix<ComplexT> b(3, 3);
  b(0) = ComplexT(1, 0);
  b(1) = ComplexT(2, 0);
  b(2) = ComplexT(4, 0);

  b(3) = ComplexT(2, 0);
  b(4) = ComplexT(3, 0);
  b(5) = ComplexT(2, 0);

  b(6) = ComplexT(1, 0);
  b(7) = ComplexT(1, 0);
  b(8) = ComplexT(2, 0);

  auto ab = a * b;
  ASSERT(ab.rows() == 4);
  ASSERT(ab.cols() == 3);
  ASSERT_NEAR(ab(0).real(), 5, eps);
  ASSERT_NEAR(ab(1).real(), 8, eps);
  ASSERT_NEAR(ab(2).real(), 6, eps);
  ASSERT_NEAR(ab(3).real(), 11, eps);

  ASSERT_NEAR(ab(4).real(), 4, eps);
  ASSERT_NEAR(ab(5).real(), 9, eps);
  ASSERT_NEAR(ab(6).real(), 5, eps);
  ASSERT_NEAR(ab(7).real(), 9, eps);

  ASSERT_NEAR(ab(8).real(), 3, eps);
  ASSERT_NEAR(ab(9).real(), 5, eps);
  ASSERT_NEAR(ab(10).real(), 3, eps);
  ASSERT_NEAR(ab(11).real(), 6, eps);

  for (Int i = 0; i < 12; ++i) {
    ASSERT_NEAR(ab(i).imag(), 0, eps);
  }

  ab.zero();
  matmul(ab, a, b);
  ASSERT(ab.rows() == 4);
  ASSERT(ab.cols() == 3);
  ASSERT_NEAR(ab(0).real(), 5, eps);
  ASSERT_NEAR(ab(1).real(), 8, eps);
  ASSERT_NEAR(ab(2).real(), 6, eps);
  ASSERT_NEAR(ab(3).real(), 11, eps);

  ASSERT_NEAR(ab(4).real(), 4, eps);
  ASSERT_NEAR(ab(5).real(), 9, eps);
  ASSERT_NEAR(ab(6).real(), 5, eps);
  ASSERT_NEAR(ab(7).real(), 9, eps);

  ASSERT_NEAR(ab(8).real(), 3, eps);
  ASSERT_NEAR(ab(9).real(), 5, eps);
  ASSERT_NEAR(ab(10).real(), 3, eps);
  ASSERT_NEAR(ab(11).real(), 6, eps);

  for (Int i = 0; i < 12; ++i) {
    ASSERT_NEAR(ab(i).imag(), 0, eps);
  }
}

#if UM2_USE_BLAS_LAPACK
template <class T>
TEST_CASE(lin_solve_complex)
{

  // Solve A * X = B
  // A = 6.80  -6.05  -0.45   8.32  -9.67
  //    -2.11  -3.30   2.58   2.71  -5.14
  //     5.66   5.36  -2.70   4.35  -7.26
  //     5.97  -4.44   0.27  -7.17   6.08
  //     8.23   1.08   9.04   2.14  -6.87
  //
  // B = 4.02  -1.56   9.81
  //     6.19   4.00  -4.09
  //    -8.22  -8.67  -4.57
  //    -7.57   1.75  -8.61
  //    -3.03   2.86   8.99

  using ComplexT = Complex<T>;
  auto constexpr eps = castIfNot<T>(1e-2);
  um2::Matrix<ComplexT> a(5, 5);
  a(0) = static_cast<T>(6.80);
  a(1) = static_cast<T>(-2.11);
  a(2) = static_cast<T>(5.66);
  a(3) = static_cast<T>(5.97);
  a(4) = static_cast<T>(8.23);

  a(5) = static_cast<T>(-6.05);
  a(6) = static_cast<T>(-3.30);
  a(7) = static_cast<T>(5.36);
  a(8) = static_cast<T>(-4.44);
  a(9) = static_cast<T>(1.08);

  a(10) = static_cast<T>(-0.45);
  a(11) = static_cast<T>(2.58);
  a(12) = static_cast<T>(-2.70);
  a(13) = static_cast<T>(0.27);
  a(14) = static_cast<T>(9.04);

  a(15) = static_cast<T>(8.32);
  a(16) = static_cast<T>(2.71);
  a(17) = static_cast<T>(4.35);
  a(18) = static_cast<T>(-7.17);
  a(19) = static_cast<T>(2.14);

  a(20) = static_cast<T>(-9.67);
  a(21) = static_cast<T>(-5.14);
  a(22) = static_cast<T>(-7.26);
  a(23) = static_cast<T>(6.08);
  a(24) = static_cast<T>(-6.87);

  um2::Matrix<ComplexT> b(5, 3);
  b(0) = static_cast<T>(4.02);
  b(1) = static_cast<T>(6.19);
  b(2) = static_cast<T>(-8.22);
  b(3) = static_cast<T>(-7.57);
  b(4) = static_cast<T>(-3.03);

  b(5) = static_cast<T>(-1.56);
  b(6) = static_cast<T>(4.00);
  b(7) = static_cast<T>(-8.67);
  b(8) = static_cast<T>(1.75);
  b(9) = static_cast<T>(2.86);

  b(10) = static_cast<T>(9.81);
  b(11) = static_cast<T>(-4.09);
  b(12) = static_cast<T>(-4.57);
  b(13) = static_cast<T>(-8.61);
  b(14) = static_cast<T>(8.99);

  // X = -0.80  -0.39   0.96
  //     -0.70  -0.55   0.22
  //      0.59   0.84   1.90
  //      1.32  -0.10   5.36
  //      0.57   0.11   4.04

  auto const x = linearSolve(a, b);
  ASSERT(x.rows() == 5);
  ASSERT(x.cols() == 3);

  ASSERT_NEAR(x(0, 0).real(), static_cast<T>(-0.80), eps);
  ASSERT_NEAR(x(1, 0).real(), static_cast<T>(-0.70), eps);
  ASSERT_NEAR(x(2, 0).real(), static_cast<T>(0.59), eps);
  ASSERT_NEAR(x(3, 0).real(), static_cast<T>(1.32), eps);
  ASSERT_NEAR(x(4, 0).real(), static_cast<T>(0.57), eps);

  ASSERT_NEAR(x(0, 1).real(), static_cast<T>(-0.39), eps);
  ASSERT_NEAR(x(1, 1).real(), static_cast<T>(-0.55), eps);
  ASSERT_NEAR(x(2, 1).real(), static_cast<T>(0.84), eps);
  ASSERT_NEAR(x(3, 1).real(), static_cast<T>(-0.10), eps);
  ASSERT_NEAR(x(4, 1).real(), static_cast<T>(0.11), eps);

  ASSERT_NEAR(x(0, 2).real(), static_cast<T>(0.96), eps);
  ASSERT_NEAR(x(1, 2).real(), static_cast<T>(0.22), eps);
  ASSERT_NEAR(x(2, 2).real(), static_cast<T>(1.90), eps);
  ASSERT_NEAR(x(3, 2).real(), static_cast<T>(5.36), eps);
  ASSERT_NEAR(x(4, 2).real(), static_cast<T>(4.04), eps);

  um2::Vector<Int> ipiv(5);
  linearSolve(a, b, ipiv);

  ASSERT(b.rows() == 5);
  ASSERT(b.cols() == 3);
  ASSERT_NEAR(b(0, 0).real(), static_cast<T>(-0.80), eps);
  ASSERT_NEAR(b(1, 0).real(), static_cast<T>(-0.70), eps);
  ASSERT_NEAR(b(2, 0).real(), static_cast<T>(0.59), eps);
  ASSERT_NEAR(b(3, 0).real(), static_cast<T>(1.32), eps);
  ASSERT_NEAR(b(4, 0).real(), static_cast<T>(0.57), eps);

  ASSERT_NEAR(b(0, 1).real(), static_cast<T>(-0.39), eps);
  ASSERT_NEAR(b(1, 1).real(), static_cast<T>(-0.55), eps);
  ASSERT_NEAR(b(2, 1).real(), static_cast<T>(0.84), eps);
  ASSERT_NEAR(b(3, 1).real(), static_cast<T>(-0.10), eps);
  ASSERT_NEAR(b(4, 1).real(), static_cast<T>(0.11), eps);

  ASSERT_NEAR(b(0, 2).real(), static_cast<T>(0.96), eps);
  ASSERT_NEAR(b(1, 2).real(), static_cast<T>(0.22), eps);
  ASSERT_NEAR(b(2, 2).real(), static_cast<T>(1.90), eps);
  ASSERT_NEAR(b(3, 2).real(), static_cast<T>(5.36), eps);
  ASSERT_NEAR(b(4, 2).real(), static_cast<T>(4.04), eps);
}

template <class T>
TEST_CASE(eigvals_complex)
{
  // A = -1.01   0.86  -4.60   3.31  -4.81
  //      3.98   0.53  -7.04   5.29   3.55
  //      3.30   8.26  -3.89   8.20  -1.51
  //      4.43   4.96  -7.66  -7.33   6.18
  //      7.31  -6.43  -6.16   2.47   5.58
  //
  // lambda =
  //   (  2.86, 10.76)
  //   (  2.86,-10.76)
  //   ( -0.69,  4.70)
  //   ( -0.69, -4.70)
  //   (-10.46,  0.00)

  using ComplexT = Complex<T>;
  auto constexpr eps = castIfNot<T>(1e-2);
  um2::Matrix<ComplexT> a(5, 5);
  a(0) = static_cast<T>(-1.01);
  a(1) = static_cast<T>(3.98);
  a(2) = static_cast<T>(3.30);
  a(3) = static_cast<T>(4.43);
  a(4) = static_cast<T>(7.31);

  a(5) = static_cast<T>(0.86);
  a(6) = static_cast<T>(0.53);
  a(7) = static_cast<T>(8.26);
  a(8) = static_cast<T>(4.96);
  a(9) = static_cast<T>(-6.43);

  a(10) = static_cast<T>(-4.60);
  a(11) = static_cast<T>(-7.04);
  a(12) = static_cast<T>(-3.89);
  a(13) = static_cast<T>(-7.66);
  a(14) = static_cast<T>(-6.16);

  a(15) = static_cast<T>(3.31);
  a(16) = static_cast<T>(5.29);
  a(17) = static_cast<T>(8.20);
  a(18) = static_cast<T>(-7.33);
  a(19) = static_cast<T>(2.47);

  a(20) = static_cast<T>(-4.81);
  a(21) = static_cast<T>(3.55);
  a(22) = static_cast<T>(-1.51);
  a(23) = static_cast<T>(6.18);
  a(24) = static_cast<T>(5.58);

  auto const lambda = eigvals(a);

  ASSERT(lambda.size() == 5);
  ASSERT_NEAR(lambda[0].real(), static_cast<T>(2.86), eps);
  ASSERT_NEAR(lambda[0].imag(), static_cast<T>(10.76), eps);

  ASSERT_NEAR(lambda[1].real(), static_cast<T>(2.86), eps);
  ASSERT_NEAR(lambda[1].imag(), static_cast<T>(-10.76), eps);

  ASSERT_NEAR(lambda[2].real(), static_cast<T>(-0.69), eps);
  ASSERT_NEAR(lambda[2].imag(), static_cast<T>(4.70), eps);

  ASSERT_NEAR(lambda[3].real(), static_cast<T>(-0.69), eps);
  ASSERT_NEAR(lambda[3].imag(), static_cast<T>(-4.70), eps);

  ASSERT_NEAR(lambda[4].real(), static_cast<T>(-10.46), eps);
  ASSERT_NEAR(lambda[4].imag(), static_cast<T>(0.00), eps);
}
#endif // UM2_USE_BLAS_LAPACK

template <class T>
TEST_CASE(transpose_complex)
{

  // A = (1, 1)  (2, -2)
  //     (3, 3)  (4, -4)
  using ComplexT = Complex<T>;
  um2::Matrix<ComplexT> a(2, 2);
  a(0) = ComplexT(1, 1);
  a(1) = ComplexT(3, 3);
  a(2) = ComplexT(2, -2);
  a(3) = ComplexT(4, -4);

  auto constexpr eps = castIfNot<T>(1e-6);
  a.transpose();
  ASSERT_NEAR(a(0).real(), 1, eps);
  ASSERT_NEAR(a(0).imag(), -1, eps);

  ASSERT_NEAR(a(1).real(), 2, eps);
  ASSERT_NEAR(a(1).imag(), 2, eps);

  ASSERT_NEAR(a(2).real(), 3, eps);
  ASSERT_NEAR(a(2).imag(), -3, eps);

  ASSERT_NEAR(a(3).real(), 4, eps);
  ASSERT_NEAR(a(3).imag(), 4, eps);
}

template <class T>
TEST_SUITE(Matrix_real)
{
  TEST(accessors_real<T>);
  TEST(mat_vec_real<T>);
  TEST(add_sub_real<T>);
  TEST(mat_mul_real<T>);
#if UM2_USE_BLAS_LAPACK
  TEST(lin_solve_real<T>);
  TEST(eigvals_real<T>);
#endif
  TEST(transpose_real<T>);
}

template <class T>
TEST_SUITE(Matrix_complex)
{
  TEST(accessors_complex<T>);
  TEST(mat_vec_complex<T>);
  TEST(add_sub_complex<T>);
  TEST(mat_mul_complex<T>);
#if UM2_USE_BLAS_LAPACK
  TEST(lin_solve_complex<T>);
  TEST(eigvals_complex<T>);
#endif
  TEST(transpose_complex<T>);
}

auto
main() -> int
{
  RUN_SUITE(Matrix_real<float>);
  RUN_SUITE(Matrix_real<double>);
  RUN_SUITE(Matrix_complex<float>);
  RUN_SUITE(Matrix_complex<double>);
  return 0;
}
