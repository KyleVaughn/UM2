#include <um2/config.hpp>
#include <um2/math/matrix.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

#include <complex>

using Complex32 = std::complex<float>;
using Complex64 = std::complex<double>;

// TODO(kcvaughn): Template of float/double and Complex32/Complex64. This should reduce
// code size by half

//=============================================================================
// float
//=============================================================================

HOSTDEV
TEST_CASE(accessors_float)
{
  using T = float;
  auto const eps = 1e-6F;
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

HOSTDEV
TEST_CASE(mat_vec_float)
{
  using T = float;
  auto const eps = 1e-6F;
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

HOSTDEV
TEST_CASE(add_sub_float)
{
  using T = float;
  auto const eps = 1e-6F;
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

TEST_CASE(mat_mul_float)
{
  using T = float;
  auto const eps = 1e-6F;
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

TEST_CASE(lin_solve_float)
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

  auto constexpr eps = 1e-2F;
  um2::Matrix<float> a(5, 5);
  a(0) = 6.80F;
  a(1) = -2.11F;
  a(2) = 5.66F;
  a(3) = 5.97F;
  a(4) = 8.23F;

  a(5) = -6.05F;
  a(6) = -3.30F;
  a(7) = 5.36F;
  a(8) = -4.44F;
  a(9) = 1.08F;

  a(10) = -0.45F;
  a(11) = 2.58F;
  a(12) = -2.70F;
  a(13) = 0.27F;
  a(14) = 9.04F;

  a(15) = 8.32F;
  a(16) = 2.71F;
  a(17) = 4.35F;
  a(18) = -7.17F;
  a(19) = 2.14F;

  a(20) = -9.67F;
  a(21) = -5.14F;
  a(22) = -7.26F;
  a(23) = 6.08F;
  a(24) = -6.87F;

  um2::Matrix<float> b(5, 3);
  b(0) = 4.02F;
  b(1) = 6.19F;
  b(2) = -8.22F;
  b(3) = -7.57F;
  b(4) = -3.03F;

  b(5) = -1.56F;
  b(6) = 4.00F;
  b(7) = -8.67F;
  b(8) = 1.75F;
  b(9) = 2.86F;

  b(10) = 9.81F;
  b(11) = -4.09F;
  b(12) = -4.57F;
  b(13) = -8.61F;
  b(14) = 8.99F;

  // X = -0.80  -0.39   0.96
  //     -0.70  -0.55   0.22
  //      0.59   0.84   1.90
  //      1.32  -0.10   5.36
  //      0.57   0.11   4.04

  auto x = linearSolve(a, b);
  ASSERT(x.rows() == 5);
  ASSERT(x.cols() == 3);

  ASSERT_NEAR(x(0, 0), -0.80F, eps);
  ASSERT_NEAR(x(1, 0), -0.70F, eps);
  ASSERT_NEAR(x(2, 0), 0.59F, eps);
  ASSERT_NEAR(x(3, 0), 1.32F, eps);
  ASSERT_NEAR(x(4, 0), 0.57F, eps);

  ASSERT_NEAR(x(0, 1), -0.39F, eps);
  ASSERT_NEAR(x(1, 1), -0.55F, eps);
  ASSERT_NEAR(x(2, 1), 0.84F, eps);
  ASSERT_NEAR(x(3, 1), -0.10F, eps);
  ASSERT_NEAR(x(4, 1), 0.11F, eps);

  ASSERT_NEAR(x(0, 2), 0.96F, eps);
  ASSERT_NEAR(x(1, 2), 0.22F, eps);
  ASSERT_NEAR(x(2, 2), 1.90F, eps);
  ASSERT_NEAR(x(3, 2), 5.36F, eps);
  ASSERT_NEAR(x(4, 2), 4.04F, eps);

  um2::Vector<Int> ipiv(5);
  linearSolve(a, b, ipiv);
  ASSERT(b.rows() == 5);
  ASSERT(b.cols() == 3);

  ASSERT_NEAR(b(0, 0), -0.80F, eps);
  ASSERT_NEAR(b(1, 0), -0.70F, eps);
  ASSERT_NEAR(b(2, 0), 0.59F, eps);
  ASSERT_NEAR(b(3, 0), 1.32F, eps);
  ASSERT_NEAR(b(4, 0), 0.57F, eps);

  ASSERT_NEAR(b(0, 1), -0.39F, eps);
  ASSERT_NEAR(b(1, 1), -0.55F, eps);
  ASSERT_NEAR(b(2, 1), 0.84F, eps);
  ASSERT_NEAR(b(3, 1), -0.10F, eps);
  ASSERT_NEAR(b(4, 1), 0.11F, eps);

  ASSERT_NEAR(b(0, 2), 0.96F, eps);
  ASSERT_NEAR(b(1, 2), 0.22F, eps);
  ASSERT_NEAR(b(2, 2), 1.90F, eps);
  ASSERT_NEAR(b(3, 2), 5.36F, eps);
  ASSERT_NEAR(b(4, 2), 4.04F, eps);
}

TEST_CASE(eigvals_float)
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

  auto constexpr eps = 1e-2F;
  um2::Matrix<float> a(5, 5);
  a(0) = -1.01F;
  a(1) = 3.98F;
  a(2) = 3.30F;
  a(3) = 4.43F;
  a(4) = 7.31F;

  a(5) = 0.86F;
  a(6) = 0.53F;
  a(7) = 8.26F;
  a(8) = 4.96F;
  a(9) = -6.43F;

  a(10) = -4.60F;
  a(11) = -7.04F;
  a(12) = -3.89F;
  a(13) = -7.66F;
  a(14) = -6.16F;

  a(15) = 3.31F;
  a(16) = 5.29F;
  a(17) = 8.20F;
  a(18) = -7.33F;
  a(19) = 2.47F;

  a(20) = -4.81F;
  a(21) = 3.55F;
  a(22) = -1.51F;
  a(23) = 6.18F;
  a(24) = 5.58F;

  auto const lambda = eigvals(a);

  ASSERT(lambda.size() == 5);
  ASSERT_NEAR(lambda[0].real(), 2.86F, eps);
  ASSERT_NEAR(lambda[0].imag(), 10.76F, eps);

  ASSERT_NEAR(lambda[1].real(), 2.86F, eps);
  ASSERT_NEAR(lambda[1].imag(), -10.76F, eps);

  ASSERT_NEAR(lambda[2].real(), -0.69F, eps);
  ASSERT_NEAR(lambda[2].imag(), 4.70F, eps);

  ASSERT_NEAR(lambda[3].real(), -0.69F, eps);
  ASSERT_NEAR(lambda[3].imag(), -4.70F, eps);

  ASSERT_NEAR(lambda[4].real(), -10.46F, eps);
  ASSERT_NEAR(lambda[4].imag(), 0.00F, eps);
}

TEST_CASE(transpose_float)
{

  // A = 1  2  3
  //     4  5  6
  //     7  8  9
  um2::Matrix<float> a(3, 3);
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
    ASSERT_NEAR(a(i), static_cast<float>(i + 1), 1e-6F);
  }
}

//=============================================================================
// double
//=============================================================================

HOSTDEV
TEST_CASE(accessors_double)
{
  using T = double;
  auto const eps = 1e-6;
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

HOSTDEV
TEST_CASE(mat_vec_double)
{
  using T = double;
  auto const eps = 1e-6;
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

HOSTDEV
TEST_CASE(add_sub_double)
{
  using T = double;
  auto const eps = 1e-6;
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

TEST_CASE(mat_mul_double)
{
  using T = double;
  auto const eps = 1e-6;
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

TEST_CASE(lin_solve_double)
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

  auto constexpr eps = 1e-2;
  um2::Matrix<double> a(5, 5);
  a(0) = 6.80;
  a(1) = -2.11;
  a(2) = 5.66;
  a(3) = 5.97;
  a(4) = 8.23;

  a(5) = -6.05;
  a(6) = -3.30;
  a(7) = 5.36;
  a(8) = -4.44;
  a(9) = 1.08;

  a(10) = -0.45;
  a(11) = 2.58;
  a(12) = -2.70;
  a(13) = 0.27;
  a(14) = 9.04;

  a(15) = 8.32;
  a(16) = 2.71;
  a(17) = 4.35;
  a(18) = -7.17;
  a(19) = 2.14;

  a(20) = -9.67;
  a(21) = -5.14;
  a(22) = -7.26;
  a(23) = 6.08;
  a(24) = -6.87;

  um2::Matrix<double> b(5, 3);
  b(0) = 4.02;
  b(1) = 6.19;
  b(2) = -8.22;
  b(3) = -7.57;
  b(4) = -3.03;

  b(5) = -1.56;
  b(6) = 4.00;
  b(7) = -8.67;
  b(8) = 1.75;
  b(9) = 2.86;

  b(10) = 9.81;
  b(11) = -4.09;
  b(12) = -4.57;
  b(13) = -8.61;
  b(14) = 8.99;

  // X = -0.80  -0.39   0.96
  //     -0.70  -0.55   0.22
  //      0.59   0.84   1.90
  //      1.32  -0.10   5.36
  //      0.57   0.11   4.04

  auto const x = linearSolve(a, b);
  ASSERT(x.rows() == 5);
  ASSERT(x.cols() == 3);

  ASSERT_NEAR(x(0, 0), -0.80, eps);
  ASSERT_NEAR(x(1, 0), -0.70, eps);
  ASSERT_NEAR(x(2, 0), 0.59, eps);
  ASSERT_NEAR(x(3, 0), 1.32, eps);
  ASSERT_NEAR(x(4, 0), 0.57, eps);

  ASSERT_NEAR(x(0, 1), -0.39, eps);
  ASSERT_NEAR(x(1, 1), -0.55, eps);
  ASSERT_NEAR(x(2, 1), 0.84, eps);
  ASSERT_NEAR(x(3, 1), -0.10, eps);
  ASSERT_NEAR(x(4, 1), 0.11, eps);

  ASSERT_NEAR(x(0, 2), 0.96, eps);
  ASSERT_NEAR(x(1, 2), 0.22, eps);
  ASSERT_NEAR(x(2, 2), 1.90, eps);
  ASSERT_NEAR(x(3, 2), 5.36, eps);
  ASSERT_NEAR(x(4, 2), 4.04, eps);

  um2::Vector<Int> ipiv(5);
  linearSolve(a, b, ipiv);

  ASSERT(b.rows() == 5);
  ASSERT(b.cols() == 3);

  ASSERT_NEAR(b(0, 0), -0.80, eps);
  ASSERT_NEAR(b(1, 0), -0.70, eps);
  ASSERT_NEAR(b(2, 0), 0.59, eps);
  ASSERT_NEAR(b(3, 0), 1.32, eps);
  ASSERT_NEAR(b(4, 0), 0.57, eps);

  ASSERT_NEAR(b(0, 1), -0.39, eps);
  ASSERT_NEAR(b(1, 1), -0.55, eps);
  ASSERT_NEAR(b(2, 1), 0.84, eps);
  ASSERT_NEAR(b(3, 1), -0.10, eps);
  ASSERT_NEAR(b(4, 1), 0.11, eps);

  ASSERT_NEAR(b(0, 2), 0.96, eps);
  ASSERT_NEAR(b(1, 2), 0.22, eps);
  ASSERT_NEAR(b(2, 2), 1.90, eps);
  ASSERT_NEAR(b(3, 2), 5.36, eps);
  ASSERT_NEAR(b(4, 2), 4.04, eps);
}

TEST_CASE(eigvals_double)
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

  auto constexpr eps = 1e-2;
  um2::Matrix<double> a(5, 5);
  a(0) = -1.01;
  a(1) = 3.98;
  a(2) = 3.30;
  a(3) = 4.43;
  a(4) = 7.31;

  a(5) = 0.86;
  a(6) = 0.53;
  a(7) = 8.26;
  a(8) = 4.96;
  a(9) = -6.43;

  a(10) = -4.60;
  a(11) = -7.04;
  a(12) = -3.89;
  a(13) = -7.66;
  a(14) = -6.16;

  a(15) = 3.31;
  a(16) = 5.29;
  a(17) = 8.20;
  a(18) = -7.33;
  a(19) = 2.47;

  a(20) = -4.81;
  a(21) = 3.55;
  a(22) = -1.51;
  a(23) = 6.18;
  a(24) = 5.58;

  auto const lambda = eigvals(a);

  ASSERT(lambda.size() == 5);
  ASSERT_NEAR(lambda[0].real(), 2.86, eps);
  ASSERT_NEAR(lambda[0].imag(), 10.76, eps);

  ASSERT_NEAR(lambda[1].real(), 2.86, eps);
  ASSERT_NEAR(lambda[1].imag(), -10.76, eps);

  ASSERT_NEAR(lambda[2].real(), -0.69, eps);
  ASSERT_NEAR(lambda[2].imag(), 4.70, eps);

  ASSERT_NEAR(lambda[3].real(), -0.69, eps);
  ASSERT_NEAR(lambda[3].imag(), -4.70, eps);

  ASSERT_NEAR(lambda[4].real(), -10.46, eps);
  ASSERT_NEAR(lambda[4].imag(), 0.00, eps);
}

//=============================================================================
// Complex32
//=============================================================================

HOSTDEV
TEST_CASE(accessors_Complex32)
{
  using T = Complex32;
  auto const eps = 1e-6F;
  um2::Matrix<T> m(3, 4);
  ASSERT(m.rows() == 3);
  ASSERT(m.cols() == 4);
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      Complex32 const c(static_cast<float>(j * 3 + i), static_cast<float>(j * 3 + i + 1));
      m(i, j) = c;
    }
  }
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      Complex32 const c(static_cast<float>(j * 3 + i), static_cast<float>(j * 3 + i + 1));
      ASSERT_NEAR(m(i, j).real(), c.real(), eps);
      ASSERT_NEAR(m(i, j).imag(), c.imag(), eps);
    }
  }
}

HOSTDEV
TEST_CASE(mat_vec_Complex32)
{
  using T = Complex32;
  auto const eps = 1e-6F;
  Int constexpr n = 3;
  um2::Matrix<T> const identity = um2::Matrix<T>::identity(n);
  um2::Vector<T> v(n);
  for (Int i = 0; i < n; ++i) {
    Complex32 const c(static_cast<float>(i), static_cast<float>(i + 1));
    v[i] = c;
  }
  auto mv = identity * v;
  ASSERT(mv.size() == n);
  for (Int i = 0; i < n; ++i) {
    Complex32 const c(static_cast<float>(i), static_cast<float>(i + 1));
    ASSERT_NEAR(mv[i].real(), c.real(), eps);
    ASSERT_NEAR(mv[i].imag(), c.imag(), eps);
  }
}

HOSTDEV
TEST_CASE(add_sub_Complex32)
{
  using T = Complex32;
  auto const eps = 1e-6F;
  um2::Matrix<T> a(3, 4);
  um2::Matrix<T> b(3, 4);
  um2::Matrix<T> c(3, 4);
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      a(i, j) = T(static_cast<float>(j * 3 + i), static_cast<float>(j * 3 + i));
      b(i, j) = T(static_cast<float>(j * 3 + i + 1), static_cast<float>(j * 3 + i + 1));
      c(i, j) = T(static_cast<float>(2 * j * 3 + 2 * i + 1),
                  static_cast<float>(2 * j * 3 + 2 * i + 1));
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
      ASSERT_NEAR(e(i, j).real(), static_cast<float>(1), eps);
      ASSERT_NEAR(e(i, j).imag(), static_cast<float>(1), eps);
    }
  }
}

TEST_CASE(mat_mul_Complex32)
{
  using T = Complex32;
  auto const eps = 1e-6F;
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
  a(0) = T(1, 0);
  a(1) = T(2, 0);
  a(2) = T(0, 0);
  a(3) = T(1, 0);

  a(4) = T(0, 0);
  a(5) = T(1, 0);
  a(6) = T(1, 0);
  a(7) = T(1, 0);

  a(8) = T(1, 0);
  a(9) = T(1, 0);
  a(10) = T(1, 0);
  a(11) = T(2, 0);
  um2::Matrix<T> b(3, 3);
  b(0) = T(1, 0);
  b(1) = T(2, 0);
  b(2) = T(4, 0);

  b(3) = T(2, 0);
  b(4) = T(3, 0);
  b(5) = T(2, 0);

  b(6) = T(1, 0);
  b(7) = T(1, 0);
  b(8) = T(2, 0);

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

TEST_CASE(lin_solve_Complex32)
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

  auto constexpr eps = 1e-2F;
  um2::Matrix<Complex32> a(5, 5);
  a(0) = 6.80F;
  a(1) = -2.11F;
  a(2) = 5.66F;
  a(3) = 5.97F;
  a(4) = 8.23F;

  a(5) = -6.05F;
  a(6) = -3.30F;
  a(7) = 5.36F;
  a(8) = -4.44F;
  a(9) = 1.08F;

  a(10) = -0.45F;
  a(11) = 2.58F;
  a(12) = -2.70F;
  a(13) = 0.27F;
  a(14) = 9.04F;

  a(15) = 8.32F;
  a(16) = 2.71F;
  a(17) = 4.35F;
  a(18) = -7.17F;
  a(19) = 2.14F;

  a(20) = -9.67F;
  a(21) = -5.14F;
  a(22) = -7.26F;
  a(23) = 6.08F;
  a(24) = -6.87F;

  um2::Matrix<Complex32> b(5, 3);
  b(0) = 4.02F;
  b(1) = 6.19F;
  b(2) = -8.22F;
  b(3) = -7.57F;
  b(4) = -3.03F;

  b(5) = -1.56F;
  b(6) = 4.00F;
  b(7) = -8.67F;
  b(8) = 1.75F;
  b(9) = 2.86F;

  b(10) = 9.81F;
  b(11) = -4.09F;
  b(12) = -4.57F;
  b(13) = -8.61F;
  b(14) = 8.99F;

  // X = -0.80  -0.39   0.96
  //     -0.70  -0.55   0.22
  //      0.59   0.84   1.90
  //      1.32  -0.10   5.36
  //      0.57   0.11   4.04

  auto const x = linearSolve(a, b);
  ASSERT(x.rows() == 5);
  ASSERT(x.cols() == 3);

  ASSERT_NEAR(x(0, 0).real(), -0.80F, eps);
  ASSERT_NEAR(x(1, 0).real(), -0.70F, eps);
  ASSERT_NEAR(x(2, 0).real(), 0.59F, eps);
  ASSERT_NEAR(x(3, 0).real(), 1.32F, eps);
  ASSERT_NEAR(x(4, 0).real(), 0.57F, eps);

  ASSERT_NEAR(x(0, 1).real(), -0.39F, eps);
  ASSERT_NEAR(x(1, 1).real(), -0.55F, eps);
  ASSERT_NEAR(x(2, 1).real(), 0.84F, eps);
  ASSERT_NEAR(x(3, 1).real(), -0.10F, eps);
  ASSERT_NEAR(x(4, 1).real(), 0.11F, eps);

  ASSERT_NEAR(x(0, 2).real(), 0.96F, eps);
  ASSERT_NEAR(x(1, 2).real(), 0.22F, eps);
  ASSERT_NEAR(x(2, 2).real(), 1.90F, eps);
  ASSERT_NEAR(x(3, 2).real(), 5.36F, eps);
  ASSERT_NEAR(x(4, 2).real(), 4.04F, eps);

  um2::Vector<Int> ipiv(5);
  linearSolve(a, b, ipiv);

  ASSERT(b.rows() == 5);
  ASSERT(b.cols() == 3);

  ASSERT_NEAR(b(0, 0).real(), -0.80F, eps);
  ASSERT_NEAR(b(1, 0).real(), -0.70F, eps);
  ASSERT_NEAR(b(2, 0).real(), 0.59F, eps);
  ASSERT_NEAR(b(3, 0).real(), 1.32F, eps);
  ASSERT_NEAR(b(4, 0).real(), 0.57F, eps);

  ASSERT_NEAR(b(0, 1).real(), -0.39F, eps);
  ASSERT_NEAR(b(1, 1).real(), -0.55F, eps);
  ASSERT_NEAR(b(2, 1).real(), 0.84F, eps);
  ASSERT_NEAR(b(3, 1).real(), -0.10F, eps);
  ASSERT_NEAR(b(4, 1).real(), 0.11F, eps);

  ASSERT_NEAR(b(0, 2).real(), 0.96F, eps);
  ASSERT_NEAR(b(1, 2).real(), 0.22F, eps);
  ASSERT_NEAR(b(2, 2).real(), 1.90F, eps);
  ASSERT_NEAR(b(3, 2).real(), 5.36F, eps);
  ASSERT_NEAR(b(4, 2).real(), 4.04F, eps);
}

TEST_CASE(eigvals_Complex32)
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

  using T = Complex32;
  auto constexpr eps = 1e-2F;
  um2::Matrix<T> a(5, 5);
  a(0) = -1.01F;
  a(1) = 3.98F;
  a(2) = 3.30F;
  a(3) = 4.43F;
  a(4) = 7.31F;

  a(5) = 0.86F;
  a(6) = 0.53F;
  a(7) = 8.26F;
  a(8) = 4.96F;
  a(9) = -6.43F;

  a(10) = -4.60F;
  a(11) = -7.04F;
  a(12) = -3.89F;
  a(13) = -7.66F;
  a(14) = -6.16F;

  a(15) = 3.31F;
  a(16) = 5.29F;
  a(17) = 8.20F;
  a(18) = -7.33F;
  a(19) = 2.47F;

  a(20) = -4.81F;
  a(21) = 3.55F;
  a(22) = -1.51F;
  a(23) = 6.18F;
  a(24) = 5.58F;

  auto const lambda = eigvals(a);

  ASSERT(lambda.size() == 5);
  ASSERT_NEAR(lambda[0].real(), 2.86F, eps);
  ASSERT_NEAR(lambda[0].imag(), 10.76F, eps);

  ASSERT_NEAR(lambda[1].real(), 2.86F, eps);
  ASSERT_NEAR(lambda[1].imag(), -10.76F, eps);

  ASSERT_NEAR(lambda[2].real(), -0.69F, eps);
  ASSERT_NEAR(lambda[2].imag(), 4.70F, eps);

  ASSERT_NEAR(lambda[3].real(), -0.69F, eps);
  ASSERT_NEAR(lambda[3].imag(), -4.70F, eps);

  ASSERT_NEAR(lambda[4].real(), -10.46F, eps);
  ASSERT_NEAR(lambda[4].imag(), 0.00F, eps);
}

TEST_CASE(transpose_Complex32)
{

  // A = (1, 1)  (2, -2)
  //     (3, 3)  (4, -4)
  um2::Matrix<Complex32> a(2, 2);
  a(0) = Complex32(1, 1);
  a(1) = Complex32(3, 3);
  a(2) = Complex32(2, -2);
  a(3) = Complex32(4, -4);

  auto constexpr eps = 1e-6F;
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

//=============================================================================
// Complex64
//=============================================================================

HOSTDEV
TEST_CASE(accessors_Complex64)
{
  using T = Complex64;
  auto const eps = 1e-6;
  um2::Matrix<T> m(3, 4);
  ASSERT(m.rows() == 3);
  ASSERT(m.cols() == 4);
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      Complex64 const c(static_cast<double>(j * 3 + i),
                        static_cast<double>(j * 3 + i + 1));
      m(i, j) = c;
    }
  }
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      Complex64 const c(static_cast<double>(j * 3 + i),
                        static_cast<double>(j * 3 + i + 1));
      ASSERT_NEAR(m(i, j).real(), c.real(), eps);
      ASSERT_NEAR(m(i, j).imag(), c.imag(), eps);
    }
  }
}

HOSTDEV
TEST_CASE(mat_vec_Complex64)
{
  using T = Complex64;
  auto const eps = 1e-6;
  Int constexpr n = 3;
  um2::Matrix<T> const identity = um2::Matrix<T>::identity(n);
  um2::Vector<T> v(n);
  for (Int i = 0; i < n; ++i) {
    Complex64 const c(static_cast<double>(i), static_cast<double>(i + 1));
    v[i] = c;
  }
  auto mv = identity * v;
  ASSERT(mv.size() == n);
  for (Int i = 0; i < n; ++i) {
    Complex64 const c(static_cast<double>(i), static_cast<double>(i + 1));
    ASSERT_NEAR(mv[i].real(), c.real(), eps);
    ASSERT_NEAR(mv[i].imag(), c.imag(), eps);
  }
}

HOSTDEV
TEST_CASE(add_sub_Complex64)
{
  using T = Complex64;
  auto const eps = 1e-6;
  um2::Matrix<T> a(3, 4);
  um2::Matrix<T> b(3, 4);
  um2::Matrix<T> c(3, 4);
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      a(i, j) = T(static_cast<double>(j * 3 + i), static_cast<double>(j * 3 + i));
      b(i, j) = T(static_cast<double>(j * 3 + i + 1), static_cast<double>(j * 3 + i + 1));
      c(i, j) = T(static_cast<double>(2 * j * 3 + 2 * i + 1),
                  static_cast<double>(2 * j * 3 + 2 * i + 1));
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
      ASSERT_NEAR(e(i, j).real(), static_cast<double>(1), eps);
      ASSERT_NEAR(e(i, j).imag(), static_cast<double>(1), eps);
    }
  }
}

TEST_CASE(mat_mul_Complex64)
{
  using T = Complex64;
  auto const eps = 1e-6;
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
  a(0) = T(1, 0);
  a(1) = T(2, 0);
  a(2) = T(0, 0);
  a(3) = T(1, 0);

  a(4) = T(0, 0);
  a(5) = T(1, 0);
  a(6) = T(1, 0);
  a(7) = T(1, 0);

  a(8) = T(1, 0);
  a(9) = T(1, 0);
  a(10) = T(1, 0);
  a(11) = T(2, 0);
  um2::Matrix<T> b(3, 3);
  b(0) = T(1, 0);
  b(1) = T(2, 0);
  b(2) = T(4, 0);

  b(3) = T(2, 0);
  b(4) = T(3, 0);
  b(5) = T(2, 0);

  b(6) = T(1, 0);
  b(7) = T(1, 0);
  b(8) = T(2, 0);

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

TEST_CASE(lin_solve_Complex64)
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

  auto constexpr eps = 1e-2;
  um2::Matrix<Complex64> a(5, 5);
  a(0) = 6.80;
  a(1) = -2.11;
  a(2) = 5.66;
  a(3) = 5.97;
  a(4) = 8.23;

  a(5) = -6.05;
  a(6) = -3.30;
  a(7) = 5.36;
  a(8) = -4.44;
  a(9) = 1.08;

  a(10) = -0.45;
  a(11) = 2.58;
  a(12) = -2.70;
  a(13) = 0.27;
  a(14) = 9.04;

  a(15) = 8.32;
  a(16) = 2.71;
  a(17) = 4.35;
  a(18) = -7.17;
  a(19) = 2.14;

  a(20) = -9.67;
  a(21) = -5.14;
  a(22) = -7.26;
  a(23) = 6.08;
  a(24) = -6.87;

  um2::Matrix<Complex64> b(5, 3);
  b(0) = 4.02;
  b(1) = 6.19;
  b(2) = -8.22;
  b(3) = -7.57;
  b(4) = -3.03;

  b(5) = -1.56;
  b(6) = 4.00;
  b(7) = -8.67;
  b(8) = 1.75;
  b(9) = 2.86;

  b(10) = 9.81;
  b(11) = -4.09;
  b(12) = -4.57;
  b(13) = -8.61;
  b(14) = 8.99;

  // X = -0.80  -0.39   0.96
  //     -0.70  -0.55   0.22
  //      0.59   0.84   1.90
  //      1.32  -0.10   5.36
  //      0.57   0.11   4.04

  auto const x = linearSolve(a, b);
  ASSERT(x.rows() == 5);
  ASSERT(x.cols() == 3);

  ASSERT_NEAR(x(0, 0).real(), -0.80, eps);
  ASSERT_NEAR(x(1, 0).real(), -0.70, eps);
  ASSERT_NEAR(x(2, 0).real(), 0.59, eps);
  ASSERT_NEAR(x(3, 0).real(), 1.32, eps);
  ASSERT_NEAR(x(4, 0).real(), 0.57, eps);

  ASSERT_NEAR(x(0, 1).real(), -0.39, eps);
  ASSERT_NEAR(x(1, 1).real(), -0.55, eps);
  ASSERT_NEAR(x(2, 1).real(), 0.84, eps);
  ASSERT_NEAR(x(3, 1).real(), -0.10, eps);
  ASSERT_NEAR(x(4, 1).real(), 0.11, eps);

  ASSERT_NEAR(x(0, 2).real(), 0.96, eps);
  ASSERT_NEAR(x(1, 2).real(), 0.22, eps);
  ASSERT_NEAR(x(2, 2).real(), 1.90, eps);
  ASSERT_NEAR(x(3, 2).real(), 5.36, eps);
  ASSERT_NEAR(x(4, 2).real(), 4.04, eps);

  um2::Vector<Int> ipiv(5);
  linearSolve(a, b, ipiv);
  ASSERT(b.rows() == 5);
  ASSERT(b.cols() == 3);

  ASSERT_NEAR(b(0, 0).real(), -0.80, eps);
  ASSERT_NEAR(b(1, 0).real(), -0.70, eps);
  ASSERT_NEAR(b(2, 0).real(), 0.59, eps);
  ASSERT_NEAR(b(3, 0).real(), 1.32, eps);
  ASSERT_NEAR(b(4, 0).real(), 0.57, eps);

  ASSERT_NEAR(b(0, 1).real(), -0.39, eps);
  ASSERT_NEAR(b(1, 1).real(), -0.55, eps);
  ASSERT_NEAR(b(2, 1).real(), 0.84, eps);
  ASSERT_NEAR(b(3, 1).real(), -0.10, eps);
  ASSERT_NEAR(b(4, 1).real(), 0.11, eps);

  ASSERT_NEAR(b(0, 2).real(), 0.96, eps);
  ASSERT_NEAR(b(1, 2).real(), 0.22, eps);
  ASSERT_NEAR(b(2, 2).real(), 1.90, eps);
  ASSERT_NEAR(b(3, 2).real(), 5.36, eps);
  ASSERT_NEAR(b(4, 2).real(), 4.04, eps);
}

TEST_CASE(eigvals_Complex64)
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

  using T = Complex64;
  auto constexpr eps = 1e-2;
  um2::Matrix<T> a(5, 5);
  a(0) = -1.01;
  a(1) = 3.98;
  a(2) = 3.30;
  a(3) = 4.43;
  a(4) = 7.31;

  a(5) = 0.86;
  a(6) = 0.53;
  a(7) = 8.26;
  a(8) = 4.96;
  a(9) = -6.43;

  a(10) = -4.60;
  a(11) = -7.04;
  a(12) = -3.89;
  a(13) = -7.66;
  a(14) = -6.16;

  a(15) = 3.31;
  a(16) = 5.29;
  a(17) = 8.20;
  a(18) = -7.33;
  a(19) = 2.47;

  a(20) = -4.81;
  a(21) = 3.55;
  a(22) = -1.51;
  a(23) = 6.18;
  a(24) = 5.58;

  auto const lambda = eigvals(a);

  ASSERT(lambda.size() == 5);
  ASSERT_NEAR(lambda[0].real(), 2.86, eps);
  ASSERT_NEAR(lambda[0].imag(), 10.76, eps);

  ASSERT_NEAR(lambda[1].real(), 2.86, eps);
  ASSERT_NEAR(lambda[1].imag(), -10.76, eps);

  ASSERT_NEAR(lambda[2].real(), -0.69, eps);
  ASSERT_NEAR(lambda[2].imag(), 4.70, eps);

  ASSERT_NEAR(lambda[3].real(), -0.69, eps);
  ASSERT_NEAR(lambda[3].imag(), -4.70, eps);

  ASSERT_NEAR(lambda[4].real(), -10.46, eps);
  ASSERT_NEAR(lambda[4].imag(), 0.00, eps);
}

TEST_SUITE(Matrix_float)
{
  TEST_HOSTDEV(accessors_float);
  TEST_HOSTDEV(mat_vec_float);
  TEST_HOSTDEV(add_sub_float);
  TEST_HOSTDEV(mat_mul_float);
  TEST_HOSTDEV(lin_solve_float);
  TEST_HOSTDEV(eigvals_float);
  TEST_HOSTDEV(transpose_float);
}

TEST_SUITE(Matrix_double)
{
  TEST_HOSTDEV(accessors_double);
  TEST_HOSTDEV(mat_vec_double);
  TEST_HOSTDEV(add_sub_double);
  TEST_HOSTDEV(mat_mul_double);
  TEST_HOSTDEV(lin_solve_double);
  TEST_HOSTDEV(eigvals_double);
}

TEST_SUITE(Matrix_Complex32)
{
  TEST_HOSTDEV(accessors_Complex32);
  TEST_HOSTDEV(mat_vec_Complex32);
  TEST_HOSTDEV(add_sub_Complex32);
  TEST_HOSTDEV(mat_mul_Complex32);
  TEST_HOSTDEV(lin_solve_Complex32);
  TEST_HOSTDEV(eigvals_Complex32);
  TEST_HOSTDEV(transpose_Complex32);
}

TEST_SUITE(Matrix_Complex64)
{
  TEST_HOSTDEV(accessors_Complex64);
  TEST_HOSTDEV(mat_vec_Complex64);
  TEST_HOSTDEV(add_sub_Complex64);
  TEST_HOSTDEV(mat_mul_Complex64);
  TEST_HOSTDEV(lin_solve_Complex64);
  TEST_HOSTDEV(eigvals_Complex64);
}

auto
main() -> int
{
  RUN_SUITE(Matrix_float);
  RUN_SUITE(Matrix_double);
  RUN_SUITE(Matrix_Complex32);
  RUN_SUITE(Matrix_Complex64);
  return 0;
}
