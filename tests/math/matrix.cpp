#include <um2/math/matrix.hpp>

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

  auto const ab = a * b;
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

  auto const ab = a * b;
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
      c(i, j) = T(static_cast<float>(2 * j * 3 + 2 * i + 1), static_cast<float>(2 * j * 3 + 2 * i + 1)); 
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

  auto const ab = a * b;
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
      Complex64 const c(static_cast<double>(j * 3 + i), static_cast<double>(j * 3 + i + 1));
      m(i, j) = c;
    }
  }
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      Complex64 const c(static_cast<double>(j * 3 + i), static_cast<double>(j * 3 + i + 1));
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
      c(i, j) = T(static_cast<double>(2 * j * 3 + 2 * i + 1), static_cast<double>(2 * j * 3 + 2 * i + 1)); 
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

  auto const ab = a * b;
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

TEST_SUITE(Matrix_float)
{
  TEST_HOSTDEV(accessors_float);
  TEST_HOSTDEV(mat_vec_float);
  TEST_HOSTDEV(add_sub_float);
  TEST_HOSTDEV(mat_mul_float);
}

TEST_SUITE(Matrix_double)
{
  TEST_HOSTDEV(accessors_double);
  TEST_HOSTDEV(mat_vec_double);
  TEST_HOSTDEV(add_sub_double);
  TEST_HOSTDEV(mat_mul_double);
}

TEST_SUITE(Matrix_Complex32)
{
  TEST_HOSTDEV(accessors_Complex32);
  TEST_HOSTDEV(mat_vec_Complex32);
  TEST_HOSTDEV(add_sub_Complex32);
  TEST_HOSTDEV(mat_mul_Complex32);
}

TEST_SUITE(Matrix_Complex64)
{
  TEST_HOSTDEV(accessors_Complex64);
  TEST_HOSTDEV(mat_vec_Complex64);
  TEST_HOSTDEV(add_sub_Complex64);
  TEST_HOSTDEV(mat_mul_Complex64);
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
