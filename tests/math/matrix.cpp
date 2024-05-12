#include <um2/math/matrix.hpp>

#include "../test_macros.hpp"

#include <complex>

using Complexf = std::complex<float>;
using Complexd = std::complex<double>;

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
TEST_CASE(accessors_Complexf)
{
  using T = Complexf;
  auto const eps = 1e-6F;
  um2::Matrix<T> m(3, 4);
  ASSERT(m.rows() == 3);
  ASSERT(m.cols() == 4);
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      Complexf const c(static_cast<float>(j * 3 + i), static_cast<float>(j * 3 + i + 1));
      m(i, j) = c;
    }
  }
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      Complexf const c(static_cast<float>(j * 3 + i), static_cast<float>(j * 3 + i + 1));
      ASSERT_NEAR(m(i, j).real(), c.real(), eps);
      ASSERT_NEAR(m(i, j).imag(), c.imag(), eps);
    }
  }
}

HOSTDEV
TEST_CASE(mat_vec_Complexf)
{
  using T = Complexf;
  auto const eps = 1e-6F;
  Int constexpr n = 3;
  um2::Matrix<T> const identity = um2::Matrix<T>::identity(n);
  um2::Vector<T> v(n);
  for (Int i = 0; i < n; ++i) {
    Complexf const c(static_cast<float>(i), static_cast<float>(i + 1));
    v[i] = c;
  }
  auto mv = identity * v;
  ASSERT(mv.size() == n);
  for (Int i = 0; i < n; ++i) {
    Complexf const c(static_cast<float>(i), static_cast<float>(i + 1));
    ASSERT_NEAR(mv[i].real(), c.real(), eps); 
    ASSERT_NEAR(mv[i].imag(), c.imag(), eps); 
  }
}

HOSTDEV
TEST_CASE(accessors_Complexd)
{
  using T = Complexd;
  auto const eps = 1e-6;
  um2::Matrix<T> m(3, 4);
  ASSERT(m.rows() == 3);
  ASSERT(m.cols() == 4);
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      Complexd const c(static_cast<double>(j * 3 + i), static_cast<double>(j * 3 + i + 1));
      m(i, j) = c;
    }
  }
  for (Int j = 0; j < 4; ++j) {
    for (Int i = 0; i < 3; ++i) {
      Complexd const c(static_cast<double>(j * 3 + i), static_cast<double>(j * 3 + i + 1));
      ASSERT_NEAR(m(i, j).real(), c.real(), eps);
      ASSERT_NEAR(m(i, j).imag(), c.imag(), eps);
    }
  }
}

HOSTDEV
TEST_CASE(mat_vec_Complexd)
{
  using T = Complexd;
  auto const eps = 1e-6;
  Int constexpr n = 3;
  um2::Matrix<T> const identity = um2::Matrix<T>::identity(n);
  um2::Vector<T> v(n);
  for (Int i = 0; i < n; ++i) {
    Complexd const c(static_cast<double>(i), static_cast<double>(i + 1));
    v[i] = c;
  }
  auto mv = identity * v;
  ASSERT(mv.size() == n);
  for (Int i = 0; i < n; ++i) {
    Complexd const c(static_cast<double>(i), static_cast<double>(i + 1));
    ASSERT_NEAR(mv[i].real(), c.real(), eps); 
    ASSERT_NEAR(mv[i].imag(), c.imag(), eps); 
  }
}

TEST_SUITE(Matrix_float)
{
  TEST_HOSTDEV(accessors_float);
  TEST_HOSTDEV(mat_vec_float);
}

TEST_SUITE(Matrix_double)
{
  TEST_HOSTDEV(accessors_float);
  TEST_HOSTDEV(mat_vec_float);
}

TEST_SUITE(Matrix_Complexf)
{
  TEST_HOSTDEV(accessors_Complexf);
  TEST_HOSTDEV(mat_vec_Complexf);
}

TEST_SUITE(Matrix_Complexd)
{
  TEST_HOSTDEV(accessors_Complexf);
  TEST_HOSTDEV(mat_vec_Complexf);
}

auto
main() -> int
{
  RUN_SUITE(Matrix_float);
  RUN_SUITE(Matrix_double);
  RUN_SUITE(Matrix_Complexf);
  RUN_SUITE(Matrix_Complexd);
  return 0;
}
