#include <um2/math/matrix.hpp>

#include <complex>

#include "cblas.h"       
#include "lapacke.h"

namespace um2 {

//==============================================================================
// Matrix-Vector Multiplication
//==============================================================================

template<>
auto
operator*(Matrix<float> const & a, Vector<float> const & x) -> Vector<float>
{
  ASSERT(a.cols() == x.size());
  Vector<float> y(a.rows());

  // Use BLAS's sgemv function to perform the matrix-vector multiplication 
  // y <-- alpha * A * x + beta * y
  cblas_sgemv(
      CblasColMajor, // Matrix is stored in column-major order
      CblasNoTrans,  // Do not transpose the matrix
      a.rows(),      // Number of rows
      a.cols(),      // Number of columns
      1.0F,          // alpha
      a.data(),      // Matrix data
      a.rows(),      // Leading dimension of matrix
      x.data(),      // Vector data
      1,             // Increment for x
      0.0F,          // beta
      y.data(),      // Output vector
      1);            // Increment for y
  return y;
}

template<>
auto
operator*(Matrix<double> const & a, Vector<double> const & x) -> Vector<double>
{
  ASSERT(a.cols() == x.size());
  Vector<double> y(a.rows());

  // Use BLAS's dgemv function to perform the matrix-vector multiplication 
  // y <-- alpha * A * x + beta * y
  cblas_dgemv(
      CblasColMajor, // Matrix is stored in column-major order
      CblasNoTrans,  // Do not transpose the matrix
      a.rows(),      // Number of rows
      a.cols(),      // Number of columns
      1.0,           // alpha
      a.data(),      // Matrix data
      a.rows(),      // Leading dimension of matrix
      x.data(),      // Vector data
      1,             // Increment for x
      0.0,           // beta
      y.data(),      // Output vector
      1);            // Increment for y
  return y;
}

// complex<float> specialization
template<>
auto
operator*(Matrix<std::complex<float>> const & a, Vector<std::complex<float>> const & x) -> Vector<std::complex<float>>
{
  using Complexf = std::complex<float>;
  ASSERT(a.cols() == x.size());
  Vector<Complexf> y(a.rows());

  Complexf const alpha(1.0F);
  Complexf const beta(0.0F);

  // Use BLAS's cgemv function to perform the matrix-vector multiplication 
  // y <-- alpha * A * x + beta * y
  cblas_cgemv(
      CblasColMajor, // Matrix is stored in column-major order
      CblasNoTrans,  // Do not transpose the matrix
      a.rows(),      // Number of rows
      a.cols(),      // Number of columns
      &alpha,        // alpha
      a.data(),      // Matrix data
      a.rows(),      // Leading dimension of matrix
      x.data(),      // Vector data
      1,             // Increment for x
      &beta,         // beta
      y.data(),      // Output vector
      1);            // Increment for y
  return y;
}

template<>
auto
operator*(Matrix<std::complex<double>> const & a, Vector<std::complex<double>> const & x) -> Vector<std::complex<double>>
{
  using Complexd = std::complex<double>;
  ASSERT(a.cols() == x.size());
  Vector<Complexd> y(a.rows());

  Complexd const alpha(1.0);
  Complexd const beta(0.0);

  // Use BLAS's zgemv function to perform the matrix-vector multiplication 
  // y <-- alpha * A * x + beta * y
  cblas_zgemv(
      CblasColMajor, // Matrix is stored in column-major order
      CblasNoTrans,  // Do not transpose the matrix
      a.rows(),      // Number of rows
      a.cols(),      // Number of columns
      &alpha,        // alpha
      a.data(),      // Matrix data
      a.rows(),      // Leading dimension of matrix
      x.data(),      // Vector data
      1,             // Increment for x
      &beta,         // beta
      y.data(),      // Output vector
      1);            // Increment for y
  return y;
}

} // namespace um2
