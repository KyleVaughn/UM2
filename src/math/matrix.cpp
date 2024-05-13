#include <um2/math/matrix.hpp>

#include <complex>

#include "cblas.h"       
#include "lapacke.h"

namespace um2 {

//==============================================================================
// Matrix-Vector Multiplication
//==============================================================================

template<>
PURE auto
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
PURE auto
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
PURE auto
operator*(Matrix<std::complex<float>> const & a, Vector<std::complex<float>> const & x) -> Vector<std::complex<float>>
{
  using Complex32 = std::complex<float>;
  ASSERT(a.cols() == x.size());
  Vector<Complex32> y(a.rows());

  Complex32 const alpha(1.0F);
  Complex32 const beta(0.0F);

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
PURE auto
operator*(Matrix<std::complex<double>> const & a, Vector<std::complex<double>> const & x) -> Vector<std::complex<double>>
{
  using Complex64 = std::complex<double>;
  ASSERT(a.cols() == x.size());
  Vector<Complex64> y(a.rows());

  Complex64 const alpha(1.0);
  Complex64 const beta(0.0);

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

//==============================================================================
// Matrix-Matrix Multiplication
//==============================================================================

template<>
PURE auto
operator*(Matrix<float> const & a, Matrix<float> const & b) -> Matrix<float>
{
  ASSERT(a.cols() == b.rows());
  Matrix<float> c(a.rows(), b.cols());

  // Use BLAS's sgemm function to perform the matrix-matrix multiplication 
  // C = alpha * A * B + beta * C
  cblas_sgemm(
      CblasColMajor, // Matrix is stored in column-major order
      CblasNoTrans,  // Do not transpose the matrix A
      CblasNoTrans,  // Do not transpose the matrix B
      a.rows(),      // Number of rows in A
      b.cols(),      // Number of columns in B
      a.cols(),      // Number of columns in A
      1.0F,          // alpha
      a.data(),      // Matrix A data
      a.rows(),      // Leading dimension of A
      b.data(),      // Matrix B data
      b.rows(),      // Leading dimension of B
      0.0F,          // beta
      c.data(),      // Output matrix
      c.rows());     // Leading dimension of C
  return c;
}

template<>
PURE auto
operator*(Matrix<double> const & a, Matrix<double> const & b) -> Matrix<double>
{
  ASSERT(a.cols() == b.rows());
  Matrix<double> c(a.rows(), b.cols());

  // Use BLAS's dgemm function to perform the matrix-matrix multiplication 
  // C = alpha * A * B + beta * C
  cblas_dgemm(
      CblasColMajor, // Matrix is stored in column-major order
      CblasNoTrans,  // Do not transpose the matrix A
      CblasNoTrans,  // Do not transpose the matrix B
      a.rows(),      // Number of rows in A
      b.cols(),      // Number of columns in B
      a.cols(),      // Number of columns in A
      1.0,           // alpha
      a.data(),      // Matrix A data
      a.rows(),      // Leading dimension of A
      b.data(),      // Matrix B data
      b.rows(),      // Leading dimension of B
      0.0,           // beta
      c.data(),      // Output matrix
      c.rows());     // Leading dimension of C
  return c;
}

template<>
PURE auto
operator*(Matrix<std::complex<float>> const & a, Matrix<std::complex<float>> const & b) -> Matrix<std::complex<float>>
{
  using Complex32 = std::complex<float>;
  ASSERT(a.cols() == b.rows());
  Matrix<Complex32> c(a.rows(), b.cols());

  Complex32 const alpha(1.0F);
  Complex32 const beta(0.0F);

  // Use BLAS's cgemm function to perform the matrix-matrix multiplication 
  // C = alpha * A * B + beta * C
  cblas_cgemm(
      CblasColMajor, // Matrix is stored in column-major order
      CblasNoTrans,  // Do not transpose the matrix A
      CblasNoTrans,  // Do not transpose the matrix B
      a.rows(),      // Number of rows in A
      b.cols(),      // Number of columns in B
      a.cols(),      // Number of columns in A
      &alpha,        // alpha
      a.data(),      // Matrix A data
      a.rows(),      // Leading dimension of A
      b.data(),      // Matrix B data
      b.rows(),      // Leading dimension of B
      &beta,         // beta
      c.data(),      // Output matrix
      c.rows());     // Leading dimension of C
  return c;
}

template<>
PURE auto
operator*(Matrix<std::complex<double>> const & a, Matrix<std::complex<double>> const & b) -> Matrix<std::complex<double>>
{
  using Complex64 = std::complex<double>;
  ASSERT(a.cols() == b.rows());
  Matrix<Complex64> c(a.rows(), b.cols());

  Complex64 const alpha(1.0);
  Complex64 const beta(0.0);

  // Use BLAS's zgemm function to perform the matrix-matrix multiplication 
  // C = alpha * A * B + beta * C
  cblas_zgemm(
      CblasColMajor, // Matrix is stored in column-major order
      CblasNoTrans,  // Do not transpose the matrix A
      CblasNoTrans,  // Do not transpose the matrix B
      a.rows(),      // Number of rows in A
      b.cols(),      // Number of columns in B
      a.cols(),      // Number of columns in A
      &alpha,        // alpha
      a.data(),      // Matrix A data
      a.rows(),      // Leading dimension of A
      b.data(),      // Matrix B data
      b.rows(),      // Leading dimension of B
      &beta,         // beta
      c.data(),      // Output matrix
      c.rows());     // Leading dimension of C
  return c;
}

} // namespace um2
