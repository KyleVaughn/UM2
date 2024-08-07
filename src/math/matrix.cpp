#include <um2/config.hpp>

#if UM2_USE_BLAS_LAPACK
#  include <um2/math/matrix.hpp>
#  include <um2/stdlib/assert.hpp>
#  include <um2/stdlib/vector.hpp>

#  include "cblas.h"
#  include "lapack.h"
#  include "lapacke.h"

#  include <cstdint>

static_assert(sizeof(Int) == sizeof(lapack_int),
              "LAPACK integer type must be the same size as um2::Int");

namespace um2
{

//==============================================================================
// Matrix-Vector Multiplication
//==============================================================================

template <>
PURE auto
operator*(Matrix<float> const & a, Vector<float> const & x) -> Vector<float>
{
  ASSERT(a.cols() == x.size());
  Vector<float> y(a.rows());

  // Use BLAS's sgemv function to perform the matrix-vector multiplication
  // y <-- alpha * A * x + beta * y
  cblas_sgemv(CblasColMajor, // Matrix is stored in column-major order
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

template <>
PURE auto
operator*(Matrix<double> const & a, Vector<double> const & x) -> Vector<double>
{
  ASSERT(a.cols() == x.size());
  Vector<double> y(a.rows());

  // Use BLAS's dgemv function to perform the matrix-vector multiplication
  // y <-- alpha * A * x + beta * y
  cblas_dgemv(CblasColMajor, // Matrix is stored in column-major order
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

// Complex<float> specialization
template <>
PURE auto
operator*(Matrix<Complex<float>> const & a,
          Vector<Complex<float>> const & x) -> Vector<Complex<float>>
{
  using Complex32 = Complex<float>;
  ASSERT(a.cols() == x.size());
  Vector<Complex32> y(a.rows());

  Complex32 const alpha(1.0F);
  Complex32 const beta(0.0F);

  // Use BLAS's cgemv function to perform the matrix-vector multiplication
  // y <-- alpha * A * x + beta * y
  cblas_cgemv(CblasColMajor, // Matrix is stored in column-major order
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

template <>
PURE auto
operator*(Matrix<Complex<double>> const & a,
          Vector<Complex<double>> const & x) -> Vector<Complex<double>>
{
  using Complex64 = Complex<double>;
  ASSERT(a.cols() == x.size());
  Vector<Complex64> y(a.rows());

  Complex64 const alpha(1.0);
  Complex64 const beta(0.0);

  // Use BLAS's zgemv function to perform the matrix-vector multiplication
  // y <-- alpha * A * x + beta * y
  cblas_zgemv(CblasColMajor, // Matrix is stored in column-major order
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

template <>
PURE auto
operator*(Matrix<float> const & a, Matrix<float> const & b) -> Matrix<float>
{
  ASSERT(a.cols() == b.rows());
  Matrix<float> c(a.rows(), b.cols());

  // Use BLAS's sgemm function to perform the matrix-matrix multiplication
  // C = alpha * A * B + beta * C
  cblas_sgemm(CblasColMajor, // Matrix is stored in column-major order
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

template <>
PURE auto
operator*(Matrix<double> const & a, Matrix<double> const & b) -> Matrix<double>
{
  ASSERT(a.cols() == b.rows());
  Matrix<double> c(a.rows(), b.cols());

  // Use BLAS's dgemm function to perform the matrix-matrix multiplication
  // C = alpha * A * B + beta * C
  cblas_dgemm(CblasColMajor, // Matrix is stored in column-major order
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

template <>
PURE auto
operator*(Matrix<Complex<float>> const & a,
          Matrix<Complex<float>> const & b) -> Matrix<Complex<float>>
{
  using Complex32 = Complex<float>;
  ASSERT(a.cols() == b.rows());
  Matrix<Complex32> c(a.rows(), b.cols());

  Complex32 const alpha(1.0F);
  Complex32 const beta(0.0F);

  // Use BLAS's cgemm function to perform the matrix-matrix multiplication
  // C = alpha * A * B + beta * C
  cblas_cgemm(CblasColMajor, // Matrix is stored in column-major order
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

template <>
PURE auto
operator*(Matrix<Complex<double>> const & a,
          Matrix<Complex<double>> const & b) -> Matrix<Complex<double>>
{
  using Complex64 = Complex<double>;
  ASSERT(a.cols() == b.rows());
  Matrix<Complex64> c(a.rows(), b.cols());

  Complex64 const alpha(1.0);
  Complex64 const beta(0.0);

  // Use BLAS's zgemm function to perform the matrix-matrix multiplication
  // C = alpha * A * B + beta * C
  cblas_zgemm(CblasColMajor, // Matrix is stored in column-major order
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

template <>
void
matmul(Matrix<float> & c, Matrix<float> const & a, Matrix<float> const & b)
{
  ASSERT(a.cols() == b.rows());
  ASSERT(c.rows() == a.rows());
  ASSERT(c.cols() == b.cols());

  // Use BLAS's sgemm function to perform the matrix-matrix multiplication
  // C = alpha * A * B + beta * C
  cblas_sgemm(CblasColMajor, // Matrix is stored in column-major order
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
}

template <>
void
matmul(Matrix<double> & c, Matrix<double> const & a, Matrix<double> const & b)
{
  ASSERT(a.cols() == b.rows());
  ASSERT(c.rows() == a.rows());
  ASSERT(c.cols() == b.cols());

  // Use BLAS's dgemm function to perform the matrix-matrix multiplication
  // C = alpha * A * B + beta * C
  cblas_dgemm(CblasColMajor, // Matrix is stored in column-major order
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
}

template <>
void
matmul(Matrix<Complex<float>> & c, Matrix<Complex<float>> const & a,
       Matrix<Complex<float>> const & b)
{
  using Complex32 = Complex<float>;
  ASSERT(a.cols() == b.rows());
  ASSERT(c.rows() == a.rows());
  ASSERT(c.cols() == b.cols());

  Complex32 const alpha(1.0F);
  Complex32 const beta(0.0F);

  // Use BLAS's cgemm function to perform the matrix-matrix multiplication
  // C = alpha * A * B + beta * C
  cblas_cgemm(CblasColMajor, // Matrix is stored in column-major order
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
}

template <>
void
matmul(Matrix<Complex<double>> & c, Matrix<Complex<double>> const & a,
       Matrix<Complex<double>> const & b)
{
  using Complex64 = Complex<double>;
  ASSERT(a.cols() == b.rows());
  ASSERT(c.rows() == a.rows());
  ASSERT(c.cols() == b.cols());

  Complex64 const alpha(1.0);
  Complex64 const beta(0.0);

  // Use BLAS's zgemm function to perform the matrix-matrix multiplication
  // C = alpha * A * B + beta * C
  cblas_zgemm(CblasColMajor, // Matrix is stored in column-major order
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
}

//==============================================================================
// Solve Linear System
//==============================================================================

// Solve A * X = B for X. X = A \ B
template <>
PURE auto
linearSolve(Matrix<float> const & a, Matrix<float> const & b) -> Matrix<float>
{
  ASSERT(a.rows() == a.cols()); // A must be square
  ASSERT(a.rows() == b.rows());

  // Solve the linear system using LAPACK's sgesv function
  // It is important to note that sgesv overwrites:
  //  - the input matrix A with the LU decomposition of A
  //  - the input matrix B with the solution matrix X
  //
  // Therefore we allocate a copy of A and a copy of B for the function to overwrite
  Int const n = a.rows();    // Number of rows in A
  Int const nrhs = b.cols(); // Number of columns in B
  Matrix<float> a_copy(a);   // Copy of A, since sgesv overwrites A
  Int const lda = a.rows();  // Leading dimension of A
  Int * ipiv = new Int[static_cast<uint64_t>(n)];
  Matrix<float> b_copy(b);  // Copy of B, since sgesv overwrites B
  Int const ldb = b.rows(); // Leading dimension of B
#  if UM2_ENABLE_ASSERTS
  Int const info = LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs, a_copy.data(), lda, ipiv,
                                 b_copy.data(), ldb);
  ASSERT(info == 0);
#  else
  LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs, a_copy.data(), lda, ipiv, b_copy.data(), ldb);
#  endif
  delete[] ipiv;
  return b_copy;
}

template <>
PURE auto
linearSolve(Matrix<double> const & a, Matrix<double> const & b) -> Matrix<double>
{
  ASSERT(a.rows() == a.cols()); // A must be square
  ASSERT(a.rows() == b.rows());

  // Solve the linear system using LAPACK's sgesv function
  // It is important to note that sgesv overwrites:
  //  - the input matrix A with the LU decomposition of A
  //  - the input matrix B with the solution matrix X
  //
  // Therefore we allocate a copy of A and a copy of B for the function to overwrite
  Int const n = a.rows();    // Number of rows in A
  Int const nrhs = b.cols(); // Number of columns in B
  Matrix<double> a_copy(a);  // Copy of A, since sgesv overwrites A
  Int const lda = a.rows();  // Leading dimension of A
  Int * ipiv = new Int[static_cast<uint64_t>(n)];
  Matrix<double> b_copy(b); // Copy of B, since sgesv overwrites B
  Int const ldb = b.rows(); // Leading dimension of B
#  if UM2_ENABLE_ASSERTS
  Int const info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a_copy.data(), lda, ipiv,
                                 b_copy.data(), ldb);
  ASSERT(info == 0);
#  else
  LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a_copy.data(), lda, ipiv, b_copy.data(), ldb);
#  endif
  delete[] ipiv;
  return b_copy;
}

template <>
PURE auto
linearSolve(Matrix<Complex<float>> const & a,
            Matrix<Complex<float>> const & b) -> Matrix<Complex<float>>
{
  using Complex32 = Complex<float>;
  ASSERT(a.rows() == a.cols()); // A must be square
  ASSERT(a.rows() == b.rows());

  // Solve the linear system using LAPACK's cgesv function
  // It is important to note that cgesv overwrites:
  //  - the input matrix A with the LU decomposition of A
  //  - the input matrix B with the solution matrix X
  //
  // Therefore we allocate a copy of A and a copy of B for the function to overwrite
  Int const n = a.rows();      // Number of rows in A
  Int const nrhs = b.cols();   // Number of columns in B
  Matrix<Complex32> a_copy(a); // Copy of A, since cgesv overwrites A
  lapack_complex_float * a_data = reinterpret_cast<lapack_complex_float *>(a_copy.data());
  Int const lda = a.rows(); // Leading dimension of A
  Int * ipiv = new Int[static_cast<uint64_t>(n)];
  Matrix<Complex32> b_copy(b); // Copy of B, since cgesv overwrites B
  lapack_complex_float * b_data = reinterpret_cast<lapack_complex_float *>(b_copy.data());
  Int const ldb = b.rows(); // Leading dimension of B
#  if UM2_ENABLE_ASSERTS
  Int const info =
      LAPACKE_cgesv(LAPACK_COL_MAJOR, n, nrhs, a_data, lda, ipiv, b_data, ldb);
  ASSERT(info == 0);
#  else
  LAPACKE_cgesv(LAPACK_COL_MAJOR, n, nrhs, a_data, lda, ipiv, b_data, ldb);
#  endif
  delete[] ipiv;
  return b_copy;
}

template <>
PURE auto
linearSolve(Matrix<Complex<double>> const & a,
            Matrix<Complex<double>> const & b) -> Matrix<Complex<double>>
{
  using Complex64 = Complex<double>;
  ASSERT(a.rows() == a.cols()); // A must be square
  ASSERT(a.rows() == b.rows());

  // Solve the linear system using LAPACK's zgesv function
  // It is important to note that zgesv overwrites:
  //  - the input matrix A with the LU decomposition of A
  //  - the input matrix B with the solution matrix X
  //
  // Therefore we allocate a copy of A and a copy of B for the function to overwrite
  Int const n = a.rows();      // Number of rows in A
  Int const nrhs = b.cols();   // Number of columns in B
  Matrix<Complex64> a_copy(a); // Copy of A, since zgesv overwrites A
  lapack_complex_double * a_data =
      reinterpret_cast<lapack_complex_double *>(a_copy.data());
  Int const lda = a.rows(); // Leading dimension of A
  Int * ipiv = new Int[static_cast<uint64_t>(n)];
  Matrix<Complex64> b_copy(b); // Copy of B, since zgesv overwrites B
  lapack_complex_double * b_data =
      reinterpret_cast<lapack_complex_double *>(b_copy.data());
  Int const ldb = b.rows(); // Leading dimension of B
#  if UM2_ENABLE_ASSERTS
  Int const info =
      LAPACKE_zgesv(LAPACK_COL_MAJOR, n, nrhs, a_data, lda, ipiv, b_data, ldb);
  ASSERT(info == 0);
#  else
  LAPACKE_zgesv(LAPACK_COL_MAJOR, n, nrhs, a_data, lda, ipiv, b_data, ldb);
#  endif
  delete[] ipiv;
  return b_copy;
}

template <>
void
linearSolve(Matrix<float> & a, Matrix<float> & b, Vector<Int> & ipiv)
{
  ASSERT(a.rows() == a.cols()); // A must be square
  ASSERT(a.rows() == b.rows());
  ASSERT(a.rows() == ipiv.size());

  // Solve the linear system using LAPACK's sgesv function
  // It is important to note that sgesv overwrites:
  //  - the input matrix A with the LU decomposition of A
  //  - the input matrix B with the solution matrix X
  //
  Int const n = a.rows();    // Number of rows in A
  Int const nrhs = b.cols(); // Number of columns in B
  Int const lda = a.rows();  // Leading dimension of A
  Int const ldb = b.rows();  // Leading dimension of B
#  if UM2_ENABLE_ASSERTS
  Int const info =
      LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs, a.data(), lda, ipiv.data(), b.data(), ldb);
  ASSERT(info == 0);
#  else
  LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs, a.data(), lda, ipiv.data(), b.data(), ldb);
#  endif
}

template <>
void
linearSolve(Matrix<double> & a, Matrix<double> & b, Vector<Int> & ipiv)
{
  ASSERT(a.rows() == a.cols()); // A must be square
  ASSERT(a.rows() == b.rows());
  ASSERT(a.rows() == ipiv.size());

  // Solve the linear system using LAPACK's sgesv function
  // It is important to note that sgesv overwrites:
  //  - the input matrix A with the LU decomposition of A
  //  - the input matrix B with the solution matrix X
  //
  Int const n = a.rows();    // Number of rows in A
  Int const nrhs = b.cols(); // Number of columns in B
  Int const lda = a.rows();  // Leading dimension of A
  Int const ldb = b.rows();  // Leading dimension of B
#  if UM2_ENABLE_ASSERTS
  Int const info =
      LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a.data(), lda, ipiv.data(), b.data(), ldb);
  ASSERT(info == 0);
#  else
  LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a.data(), lda, ipiv.data(), b.data(), ldb);
#  endif
}

template <>
void
linearSolve(Matrix<Complex<float>> & a, Matrix<Complex<float>> & b, Vector<Int> & ipiv)
{
  ASSERT(a.rows() == a.cols()); // A must be square
  ASSERT(a.rows() == b.rows());
  ASSERT(a.rows() == ipiv.size());

  // Solve the linear system using LAPACK's cgesv function
  // It is important to note that cgesv overwrites:
  //  - the input matrix A with the LU decomposition of A
  //  - the input matrix B with the solution matrix X
  //
  // Therefore we allocate a copy of A and a copy of B for the function to overwrite
  Int const n = a.rows();    // Number of rows in A
  Int const nrhs = b.cols(); // Number of columns in B
  lapack_complex_float * a_data = reinterpret_cast<lapack_complex_float *>(a.data());
  Int const lda = a.rows(); // Leading dimension of A
  lapack_complex_float * b_data = reinterpret_cast<lapack_complex_float *>(b.data());
  Int const ldb = b.rows(); // Leading dimension of B
#  if UM2_ENABLE_ASSERTS
  Int const info =
      LAPACKE_cgesv(LAPACK_COL_MAJOR, n, nrhs, a_data, lda, ipiv.data(), b_data, ldb);
  ASSERT(info == 0);
#  else
  LAPACKE_cgesv(LAPACK_COL_MAJOR, n, nrhs, a_data, lda, ipiv.data(), b_data, ldb);
#  endif
}

template <>
void
linearSolve(Matrix<Complex<double>> & a, Matrix<Complex<double>> & b, Vector<Int> & ipiv)
{
  ASSERT(a.rows() == a.cols()); // A must be square
  ASSERT(a.rows() == b.rows());
  ASSERT(a.rows() == ipiv.size());

  // Solve the linear system using LAPACK's zgesv function
  // It is important to note that zgesv overwrites:
  //  - the input matrix A with the LU decomposition of A
  //  - the input matrix B with the solution matrix X
  //
  // Therefore we allocate a copy of A and a copy of B for the function to overwrite
  Int const n = a.rows();    // Number of rows in A
  Int const nrhs = b.cols(); // Number of columns in B
  lapack_complex_double * a_data = reinterpret_cast<lapack_complex_double *>(a.data());
  Int const lda = a.rows(); // Leading dimension of A
  lapack_complex_double * b_data = reinterpret_cast<lapack_complex_double *>(b.data());
  Int const ldb = b.rows(); // Leading dimension of B
#  if UM2_ENABLE_ASSERTS
  Int const info =
      LAPACKE_zgesv(LAPACK_COL_MAJOR, n, nrhs, a_data, lda, ipiv.data(), b_data, ldb);
  ASSERT(info == 0);
#  else
  LAPACKE_zgesv(LAPACK_COL_MAJOR, n, nrhs, a_data, lda, ipiv.data(), b_data, ldb);
#  endif
}

//==============================================================================
// Eigenvalues
//==============================================================================

PURE auto
eigvals(Matrix<float> const & a) -> Vector<Complex<float>>
{
  ASSERT(a.rows() == a.cols()); // A must be square

  // Compute the eigenvalues using LAPACK's sgeev function
  // It is important to note that sgeev overwrites:
  // - the input matrix A

  char const jobvl = 'N';   // Do not compute left eigenvectors
  char const jobvr = 'N';   // Do not compute right eigenvectors
  Int const n = a.rows();   // Number of rows in A
  Matrix<float> a_copy(a);  // Copy of A, since sgeev overwrites A
  Int const lda = a.rows(); // Leading dimension of A
  Vector<float> wr(n);      // Real part of the eigenvalues
  Vector<float> wi(n);      // Imaginary part of the eigenvalues
  float * vl = nullptr;     // Left eigenvectors
  Int const ldvl = 1;       // Leading dimension of left eigenvectors
  float * vr = nullptr;     // Right eigenvectors
  Int const ldvr = 1;       // Leading dimension of right eigenvectors
  auto * work = new float[static_cast<uint64_t>(4 * n)]; // Workspace
  Int const lwork = 4 * n;                               // Size of the workspace
#  if UM2_ENABLE_ASSERTS
  Int const info =
      LAPACKE_sgeev_work(LAPACK_COL_MAJOR, jobvl, jobvr, n, a_copy.data(), lda, wr.data(),
                         wi.data(), vl, ldvl, vr, ldvr, work, lwork);
  ASSERT(info == 0);
#  else
  LAPACKE_sgeev_work(LAPACK_COL_MAJOR, jobvl, jobvr, n, a_copy.data(), lda, wr.data(),
                     wi.data(), vl, ldvl, vr, ldvr, work, lwork);
#  endif
  delete[] work;

  Vector<Complex<float>> w(n);
  for (Int i = 0; i < n; ++i) {
    w[i] = Complex<float>(wr[i], wi[i]);
  }
  return w;
}

PURE auto
eigvals(Matrix<double> const & a) -> Vector<Complex<double>>
{
  ASSERT(a.rows() == a.cols()); // A must be square

  // Compute the eigenvalues using LAPACK's dgeev function
  // It is important to note that dgeev overwrites:
  // - the input matrix A

  char const jobvl = 'N';   // Do not compute left eigenvectors
  char const jobvr = 'N';   // Do not compute right eigenvectors
  Int const n = a.rows();   // Number of rows in A
  Matrix<double> a_copy(a); // Copy of A, since dgeev overwrites A
  Int const lda = a.rows(); // Leading dimension of A
  Vector<double> wr(n);     // Real part of the eigenvalues
  Vector<double> wi(n);     // Imaginary part of the eigenvalues
  double * vl = nullptr;    // Left eigenvectors
  Int const ldvl = 1;       // Leading dimension of left eigenvectors
  double * vr = nullptr;    // Right eigenvectors
  Int const ldvr = 1;       // Leading dimension of right eigenvectors
  auto * work = new double[static_cast<uint64_t>(4 * n)]; // Workspace
  Int const lwork = 4 * n;                                // Size of the workspace
#  if UM2_ENABLE_ASSERTS
  Int const info =
      LAPACKE_dgeev_work(LAPACK_COL_MAJOR, jobvl, jobvr, n, a_copy.data(), lda, wr.data(),
                         wi.data(), vl, ldvl, vr, ldvr, work, lwork);
  ASSERT(info == 0);
#  else
  LAPACKE_dgeev_work(LAPACK_COL_MAJOR, jobvl, jobvr, n, a_copy.data(), lda, wr.data(),
                     wi.data(), vl, ldvl, vr, ldvr, work, lwork);
#  endif
  delete[] work;

  Vector<Complex<double>> w(n);
  for (Int i = 0; i < n; ++i) {
    w[i] = Complex<double>(wr[i], wi[i]);
  }
  return w;
}

PURE auto
eigvals(Matrix<Complex<float>> const & a) -> Vector<Complex<float>>
{
  using Complex32 = Complex<float>;
  ASSERT(a.rows() == a.cols()); // A must be square

  // Compute the eigenvalues using LAPACK's cgeev function
  // It is important to note that cgeev overwrites:
  // - the input matrix A

  char const jobvl = 'N';      // Do not compute left eigenvectors
  char const jobvr = 'N';      // Do not compute right eigenvectors
  Int const n = a.rows();      // Number of rows in A
  Matrix<Complex32> a_copy(a); // Copy of A, since cgeev overwrites A
  lapack_complex_float * a_data = reinterpret_cast<lapack_complex_float *>(a_copy.data());
  Int const lda = a.rows(); // Leading dimension of A
  Vector<Complex32> w(n);   // Eigenvalues
  lapack_complex_float * w_data = reinterpret_cast<lapack_complex_float *>(w.data());
  lapack_complex_float * vl = nullptr; // Left eigenvectors
  Int const ldvl = 1;                  // Leading dimension of left eigenvectors
  lapack_complex_float * vr = nullptr; // Right eigenvectors
  Int const ldvr = 1;                  // Leading dimension of right eigenvectors
  Vector<Complex32> work(4 * n);       // Workspace
  lapack_complex_float * work_data =
      reinterpret_cast<lapack_complex_float *>(work.data());
  Int const lwork = 4 * n;    // Size of the workspace
  Vector<float> rwork(2 * n); // Real workspace
#  if UM2_ENABLE_ASSERTS
  Int const info =
      LAPACKE_cgeev_work(LAPACK_COL_MAJOR, jobvl, jobvr, n, a_data, lda, w_data, vl, ldvl,
                         vr, ldvr, work_data, lwork, rwork.data());
  ASSERT(info == 0);
#  else
  LAPACKE_cgeev_work(LAPACK_COL_MAJOR, jobvl, jobvr, n, a_data, lda, w_data, vl, ldvl, vr,
                     ldvr, work_data, lwork, rwork.data());
#  endif

  return w;
}

PURE auto
eigvals(Matrix<Complex<double>> const & a) -> Vector<Complex<double>>
{
  using Complex64 = Complex<double>;
  ASSERT(a.rows() == a.cols()); // A must be square

  // Compute the eigenvalues using LAPACK's zgeev function
  // It is important to note that zgeev overwrites:
  // - the input matrix A

  char const jobvl = 'N';      // Do not compute left eigenvectors
  char const jobvr = 'N';      // Do not compute right eigenvectors
  Int const n = a.rows();      // Number of rows in A
  Matrix<Complex64> a_copy(a); // Copy of A, since zgeev overwrites A
  lapack_complex_double * a_data =
      reinterpret_cast<lapack_complex_double *>(a_copy.data());
  Int const lda = a.rows(); // Leading dimension of A
  Vector<Complex64> w(n);   // Eigenvalues
  lapack_complex_double * w_data = reinterpret_cast<lapack_complex_double *>(w.data());
  lapack_complex_double * vl = nullptr; // Left eigenvectors
  Int const ldvl = 1;                   // Leading dimension of left eigenvectors
  lapack_complex_double * vr = nullptr; // Right eigenvectors
  Int const ldvr = 1;                   // Leading dimension of right eigenvectors
  Vector<Complex64> work(4 * n);        // Workspace
  lapack_complex_double * work_data =
      reinterpret_cast<lapack_complex_double *>(work.data());
  Int const lwork = 4 * n;     // Size of the workspace
  Vector<double> rwork(2 * n); // Real workspace
#  if UM2_ENABLE_ASSERTS
  Int const info =
      LAPACKE_zgeev_work(LAPACK_COL_MAJOR, jobvl, jobvr, n, a_data, lda, w_data, vl, ldvl,
                         vr, ldvr, work_data, lwork, rwork.data());
  ASSERT(info == 0);
#  else
  LAPACKE_zgeev_work(LAPACK_COL_MAJOR, jobvl, jobvr, n, a_data, lda, w_data, vl, ldvl, vr,
                     ldvr, work_data, lwork, rwork.data());
#  endif

  return w;
}

} // namespace um2

#endif // UM2_USE_BLAS_LAPACK
