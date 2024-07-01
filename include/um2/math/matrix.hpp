#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/algorithm/fill.hpp>
#include <um2/stdlib/vector.hpp>

//==============================================================================
// MATRIX
//==============================================================================
// A matrix class with dynamic size and column-major storage. If small, fixed
// size matrices are needed, see Mat.hpp.
//
// Uses OpenBLAS for BLAS and LAPACK operations.

namespace um2
{

template <class T>
class Matrix
{
  Int _rows;
  Int _cols;

  Vector<T> _data;

public:
  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  rows() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cols() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  asVector() noexcept -> Vector<T> &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  asVector() const noexcept -> Vector<T> const &;

  PURE HOSTDEV constexpr auto
  operator()(Int i) noexcept -> T &;

  PURE HOSTDEV constexpr auto
  operator()(Int i) const noexcept -> T const &;

  PURE HOSTDEV constexpr auto
  operator()(Int i, Int j) noexcept -> T &;

  PURE HOSTDEV constexpr auto
  operator()(Int i, Int j) const noexcept -> T const &;

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Matrix() noexcept = default;

  Matrix(Int rows, Int cols) noexcept;

  Matrix(Int rows, Int cols, T const & value) noexcept;

  static auto
  identity(Int n) -> Matrix<T>;

  //==============================================================================
  // Operators
  //==============================================================================

  auto
  operator*=(T scalar) -> Matrix<T> &;

  auto
  operator+=(Matrix<T> const & other) -> Matrix<T> &;

  auto
  operator-=(Matrix<T> const & other) -> Matrix<T> &;

  //==============================================================================
  // Methods
  //==============================================================================

  void
  zero() noexcept;

  void
  transpose() noexcept;
};

//==============================================================================
// Free functions
//==============================================================================

// Matrix-vector operations
//------------------------------------------------------------------------------
template <class T>
PURE auto
operator*(Matrix<T> const & a, Vector<T> const & x) -> Vector<T>;

// Matrix-matrix operations
//------------------------------------------------------------------------------

template <class T>
PURE auto
operator+(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>;

template <class T>
PURE auto
operator-(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>;

template <class T>
PURE auto
operator*(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>;

// Non-allocating matrix-matrix multiplication. C = A * B.
template <class T>
void
matmul(Matrix<T> & c, Matrix<T> const & a, Matrix<T> const & b);

#if UM2_USE_BLAS_LAPACK

// Solver
//------------------------------------------------------------------------------
// Solve A * X = B for X. X = A \ B
template <class T>
PURE auto
linearSolve(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>;

// Same as above, but non-allocating. ipiv is of size n.
// On exit:
// - A is overwritten with its LU decomposition.
// - B is overwritten with the solution X.
template <class T>
void
linearSolve(Matrix<T> & a, Matrix<T> & b, Vector<Int> & ipiv);

// Eigenvalues
//------------------------------------------------------------------------------
PURE auto
eigvals(Matrix<float> const & a) -> Vector<Complex<float>>;

PURE auto
eigvals(Matrix<double> const & a) -> Vector<Complex<double>>;

PURE auto
eigvals(Matrix<Complex<float>> const & a) -> Vector<Complex<float>>;

PURE auto
eigvals(Matrix<Complex<double>> const & a) -> Vector<Complex<double>>;

#endif // UM2_USE_BLAS_LAPACK

//==============================================================================
// Accessors
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::rows() const noexcept -> Int
{
  return _rows;
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::cols() const noexcept -> Int
{
  return _cols;
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::data() noexcept -> T *
{
  return _data.data();
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::data() const noexcept -> T const *
{
  return _data.data();
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::begin() noexcept -> T *
{
  return _data.begin();
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::begin() const noexcept -> T const *
{
  return _data.begin();
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::end() noexcept -> T *
{
  return _data.end();
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::end() const noexcept -> T const *
{
  return _data.end();
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::asVector() noexcept -> Vector<T> &
{
  return _data;
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::asVector() const noexcept -> Vector<T> const &
{
  return _data;
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::operator()(Int i) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < _rows * _cols);
  return _data[i];
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::operator()(Int i) const noexcept -> T const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < _rows * _cols);
  return _data[i];
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::operator()(Int i, Int j) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(0 <= j);
  ASSERT(i < _rows);
  ASSERT(j < _cols);
  return _data[j * _rows + i];
}

template <class T>
PURE HOSTDEV constexpr auto
Matrix<T>::operator()(Int i, Int j) const noexcept -> T const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(0 <= j);
  ASSERT(i < _rows);
  ASSERT(j < _cols);
  return _data[j * _rows + i];
}

//==============================================================================
// Constructors
//==============================================================================

template <class T>
Matrix<T>::Matrix(Int rows, Int cols) noexcept
    : _rows{rows},
      _cols{cols},
      _data(rows * cols)
{
  ASSERT(rows >= 0);
  ASSERT(cols >= 0);
}

template <class T>
Matrix<T>::Matrix(Int rows, Int cols, T const & value) noexcept
    : _rows{rows},
      _cols{cols},
      _data(rows * cols, value)
{
  ASSERT(rows >= 0);
  ASSERT(cols >= 0);
}

template <class T>
auto
Matrix<T>::identity(Int n) -> Matrix<T>
{
  Matrix<T> result(n, n);
  um2::fill(result.begin(), result.end(), static_cast<T>(0));
  for (Int i = 0; i < n; ++i) {
    result(i, i) = static_cast<T>(1);
  }
  return result;
}

//==============================================================================
// Operators
//==============================================================================

template <class T>
auto
Matrix<T>::operator*=(T const scalar) -> Matrix<T> &
{
  for (Int i = 0; i < _rows * _cols; ++i) {
    _data[i] *= scalar;
  }
  return *this;
}

template <class T>
auto
Matrix<T>::operator+=(Matrix<T> const & other) -> Matrix<T> &
{
  ASSERT_ASSUME(_rows == other._rows);
  ASSERT_ASSUME(_cols == other._cols);

  for (Int i = 0; i < _rows * _cols; ++i) {
    _data[i] += other._data[i];
  }
  return *this;
}

template <class T>
auto
Matrix<T>::operator-=(Matrix<T> const & other) -> Matrix<T> &
{
  ASSERT_ASSUME(_rows == other._rows);
  ASSERT_ASSUME(_cols == other._cols);

  for (Int i = 0; i < _rows * _cols; ++i) {
    _data[i] -= other._data[i];
  }
  return *this;
}

//==============================================================================
// Methods
//==============================================================================

template <class T>
void
Matrix<T>::zero() noexcept
{
  um2::fill(_data.begin(), _data.end(), static_cast<T>(0));
}

// THIS IS HACKED TOGETHER. DELETE AND BE SMARTER.
// N x M matrix fails
// differentiate between conjugate transpose and transpose
template <class T>
void
Matrix<T>::transpose() noexcept
{
  ASSERT(_rows == _cols);
  for (Int j = 0; j < _cols - 1; ++j) {
    for (Int i = j + 1; i < _rows; ++i) {
      // Fix this.
      T aij = _data[j * _rows + i];
      T aji = _data[i * _rows + j];
      if constexpr (std::is_same_v<T, Complex<float>> ||
                    std::is_same_v<T, Complex<double>>) {
        aij = conj(aij);
        aji = conj(aji);
      }
      _data[j * _rows + i] = aji;
      _data[i * _rows + j] = aij;
    }
  }
  if constexpr (std::is_same_v<T, Complex<float>> || std::is_same_v<T, Complex<double>>) {
    for (Int i = 0; i < _rows; ++i) {
      _data[i * _rows + i] = conj(_data[i * _rows + i]);
    }
  }
}

//==============================================================================
// Free functions
//==============================================================================

template <class T>
PURE auto
operator+(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>
{
  ASSERT_ASSUME(a.rows() == b.rows());
  ASSERT_ASSUME(a.cols() == b.cols());

  Matrix<T> result(a.rows(), a.cols());
  for (Int i = 0; i < a.rows() * a.cols(); ++i) {
    result(i) = a(i) + b(i);
  }
  return result;
}

template <class T>
PURE auto
operator-(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>
{
  ASSERT_ASSUME(a.rows() == b.rows());
  ASSERT_ASSUME(a.cols() == b.cols());

  Matrix<T> result(a.rows(), a.cols());
  for (Int i = 0; i < a.rows() * a.cols(); ++i) {
    result(i) = a(i) - b(i);
  }
  return result;
}

// If we aren't using BLAS, we need to implement mat-vec and mat-mat
#if !UM2_USE_BLAS_LAPACK

template <class T>
PURE auto
operator*(Matrix<T> const & a, Vector<T> const & x) -> Vector<T>
{
  ASSERT(a.cols() == x.size());
  Vector<T> y(a.rows(), static_cast<T>(0));

  for (Int i = 0; i < a.cols(); ++i) {
    for (Int j = 0; j < a.rows(); ++j) {
      y[j] += a(j, i) * x[i];
    }
  }
  return y;
}

template <class T>
PURE auto
operator*(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>
{
  ASSERT(a.cols() == b.rows());
  Matrix<T> c(a.rows(), b.cols(), static_cast<T>(0));

  for (Int i = 0; i < a.rows(); ++i) {
    for (Int j = 0; j < b.cols(); ++j) {
      for (Int k = 0; k < a.cols(); ++k) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return c;
}

template <class T>
void
matmul(Matrix<T> & c, Matrix<T> const & a, Matrix<T> const & b)
{
  ASSERT(a.cols() == b.rows());
  ASSERT(c.rows() == a.rows());
  ASSERT(c.cols() == b.cols());

  for (Int i = 0; i < a.rows(); ++i) {
    for (Int j = 0; j < b.cols(); ++j) {
      for (Int k = 0; k < a.cols(); ++k) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

#endif // !UM2_USE_BLAS_LAPACK

} // namespace um2
