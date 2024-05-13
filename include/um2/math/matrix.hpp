#pragma once

#include <um2/stdlib/vector.hpp>
#include <um2/stdlib/algorithm/fill.hpp>

#include <complex>

//==============================================================================
// MATRIX
//==============================================================================
// A matrix class with dynamic size and column-major storage. If small, fixed
// size matrices are needed, see Mat.hpp.
//
// Uses OpenBLAS for BLAS and LAPACK operations.
// TODO(kcvaughn): Add static check that sizeof(Int) == sizeof(lapack_int) or 
// other types used in OpenBLAS.

namespace um2
{

template <typename T>
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

  static auto
  identity(Int n) -> Matrix<T>;

  //==============================================================================
  // Operators
  //==============================================================================

  auto
  operator+=(Matrix<T> const & other) -> Matrix<T> &;

  auto
  operator-=(Matrix<T> const & other) -> Matrix<T> &;

  //==============================================================================
  // Methods
  //==============================================================================

  void
  zero() noexcept;
};

//==============================================================================
// Free functions 
//==============================================================================

// Matrix-vector operations 
//------------------------------------------------------------------------------
template <typename T>
PURE auto
operator*(Matrix<T> const & a, Vector<T> const & x) -> Vector<T>;

// Matrix-matrix operations
//------------------------------------------------------------------------------

template <typename T>
PURE auto
operator+(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>;

template <typename T>
PURE auto
operator-(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>;

template <typename T>
PURE auto
operator*(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>;

// Solver
//------------------------------------------------------------------------------
// Solve A X = B for X. X = A \ B
template <typename T>
PURE auto
linearSolve(Matrix<T> const & a, Matrix<T> const & b) -> Matrix<T>;

// Eigenvalues
//------------------------------------------------------------------------------
PURE auto
eig(Matrix<float> const & a) -> Vector<std::complex<float>>;

PURE auto
eig(Matrix<double> const & a) -> Vector<std::complex<double>>;

PURE auto
eig(Matrix<std::complex<float>> const & a) -> Vector<std::complex<float>>;

PURE auto
eig(Matrix<std::complex<double>> const & a) -> Vector<std::complex<double>>;

//==============================================================================
// Accessors
//==============================================================================

template <typename T>
PURE HOSTDEV constexpr auto
Matrix<T>::rows() const noexcept -> Int
{
  return _rows;
}

template <typename T>
PURE HOSTDEV constexpr auto
Matrix<T>::cols() const noexcept -> Int
{
  return _cols;
}

template <typename T>
PURE HOSTDEV constexpr auto
Matrix<T>::data() noexcept -> T *
{
  return _data.data();
}

template <typename T>
PURE HOSTDEV constexpr auto
Matrix<T>::data() const noexcept -> T const *
{
  return _data.data();
}

template <typename T>
PURE HOSTDEV constexpr auto
Matrix<T>::operator()(Int i) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < _rows * _cols);
  return _data[i];
}

template <typename T>
PURE HOSTDEV constexpr auto
Matrix<T>::operator()(Int i) const noexcept -> T const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < _rows * _cols);
  return _data[i];
}

template <typename T>
PURE HOSTDEV constexpr auto
Matrix<T>::operator()(Int i, Int j) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i); 
  ASSERT_ASSUME(0 <= j);
  ASSERT(i < _rows);
  ASSERT(j < _cols);
  return _data[j * _rows + i];
}

template <typename T>
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

template <typename T>
Matrix<T>::Matrix(Int rows, Int cols) noexcept
  : _rows{rows}, _cols{cols}, _data(rows * cols)
{
  ASSERT(rows >= 0);
  ASSERT(cols >= 0);
}

template <typename T>
auto
Matrix<T>::identity(Int n) -> Matrix<T>
{
  Matrix<T> result(n, n);
  um2::fill(result._data.begin(), result._data.end(), static_cast<T>(0)); 
  for (Int i = 0; i < n; ++i) {
    result(i, i) = static_cast<T>(1); 
  }
  return result;
}

//==============================================================================
// Operators
//==============================================================================

template <typename T>
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

template <typename T>
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

template <typename T>
void
Matrix<T>::zero() noexcept
{
  um2::fill(_data.begin(), _data.end(), static_cast<T>(0));
}

//==============================================================================
// Free functions 
//==============================================================================

template <typename T>
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

template <typename T>
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

} // namespace um2
