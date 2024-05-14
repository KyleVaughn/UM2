#pragma once

#include <um2/stdlib/vector.hpp>
#include <um2/stdlib/algorithm/fill.hpp>

#include <iostream>
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
eigvals(Matrix<float> const & a) -> Vector<std::complex<float>>;

PURE auto
eigvals(Matrix<double> const & a) -> Vector<std::complex<double>>;

PURE auto
eigvals(Matrix<std::complex<float>> const & a) -> Vector<std::complex<float>>;

PURE auto
eigvals(Matrix<std::complex<double>> const & a) -> Vector<std::complex<double>>;

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
Matrix<T>::operator*=(T const scalar) -> Matrix<T> &
{
  for (Int i = 0; i < _rows * _cols; ++i) {
    _data[i] *= scalar;
  }
  return *this;
}

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

// THIS IS HACKED TOGETHER. DELETE AND BE SMARTER.
// N x M matrix fails
// differentiate between conjugate transpose and transpose
template <typename T>
void
Matrix<T>::transpose() noexcept
{
  ASSERT(_rows == _cols);
  for (Int j = 0; j < _cols - 1; ++j) {
    for (Int i = j + 1; i < _rows; ++i) {
      // Fix this.
      T  aij = _data[j * _rows + i]; 
      T  aji = _data[i * _rows + j]; 
      if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        aij = std::conj(aij);
        aji = std::conj(aji);
      }
      _data[j * _rows + i] = aji;
      _data[i * _rows + j] = aij;
    }
  }
  if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
    for (Int i = 0; i < _rows; ++i) {
      _data[i * _rows + i] = std::conj(_data[i * _rows + i]);
    }
  }
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
