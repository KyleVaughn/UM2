#pragma once

#include <um2/stdlib/vector.hpp>
#include <um2/stdlib/algorithm/fill.hpp>

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

};

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
// Free functions 
//==============================================================================

// Matrix-vector operations 
//------------------------------------------------------------------------------
template <typename T>
auto
operator*(Matrix<T> const & a, Vector<T> const & x) -> Vector<T>;


} // namespace um2
