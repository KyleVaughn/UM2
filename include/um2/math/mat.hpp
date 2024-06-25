#pragma once

#include <um2/config.hpp>

#include <um2/math/vec.hpp>
#include <um2/stdlib/math/trigonometric_functions.hpp>
#include <um2/stdlib/memory/addressof.hpp>

//==============================================================================
// MAT
//==============================================================================
// An M by N matrix.
//
// This class is used for VERY small matrices, where the matrix size is known
// at compile time. The matrix is stored in column-major order.
//
// For matrices larger than 16x16, use Matrix.

namespace um2
{

template <Int M, Int N, typename T>
class Mat
{
  static_assert(M > 0);
  static_assert(N > 0);

  using Col = Vec<M, T>;

  // Stored column-major
  // 0 3
  // 1 4
  // 2 5

  Vec<N, Col> _cols;

public:
  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV constexpr auto
  col(Int i) noexcept -> Col &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  col(Int i) const noexcept -> Col const &;

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

  constexpr Mat() noexcept = default;

  template <std::same_as<Col>... Cols>
    requires(sizeof...(Cols) == N)
  HOSTDEV constexpr explicit Mat(Cols... cols) noexcept;

  //==============================================================================
  // Methods
  //==============================================================================

  HOSTDEV [[nodiscard]] static constexpr auto
  zero() noexcept -> Mat<M, N, T>;

  HOSTDEV [[nodiscard]] static constexpr auto
  identity() noexcept -> Mat<M, N, T>
    requires(M == N);
};

//==============================================================================
// Aliases
//==============================================================================

template <typename T>
using Mat2x2 = Mat<2, 2, T>;

template <typename T>
using Mat3x3 = Mat<3, 3, T>;

using Mat2x2f = Mat2x2<float>;
using Mat2x2d = Mat2x2<double>;

//==============================================================================
// Accessors
//==============================================================================

template <Int M, Int N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::col(Int i) noexcept -> typename Mat<M, N, T>::Col &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _cols[i];
}

template <Int M, Int N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::col(Int i) const noexcept -> typename Mat<M, N, T>::Col const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _cols[i];
}

template <Int M, Int N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::operator()(Int i) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < M * N);
  return reinterpret_cast<T *>(um2::addressof(_cols))[i];
}

template <Int M, Int N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::operator()(Int i) const noexcept -> T const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < M * N);
  return reinterpret_cast<T const *>(um2::addressof(_cols))[i];
}

template <Int M, Int N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::operator()(Int i, Int j) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(0 <= j);
  ASSERT_ASSUME(i < M);
  ASSERT_ASSUME(j < N);
  return _cols[j][i];
}

template <Int M, Int N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::operator()(Int i, Int j) const noexcept -> T const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(0 <= j);
  ASSERT_ASSUME(i < M);
  ASSERT_ASSUME(j < N);
  return _cols[j][i];
}

//==============================================================================
// Constructors
//==============================================================================

template <Int M, Int N, typename T>
template <std::same_as<Vec<M, T>>... Cols>
  requires(sizeof...(Cols) == N)
HOSTDEV constexpr Mat<M, N, T>::Mat(Cols... cols) noexcept
    : _cols{cols...}
{
}

//=============================================================================
//  Member functions
//=============================================================================

template <Int M, Int N, typename T>
HOSTDEV [[nodiscard]] constexpr auto
Mat<M, N, T>::zero() noexcept -> Mat<M, N, T>
{
  Mat<M, N, T> result;
  for (Int i = 0; i < N; ++i) {
    result._cols[i] = Vec<M, T>::zero();
  }
  return result;
}

template <Int M, Int N, typename T>
HOSTDEV [[nodiscard]] constexpr auto
Mat<M, N, T>::identity() noexcept -> Mat<M, N, T>
  requires(M == N)
{
  Mat<M, N, T> result = Mat<M, N, T>::zero();
  for (Int i = 0; i < N; ++i) {
    result(i, i) = static_cast<T>(1);
  }
  return result;
}

//==============================================================================
// Free functions
//==============================================================================

template <Int M, Int N, typename T>
PURE HOSTDEV constexpr auto
operator+(Mat<M, N, T> const & a, Mat<M, N, T> const & b) noexcept -> Mat<M, N, T>
{
  Mat<M, N, T> result;
  for (Int i = 0; i < N; ++i) {
    result.col(i) = a.col(i) + b.col(i);
  }
  return result;
}

template <Int M, Int N, typename T>
PURE HOSTDEV constexpr auto
operator-(Mat<M, N, T> const & a, Mat<M, N, T> const & b) noexcept -> Mat<M, N, T>
{
  Mat<M, N, T> result;
  for (Int i = 0; i < N; ++i) {
    result.col(i) = a.col(i) - b.col(i);
  }
  return result;
}

template <Int M, Int N, typename T>
PURE HOSTDEV constexpr auto
operator*(Mat<M, N, T> const & a, Vec<N, T> const & x) noexcept -> Vec<M, T>
{
  Vec<M, T> res = Vec<M, T>::zero();
  for (Int i = 0; i < N; ++i) {
    res += x[i] * a.col(i);
  }
  return res;
}

template <Int M, Int N, Int P, typename T>
PURE HOSTDEV constexpr auto
operator*(Mat<M, N, T> const & a, Mat<N, P, T> const & b) noexcept -> Mat<M, P, T>
{
// False positive warning for uninitialized variable, since N > 0
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
  Mat<M, P, T> result;
  for (Int i = 0; i < N; ++i) {
    result.col(i) = a * b.col(i);
  }
  return result;
#pragma GCC diagnostic pop
}

// angle in radians
template <typename T>
PURE HOSTDEV constexpr auto
makeRotationMatrix(T angle) noexcept -> Mat2x2<T>
{
  T const c = um2::cos(angle);
  T const s = um2::sin(angle);
  Mat2x2<T> result;
  // [c -s
  //  s  c]
  result(0) = c;
  result(1) = s;
  result(2) = -s;
  result(3) = c;
  return result;
}

template <typename T>
PURE HOSTDEV constexpr auto
inv(Mat2x2<T> const & m) noexcept -> Mat2x2<T>
{
  // [a b
  //  c d]
  T const a = m(0);
  T const c = m(1);
  T const b = m(2);
  T const d = m(3);
  T const det = det2x2(a, b, c, d); // Kahan's alg. Included from Vec.hpp
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
  // NOLINTNEXTLINE(clang-diagnostic-float-equal)
  ASSERT(det != 0);
#pragma GCC diagnostic pop
  Mat2x2<T> result;
  // [ d -b
  //  -c  a]
  result(0) = d / det;
  result(1) = -c / det;
  result(2) = -b / det;
  result(3) = a / det;
  return result;
}

} // namespace um2
