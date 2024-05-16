#pragma once

#include <um2/config.hpp>

#include <um2/math/vec.hpp>
#include <um2/stdlib/math/trigonometric_functions.hpp>

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

  using Col = Vec<M, T>;

  // Stored column-major
  // 0 3
  // 1 4
  // 2 5

  Col _cols[N];

public:
  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV constexpr auto
  col(Int i) noexcept -> Col &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  col(Int i) const noexcept -> Col const &;

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
};

//==============================================================================
// Aliases
//==============================================================================

template <typename T>
using Mat2x2 = Mat<2, 2, T>;

template <typename T>
using Mat3x3 = Mat<3, 3, T>;

using Mat2x2F = Mat2x2<Float>;

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

// From a list of columns
template <Int M, Int N, typename T>
template <std::same_as<Vec<M, T>>... Cols>
  requires(sizeof...(Cols) == N)
HOSTDEV constexpr Mat<M, N, T>::Mat(Cols... cols) noexcept
    : _cols{cols...}
{
}

//==============================================================================
// Free functions
//==============================================================================

template <typename T>
PURE HOSTDEV constexpr auto
operator*(Mat2x2<T> const & a, Vec2<T> const & x) noexcept -> Vec2<T>
{
#if UM2_ENABLE_SIMD_VEC
  return x[0] * a.col(0) + x[1] * a.col(1);
#else
  return Vec2<T>{a(0, 0) * x[0] + a(0, 1) * x[1], a(1, 0) * x[0] + a(1, 1) * x[1]};
#endif
}

template <typename T>
PURE HOSTDEV constexpr auto
operator*(Mat2x2<T> const & a, Mat2x2<T> const & b) noexcept -> Mat2x2<T>
{
  return Mat2x2<T>{a * b.col(0), a * b.col(1)};
}

template <typename T>
PURE HOSTDEV constexpr auto
operator*(Mat3x3<T> const & a, Vec3<T> const & x) noexcept -> Vec3<T>
{
  return Vec3<T>{a(0, 0) * x[0] + a(0, 1) * x[1] + a(0, 2) * x[2],
                 a(1, 0) * x[0] + a(1, 1) * x[1] + a(1, 2) * x[2],
                 a(2, 0) * x[0] + a(2, 1) * x[1] + a(2, 2) * x[2]};
}

// angle in radians
template <typename T>
PURE HOSTDEV constexpr auto
makeRotationMatrix(T angle) noexcept -> Mat2x2<T>
{
  T const c = um2::cos(angle);
  T const s = um2::sin(angle);
  return Mat2x2<T>{
      Vec2<T>{ c, s},
      Vec2<T>{-s, c}
  };
}

template <typename T>
PURE HOSTDEV constexpr auto
inv(Mat2x2<T> const & m) noexcept -> Mat2x2<T>
{
  // [a b
  //  c d]
  T const a = m(0, 0);
  T const b = m(0, 1);
  T const c = m(1, 0);
  T const d = m(1, 1);
  T const det = det2x2(a, b, c, d);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
  // NOLINTNEXTLINE(clang-diagnostic-float-equal)
  ASSERT(det != 0);
#pragma GCC diagnostic pop
  return Mat2x2<T>{
      Vec2<T>{ d / det, -c / det},
      Vec2<T>{-b / det,  a / det}
  };
}

} // namespace um2
