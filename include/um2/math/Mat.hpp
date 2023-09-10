#pragma once

#include <um2/math/Vec.hpp>

namespace um2
{

//==============================================================================
// MAT
//==============================================================================
//
// An M by N matrix.
//
// This struct is used for VERY small matrices, where the matrix size is known
// at compile time. The matrix is stored in column-major order.
//
// Note that there is not a general matmul or matvec function. Anything beyond
// very small matrices should be done using something like OpenBLAS, cuBLAS, Eigen,
// etc.

template <Size M, Size N, typename T>
struct Mat {

  using Col = Vec<M, T>;

  // Stored column-major
  // 0 3
  // 1 4
  // 2 5

  Col cols[N];

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV constexpr auto
  col(Size i) noexcept -> Col &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  col(Size i) const noexcept -> Col const &;

  PURE HOSTDEV constexpr auto
  operator()(Size i, Size j) noexcept -> T &;

  PURE HOSTDEV constexpr auto
  operator()(Size i, Size j) const noexcept -> T const &;

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Mat() noexcept = default;

  template <std::same_as<Col>... Cols>
    requires(sizeof...(Cols) == N)
  HOSTDEV constexpr explicit Mat(Cols... in_cols) noexcept;
};

//==============================================================================
// Aliases
//==============================================================================

template <typename T>
using Mat2x2 = Mat<2, 2, T>;

template <typename T>
using Mat3x3 = Mat<3, 3, T>;

//==============================================================================
// Methods
//==============================================================================

template <Size M, Size N, typename T>
PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) justification: capitalize matrix var.
operator*(Mat<M, N, T> const & A, Vec<N, T> const & x) noexcept -> Vec<M, T>;

} // namespace um2

#include "Mat.inl"
