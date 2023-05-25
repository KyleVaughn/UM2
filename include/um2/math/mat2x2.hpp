#pragma once

#include <um2/common/config.hpp>
#include <um2/math/mat.hpp>
#include <um2/math/vec2.hpp>

namespace um2
{
// We want to use capital letters for matrix variables
// NOLINTBEGIN(readability-identifier-naming)

template <typename T>
using Mat2x2 = Mat<2, 2, T>;

using Mat2x2f = Mat2x2<float>;
using Mat2x2d = Mat2x2<double>;

template <typename T>
requires(std::is_arithmetic_v<T>) struct Mat<2, 2, T> {

  // Stored column-major
  // x0 x1
  // y0 y1
  // NOLINTNEXTLINE(*-avoid-c-arrays)
  Vec2<T> cols[2];

  // -- Accessors --

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) -> Vec2<T> &
  {
    assert(0 <= i && i < 2);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    return cols[i];
  }

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) const -> Vec2<T> const &
  {
    assert(0 <= i && i < 2);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    return cols[i];
  }

  // -- Constructors --

  constexpr Mat() = default;

  UM2_HOSTDEV constexpr Mat(Vec2<T> const & c0, Vec2<T> const & c1) : cols{c0, c1} {}
};

// -- Unary operators --

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator-(Mat2x2<T> A) -> Mat2x2<T>;

// -- Binary operators --

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator+(Mat2x2<T> A, Mat2x2<T> const & B)
    -> Mat2x2<T>;

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator-(Mat2x2<T> A, Mat2x2<T> const & B)
    -> Mat2x2<T>;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator*=(Mat2x2<T> & A, S s) -> Mat2x2<T> &;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator/=(Mat2x2<T> & A, S s) -> Mat2x2<T> &;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator*(S s, Mat2x2<T> A) -> Mat2x2<T>;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator*(Mat2x2<T> A, S s) -> Mat2x2<T>;

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator*(Mat2x2<T> const & A, Vec2<T> const & v)
    -> Vec2<T>;

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator*(Mat2x2<T> const & A, Mat2x2<T> const & B)
    -> Mat2x2<T>;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator/(Mat2x2<T> A, S s) -> Mat2x2<T>;

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto det(Mat2x2<T> const & A) -> T;

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto inv(Mat2x2<T> const & A) -> Mat2x2<T>;

// NOLINTEND(readability-identifier-naming)
} // namespace um2

#include "mat2x2.inl"