#pragma once

#include <um2/common/config.hpp>
#include <um2/math/vec.hpp>

#include <cmath>            // std::sqrt
#include <thrust/extrema.h> // thrust::min, thrust::max

namespace um2
{

template <typename T>
using Vec2 = Vec<2, T>;

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2i = Vec2<int32_t>;
using Vec2u = Vec2<uint32_t>;

template <typename T>
requires(std::is_arithmetic_v<T>) struct Vec<2, T> {
  T x, y;

  // -- Accessors --

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) -> T &
  {
    assert(0 <= i && i < 2);
    if (i == 0) {
      return x;
    }
    return y;
  }

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) const -> T const &
  {
    assert(0 <= i && i < 2);
    if (i == 0) {
      return x;
    }
    return y;
  }

  // -- Constructors --

  constexpr Vec() = default;

  UM2_HOSTDEV constexpr Vec(T x_in, T y_in) : x{x_in}, y{y_in} {}

  // Allow implicit conversion from integral types.
  // Otherwise, require explicit conversion to avoid accidental loss of
  // precision/performance.
  template <std::integral I>
  UM2_HOSTDEV constexpr Vec(I x_in, I y_in)
      : x{static_cast<T>(x_in)}, y{static_cast<T>(y_in)}
  {
  }
};

// Zero vector created using value initialization.
template <typename T>
constexpr Vec2<T> zero_vec = Vec2<T>{};

// Many of the functions below use UM2_CONST instead of UM2_PURE, since
// passing a Vec2<T> by value is can be more efficient than passing by const
// reference.

// -- Unary operators --

template <typename T>
requires(!std::unsigned_integral<T>) UM2_CONST UM2_HOSTDEV constexpr auto
operator-(Vec2<T> v) -> Vec2<T>;

// -- Binary operators --

template <typename T>
UM2_HOSTDEV constexpr auto operator+=(Vec2<T> & u, Vec2<T> v) -> Vec2<T> &;

template <typename T>
UM2_HOSTDEV constexpr auto operator-=(Vec2<T> & u, Vec2<T> v) -> Vec2<T> &;

template <typename T>
UM2_HOSTDEV constexpr auto operator*=(Vec2<T> & u, Vec2<T> v) -> Vec2<T> &;

template <typename T>
UM2_HOSTDEV constexpr auto operator/=(Vec2<T> & u, Vec2<T> v) -> Vec2<T> &;

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator+(Vec2<T> u, Vec2<T> v) -> Vec2<T>;

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator-(Vec2<T> u, Vec2<T> v) -> Vec2<T>;

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator*(Vec2<T> u, Vec2<T> v) -> Vec2<T>;

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator/(Vec2<T> u, Vec2<T> v) -> Vec2<T>;

// -- Scalar operators --

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator+=(Vec2<T> & u, S s) -> Vec2<T> &;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator-=(Vec2<T> & u, S s) -> Vec2<T> &;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator*=(Vec2<T> & u, S s) -> Vec2<T> &;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator/=(Vec2<T> & u, S s) -> Vec2<T> &;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator+(Vec2<T> u, S s) -> Vec2<T>;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator-(Vec2<T> u, S s) -> Vec2<T>;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator*(Vec2<T> u, S s) -> Vec2<T>;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator*(S s, Vec2<T> u) -> Vec2<T>;

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator/(Vec2<T> u, S s) -> Vec2<T>;

// -- Methods --

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto min(Vec2<T> u, Vec2<T> v) -> Vec2<T>;

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto max(Vec2<T> u, Vec2<T> v) -> Vec2<T>;

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto dot(Vec2<T> u, Vec2<T> v) -> T;

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto cross(Vec2<T> u, Vec2<T> v) -> T;

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto norm2(Vec2<T> u) -> T;

template <std::floating_point T>
UM2_CONST UM2_HOSTDEV constexpr auto norm(Vec2<T> u) -> T;

template <std::floating_point T>
UM2_CONST UM2_HOSTDEV constexpr auto normalize(Vec2<T> u) -> Vec2<T>;

} // namespace um2

#include "vec2.inl"