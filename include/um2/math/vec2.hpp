#pragma once

#include <um2/common/config.hpp>
#include <um2/math/vec.hpp>

namespace um2
{

template <typename T>
using Vec2 = Vec<2, T>;

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2i = Vec2<int32_t>;
using Vec2u = Vec2<uint32_t>;

template <typename T>
requires(std::is_arithmetic_v<T>)
struct Vec<2, T> {
  T x, y;

  // -- Constructors --

  UM2_HOSTDEV constexpr Vec() = default;

  // Allow implicit conversion from integral types.
  // Otherwise, require explicit conversion to avoid accidental loss of
  // precision/performance.
  template <std::integral I>
  requires(!std::same_as<I, T>)
  UM2_HOSTDEV constexpr Vec(I x_in, I y_in) : x{static_cast<T>(x_in)},
                                              y{static_cast<T>(y_in)}
  { 
  };

}; 

// Zero vector created using value initialization.
template <typename T>
constexpr Vec2<T> zero_vec = Vec2<T>{};

// -- Unary operators --

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator-(Vec2<T> v) -> Vec2<T>;

//// -- Binary operators --
//
//template <length_t D, typename T, Qualifier Q>
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator+=(Vec<D, T, Q> &, Vec<D, T, Q> const &);
//
//template <length_t D, typename T, Qualifier Q>
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator-=(Vec<D, T, Q> &, Vec<D, T, Q> const &);
//
//template <length_t D, typename T, Qualifier Q>
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator*=(Vec<D, T, Q> &, Vec<D, T, Q> const &);
//
//template <length_t D, typename T, Qualifier Q>
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator/=(Vec<D, T, Q> &,
//                                                Vec<D, T, Q> const &); // div by 0
//
//template <length_t D, typename T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator+(Vec<D, T, Q>, Vec<D, T, Q> const &);
//
//template <length_t D, typename T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator-(Vec<D, T, Q>, Vec<D, T, Q> const &);
//
//template <length_t D, typename T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator*(Vec<D, T, Q>, Vec<D, T, Q> const &);
//
//template <length_t D, typename T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator/(Vec<D, T, Q>,
//                                                      Vec<D, T, Q> const &); // div by 0
//
//// -- Scalar operators --
//
//template <length_t D, typename T, Qualifier Q, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator+=(Vec<D, T, Q> &, S const &);
//
//template <length_t D, typename T, Qualifier Q, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator-=(Vec<D, T, Q> &, S const &);
//
//template <length_t D, typename T, Qualifier Q, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator*=(Vec<D, T, Q> &, S const &);
//
//template <length_t D, typename T, Qualifier Q, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator/=(Vec<D, T, Q> &, S const &);
//
//template <length_t D, typename T, Qualifier Q, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator+(Vec<D, T, Q>, S const &);
//
//template <length_t D, typename T, Qualifier Q, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator*(S const &, Vec<D, T, Q>);
//
//template <length_t D, typename T, Qualifier Q, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator*(Vec<D, T, Q>, S const &);
//
//template <length_t D, typename T, Qualifier Q, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator/(Vec<D, T, Q>, S const &);
//
//// -- Methods --
//
//template <length_t D, typename T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> min(Vec<D, T, Q>, Vec<D, T, Q> const &);
//
//template <length_t D, typename T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> max(Vec<D, T, Q>, Vec<D, T, Q> const &);
//
//template <length_t D, typename T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr T dot(Vec<D, T, Q> const &, Vec<D, T, Q> const &);
//
//template <length_t D, typename T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr T norm2(Vec<D, T, Q> const &);
//
//template <length_t D, std::floating_point T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr T norm(Vec<D, T, Q> const &);
//
//template <length_t D, std::floating_point T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> normalize(Vec<D, T, Q> const &);
//
//// -- Vec2 --
//
//template <typename T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr T cross(Vec2<T, Q> const &, Vec2<T, Q> const &);
//
//// -- Vec3 --
//
//template <typename T, Qualifier Q>
//UM2_PURE UM2_HOSTDEV constexpr Vec3<T, Q> cross(Vec3<T, Q> const &, Vec3<T, Q> const &);
//
} // namespace um2

#include "vec2.inl"