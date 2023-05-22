#pragma once

#include <ostream>
#include <um2/common/array.hpp>
namespace um2
{

// -----------------------------------------------------------------------------
// VEC
// -----------------------------------------------------------------------------
// A D-dimensional vector with data type T.
//
// This struct is used for small vectors, where the number of elements is known
// at compile time. Depending on the type of T and length D, the vector may be
// stored in a compiler-specific vector extension, or in an array. Depending on
// the value of Q, the vector may be aligned or unaligned.

template <len_t D, typename T>
struct Vec {

  // -- Implementation --

  using array_type = Array<D, T>;
  using value_type = T;

  array_type data;

  // -- Accessors --

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t /*i*/) -> T &;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t /*i*/) const -> T const &;

  //    // Return by value to avoid dangling reference when vector extensions are
  //    // used. This should only affect SIMD sized vectors of fundamental types.
  //    // https://clang.llvm.org/docs/LanguageExtensions.html#vector-operations
  //    UM2_NDEBUG_PURE UM2_HOSTDEV constexpr T operator [] (len_t const i) const
  //    requires (is_simd_vector<D, T>);

  // -- Constructors --

  UM2_HOSTDEV constexpr Vec() = default;

  // Allow implicit conversion from integral types.
  // Otherwise, require explicit conversion to avoid accidental loss of
  // precision/performance.
  template <typename... Is>
  requires(sizeof...(Is) == D && (std::integral<Is> && ...) &&
           !(std::same_as<T, Is> && ...)) UM2_HOSTDEV
      constexpr explicit Vec(Is const... /*args*/);

  template <typename... Ts>
  requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...)) UM2_HOSTDEV
      constexpr explicit Vec(Ts const... /*args*/);

}; // struct Vec

// Zero vector created using value initialization.
template <len_t D, typename T>
constexpr Vec<D, T> zero_vec = Vec<D, T>{};

// -- Aliases --

template <typename T>
using vec1 = Vec<1, T>;
template <typename T>
using vec2 = Vec<2, T>;
template <typename T>
using vec3 = Vec<3, T>;
template <typename T>
using vec4 = Vec<4, T>;

using vec1f = vec1<float>;
using vec2f = vec2<float>;
using vec3f = vec3<float>;
using vec4f = vec4<float>;

using vec1d = vec1<double>;
using vec2d = vec2<double>;
using vec3d = vec3<double>;
using vec4d = vec4<double>;

using vec1i = vec1<int>;
using vec2i = vec2<int>;
using vec3i = vec3<int>;
using vec4i = vec4<int>;

using vec1u = vec1<unsigned>;
using vec2u = vec2<unsigned>;
using vec3u = vec3<unsigned>;
using vec4u = vec4<unsigned>;

// -- IO --

template <len_t D, typename T>
auto operator<<(std::ostream & /*os*/, Vec<D, T> const & /*v*/) -> std::ostream &;

// -- Unary operators --

template <len_t D, typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator-(Vec<D, T> /*v*/) -> Vec<D, T>;

// -- Binary operators --

template <len_t D, typename T>
UM2_HOSTDEV constexpr auto operator+=(Vec<D, T> & /*u*/, Vec<D, T> const & /*v*/)
    -> Vec<D, T> &;

template <len_t D, typename T>
UM2_HOSTDEV constexpr auto operator-=(Vec<D, T> & /*u*/, Vec<D, T> const & /*v*/)
    -> Vec<D, T> &;

template <len_t D, typename T>
UM2_HOSTDEV constexpr auto operator*=(Vec<D, T> & /*u*/, Vec<D, T> const & /*v*/)
    -> Vec<D, T> &;

template <len_t D, typename T>
UM2_HOSTDEV constexpr auto operator/=(Vec<D, T> & /*u*/, Vec<D, T> const & /*v*/)
    -> Vec<D, T> &; // div by 0

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator+(Vec<D, T> /*u*/, Vec<D, T> const & /*v*/)
    -> Vec<D, T>;

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator-(Vec<D, T> /*u*/, Vec<D, T> const & /*v*/)
    -> Vec<D, T>;

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator*(Vec<D, T> /*u*/, Vec<D, T> const & /*v*/)
    -> Vec<D, T>;

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator/(Vec<D, T> /*u*/, Vec<D, T> const & /*v*/)
    -> Vec<D, T>; // div by 0

// -- Scalar operators --

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator+=(Vec<D, T> & /*u*/, S const & /*s*/) -> Vec<D, T> &;

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator-=(Vec<D, T> & /*u*/, S const & /*s*/) -> Vec<D, T> &;

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator*=(Vec<D, T> & /*u*/, S const & /*s*/) -> Vec<D, T> &;

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator/=(Vec<D, T> & /*u*/, S const & /*s*/) -> Vec<D, T> &;

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> ||
         (std::floating_point<T> && std::integral<S>)) UM2_PURE UM2_HOSTDEV constexpr auto
operator+(Vec<D, T>, S const &) -> Vec<D, T>;

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> ||
         (std::floating_point<T> && std::integral<S>)) UM2_PURE UM2_HOSTDEV constexpr auto
operator*(S const & /*s*/, Vec<D, T> /*v*/) -> Vec<D, T>;

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> ||
         (std::floating_point<T> && std::integral<S>)) UM2_PURE UM2_HOSTDEV constexpr auto
operator*(Vec<D, T> /*v*/, S const & /*s*/) -> Vec<D, T>;

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> ||
         (std::floating_point<T> && std::integral<S>)) UM2_PURE UM2_HOSTDEV constexpr auto
operator/(Vec<D, T> /*v*/, S const & /*s*/) -> Vec<D, T>;

// -- Methods --

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto min(Vec<D, T> /*u*/, Vec<D, T> const & /*v*/)
    -> Vec<D, T>;

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto max(Vec<D, T> /*u*/, Vec<D, T> const & /*v*/)
    -> Vec<D, T>;

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto dot(Vec<D, T> const & /*u*/, Vec<D, T> const & /*v*/)
    -> T;

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto norm2(Vec<D, T> const & /*v*/) -> T;

template <len_t D, std::floating_point T>
UM2_PURE UM2_HOSTDEV constexpr auto norm(Vec<D, T> const & /*v*/) -> T;

template <len_t D, std::floating_point T>
UM2_PURE UM2_HOSTDEV constexpr auto normalize(Vec<D, T> const & /*v*/) -> Vec<D, T>;

// -- Vec2 --

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto cross(vec2<T> const & /*u*/, vec2<T> const & /*v*/)
    -> T;

// -- vec3 --

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto cross(vec3<T> const & /*u*/, vec3<T> const & /*v*/)
    -> vec3<T>;

} // namespace um2

#include "vec.inl"
