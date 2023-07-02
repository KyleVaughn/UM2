#pragma once

#include <um2/config.hpp>

#include <um2/math/math_functions.hpp> // um2::sqrt

#include <thrust/extrema.h> // thrust::min, thrust::max

#include <concepts>

namespace um2 {

// -----------------------------------------------------------------------------
// VEC
// -----------------------------------------------------------------------------
// A D-dimensional vector with data of type T.
//
// This struct is used for small vectors, where the number of elements is known
// at compile time.
//
// Many arithmetic operators are purposely not defined, to avoid accidental
// loss of performance through creation of temporaries, poor vectorization, etc.

template <Size D, class T>
struct Vec {

  T data[D];

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------
  
  PURE HOSTDEV constexpr auto 
  // cppcheck-suppress functionConst
  operator [](Size i) noexcept -> T &;

  PURE HOSTDEV constexpr auto 
  operator [](Size i) const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto    
  // cppcheck-suppress functionConst    
  begin() noexcept -> T *;    
    
  PURE HOSTDEV [[nodiscard]] constexpr auto    
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto    
  // cppcheck-suppress functionConst    
  end() noexcept -> T *;    
    
  PURE HOSTDEV [[nodiscard]] constexpr auto    
  end() const noexcept -> T const *;

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr Vec() = default;

  // NOLINTBEGIN(google-explicit-constructor)
  // Allow implicit conversion from integral types.
  // Otherwise, require explicit conversion to avoid accidental loss of
  // precision/performance.
  template <class ...Is>
  requires (sizeof...(Is) == D && (std::integral<Is> && ...) && !(std::same_as<T, Is> && ...))
  // cppcheck-suppress noExplicitConstructor
  HOSTDEV constexpr Vec(Is const ...args) noexcept;

  template <class ...Ts>
  requires (sizeof...(Ts) == D && (std::same_as<T, Ts> && ...)) 
  // cppcheck-suppress noExplicitConstructor
  HOSTDEV constexpr Vec(Ts const ...args) noexcept;
  // NOLINTEND(google-explicit-constructor)
  
  // -----------------------------------------------------------------------------
  // Binary operators
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr auto 
  operator += (Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  operator -= (Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  operator *= (Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  operator /= (Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  template <class S>
  requires (std::same_as<T, S> || std::integral<S>)
  HOSTDEV constexpr auto
  operator += (S const & s) noexcept -> Vec<D, T> &;

  template <class S>
  requires (std::same_as<T, S> || std::integral<S>)
  HOSTDEV constexpr auto
  operator -= (S const & s) noexcept -> Vec<D, T> &;

  template <class S>
  requires (std::same_as<T, S> || std::integral<S>)
  HOSTDEV constexpr auto
  operator *= (S const & s) noexcept -> Vec<D, T> &;

  template <class S>
  requires (std::same_as<T, S> || std::integral<S>)
  HOSTDEV constexpr auto
  operator /= (S const & s) noexcept -> Vec<D, T> &;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr auto
  min(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  max(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  PURE HOSTDEV constexpr auto
  dot(Vec<D, T> const & v) const noexcept -> T;

  PURE HOSTDEV constexpr auto
  squaredNorm() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  norm() const noexcept -> T;

  HOSTDEV constexpr void
  normalize() noexcept;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cross(Vec<2, T> const & v) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cross(Vec<3, T> const & v) const noexcept -> Vec<3, T>;

}; // struct Vec

// Zero vector
template <Size D, class T>
HOSTDEV constexpr auto zeroVec() -> Vec<D, T> 
{
  if constexpr (D == 1) {
    return Vec<D, T>(0);
  } else if constexpr (D == 2) {
    return Vec<D, T>(0, 0);
  } else if constexpr (D == 3) {
    return Vec<D, T>(0, 0, 0);
  } else if constexpr (D == 4) {
    return Vec<D, T>(0, 0, 0, 0);
  } else {
    static_assert(D == 1 || D == 2 || D == 3 || D == 4, "Invalid dimension");
    return Vec<D, T>();
  }
}

// -----------------------------------------------------------------------------
// Aliases
// -----------------------------------------------------------------------------

template <class T> using Vec1 = Vec<1, T>;
template <class T> using Vec2 = Vec<2, T>;
template <class T> using Vec3 = Vec<3, T>;
template <class T> using Vec4 = Vec<4, T>;

using Vec1f = Vec1<float>;
using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;

using Vec1d = Vec1<double>;
using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;
using Vec4d = Vec4<double>;

using Vec1i = Vec1<int>;
using Vec2i = Vec2<int>;
using Vec3i = Vec3<int>;
using Vec4i = Vec4<int>;

using Vec1u = Vec1<unsigned>;
using Vec2u = Vec2<unsigned>;
using Vec3u = Vec3<unsigned>;
using Vec4u = Vec4<unsigned>;

//// -- Methods --
//
//template <Size D, class T>
//PURE HOSTDEV constexpr Vec<D, T> min(Vec<D, T>, Vec<D, T> const &);
//
//template <Size D, class T>
//PURE HOSTDEV constexpr Vec<D, T> max(Vec<D, T>, Vec<D, T> const &);
//
//template <Size D, class T>
//PURE HOSTDEV constexpr T dot(Vec<D, T> const &, Vec<D, T> const &);
//
//template <Size D, class T>
//PURE HOSTDEV constexpr T norm2(Vec<D, T> const &);
//
//template <Size D, std::floating_point T>
//PURE HOSTDEV constexpr T norm(Vec<D, T> const &);
//
////// Also do inplace
////template <Size D, std::floating_point T>
////PURE HOSTDEV constexpr Vec<D, T> normalize(Vec<D, T> const &);

} // namespace um2

#include "Vec.inl"
