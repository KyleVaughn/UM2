#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/algorithm.hpp> // um2::min, um2::max
#include <um2/stdlib/math.hpp>      // um2::sqrt
#include <um2/stdlib/memory.hpp>    // addressof

#include <concepts>

namespace um2
{

//==============================================================================
// VEC
//==============================================================================
//
// A D-dimensional vector with data of type T.
//
// This struct is used for small vectors, where the number of elements is known
// at compile time.
//
// Many arithmetic operators are purposely not defined to avoid accidental
// loss of performance through creation of temporaries, poor vectorization, etc.
// Ideally we should use expression templates to avoid this, but Eigen + CUDA
// is not a good combination.

template <Size D, class T>
struct Vec {

  T data[D];

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV constexpr auto
  operator[](Size i) noexcept -> T &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() const noexcept -> T const *;

  //==============================================================================
  // Constructors
  //==============================================================================

  // cppcheck-suppress uninitMemberVar; justification: it shouldn't be
  constexpr Vec() noexcept = default;

  // Allow implicit conversion from integral types.
  // Otherwise, require explicit conversion to avoid accidental loss of
  // precision/performance.
  // NOLINTBEGIN(google-explicit-constructor) justified
  template <class... Is>
    requires(sizeof...(Is) == D && (std::integral<Is> && ...) &&
             !(std::same_as<T, Is> && ...))
  HOSTDEV constexpr Vec(Is const... args) noexcept;

  template <class... Ts>
    requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...))
  HOSTDEV constexpr Vec(Ts const... args) noexcept;
  // NOLINTEND(google-explicit-constructor)

  //==============================================================================
  // Binary operators
  //==============================================================================

  HOSTDEV constexpr auto
  operator+=(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  operator-=(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  operator*=(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  operator/=(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  template <class S>
    requires(std::same_as<T, S> || std::integral<S>)
  HOSTDEV constexpr auto
  operator+=(S const & s) noexcept -> Vec<D, T> &;

  template <class S>
    requires(std::same_as<T, S> || std::integral<S>)
  HOSTDEV constexpr auto
  operator-=(S const & s) noexcept -> Vec<D, T> &;

  template <class S>
    requires(std::same_as<T, S> || std::integral<S>)
  HOSTDEV constexpr auto
  operator*=(S const & s) noexcept -> Vec<D, T> &;

  template <class S>
    requires(std::same_as<T, S> || std::integral<S>)
  HOSTDEV constexpr auto
  operator/=(S const & s) noexcept -> Vec<D, T> &;

  //==============================================================================
  // Methods
  //==============================================================================

  HOSTDEV constexpr auto
  min(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  max(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  dot(Vec<D, T> const & v) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredNorm() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  norm() const noexcept -> T;

  HOSTDEV constexpr void
  normalize() noexcept;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  normalized() const noexcept -> Vec<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cross(Vec<2, T> const & v) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cross(Vec<3, T> const & v) const noexcept -> Vec<3, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredDistanceTo(Vec<D, T> const & v) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  distanceTo(Vec<D, T> const & v) const noexcept -> T;

}; // struct Vec

// Zero vector
template <Size D, class T>
HOSTDEV constexpr auto
zeroVec() noexcept -> Vec<D, T>
{
  // There has to be a better way to do this...
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

//==============================================================================
// Aliases
//==============================================================================

template <class T>
using Vec1 = Vec<1, T>;
template <class T>
using Vec2 = Vec<2, T>;
template <class T>
using Vec3 = Vec<3, T>;
template <class T>
using Vec4 = Vec<4, T>;

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

//==============================================================================
// Binary operators
//==============================================================================

template <Size D, class T>
HOSTDEV constexpr auto
operator+(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>;

template <Size D, class T>
HOSTDEV constexpr auto
operator-(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>;

template <Size D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
HOSTDEV constexpr auto
operator*(Scalar s, Vec<D, T> u) noexcept -> Vec<D, T>;

template <Size D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
HOSTDEV constexpr auto
operator/(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>;

//==============================================================================
// Methods
//==============================================================================

template <Size D, class T>
HOSTDEV constexpr auto
min(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>;

template <Size D, class T>
HOSTDEV constexpr auto
max(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>;

template <Size D, class T>
PURE HOSTDEV constexpr auto
dot(Vec<D, T> const & u, Vec<D, T> const & v) noexcept -> T;

template <Size D, class T>
PURE HOSTDEV constexpr auto
squaredNorm(Vec<D, T> const & v) noexcept -> T;

template <Size D, class T>
PURE HOSTDEV constexpr auto
norm(Vec<D, T> const & v) noexcept -> T;

template <Size D, class T>
PURE HOSTDEV constexpr auto
normalized(Vec<D, T> v) noexcept -> Vec<D, T>;

} // namespace um2

#include "Vec.inl"
