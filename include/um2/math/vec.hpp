#pragma once

#include <um2/config.hpp>

#include <um2/common/cast_if_not.hpp>
#include <um2/stdlib/algorithm/max.hpp>
#include <um2/stdlib/algorithm/min.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/math/fma.hpp>
#include <um2/stdlib/math/roots.hpp>

#include <concepts>

// We alias Vec<D> to Point<D> for D-dimensional points. Hence, we need to
// define some constants for points in this file.

//==============================================================================
// Constants
//==============================================================================
// epsDistance:
//   Distance between two points, below which they are considered to be equal.
// epsDistance2:
//   Squared distance between two points, below which they are considered to be
//   equal.
// infDistance:
//  Distance between two points, above which they are considered to be
//  infinitely far apart. Typically used for invalid points and values.
//
// NOTE: fast-math assumes no infinities, so we need inf_distance to be finite.

namespace um2
{

// EPS_FLOAT = 1.1920928955078125e-07F;
// EPS_DOUBLE = 2.2204460492503131e-16;
// Float32 has about 7 decimal digits of precision.
// If the max coordinate is on the order of 100 or so, then
// 123.4567 means 1e-4 is about all we can expect in terms of absolute error.
// This number is likely too small, since many algorithms yield 1 to 3 ULPs of
// error.

template <class T>
HOSTDEV constexpr auto
epsDistance() -> T
{
  if constexpr (std::same_as<T, float>) {
    return castIfNot<T>(1e-4); // 1 um
  } else if constexpr (std::same_as<T, double>) {
    return castIfNot<T>(1e-7); // 1 nm
  } else {
    static_assert(always_false<T>, "Unsupported type");
  }
}

template <class T>
HOSTDEV constexpr auto
epsDistance2() -> T
{
  return epsDistance<T>() * epsDistance<T>();
}

template <class T>
HOSTDEV constexpr auto
infDistance() -> T
{
  return castIfNot<T>(1e8); // 1000 km
}

} // namespace um2

//==============================================================================
// VEC
//==============================================================================
// A D-dimensional vector with data of type T.

namespace um2
{

#ifndef __CUDA_ARCH__
static constexpr bool gcc_vec_ext_enabled = (UM2_ENABLE_SIMD_VEC == 1);
#else
static constexpr bool gcc_vec_ext_enabled = false;
#endif

static consteval auto
isPowerOf2(Int x) noexcept -> bool
{
  return (x & (x - 1)) == 0;
};

template <Int D, class T>
concept is_simd_vec = isPowerOf2(D) && std::is_arithmetic_v<T>;

template <Int D, class T, bool B = is_simd_vec<D, T>>
struct VecData;

template <Int D, class T>
struct VecData<D, T, false> {
  using Data = T[D];
};

template <Int D, class T>
struct VecData<D, T, true> {
#if UM2_ENABLE_SIMD_VEC && !defined(__CUDA_ARCH__)
  using Data __attribute__((vector_size(D * sizeof(T)))) = T;
#else
  using Data = T[D];
#endif
};

template <Int D, class T>
class Vec;

template <class T>
struct IsSIMDVec {
  static constexpr bool value = false;
};

template <Int D, class T>
struct IsSIMDVec<Vec<D, T>> {
  static constexpr bool value = is_simd_vec<D, T>;
};

// Align if power of 2 and arithmetic, or if the data otherwise maps to a SIMD type
template <Int D, class T>
static consteval auto
vecAlignment() noexcept -> Int
{
  if constexpr (isPowerOf2(D) && (std::is_arithmetic_v<T> || IsSIMDVec<T>::value)) {
    return D * alignof(T);
  } else {
    return alignof(T);
  }
};

template <Int D, class T>
class Vec
{
  static_assert(D > 0);

  using Data = typename VecData<D, T>::Data;
  alignas(vecAlignment<D, T>()) Data _data;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getPointer() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getConstPointer() const noexcept -> T const *;

public:
  //==============================================================================
  // Element access
  //==============================================================================

  PURE HOSTDEV constexpr auto
  operator[](Int i) noexcept -> T &;

  PURE HOSTDEV constexpr auto
  operator[](Int i) const noexcept -> T const &;

  //==============================================================================
  // Iterators
  //==============================================================================

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

  constexpr Vec() noexcept = default;

  // Allow implicit conversion from integral types.
  // Otherwise, require explicit conversion to avoid accidental loss of
  // precision/performance.
  // NOLINTBEGIN(google-explicit-constructor)
  template <class... Is>
    requires(sizeof...(Is) == D && (std::integral<Is> && ...) &&
             !(std::same_as<T, Is> && ...))
  HOSTDEV constexpr Vec(Is const... args) noexcept;

  template <class... Ts>
    requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...))
  HOSTDEV constexpr Vec(Ts const... args) noexcept;
  // NOLINTEND(google-explicit-constructor)

  //==============================================================================
  // Unary operators
  //==============================================================================

  HOSTDEV constexpr auto
  operator-() const noexcept -> Vec<D, T>;

  //==============================================================================
  // Binary operators
  //==============================================================================

  // Element-wise operators with Vecs
  HOSTDEV constexpr auto
  operator+=(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  operator-=(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  operator*=(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  HOSTDEV constexpr auto
  operator/=(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  // Element-wise operators with scalars
  // Require that the scalar type is either the same as the vector type or an
  // integral type.

  template <class S>
    requires(std::same_as<T, S> || std::integral<S>)
  HOSTDEV constexpr auto
  operator=(S const & s) noexcept -> Vec<D, T> &;

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
  // Other member functions
  //==============================================================================

  HOSTDEV [[nodiscard]] static constexpr auto
  zero() noexcept -> Vec<D, T>;

  HOSTDEV constexpr void
  min(Vec<D, T> const & v) noexcept;

  HOSTDEV constexpr void
  max(Vec<D, T> const & v) noexcept;

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
  cross(Vec<2, T> const & v) const noexcept -> T
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cross(Vec<3, T> const & v) const noexcept -> Vec<3, T>
    requires(D == 3);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredDistanceTo(Vec<D, T> const & v) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  distanceTo(Vec<D, T> const & v) const noexcept -> T;

  // eps2 is the squared distance below which two points are considered to be
  // equal.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isApprox(Vec<D, T> const & v,
           T const & eps2 = epsDistance2<T>()) const noexcept -> bool;

}; // class Vec

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

using Vec2I = Vec2<Int>;
using Vec3I = Vec3<Int>;

using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;

using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;

using Vec2F = Vec2<Float>;
using Vec3F = Vec3<Float>;

//==============================================================================
// Free functions
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
min(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>;

template <Int D, class T>
PURE HOSTDEV constexpr auto
max(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>;

template <Int D, class T>
PURE HOSTDEV constexpr auto
dot(Vec<D, T> const & u, Vec<D, T> const & v) noexcept -> T;

template <Int D, class T>
PURE HOSTDEV constexpr auto
squaredNorm(Vec<D, T> const & v) noexcept -> T;

template <Int D, class T>
PURE HOSTDEV constexpr auto
norm(Vec<D, T> const & v) noexcept -> T;

template <Int D, class T>
PURE HOSTDEV constexpr auto
normalized(Vec<D, T> v) noexcept -> Vec<D, T>;

//==============================================================================
// Non-member operators
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator==(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool;

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator<(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool;

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator!=(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool;

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator>(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool;

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator<=(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool;

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator>=(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool;

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator+(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>;

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator-(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>;

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator*(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>;

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator/(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>;

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator+(Scalar s, Vec<D, T> u) noexcept -> Vec<D, T>;

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator+(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>;

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator-(Scalar s, Vec<D, T> u) noexcept -> Vec<D, T>;

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator-(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>;

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator*(Scalar s, Vec<D, T> u) noexcept -> Vec<D, T>;

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator*(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>;

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator/(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>;

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator/(Scalar s, Vec<D, T> const & u) noexcept -> Vec<D, T>;

//==============================================================================
// Private functions
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::getPointer() noexcept -> T *
{
  return reinterpret_cast<T *>(&_data);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::getConstPointer() const noexcept -> T const *
{
  return reinterpret_cast<T const *>(&_data);
}

//==============================================================================
// Element access
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::operator[](Int i) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    return getPointer()[i];
  } else {
    return _data[i];
  }
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::operator[](Int i) const noexcept -> T const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    return getConstPointer()[i];
  } else {
    return _data[i];
  }
}

//==============================================================================
// Iterators
//==============================================================================

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::begin() noexcept -> T *
{
  return getPointer();
}

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::begin() const noexcept -> T const *
{
  return getConstPointer();
}

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::end() noexcept -> T *
{
  return getPointer() + D;
}

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::end() const noexcept -> T const *
{
  return getConstPointer() + D;
}

//==============================================================================
// Constructors
//==============================================================================

template <Int D, class T>
template <class... Is>
  requires(sizeof...(Is) == D && (std::integral<Is> && ...) &&
           !(std::same_as<T, Is> && ...))
HOSTDEV constexpr Vec<D, T>::Vec(Is const... args) noexcept
    : _data{static_cast<T>(args)...}
{
}

template <Int D, class T>
template <class... Ts>
  requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...))
HOSTDEV constexpr Vec<D, T>::Vec(Ts const... args) noexcept
    : _data{args...}
{
}

//==============================================================================
// Unary operators
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator-() const noexcept -> Vec<D, T>
{
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    Vec<D, T> result;
    result._data = -_data;
    return result;
  } else {
    Vec<D, T> result;
    for (Int i = 0; i < D; ++i) {
      result[i] = -_data[i];
    }
    return result;
  }
}

//==============================================================================
// Binary operators
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator+=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    _data += v._data;
  } else {
    for (Int i = 0; i < D; ++i) {
      _data[i] += v[i];
    }
  }
  return *this;
}

template <Int D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator-=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    _data -= v._data;
  } else {
    for (Int i = 0; i < D; ++i) {
      _data[i] -= v[i];
    }
  }
  return *this;
}

template <Int D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator*=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    _data *= v._data;
  } else {
    for (Int i = 0; i < D; ++i) {
      _data[i] *= v[i];
    }
  }
  return *this;
}

template <Int D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator/=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    _data /= v._data;
  } else {
    for (Int i = 0; i < D; ++i) {
      _data[i] /= v[i];
    }
  }
  return *this;
}

template <Int D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto
Vec<D, T>::operator=(S const & s) noexcept -> Vec<D, T> &
{
  for (Int i = 0; i < D; ++i) {
    _data[i] = static_cast<T>(s);
  }
  return *this;
}

template <Int D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto
Vec<D, T>::operator+=(S const & s) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    _data += static_cast<T>(s);
  } else {
    for (Int i = 0; i < D; ++i) {
      _data[i] += static_cast<T>(s);
    }
  }
  return *this;
}

template <Int D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto
Vec<D, T>::operator-=(S const & s) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    _data -= static_cast<T>(s);
  } else {
    for (Int i = 0; i < D; ++i) {
      _data[i] -= static_cast<T>(s);
    }
  }
  return *this;
}

template <Int D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto
Vec<D, T>::operator*=(S const & s) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    _data *= static_cast<T>(s);
  } else {
    for (Int i = 0; i < D; ++i) {
      _data[i] *= static_cast<T>(s);
    }
  }
  return *this;
}

template <Int D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto
Vec<D, T>::operator/=(S const & s) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T> && gcc_vec_ext_enabled) {
    _data /= static_cast<T>(s);
  } else {
    for (Int i = 0; i < D; ++i) {
      _data[i] /= static_cast<T>(s);
    }
  }
  return *this;
}

//==============================================================================
// Other member functions
//==============================================================================

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::zero() noexcept -> Vec<D, T>
{
  Vec<D, T> result;
  for (Int i = 0; i < D; ++i) {
    result[i] = static_cast<T>(0);
  }
  return result;
}

template <Int D, class T>
HOSTDEV constexpr void
Vec<D, T>::min(Vec<D, T> const & v) noexcept
{
  for (Int i = 0; i < D; ++i) {
    _data[i] = um2::min(_data[i], v._data[i]);
  }
}

template <Int D, class T>
HOSTDEV constexpr void
Vec<D, T>::max(Vec<D, T> const & v) noexcept
{
  for (Int i = 0; i < D; ++i) {
    _data[i] = um2::max(_data[i], v._data[i]);
  }
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::dot(Vec<D, T> const & v) const noexcept -> T
{
  T result = _data[0] * v._data[0];
  for (Int i = 1; i < D; ++i) {
    result += _data[i] * v._data[i];
  }
  return result;
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::squaredNorm() const noexcept -> T
{
  T result = _data[0] * _data[0];
  for (Int i = 1; i < D; ++i) {
    result += _data[i] * _data[i];
  }
  return result;
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::norm() const noexcept -> T
{
  return um2::sqrt(squaredNorm());
}

template <Int D, class T>
HOSTDEV constexpr void
Vec<D, T>::normalize() noexcept
{
  *this /= norm();
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::normalized() const noexcept -> Vec<D, T>
{
  Vec<D, T> result = *this;
  result.normalize();
  return result;
}

template <class T>
HOSTDEV [[nodiscard]] constexpr auto
det2x2(T const a, T const b, T const c, T const d) noexcept -> T
{
  // Kahan's algorithm for accurate 2 x 2 determinants.
  // Returns a * d - b * c, but more accurately.
  T const w = b * c;
  T const e = um2::fma(-b, c, w);
  T const f = um2::fma(a, d, -w);
  return f + e;
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::cross(Vec<2, T> const & v) const noexcept -> T
  requires(D == 2)
{
  // It's important to use the slightly slower, but much more accurate Kahan's
  // algorithm for the 2 x 2 determinant here, since the sign of the result is
  // used in many geometric algorithms.
  return det2x2(_data[0], _data[1], v[0], v[1]);
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::cross(Vec<3, T> const & v) const noexcept -> Vec<3, T>
  requires(D == 3)
{
  return {det2x2(_data[1], _data[2], v[1], v[2]), det2x2(_data[2], _data[0], v[2], v[0]),
          det2x2(_data[0], _data[1], v[0], v[1])};
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::squaredDistanceTo(Vec<D, T> const & v) const noexcept -> T
{
  auto const diff = *this - v;
  return diff.squaredNorm();
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::distanceTo(Vec<D, T> const & v) const noexcept -> T
{
  return um2::sqrt(squaredDistanceTo(v));
}

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::isApprox(Vec<D, T> const & v, T const & eps2) const noexcept -> bool
{
  return squaredDistanceTo(v) < eps2;
}

//==============================================================================
// Free functions
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
min(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  u.min(v);
  return u;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
max(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  u.max(v);
  return u;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
dot(Vec<D, T> const & u, Vec<D, T> const & v) noexcept -> T
{
  return u.dot(v);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
squaredNorm(Vec<D, T> const & v) noexcept -> T
{
  return v.squaredNorm();
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
norm(Vec<D, T> const & v) noexcept -> T
{
  return v.norm();
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
normalized(Vec<D, T> v) noexcept -> Vec<D, T>
{
  return v.normalized();
}

template <class T>
PURE HOSTDEV constexpr auto
cross(Vec2<T> const & u, Vec2<T> const & v) noexcept -> T
{
  return u.cross(v);
}

template <class T>
PURE HOSTDEV constexpr auto
cross(Vec3<T> const & u, Vec3<T> const & v) noexcept -> Vec3<T>
{
  return u.cross(v);
}

//==============================================================================
// Non-member operators
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator==(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool
{
  for (Int i = 0; i < D; ++i) {
    if (l[i] != r[i]) {
      return false;
    }
  }
  return true;
}

// Lexicographical comparison
template <Int D, class T>
PURE HOSTDEV constexpr auto
operator<(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool
{
  for (Int i = 0; i < D; ++i) {
    if (l[i] < r[i]) {
      return true;
    }
    if (r[i] < l[i]) {
      return false;
    }
  }
  return false;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator!=(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool
{
  return !(l == r);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator>(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool
{
  return r < l;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator<=(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool
{
  return !(r < l);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator>=(Vec<D, T> const & l, Vec<D, T> const & r) noexcept -> bool
{
  return !(l < r);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator+(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u += v;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator-(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u -= v;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator*(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u *= v;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator/(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u /= v;
}

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator+(Scalar s, Vec<D, T> u) noexcept -> Vec<D, T>
{
  return u += s;
}

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator+(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>
{
  return u += s;
}

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator-(Scalar s, Vec<D, T> u) noexcept -> Vec<D, T>
{
  Vec<D, T> result;
  for (Int i = 0; i < D; ++i) {
    result[i] = static_cast<T>(s) - u[i];
  }
  return result;
}

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator-(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>
{
  return u -= s;
}

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator*(Scalar s, Vec<D, T> u) noexcept -> Vec<D, T>
{
  return u *= s;
}

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator*(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>
{
  return u *= s;
}

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator/(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>
{
  return u /= s;
}

template <Int D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
PURE HOSTDEV constexpr auto
operator/(Scalar s, Vec<D, T> const & u) noexcept -> Vec<D, T>
{
  Vec<D, T> result;
  for (Int i = 0; i < D; ++i) {
    result[i] = static_cast<T>(s) / u[i];
  }
  return result;
}

} // namespace um2
