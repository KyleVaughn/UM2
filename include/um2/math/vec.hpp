#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/algorithm/max.hpp>
#include <um2/stdlib/algorithm/min.hpp>
#include <um2/stdlib/math/roots.hpp>
#include <um2/common/cast_if_not.hpp>

#include <concepts>

// We alias Vec<D, Float> to Point<D> for D-dimensional points. Hence, we need to
// define some constants for points in this file.

//==============================================================================    
// Constants    
//==============================================================================    
// eps_distance:    
//   Distance between two points, below which they are considered to be equal.    
// eps_distance2:    
//   Squared distance between two points, below which they are considered to be    
//   equal.    
// inf_distance:    
//  Distance between two points, above which they are considered to be
//  infinitely far apart. Typically used for invalid points and values.
//
// NOTE: fast-math assumes no infinities, so we need inf_distance to be finite.

namespace um2
{

// NOTE: If you change these 3 values, you had better find every instance
// of them in the code and tests and make sure the values are appropriate.
inline constexpr Float eps_distance = castIfNot<Float>(1e-6); // 0.1 micron
inline constexpr Float eps_distance2 = castIfNot<Float>(1e-12);
inline constexpr Float inf_distance = castIfNot<Float>(1e8); // 1000 km

} // namespace um2

//==============================================================================
// VEC
//==============================================================================
// A D-dimensional vector with data of type T.
//
// if UM2_ENABLE_SIMD_VEC, then if D is a power of 2 and T is an arithmetic type,
// we use GCC's vector extensions to store the data. This allows for automatic
// vectorization of operations.

namespace um2
{

#if UM2_ENABLE_SIMD_VEC

static consteval auto
isPowerOf2(Int x) noexcept -> bool
{
  return (x & (x - 1)) == 0;
};

template <Int D, class T>
concept is_simd_vec = isPowerOf2(D) && std::is_arithmetic_v<T>;

#else

template <Int D, class T>
concept is_simd_vec = false;

#endif

template <Int D, class T, bool B = is_simd_vec<D, T>>
struct VecData;

template <Int D, class T>
struct VecData<D, T, false>
{
  using Data = T[D]; 
};

template <Int D, class T>
struct VecData<D, T, true>
{
  // If we use "using" we get "ignoring attributes applied to dependent type 'T' without an 
  // associated declaration" warning. So we use typedef instead. Then we silence clang-tidy
  // trying to tell us to use "using" instead of "typedef".
  // NOLINTNEXTLINE(modernize-use-using,readability-identifier-naming)
  typedef T Data __attribute__((vector_size(D * sizeof(T))));
//  using Data = T __attribute__((vector_size(D * sizeof(T))));
};

template <Int D, class T>
class Vec
{

  using Data = typename VecData<D, T>::Data;
  Data _data;

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
           !(std::same_as<T, Is> && ...)) HOSTDEV
      constexpr Vec(Is const... args) noexcept;

  template <class... Ts>
  requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...)) HOSTDEV
      constexpr Vec(Ts const... args) noexcept;
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
  requires(std::same_as<T, S> || std::integral<S>) HOSTDEV constexpr auto
  operator+=(S const & s) noexcept -> Vec<D, T> &;

  template <class S>
  requires(std::same_as<T, S> || std::integral<S>) HOSTDEV constexpr auto
  operator-=(S const & s) noexcept -> Vec<D, T> &;

  template <class S>
  requires(std::same_as<T, S> || std::integral<S>) HOSTDEV constexpr auto
  operator*=(S const & s) noexcept -> Vec<D, T> &;

  template <class S>
  requires(std::same_as<T, S> || std::integral<S>) HOSTDEV constexpr auto
  operator/=(S const & s) noexcept -> Vec<D, T> &;

  //==============================================================================
  // Other member functions
  //==============================================================================

  HOSTDEV [[nodiscard]] static constexpr auto
  isSIMD() noexcept -> bool;

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
  isApprox(Vec<D, T> const & v, T const & eps2 = eps_distance2) const noexcept -> bool
  requires(std::is_floating_point_v<T>);

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

using Vec2F = Vec2<Float>;
using Vec3F = Vec3<Float>;

using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;

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
  if constexpr (is_simd_vec<D, T>) {
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
  if constexpr (is_simd_vec<D, T>) {
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
         !(std::same_as<T, Is> && ...)) HOSTDEV
    constexpr Vec<D, T>::Vec(Is const... args) noexcept
    : _data{static_cast<T>(args)...}
{
}

template <Int D, class T>
template <class... Ts>
requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...)) HOSTDEV
    constexpr Vec<D, T>::Vec(Ts const... args) noexcept
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
  if constexpr (is_simd_vec<D, T>) {
    return -_data;
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
  if constexpr (is_simd_vec<D, T>) {
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
  if constexpr (is_simd_vec<D, T>) {
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
  if constexpr (is_simd_vec<D, T>) {
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
  if constexpr (is_simd_vec<D, T>) {
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
requires(std::same_as<T, S> || std::integral<S>) HOSTDEV
    constexpr auto Vec<D, T>::operator+=(S const & s) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T>) {
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
requires(std::same_as<T, S> || std::integral<S>) HOSTDEV
    constexpr auto Vec<D, T>::operator-=(S const & s) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T>) {
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
requires(std::same_as<T, S> || std::integral<S>) HOSTDEV
    constexpr auto Vec<D, T>::operator*=(S const & s) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T>) {
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
requires(std::same_as<T, S> || std::integral<S>) HOSTDEV
    constexpr auto Vec<D, T>::operator/=(S const & s) noexcept -> Vec<D, T> &
{
  if constexpr (is_simd_vec<D, T>) {
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
HOSTDEV constexpr auto
Vec<D, T>::isSIMD() noexcept -> bool
{
  return is_simd_vec<D, T>;
}

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
  static_assert(std::is_floating_point_v<T>);
  return um2::sqrt(squaredNorm());
}

template <Int D, class T>
HOSTDEV constexpr void
Vec<D, T>::normalize() noexcept
{
  static_assert(std::is_floating_point_v<T>);
  *this /= norm();
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::normalized() const noexcept -> Vec<D, T>
{
  static_assert(std::is_floating_point_v<T>);
  Vec<D, T> result = *this;
  result.normalize();
  return result;
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::cross(Vec<2, T> const & v) const noexcept -> T
requires(D == 2)
{
  static_assert(std::is_floating_point_v<T>);
  return _data[0] * v[1] - _data[1] * v[0];
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::cross(Vec<3, T> const & v) const noexcept -> Vec<3, T>
requires(D == 3)
{
  static_assert(std::is_floating_point_v<T>);
  return {(_data[1] * v[2]) - (_data[2] * v[1]),
          (_data[2] * v[0]) - (_data[0] * v[2]),
          (_data[0] * v[1]) - (_data[1] * v[0])};
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
  static_assert(std::is_floating_point_v<T>);
  return um2::sqrt(squaredDistanceTo(v));
}

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::isApprox(Vec<D, T> const & v, T const & eps2) const noexcept -> bool
requires(std::is_floating_point_v<T>)
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
  static_assert(std::is_floating_point_v<T>);
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
  static_assert(std::is_floating_point_v<T>);
  return u.cross(v);
}

template <class T>
PURE HOSTDEV constexpr auto
cross(Vec3<T> const & u, Vec3<T> const & v) noexcept -> Vec3<T>
{
  static_assert(std::is_floating_point_v<T>);
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
