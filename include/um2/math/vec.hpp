#pragma once

#include <um2/stdlib/algorithm.hpp> // um2::min, um2::max
#include <um2/stdlib/math.hpp>      // um2::sqrt
#include <um2/stdlib/memory.hpp>    // addressof

//==============================================================================
// VEC
//==============================================================================
// A D-dimensional vector with data of type T.
//
// This class is used for small vectors, where the number of elements is known
// at compile time. The Vec is not aligned, so the compiler does not typically
// emit vectorized instructions for it. Also, be careful about an accidental
// loss of performance through the creation of temporaries. In other words,
// operations like a = b + c + 4 * d are not efficient and should be done element
// by element in a loop. It is possible to use expression templates to avoid
// such issues, but common solutions like Eigen don't play well with CUDA.

namespace um2
{

template <I D, class T>
class Vec
{

  T _data[D];

public:
  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV constexpr auto
  operator[](I i) noexcept -> T &;

  PURE HOSTDEV constexpr auto
  operator[](I i) const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() const noexcept -> T const *;

  CONST HOSTDEV [[nodiscard]] static constexpr auto
  size() noexcept -> I;

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

  // The zero vector.
  HOSTDEV [[nodiscard]] static constexpr auto
  zero() noexcept -> Vec<D, T>;

  // Element-wise minimum.
  HOSTDEV constexpr auto
  min(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  // Element-wise maximum.
  HOSTDEV constexpr auto
  max(Vec<D, T> const & v) noexcept -> Vec<D, T> &;

  // Dot product.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  dot(Vec<D, T> const & v) const noexcept -> T;

  // 2-norm squared.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredNorm() const noexcept -> T;

  // 2-norm.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  norm() const noexcept -> T;

  // Normalize the vector in place.
  HOSTDEV constexpr void
  normalize() noexcept;

  // Return a normalized copy of the vector.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  normalized() const noexcept -> Vec<D, T>;

  // Return the z-component of two planar vectors.
  // (v0.x * v1.y) - (v0.y * v1.x)
  PURE HOSTDEV [[nodiscard]] constexpr auto
  cross(Vec<2, T> const & v) const noexcept -> T;

  // Cross product.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  cross(Vec<3, T> const & v) const noexcept -> Vec<3, T>;

  // Squared distance to another vector.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredDistanceTo(Vec<D, T> const & v) const noexcept -> T;

  // Distance to another vector.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  distanceTo(Vec<D, T> const & v) const noexcept -> T;

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

using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;

//==============================================================================
// Accessors
//==============================================================================

template <I D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::operator[](I i) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _data[i];
}

template <I D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::operator[](I i) const noexcept -> T const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _data[i];
}

template <I D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::begin() noexcept -> T *
{
  return addressof(_data[0]);
}

template <I D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::begin() const noexcept -> T const *
{
  return addressof(_data[0]);
}

template <I D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::end() noexcept -> T *
{
  return addressof(_data[0]) + D;
}

template <I D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::end() const noexcept -> T const *
{
  return addressof(_data[0]) + D;
}

template <I D, class T>
CONST HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::size() noexcept -> I
{
  return D;
}

//==============================================================================
// Constructors
//==============================================================================

template <I D, class T>
template <class... Is>
  requires(sizeof...(Is) == D && (std::integral<Is> && ...) &&
           !(std::same_as<T, Is> && ...))
HOSTDEV constexpr Vec<D, T>::Vec(Is const... args) noexcept
    : _data{static_cast<T>(args)...}
{
}

template <I D, class T>
template <class... Ts>
  requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...))
HOSTDEV constexpr Vec<D, T>::Vec(Ts const... args) noexcept
    : _data{args...}
{
}

//==============================================================================
// Unary operators
//==============================================================================

template <I D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator-() const noexcept -> Vec<D, T>
{
  Vec<D, T> result;
  for (I i = 0; i < D; ++i) {
    result[i] = -_data[i];
  }
  return result;
}

//==============================================================================
// Binary operators
//==============================================================================

template <I D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator+=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (I i = 0; i < D; ++i) {
    _data[i] += v[i];
  }
  return *this;
}

template <I D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator-=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (I i = 0; i < D; ++i) {
    _data[i] -= v[i];
  }
  return *this;
}

template <I D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator*=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (I i = 0; i < D; ++i) {
    _data[i] *= v[i];
  }
  return *this;
}

template <I D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator/=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (I i = 0; i < D; ++i) {
    _data[i] /= v[i];
  }
  return *this;
}

template <I D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto Vec<D, T>::operator+=(S const & s) noexcept -> Vec<D, T> &
{
  for (I i = 0; i < D; ++i) {
    _data[i] += static_cast<T>(s);
  }
  return *this;
}

template <I D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto Vec<D, T>::operator-=(S const & s) noexcept -> Vec<D, T> &
{
  for (I i = 0; i < D; ++i) {
    _data[i] -= static_cast<T>(s);
  }
  return *this;
}

template <I D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto Vec<D, T>::operator*=(S const & s) noexcept -> Vec<D, T> &
{
  for (I i = 0; i < D; ++i) {
    _data[i] *= static_cast<T>(s);
  }
  return *this;
}

template <I D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto Vec<D, T>::operator/=(S const & s) noexcept -> Vec<D, T> &
{
  for (I i = 0; i < D; ++i) {
    _data[i] /= static_cast<T>(s);
  }
  return *this;
}

//==============================================================================
// Non-member operators
//==============================================================================

template <I D, std::integral T>
HOSTDEV constexpr auto
operator==(Vec<D, T> const & u, Vec<D, T> const & v) noexcept -> bool
{
  for (I i = 0; i < D; ++i) {
    if (u[i] != v[i]) {
      return false;
    }
  }
  return true;
}

template <I D, std::integral T>
HOSTDEV constexpr auto
operator!=(Vec<D, T> const & u, Vec<D, T> const & v) noexcept -> bool
{
  return !(u == v);
}

template <I D, class T>
HOSTDEV constexpr auto
operator+(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u += v;
}

template <I D, class T>
HOSTDEV constexpr auto
operator-(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u -= v;
}

template <I D, class T>
HOSTDEV constexpr auto
operator*(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u *= v;
}

template <I D, class T>
HOSTDEV constexpr auto
operator/(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u /= v;
}

template <I D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
HOSTDEV constexpr auto
operator*(Scalar s, Vec<D, T> u) noexcept -> Vec<D, T>
{
  return u *= s;
}

template <I D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
HOSTDEV constexpr auto
operator/(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>
{
  return u /= s;
}

template <I D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
HOSTDEV constexpr auto
operator/(Scalar s, Vec<D, T> const & u) noexcept -> Vec<D, T>
{
  Vec<D, T> result;
  for (I i = 0; i < D; ++i) {
    result[i] = s / u[i];
  }
  return result;
}

//==============================================================================
// Methods
//==============================================================================

template <I D, class T>
HOSTDEV constexpr auto
dot(Vec<D, T> const & u, Vec<D, T> const & v) noexcept -> T
{
  T result = u[0] * v[0];
  for (I i = 1; i < D; ++i) {
    result += u[i] * v[i];
  }
  return result;
}

template <I D, class T>
HOSTDEV constexpr auto
squaredNorm(Vec<D, T> const & v) noexcept -> T
{
  T result = v[0] * v[0];
  for (I i = 1; i < D; ++i) {
    result += v[i] * v[i];
  }
  return result;
}

template <I D, class T>
HOSTDEV constexpr auto
norm(Vec<D, T> const & v) noexcept -> T
{
  static_assert(std::is_floating_point_v<T>);
  return um2::sqrt(squaredNorm(v));
}

template <I D, class T>
PURE HOSTDEV constexpr auto
normalized(Vec<D, T> v) noexcept -> Vec<D, T>
{
  static_assert(std::is_floating_point_v<T>);
  v.normalize();
  return v;
}

template <I D, class T>
HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::zero() noexcept -> Vec<D, T>
{
  return Vec<D, T>{}; // Zero-initialize.
}

template <I D, class T>
HOSTDEV constexpr auto
Vec<D, T>::min(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (I i = 0; i < D; ++i) {
    _data[i] = um2::min(_data[i], v[i]);
  }
  return *this;
}

template <I D, class T>
HOSTDEV constexpr auto
Vec<D, T>::max(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (I i = 0; i < D; ++i) {
    _data[i] = um2::max(_data[i], v[i]);
  }
  return *this;
}

template <I D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::dot(Vec<D, T> const & v) const noexcept -> T
{
  return um2::dot(*this, v);
}

template <I D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::squaredNorm() const noexcept -> T
{
  return um2::squaredNorm(*this);
}

template <I D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::norm() const noexcept -> T
{
  return um2::norm(*this);
}

template <I D, class T>
HOSTDEV constexpr void
Vec<D, T>::normalize() noexcept
{
  static_assert(std::is_floating_point_v<T>);
  *this /= norm();
}

template <I D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::normalized() const noexcept -> Vec<D, T>
{
  return um2::normalized(*this);
}

template <I D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::cross(Vec2<T> const & v) const noexcept -> T
{
  static_assert(D == 2);
  static_assert(std::is_floating_point_v<T>);
  return _data[0] * v[1] - _data[1] * v[0];
}

template <I D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::cross(Vec3<T> const & v) const noexcept -> Vec3<T>
{
  static_assert(D == 3);
  static_assert(std::is_floating_point_v<T>);
  return {_data[1] * v[2] - _data[2] * v[1], _data[2] * v[0] - _data[0] * v[2],
          _data[0] * v[1] - _data[1] * v[0]};
}

template <I D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::squaredDistanceTo(Vec<D, T> const & v) const noexcept -> T
{
  T const d0 = _data[0] - v[0];
  T result = d0 * d0;
  for (I i = 1; i < D; ++i) {
    T const di = _data[i] - v[i];
    result += di * di;
  }
  return result;
}

template <I D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::distanceTo(Vec<D, T> const & v) const noexcept -> T
{
  static_assert(std::is_floating_point_v<T>);
  return um2::sqrt(squaredDistanceTo(v));
}

template <I D, class T>
HOSTDEV constexpr auto
min(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u.min(v);
}

template <I D, class T>
HOSTDEV constexpr auto
max(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u.max(v);
}

} // namespace um2
