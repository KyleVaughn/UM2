#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/algorithm/max.hpp>
#include <um2/stdlib/algorithm/min.hpp>
#include <um2/stdlib/math/roots.hpp>

#include <type_traits> // std::is_arithmetic_v

//==============================================================================
// SIMD
//==============================================================================
//
// For developers:
// To wrap or not to wrap? That is the question.
//
// If we don't wrap the GCC vector extension in a class, the compiler is 
// very bad at deducing the size of the SIMD type. It will sometimes silently 
// deduce a bad size, leading to subtle bugs. However, if we wrap the
// GCC vector extension in a class, then the operator[] cannot be used to mutate
// the data without some trouble. The compiler will complain that a "non-const reference 
// cannot bind to vector element". Therefore, we have to either use a union or 
// reinterpret_cast to get a mutable reference to the data. 
//
// If we use a union, the compiler tends to throw away all the SIMD instructions.
// But, if we use reinterpret_cast, then we have to be careful about aliasing.
//
// The best solution is to use a class and reinterpret_cast.

static consteval auto
isPowerOf2(Int x) noexcept -> bool
{
  return (x & (x - 1)) == 0; 
};

namespace um2
{

template <Int D, class T>
class SIMD
{
  static_assert(std::is_arithmetic_v<T>);
  static_assert(isPowerOf2(D));

  using Data = T __attribute__((vector_size(D * sizeof(T))));

  Data _data;

  //==============================================================================
  // Private functions
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getPointer() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getConstPointer() const noexcept -> T const *;

public:

  //==============================================================================    
  // Element Access    
  //==============================================================================    
    
  PURE HOSTDEV constexpr auto    
  operator[](Int i) noexcept -> T &;    
    
  PURE HOSTDEV constexpr auto    
  operator[](Int i) const noexcept -> T const &;

  //==============================================================================    
  // Constructors    
  //==============================================================================    
    
  constexpr SIMD() noexcept = default;    
    
  // Allow implicit conversion from integral types.    
  // Otherwise, require explicit conversion to avoid accidental loss of    
  // precision/performance.    
  // NOLINTBEGIN(google-explicit-constructor)    
  template <class... Is>    
  requires(sizeof...(Is) == D && (std::integral<Is> && ...) &&    
           !(std::same_as<T, Is> && ...)) HOSTDEV    
      constexpr SIMD(Is const... args) noexcept;    
    
  template <class... Ts>
  requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...)) HOSTDEV
      constexpr SIMD(Ts const... args) noexcept;
  // NOLINTEND(google-explicit-constructor)

  //==============================================================================
  // Unary operators
  //==============================================================================

  HOSTDEV constexpr auto
  operator-() const noexcept -> SIMD<D, T>;

  //==============================================================================
  // Binary operators
  //==============================================================================

  // Element-wise operators with SIMDs
  HOSTDEV constexpr auto
  operator+=(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &;

  HOSTDEV constexpr auto
  operator-=(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &;

  HOSTDEV constexpr auto
  operator*=(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &;

  HOSTDEV constexpr auto
  operator/=(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &;

  // Element-wise operators with scalars
  // Require that the scalar type is either the same as the vector type or an
  // integral type.
  template <class S>
  requires(std::same_as<T, S> || std::integral<S>) HOSTDEV constexpr auto
  operator+=(S const & s) noexcept -> SIMD<D, T> &;

  template <class S>
  requires(std::same_as<T, S> || std::integral<S>) HOSTDEV constexpr auto
  operator-=(S const & s) noexcept -> SIMD<D, T> &;

  template <class S>
  requires(std::same_as<T, S> || std::integral<S>) HOSTDEV constexpr auto
  operator*=(S const & s) noexcept -> SIMD<D, T> &;

  template <class S>
  requires(std::same_as<T, S> || std::integral<S>) HOSTDEV constexpr auto
  operator/=(S const & s) noexcept -> SIMD<D, T> &;

  //==============================================================================
  // Other member functions
  //==============================================================================

  HOSTDEV [[nodiscard]] constexpr auto
  min(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &;

  HOSTDEV [[nodiscard]] constexpr auto
  max(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &;

  HOSTDEV [[nodiscard]] constexpr auto
  dot(SIMD<D, T> const & v) const noexcept -> T;

  HOSTDEV [[nodiscard]] constexpr auto
  squaredNorm() const noexcept -> T;

  HOSTDEV [[nodiscard]] constexpr auto
  norm() const noexcept -> T;

  HOSTDEV constexpr void 
  normalize() noexcept;

  HOSTDEV [[nodiscard]] constexpr auto
  normalized() const noexcept -> SIMD<D, T>;

  HOSTDEV [[nodiscard]] constexpr auto
  cross(SIMD<2, T> const & v) const noexcept -> T
  requires(D == 2);

}; // class SIMD

//==============================================================================
// Free functions
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
min(SIMD<D, T> u, SIMD<D, T> const & v) noexcept -> SIMD<D, T>;

template <Int D, class T>
PURE HOSTDEV constexpr auto
max(SIMD<D, T> u, SIMD<D, T> const & v) noexcept -> SIMD<D, T>;

template <Int D, class T>
PURE HOSTDEV constexpr auto
dot(SIMD<D, T> const & u, SIMD<D, T> const & v) noexcept -> T;

template <Int D, class T>
PURE HOSTDEV constexpr auto
squaredNorm(SIMD<D, T> const & v) noexcept -> T;

template <Int D, class T>
PURE HOSTDEV constexpr auto
norm(SIMD<D, T> const & v) noexcept -> T;

template <Int D, class T>
PURE HOSTDEV constexpr auto
normalized(SIMD<D, T> v) noexcept -> SIMD<D, T>;

//==============================================================================
// Private functions
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
SIMD<D, T>::getPointer() noexcept -> T *
{
  return reinterpret_cast<T *>(&_data);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
SIMD<D, T>::getConstPointer() const noexcept -> T const *
{
  return reinterpret_cast<T const *>(&_data);
}

//==============================================================================
// Element Access
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
SIMD<D, T>::operator[](Int i) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return getPointer()[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
SIMD<D, T>::operator[](Int i) const noexcept -> T const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return getConstPointer()[i];
}

//==============================================================================
// Constructors
//==============================================================================

template <Int D, class T>    
template <class... Is>    
requires(sizeof...(Is) == D && (std::integral<Is> && ...) &&    
         !(std::same_as<T, Is> && ...)) HOSTDEV    
    constexpr SIMD<D, T>::SIMD(Is const... args) noexcept    
    : _data{static_cast<T>(args)...}    
{    
}    
    
template <Int D, class T>    
template <class... Ts>    
requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...)) HOSTDEV    
    constexpr SIMD<D, T>::SIMD(Ts const... args) noexcept    
    : _data{args...}    
{    
}

//==============================================================================
// Unary operators
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::operator-() const noexcept -> SIMD<D, T>
{
  return -_data;
}

//==============================================================================
// Binary operators
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::operator+=(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &
{
  _data += v._data;
  return *this;
}

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::operator-=(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &
{
  _data -= v._data;
  return *this;
}

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::operator*=(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &
{
  _data *= v._data;
  return *this;
}

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::operator/=(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &
{
  _data /= v._data;
  return *this;
}

template <Int D, class T>
template <class S>
requires(std::same_as<T, S> || std::integral<S>) HOSTDEV
    constexpr auto SIMD<D, T>::operator+=(S const & s) noexcept -> SIMD<D, T> &
{
  _data += static_cast<T>(s);
  return *this;
}

template <Int D, class T>
template <class S>
requires(std::same_as<T, S> || std::integral<S>) HOSTDEV
    constexpr auto SIMD<D, T>::operator-=(S const & s) noexcept -> SIMD<D, T> &
{
  _data -= static_cast<T>(s);
  return *this;
}

template <Int D, class T>
template <class S>
requires(std::same_as<T, S> || std::integral<S>) HOSTDEV
    constexpr auto SIMD<D, T>::operator*=(S const & s) noexcept -> SIMD<D, T> &
{
  _data *= static_cast<T>(s);
  return *this;
}

template <Int D, class T>
template <class S>
requires(std::same_as<T, S> || std::integral<S>) HOSTDEV
    constexpr auto SIMD<D, T>::operator/=(S const & s) noexcept -> SIMD<D, T> &
{
  _data /= static_cast<T>(s);
  return *this;
}

//==============================================================================
// Other member functions
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::min(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &
{
  // This gets optimized to a single instruction.
  for (Int i = 0; i < D; ++i) {
    _data[i] = um2::min(_data[i], v._data[i]);
  }
  return *this;
}

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::max(SIMD<D, T> const & v) noexcept -> SIMD<D, T> &
{
  // This gets optimized to a single instruction.
  for (Int i = 0; i < D; ++i) {
    _data[i] = um2::max(_data[i], v._data[i]);
  }
  return *this;
}

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::dot(SIMD<D, T> const & v) const noexcept -> T
{
  T result = _data[0] * v._data[0];
  for (Int i = 1; i < D; ++i) {
    result += _data[i] * v._data[i];
  }
  return result;
}

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::squaredNorm() const noexcept -> T
{
  T result = _data[0] * _data[0];
  for (Int i = 1; i < D; ++i) {
    result += _data[i] * _data[i];
  }
  return result;
}

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::norm() const noexcept -> T
{
  static_assert(std::is_floating_point_v<T>);
  return um2::sqrt(squaredNorm());
}

template <Int D, class T>
HOSTDEV constexpr void
SIMD<D, T>::normalize() noexcept
{
  static_assert(std::is_floating_point_v<T>);
  T const n = norm();
  ASSERT(n > 0);
  *this /= n;
}

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::normalized() const noexcept -> SIMD<D, T>
{
  static_assert(std::is_floating_point_v<T>);
  SIMD<D, T> result = *this;
  result.normalize();
  return result;
}

template <Int D, class T>
HOSTDEV constexpr auto
SIMD<D, T>::cross(SIMD<2, T> const & v) const noexcept -> T
requires(D == 2)
{
  static_assert(std::is_floating_point_v<T>);
  return _data[0] * v[1] - _data[1] * v[0];
}

//////==============================================================================
////// Non-member operators
//////==============================================================================
////
////template <Int D, std::integral T>
////PURE HOSTDEV constexpr auto
////operator==(SIMD<D, T> const & u, SIMD<D, T> const & v) noexcept -> bool
////{
////  for (Int i = 0; i < D; ++i) {
////    if (u[i] != v[i]) {
////      return false;
////    }
////  }
////  return true;
////}
////
////template <Int D, std::integral T>
////PURE HOSTDEV constexpr auto
////operator!=(SIMD<D, T> const & u, SIMD<D, T> const & v) noexcept -> bool
////{
////  return !(u == v);
////}
////

////template <Int D, class T>
////HOSTDEV [[nodiscard]] constexpr auto
////SIMD<D, T>::zero() noexcept -> SIMD<D, T>
////{
////  return SIMD<D, T>{}; // Zero-initialize.
////}
////

////template <Int D, class T>
////PURE HOSTDEV constexpr auto
////SIMD<D, T>::squaredDistanceTo(SIMD<D, T> const & v) const noexcept -> T
////{
////  T const d0 = _data[0] - v[0];
////  T result = d0 * d0;
////  for (Int i = 1; i < D; ++i) {
////    T const di = _data[i] - v[i];
////    result += di * di;
////  }
////  return result;
////}
////
////template <Int D, class T>
////PURE HOSTDEV constexpr auto
////SIMD<D, T>::distanceTo(SIMD<D, T> const & v) const noexcept -> T
////{
////  static_assert(std::is_floating_point_v<T>);
////  return um2::sqrt(squaredDistanceTo(v));
////}
////
template <Int D, class T>
PURE HOSTDEV constexpr auto
min(SIMD<D, T> u, SIMD<D, T> const & v) noexcept -> SIMD<D, T>
{
  return u.min(v);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
max(SIMD<D, T> u, SIMD<D, T> const & v) noexcept -> SIMD<D, T>
{
  return u.max(v);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
dot(SIMD<D, T> const & u, SIMD<D, T> const & v) noexcept -> T
{
  return u.dot(v);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
squaredNorm(SIMD<D, T> const & v) noexcept -> T
{
  return v.squaredNorm();
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
norm(SIMD<D, T> const & v) noexcept -> T
{
  static_assert(std::is_floating_point_v<T>);
  return v.norm();
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
normalized(SIMD<D, T> v) noexcept -> SIMD<D, T>
{
  static_assert(std::is_floating_point_v<T>);
  v.normalize();
  return v;
}

template <class T>
PURE HOSTDEV constexpr auto
cross(SIMD<2, T> const & u, SIMD<2, T> const & v) noexcept -> T
{
  static_assert(std::is_floating_point_v<T>);
  return u.cross(v);
}

} // namespace um2
