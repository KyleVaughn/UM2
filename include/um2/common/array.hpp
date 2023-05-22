#pragma once

#include <um2/common/config.hpp>

#include <cmath>

namespace um2
{

template <len_t L, typename T>
struct Array {
  // -- Implementation --

  using value_type = T;
  using iterator = T *;
  using const_iterator = const T *;

  T data[L];

  // -- Accessors --

  UM2_PURE UM2_HOSTDEV constexpr auto operator[](len_t /*i*/) -> T &;
  UM2_PURE UM2_HOSTDEV constexpr auto operator[](len_t /*i*/) const -> const T &;
};

// -- Accessors --

template <len_t L, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Array<L, T>::operator[](len_t const i) -> T &
{
  return this->data[i];
}

template <len_t L, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Array<L, T>::operator[](len_t const i) const
    -> T const &
{
  return this->data[i];
}

// -- Unary operators --

template <len_t L, typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator-(Array<L, T> a) -> Array<L, T>
{
  for (len_t i = 0; i < L; ++i) {
    a[i] = -a[i];
  }
  return a;
}

// -- Binary operators --

template <len_t L, typename T>
UM2_HOSTDEV constexpr auto operator+=(Array<L, T> & u, Array<L, T> const & v)
    -> Array<L, T> &
{
  for (len_t i = 0; i < L; ++i) {
    u[i] += v[i];
  }
  return u;
}

template <len_t L, typename T>
UM2_HOSTDEV constexpr auto operator-=(Array<L, T> & u, Array<L, T> const & v)
    -> Array<L, T> &
{
  for (len_t i = 0; i < L; ++i) {
    u[i] -= v[i];
  }
  return u;
}

template <len_t L, typename T>
UM2_HOSTDEV constexpr auto operator*=(Array<L, T> & u, Array<L, T> const & v)
    -> Array<L, T> &
{
  for (len_t i = 0; i < L; ++i) {
    u[i] *= v[i];
  }
  return u;
}

template <len_t L, typename T>
UM2_HOSTDEV constexpr auto operator/=(Array<L, T> & u, Array<L, T> const & v)
    -> Array<L, T> &
{
  for (len_t i = 0; i < L; ++i) {
    u[i] /= v[i];
  }
  return u;
}

template <len_t L, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator+(Array<L, T> u, Array<L, T> const & v)
    -> Array<L, T>
{
  u += v;
  return u;
}

template <len_t L, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator-(Array<L, T> u, Array<L, T> const & v)
    -> Array<L, T>
{
  u -= v;
  return u;
}

template <len_t L, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator*(Array<L, T> u, Array<L, T> const & v)
    -> Array<L, T>
{
  u *= v;
  return u;
}

template <len_t L, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator/(Array<L, T> u, Array<L, T> const & v)
    -> Array<L, T>
{
  u /= v;
  return u;
}

// -- Scalar operators --

template <len_t L, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator+=(Array<L, T> & u, S const & s) -> Array<L, T> &
{
  for (len_t i = 0; i < L; ++i) {
    u[i] += static_cast<T>(s);
  }
  return u;
}

template <len_t L, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator-=(Array<L, T> & u, S const & s) -> Array<L, T> &
{
  for (len_t i = 0; i < L; ++i) {
    u[i] -= static_cast<T>(s);
  }
  return u;
}

template <len_t L, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator*=(Array<L, T> & u, S const & s) -> Array<L, T> &
{
  for (len_t i = 0; i < L; ++i) {
    u[i] *= static_cast<T>(s);
  }
  return u;
}

template <len_t L, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator/=(Array<L, T> & u, S const & s) -> Array<L, T> &
{
  for (len_t i = 0; i < L; ++i) {
    u[i] /= static_cast<T>(s);
  }
  return u;
}

template <len_t L, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator*(S const & s, Array<L, T> u) -> Array<L, T>
{
  return u *= static_cast<T>(s);
}

template <len_t L, typename T, typename S>
requires(std::same_as<T, S> ||
         (std::floating_point<T> && std::integral<S>)) UM2_PURE UM2_HOSTDEV constexpr auto
operator*(Array<L, T> u, S const & s) -> Array<L, T>
{
  return u *= static_cast<T>(s);
}

template <len_t L, typename T, typename S>
requires(std::same_as<T, S> ||
         (std::floating_point<T> && std::integral<S>)) UM2_PURE UM2_HOSTDEV constexpr auto
operator/(Array<L, T> u, S const & s) -> Array<L, T>
{
  return u /= static_cast<T>(s);
}

// -- Methods --

template <len_t L, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto min(Array<L, T> u, Array<L, T> const & v)
    -> Array<L, T>
{
  for (len_t i = 0; i < L; ++i) {
    u[i] = std::min(u[i], v[i]);
  }
  return u;
}

template <len_t L, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto max(Array<L, T> u, Array<L, T> const & v)
    -> Array<L, T>
{
  for (len_t i = 0; i < L; ++i) {
    u[i] = std::max(u[i], v[i]);
  }
  return u;
}

template <len_t L, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto dot(Array<L, T> const & u, Array<L, T> const & v) -> T
{
  T r = 0;
  for (len_t i = 0; i < L; ++i) {
    r += u[i] * v[i];
  }
  return r;
}

template <len_t L, typename T>
UM2_CONST UM2_HOSTDEV constexpr auto norm2(Array<L, T> u) -> T
{
  u *= u;
  T r = 0;
  for (len_t i = 0; i < L; ++i) {
    r += u[i];
  }
  return r;
}

} // namespace um2
