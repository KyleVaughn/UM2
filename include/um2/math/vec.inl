#include <cassert>
namespace um2
{

// -- Accessors --

template <len_t D, typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto Vec<D, T>::operator[](len_t const i) -> T &
{
  assert(0 <= i && i < D);
  return this->data[i];
}

template <len_t D, typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto Vec<D, T>::operator[](len_t const i) const
    -> T const &
{
  assert(0 <= i && i < D);
  return this->data[i];
}

// -- Constructors --

template <len_t D, typename T>
template <typename... Is>
requires(sizeof...(Is) == D && (std::integral<Is> && ...) &&
         !(std::same_as<T, Is> && ...)) UM2_HOSTDEV
    constexpr Vec<D, T>::Vec(Is const... args)
    : data{static_cast<T>(args)...}
{
}

template <len_t D, typename T>
template <typename... Ts>
requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...)) UM2_HOSTDEV
    constexpr Vec<D, T>::Vec(Ts const... args)
    : data{args...}
{
}

// -- IO --

template <len_t D, typename T>
auto operator<<(std::ostream & os, Vec<D, T> const & v) -> std::ostream &
{
  os << '(';
  for (len_t i = 0; i < D; ++i) {
    os << v[i];
    if (i < D - 1) {
      os << ", ";
    }
  }
  os << ')';
  return os;
}

// -- Unary operators --

template <len_t D, typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator-(Vec<D, T> v) -> Vec<D, T>
{
  v.data = -v.data;
  return v;
}

// -- Binary operators --

template <len_t D, typename T>
UM2_HOSTDEV constexpr auto operator+=(Vec<D, T> & u, Vec<D, T> const & v) -> Vec<D, T> &
{
  u.data += v.data;
  return u;
}

template <len_t D, typename T>
UM2_HOSTDEV constexpr auto operator-=(Vec<D, T> & u, Vec<D, T> const & v) -> Vec<D, T> &
{
  u.data -= v.data;
  return u;
}

template <len_t D, typename T>
UM2_HOSTDEV constexpr auto operator*=(Vec<D, T> & u, Vec<D, T> const & v) -> Vec<D, T> &
{
  u.data *= v.data;
  return u;
}

template <len_t D, typename T>
UM2_HOSTDEV constexpr auto operator/=(Vec<D, T> & u, Vec<D, T> const & v) -> Vec<D, T> &
{
  u.data /= v.data;
  return u;
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator+(Vec<D, T> u, Vec<D, T> const & v)
    -> Vec<D, T>
{
  return u += v;
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator-(Vec<D, T> u, Vec<D, T> const & v)
    -> Vec<D, T>
{
  return u -= v;
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator*(Vec<D, T> u, Vec<D, T> const & v)
    -> Vec<D, T>
{
  return u *= v;
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator/(Vec<D, T> u, Vec<D, T> const & v)
    -> Vec<D, T>
{
  return u /= v;
}

// -- Scalar operators --

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator+=(Vec<D, T> & u, S const & s) -> Vec<D, T> &
{
  u.data += static_cast<T>(s);
  return u;
}

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator-=(Vec<D, T> & u, S const & s) -> Vec<D, T> &
{
  u.data -= static_cast<T>(s);
  return u;
}

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator*=(Vec<D, T> & u, S const & s) -> Vec<D, T> &
{
  u.data *= static_cast<T>(s);
  return u;
}

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>)) UM2_HOSTDEV
    constexpr auto
    operator/=(Vec<D, T> & u, S const & s) -> Vec<D, T> &
{
  u.data /= static_cast<T>(s);
  return u;
}

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> ||
         (std::floating_point<T> && std::integral<S>)) UM2_PURE UM2_HOSTDEV constexpr auto
operator*(S const & s, Vec<D, T> v) -> Vec<D, T>
{
  return v *= static_cast<T>(s);
}

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> ||
         (std::floating_point<T> && std::integral<S>)) UM2_PURE UM2_HOSTDEV constexpr auto
operator*(Vec<D, T> v, S const & s) -> Vec<D, T>
{
  return v *= static_cast<T>(s);
}

template <len_t D, typename T, typename S>
requires(std::same_as<T, S> ||
         (std::floating_point<T> && std::integral<S>)) UM2_PURE UM2_HOSTDEV constexpr auto
operator/(Vec<D, T> v, S const & s) -> Vec<D, T>
{
  return v /= static_cast<T>(s);
}

// -- Methods --

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto min(Vec<D, T> u, Vec<D, T> const & v) -> Vec<D, T>
{
  u.data = min(u.data, v.data);
  return u;
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto max(Vec<D, T> u, Vec<D, T> const & v) -> Vec<D, T>
{
  u.data = max(u.data, v.data);
  return u;
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto dot(Vec<D, T> const & u, Vec<D, T> const & v) -> T
{
  return dot(u.data, v.data);
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto norm2(Vec<D, T> const & v) -> T
{
  return norm2(v.data);
}

template <len_t D, std::floating_point T>
UM2_PURE UM2_HOSTDEV constexpr auto norm(Vec<D, T> const & v) -> T
{
  return std::sqrt(norm2(v));
}

template <len_t D, std::floating_point T>
UM2_PURE UM2_HOSTDEV constexpr auto normalize(Vec<D, T> const & v) -> Vec<D, T>
{
  return v / norm(v);
}

// -- vec2 --

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto cross(vec2<T> const & u, vec2<T> const & v) -> T
{
  return u[0] * v[1] - u[1] * v[0];
}

// -- vec3 --

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto cross(vec3<T> const & u, vec3<T> const & v) -> vec3<T>
{
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0]};
}

} // namespace um2
