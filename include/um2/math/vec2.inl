namespace um2
{

// -- Unary operators --

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator-(Vec<2, T> v) -> Vec<2, T> 
{
  v.x = -v.x;
  v.y = -v.y;
  return v;
}

//// -- Binary operators --
//
//template <typename T>
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator+=(Vec<D, T, Q> & u, Vec<D, T, Q> const & v)
//{
//  u.data += v.data;
//  return u;
//}
//
//template <typename T>
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator-=(Vec<D, T, Q> & u, Vec<D, T, Q> const & v)
//{
//  u.data -= v.data;
//  return u;
//}
//
//template <typename T>
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator*=(Vec<D, T, Q> & u, Vec<D, T, Q> const & v)
//{
//  u.data *= v.data;
//  return u;
//}
//
//template <typename T>
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator/=(Vec<D, T, Q> & u, Vec<D, T, Q> const & v)
//{
//  u.data /= v.data;
//  return u;
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator+(Vec<D, T, Q> u,
//                                                      Vec<D, T, Q> const & v)
//{
//  return u += v;
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator-(Vec<D, T, Q> u,
//                                                      Vec<D, T, Q> const & v)
//{
//  return u -= v;
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator*(Vec<D, T, Q> u,
//                                                      Vec<D, T, Q> const & v)
//{
//  return u *= v;
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator/(Vec<D, T, Q> u,
//                                                      Vec<D, T, Q> const & v)
//{
//  return u /= v;
//}
//
//// -- Scalar operators --
//
//template <typename T, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator+=(Vec<D, T, Q> & u, S const & s)
//{
//  u.data += static_cast<T>(s);
//  return u;
//}
//
//template <typename T, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator-=(Vec<D, T, Q> & u, S const & s)
//{
//  u.data -= static_cast<T>(s);
//  return u;
//}
//
//template <typename T, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator*=(Vec<D, T, Q> & u, S const & s)
//{
//  u.data *= static_cast<T>(s);
//  return u;
//}
//
//template <typename T, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr Vec<D, T, Q> & operator/=(Vec<D, T, Q> & u, S const & s)
//{
//  u.data /= static_cast<T>(s);
//  return u;
//}
//
//template <typename T, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator*(S const & s, Vec<D, T, Q> v)
//{
//  return v *= static_cast<T>(s);
//}
//
//template <typename T, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator*(Vec<D, T, Q> v, S const & s)
//{
//  return v *= static_cast<T>(s);
//}
//
//template <typename T, typename S>
//  requires(std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> operator/(Vec<D, T, Q> v, S const & s)
//{
//  return v /= static_cast<T>(s);
//}
//
//// -- Methods --
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> min(Vec<D, T, Q> u, Vec<D, T, Q> const & v)
//{
//  u.data = min(u.data, v.data);
//  return u;
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> max(Vec<D, T, Q> u, Vec<D, T, Q> const & v)
//{
//  u.data = max(u.data, v.data);
//  return u;
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr T dot(Vec<D, T, Q> const & u, Vec<D, T, Q> const & v)
//{
//  return dot(u.data, v.data);
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr T norm2(Vec<D, T, Q> const & v)
//{
//  return norm2(v.data);
//}
//
//template <std::floating_point T>
//UM2_PURE UM2_HOSTDEV constexpr T norm(Vec<D, T, Q> const & v)
//{
//  return std::sqrt(norm2(v));
//}
//
//template <std::floating_point T>
//UM2_PURE UM2_HOSTDEV constexpr Vec<D, T, Q> normalize(Vec<D, T, Q> const & v)
//{
//  return v / norm(v);
//}
//
//// -- Vec2 --
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr T cross(Vec2<T, Q> const & u, Vec2<T, Q> const & v)
//{
//  return u[0] * v[1] - u[1] * v[0];
//}
//
//// -- Vec3 --
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr Vec3<T, Q> cross(Vec3<T, Q> const & u,
//                                                Vec3<T, Q> const & v)
//{
//  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
//          u[0] * v[1] - u[1] * v[0]};
//}
//
} // namespace um2