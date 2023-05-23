namespace um2
{

// -----------------------------------------------------------------------------
// Unary operators
// -----------------------------------------------------------------------------

template <typename T>
requires(std::floating_point<T> || std::signed_integral<T>) UM2_CONST UM2_HOSTDEV
    constexpr auto
    operator-(Vec<2, T> v) -> Vec<2, T>
{
  v.x = -v.x;
  v.y = -v.y;
  return v;
}

// -----------------------------------------------------------------------------
// Binary operators
// -----------------------------------------------------------------------------

template <typename T>
UM2_HOSTDEV constexpr auto operator+=(Vec2<T> & u, Vec2<T> const & v) -> Vec2<T> &
{
  u.x += v.x;
  u.y += v.y;
  return u;
}

template <typename T>
UM2_HOSTDEV constexpr auto operator-=(Vec2<T> & u, Vec2<T> const & v) -> Vec2<T> &
{
  u.x -= v.x;
  u.y -= v.y;
  return u;
}

template <typename T>
UM2_HOSTDEV constexpr auto operator*=(Vec2<T> & u, Vec2<T> const & v) -> Vec2<T> &
{
  u.x *= v.x;
  u.y *= v.y;
  return u;
}

template <typename T>
UM2_HOSTDEV constexpr auto operator/=(Vec2<T> & u, Vec2<T> const & v) -> Vec2<T> &
{
  u.x /= v.x;
  u.y /= v.y;
  return u;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator+(Vec2<T> u, Vec2<T> const & v) -> Vec2<T>
{
  return u += v;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator-(Vec2<T> u, Vec2<T> const & v) -> Vec2<T>
{
  return u -= v;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator*(Vec2<T> u, Vec2<T> const & v) -> Vec2<T>
{
  return u *= v;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator/(Vec2<T> u, Vec2<T> const & v) -> Vec2<T>
{
  return u /= v;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator+=(Vec2<T> & u, S const & s) -> Vec2<T> &
{
  u.x += static_cast<T>(s);
  u.y += static_cast<T>(s);
  return u;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator-=(Vec2<T> & u, S const & s) -> Vec2<T> &
{
  u.x -= static_cast<T>(s);
  u.y -= static_cast<T>(s);
  return u;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator*=(Vec2<T> & u, S const & s) -> Vec2<T> &
{
  u.x *= static_cast<T>(s);
  u.y *= static_cast<T>(s);
  return u;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator/=(Vec2<T> & u, S const & s) -> Vec2<T> &
{
  u.x /= static_cast<T>(s);
  u.y /= static_cast<T>(s);
  return u;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_PURE UM2_HOSTDEV constexpr auto
operator+(Vec2<T> u, S const & s) -> Vec2<T>
{
  return u += s;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_PURE UM2_HOSTDEV constexpr auto
operator-(Vec2<T> u, S const & s) -> Vec2<T>
{
  return u -= s;
}
template <typename T, typename S>

requires(std::same_as<T, S> || std::integral<S>) UM2_PURE UM2_HOSTDEV constexpr auto
operator*(Vec2<T> u, S const & s) -> Vec2<T>
{
  return u *= s;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_PURE UM2_HOSTDEV constexpr auto
operator*(S const & s, Vec2<T> u) -> Vec2<T>
{
  return u *= s;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_PURE UM2_HOSTDEV constexpr auto
operator/(Vec2<T> u, S const & s) -> Vec2<T>
{
  return u /= s;
}

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto min(Vec2<T> u, Vec2<T> const & v) -> Vec2<T>
{
  u.x = thrust::min(u.x, v.x);
  u.y = thrust::min(u.y, v.y);
  return u;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto max(Vec2<T> u, Vec2<T> const & v) -> Vec2<T>
{
  u.x = thrust::max(u.x, v.x);
  u.y = thrust::max(u.y, v.y);
  return u;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto dot(Vec2<T> const & u, Vec2<T> const & v) -> T
{
  return u.x * v.x + u.y * v.y;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto cross(Vec2<T> const & u, Vec2<T> const & v) -> T
{
  return u.x * v.y - u.y * v.x;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto norm2(Vec2<T> const & u) -> T
{
  return u.x * u.x + u.y * u.y;
}

template <std::floating_point T>
UM2_PURE UM2_HOSTDEV constexpr auto norm(Vec2<T> const & u) -> T
{
  return std::sqrt(norm2(u));
}

template <std::floating_point T>
UM2_PURE UM2_HOSTDEV constexpr auto normalize(Vec2<T> u) -> Vec2<T>
{
  return u /= norm(u);
}

} // namespace um2