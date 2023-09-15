namespace um2
{

//==============================================================================
// Accessors
//==============================================================================

template <Size D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::operator[](Size i) noexcept -> T &
{
  assert(i < D);
  return data[i];
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::operator[](Size i) const noexcept -> T const &
{
  assert(i < D);
  return data[i];
}

template <Size D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::begin() noexcept -> T *
{
  return addressof(data[0]);
}

template <Size D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::begin() const noexcept -> T const *
{
  return addressof(data[0]);
}

template <Size D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::end() noexcept -> T *
{
  return addressof(data[0]) + D;
}

template <Size D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vec<D, T>::end() const noexcept -> T const *
{
  return addressof(data[0]) + D;
}

//==============================================================================
// Constructors
//==============================================================================

template <Size D, class T>
template <class... Is>
  requires(sizeof...(Is) == D && (std::integral<Is> && ...) &&
           !(std::same_as<T, Is> && ...))
HOSTDEV constexpr Vec<D, T>::Vec(Is const... args) noexcept
    : data{static_cast<T>(args)...}
{
}

template <Size D, class T>
template <class... Ts>
  requires(sizeof...(Ts) == D && (std::same_as<T, Ts> && ...))
HOSTDEV constexpr Vec<D, T>::Vec(Ts const... args) noexcept
    : data{args...}
{
}

//==============================================================================
// Binary operators
//==============================================================================

template <Size D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator+=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (Size i = 0; i < D; ++i) {
    data[i] += v[i];
  }
  return *this;
}

template <Size D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator-=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (Size i = 0; i < D; ++i) {
    data[i] -= v[i];
  }
  return *this;
}

template <Size D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator*=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (Size i = 0; i < D; ++i) {
    data[i] *= v[i];
  }
  return *this;
}

template <Size D, class T>
HOSTDEV constexpr auto
Vec<D, T>::operator/=(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (Size i = 0; i < D; ++i) {
    data[i] /= v[i];
  }
  return *this;
}

template <Size D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto
Vec<D, T>::operator+=(S const & s) noexcept -> Vec<D, T> &
{
  for (Size i = 0; i < D; ++i) {
    data[i] += static_cast<T>(s);
  }
  return *this;
}

template <Size D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto
Vec<D, T>::operator-=(S const & s) noexcept -> Vec<D, T> &
{
  for (Size i = 0; i < D; ++i) {
    data[i] -= static_cast<T>(s);
  }
  return *this;
}

template <Size D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto
Vec<D, T>::operator*=(S const & s) noexcept -> Vec<D, T> &
{
  for (Size i = 0; i < D; ++i) {
    data[i] *= static_cast<T>(s);
  }
  return *this;
}

template <Size D, class T>
template <class S>
  requires(std::same_as<T, S> || std::integral<S>)
HOSTDEV constexpr auto
Vec<D, T>::operator/=(S const & s) noexcept -> Vec<D, T> &
{
  for (Size i = 0; i < D; ++i) {
    data[i] /= static_cast<T>(s);
  }
  return *this;
}

template <Size D, class T>
HOSTDEV constexpr auto
operator+(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u += v;
}

template <Size D, class T>
HOSTDEV constexpr auto
operator-(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u -= v;
}

template <Size D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
HOSTDEV constexpr auto
operator*(Scalar s, Vec<D, T> u) noexcept -> Vec<D, T>
{
  return u *= s;
}

template <Size D, class T, typename Scalar>
  requires(std::same_as<T, Scalar> || std::integral<Scalar>)
HOSTDEV constexpr auto
operator/(Vec<D, T> u, Scalar s) noexcept -> Vec<D, T>
{
  return u /= s;
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, class T>
HOSTDEV constexpr auto
Vec<D, T>::zero() noexcept -> Vec<D, T>
{
  return Vec<D, T>{}; // Zero-initialize.
}

template <Size D, class T>
HOSTDEV constexpr auto
min(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u.min(v);
}

template <Size D, class T>
HOSTDEV constexpr auto
max(Vec<D, T> u, Vec<D, T> const & v) noexcept -> Vec<D, T>
{
  return u.max(v);
}

template <Size D, class T>
HOSTDEV constexpr auto
dot(Vec<D, T> const & u, Vec<D, T> const & v) noexcept -> T
{
  T result = u[0] * v[0];
  for (Size i = 1; i < D; ++i) {
    result += u[i] * v[i];
  }
  return result;
}

template <Size D, class T>
HOSTDEV constexpr auto
squaredNorm(Vec<D, T> const & v) noexcept -> T
{
  T result = v[0] * v[0];
  for (Size i = 1; i < D; ++i) {
    result += v[i] * v[i];
  }
  return result;
}

template <Size D, class T>
HOSTDEV constexpr auto
norm(Vec<D, T> const & v) noexcept -> T
{
  static_assert(std::is_floating_point_v<T>);
  return um2::sqrt(squaredNorm(v));
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
normalized(Vec<D, T> v) noexcept -> Vec<D, T>
{
  static_assert(std::is_floating_point_v<T>);
  v.normalize();
  return v;
}

template <Size D, class T>
HOSTDEV constexpr auto
Vec<D, T>::min(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (Size i = 0; i < D; ++i) {
    data[i] = um2::min(data[i], v[i]);
  }
  return *this;
}

template <Size D, class T>
HOSTDEV constexpr auto
Vec<D, T>::max(Vec<D, T> const & v) noexcept -> Vec<D, T> &
{
  for (Size i = 0; i < D; ++i) {
    data[i] = um2::max(data[i], v[i]);
  }
  return *this;
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::dot(Vec<D, T> const & v) const noexcept -> T
{
  return um2::dot(*this, v);
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::squaredNorm() const noexcept -> T
{
  return um2::squaredNorm(*this);
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::norm() const noexcept -> T
{
  return um2::norm(*this);
}

template <Size D, class T>
HOSTDEV constexpr void
Vec<D, T>::normalize() noexcept
{
  static_assert(std::is_floating_point_v<T>);
  *this /= norm();
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::normalized() const noexcept -> Vec<D, T>
{
  return um2::normalized(*this);
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::cross(Vec2<T> const & v) const noexcept -> T
{
  static_assert(D == 2);
  static_assert(std::is_floating_point_v<T>);
  return data[0] * v[1] - data[1] * v[0];
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::cross(Vec3<T> const & v) const noexcept -> Vec3<T>
{
  static_assert(D == 3);
  static_assert(std::is_floating_point_v<T>);
  return {data[1] * v[2] - data[2] * v[1], data[2] * v[0] - data[0] * v[2],
          data[0] * v[1] - data[1] * v[0]};
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::squaredDistanceTo(Vec<D, T> const & v) const noexcept -> T
{
  T const d0 = data[0] - v[0];
  T result = d0 * d0;
  for (Size i = 1; i < D; ++i) {
    T const di = data[i] - v[i];
    result += di * di;
  }
  return result;
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
Vec<D, T>::distanceTo(Vec<D, T> const & v) const noexcept -> T
{
  static_assert(std::is_floating_point_v<T>);
  return um2::sqrt(squaredDistanceTo(v));
}

} // namespace um2
