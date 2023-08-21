namespace um2
{

template <typename T>
PURE HOSTDEV constexpr auto
isCCW(Triangle2<T> const & t) noexcept -> bool
{
  return areCCW(t[0], t[1], t[2]);
}

template <typename T>
PURE HOSTDEV constexpr auto
isCCW(Quadrilateral2<T> const & q) noexcept -> bool
{
  bool const b0 = areCCW(q[0], q[1], q[2]);
  bool const b1 = areCCW(q[0], q[2], q[3]);
  return b0 && b1;
}

template <typename T>
PURE HOSTDEV constexpr auto
isCCW(QuadraticTriangle2<T> const & q) noexcept -> bool
{
  return isCCW(linearPolygon(q));
}

template <typename T>
PURE HOSTDEV constexpr auto
isCCW(QuadraticQuadrilateral2<T> const & q) noexcept -> bool
{
  return isCCW(linearPolygon(q));
}

} // namespace um2
