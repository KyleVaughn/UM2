namespace um2
{

template <Size D, typename T>
HOSTDEV constexpr void
flipFace(Triangle<D, T> & t) noexcept
{
  um2::swap(t[1], t[2]);
}

template <Size D, typename T>
HOSTDEV constexpr void
flipFace(Quadrilateral<D, T> & q) noexcept
{
  um2::swap(q[1], q[3]);
}

template <Size D, typename T>
HOSTDEV constexpr void
flipFace(QuadraticTriangle<D, T> & q) noexcept
{
  um2::swap(q[1], q[2]);
  um2::swap(q[3], q[5]);
}

template <Size D, typename T>
HOSTDEV constexpr void
flipFace(QuadraticQuadrilateral<D, T> & q) noexcept
{
  um2::swap(q[1], q[3]);
  um2::swap(q[4], q[7]);
}

} // namespace um2
