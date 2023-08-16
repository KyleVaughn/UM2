namespace um2
{

// --------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::xMin() const noexcept -> T
{
  return minima[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::xMax() const noexcept -> T
{
  return maxima[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::yMin() const noexcept -> T
{
  static_assert(2 <= D);
  return minima[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::yMax() const noexcept -> T
{
  static_assert(2 <= D);
  return maxima[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::zMin() const noexcept -> T
{
  static_assert(3 <= D);
  return minima[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::zMax() const noexcept -> T
{
  static_assert(3 <= D);
  return maxima[2];
}

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

template <Size D, typename T>
// NOLINTBEGIN(misc-unused-parameters)
HOSTDEV constexpr AxisAlignedBox<D, T>::AxisAlignedBox(Point<D, T> const & min,
                                                       Point<D, T> const & max) noexcept
    : minima(min),
      maxima(max)
{
  for (Size i = 0; i < D; ++i) {
    assert(minima[i] <= maxima[i]);
  }
}
// NOLINTEND(misc-unused-parameters)

// ------------------------------------------------------------------------------
// Methods
// ------------------------------------------------------------------------------
template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::width() const noexcept -> T
{
  return xMax() - xMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::height() const noexcept -> T
{
  static_assert(2 <= D);
  return yMax() - yMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::depth() const noexcept -> T
{
  static_assert(3 <= D);
  return zMax() - zMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::centroid() const noexcept -> Point<D, T>
{
  return midpoint(minima, maxima);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(misc-unused-parameters)
AxisAlignedBox<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  for (Size i = 0; i < D; ++i) {
    if (p[i] < minima[i] || maxima[i] < p[i]) {
      return false;
    }
  }
  return true;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
isApprox(AxisAlignedBox<D, T> const & a, AxisAlignedBox<D, T> const & b) noexcept -> bool
{
  return isApprox(a.minima, b.minima) && isApprox(a.maxima, b.maxima);
}

// ------------------------------------------------------------------------------
// Bounding Box
// ------------------------------------------------------------------------------
template <Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(AxisAlignedBox<D, T> const & a, AxisAlignedBox<D, T> const & b) noexcept
    -> AxisAlignedBox<D, T>
{

  Point<D, T> minima;
  Point<D, T> maxima;
  for (Size i = 0; i < D; ++i) {
    minima[i] = um2::min(a.minima[i], b.minima[i]);
    maxima[i] = um2::max(a.maxima[i], b.maxima[i]);
  }
  return AxisAlignedBox<D, T>{minima, maxima};
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(AxisAlignedBox<D, T> const & box, Point<D, T> const & p) noexcept
    -> AxisAlignedBox<D, T>
{
  auto result = box;
  result.minima.min(p);
  result.maxima.max(p);
  return result;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(Point<D, T> const & p, AxisAlignedBox<D, T> const & box) noexcept
    -> AxisAlignedBox<D, T>
{
  return boundingBox(box, p);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(Point<D, T> const & a, Point<D, T> const & b) noexcept -> AxisAlignedBox<D, T>
{
  return AxisAlignedBox<D, T>{um2::min(a, b), um2::max(a, b)};
}

template <Size D, typename T, Size N>
PURE HOSTDEV constexpr auto
boundingBox(Point<D, T> const (&points)[N]) noexcept -> AxisAlignedBox<D, T>
{
  Point<D, T> minima = points[0];
  Point<D, T> maxima = points[0];
  for (Size i = 1; i < N; ++i) {
    minima.min(points[i]);
    maxima.max(points[i]);
  }
  return AxisAlignedBox<D, T>(minima, maxima);
}

template <Size D, typename T>
PURE auto
boundingBox(Vector<Point<D, T>> const & points) noexcept -> AxisAlignedBox<D, T>
{
  return std::reduce(std::execution::par, points.begin(), points.end(),
                     AxisAlignedBox<D, T>{points[0], points[0]},
                     [](auto const & a, auto const & b) { return boundingBox(a, b); });
}

} // namespace um2
