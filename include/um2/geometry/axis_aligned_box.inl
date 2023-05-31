namespace um2
{

// --------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::xmin() const -> T
{
  return minima[0];
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::xmax() const -> T
{
  return maxima[0];
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::ymin() const -> T
{
  static_assert(2 <= D);
  return minima[1];
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::ymax() const -> T
{
  static_assert(2 <= D);
  return maxima[1];
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::zmin() const -> T
{
  static_assert(3 <= D);
  return minima[2];
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::zmax() const -> T
{
  static_assert(3 <= D);
  return maxima[2];
}

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV constexpr AABox<D, T>::AABox(Point<D, T> const & min, Point<D, T> const & max)
    : minima(min),
      maxima(max)
{
  for (len_t i = 0; i < D; ++i) {
    assert(minima[i] <= maxima[i]);
  }
}

// ------------------------------------------------------------------------------
// Methods
// ------------------------------------------------------------------------------
template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::width() const -> T
{
  return xmax() - xmin();
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::height() const -> T
{
  static_assert(2 <= D);
  return ymax() - ymin();
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::depth() const -> T
{
  static_assert(3 <= D);
  return zmax() - zmin();
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::centroid() const -> Point<D, T>
{
  return midpoint(minima, maxima);
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto AABox<D, T>::contains(Point<D, T> const & p) const
    -> bool
{
  Vec<D, T> const eps_vec = Vec<D, T>::Constant(epsilonDistance<T>());
  Vec<D, T> const minima_vec = minima - eps_vec;
  Vec<D, T> const maxima_vec = maxima + eps_vec;
  return (minima_vec.array() <= p.array()).all() &&
         (p.array() <= maxima_vec.array()).all();
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto isApprox(AABox<D, T> const & a, AABox<D, T> const & b)
    -> bool
{
  return isApprox(a.minima, b.minima) && isApprox(a.maxima, b.maxima);
}

// ------------------------------------------------------------------------------
// Bounding Box
// ------------------------------------------------------------------------------
template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto boundingBox(AABox<D, T> const & a,
                                                AABox<D, T> const & b) -> AABox<D, T>
{
  Point<D, T> const minima = a.minima.cwiseMin(b.minima);
  Point<D, T> const maxima = a.maxima.cwiseMax(b.maxima);
  return AABox<D, T>{minima, maxima};
}

template <len_t D, typename T, len_t N>
UM2_PURE UM2_HOSTDEV constexpr auto boundingBox(Point<D, T> const (&points)[N])
    -> AABox<D, T>
{
  Point<D, T> minima = points[0];
  Point<D, T> maxima = points[0];
  for (len_t i = 1; i < N; ++i) {
    minima = minima.cwiseMin(points[i]);
    maxima = maxima.cwiseMax(points[i]);
  }
  return AABox<D, T>(minima, maxima);
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto boundingBox(Vector<Point<D, T>> const & points)
    -> AABox<D, T>
{
  Point<D, T> minima = points[0];
  Point<D, T> maxima = points[0];
  for (len_t i = 1; i < points.size(); ++i) {
    minima = minima.cwiseMin(points[i]);
    maxima = maxima.cwiseMax(points[i]);
  }
  return AABox<D, T>(minima, maxima);
}

} // namespace um2
