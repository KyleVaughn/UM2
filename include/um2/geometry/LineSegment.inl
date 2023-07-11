namespace um2
{

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr LineSegment<D, T>::Polytope(Point<D, T> const & p0,
                                              Point<D, T> const & p1) noexcept
{
  w[0] = p0;
  for (Size i = 1; i < D; ++i) {
    w[1] = p1[i] - p0[i];
  }
}

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::getVertex(Size const i) const noexcept -> Point<D, T>
{
  return this->operator()(i);
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::operator()(R const r) const noexcept -> Point<D, T>
{
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w[0][i] + static_cast<T>(r) * w[1][i];
  }
  return result;
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::jacobian(R /*r*/) const noexcept -> Vec<D, T>
{
  // L(r) = = w0 + r * w1  
  // L'(r) = w1
  return w[1]; 
}

// -------------------------------------------------------------------
// isLeft
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::isLeft(Point<D, T> const & p) const noexcept -> bool
{
  static_assert(D == 2, "isLeft is only defined for 2D line segments");
  // If the cross product of the vector from the first vertex to the 
  // second vertex and the vector from the first vertex to the point
  // is positive, then the point is to the left of the line segment.
  Vec<D, T> dp;
  for (Size i = 0; i < D; ++i) {
    dp[i] = p[i] - w[0][i];
  }
  return 0 <= w[1].cross(dp); 
}

// -------------------------------------------------------------------
// length
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::length() const noexcept -> T
{
  return w[1].norm();
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  Point<D, T> minima = w[0];
  Point<D, T> maxima = w[0];
  Point<D, T> v1 = getVertex(1);
  minima.min(v1);
  maxima.max(v1);
  return AxisAlignedBox<D, T>{minima, maxima}; 
}

} // namespace um2
