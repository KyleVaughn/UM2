namespace um2
{

//==============================================================================
// Accessors
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return v[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return v[i];
}

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr QuadraticSegment<D, T>::Polytope(Point<D, T> const & p0,
                                                   Point<D, T> const & p1,
                                                   Point<D, T> const & p2) noexcept
    : v{p0, p1, p2}
{
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator()(R const r) const noexcept -> Point<D, T>
{
  return interpolate(*this, r);
}

//==============================================================================
// jacobian
//==============================================================================

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::jacobian(R r) const noexcept -> Vec<D, T>
{
  return um2::jacobian(*this, r);
}

//==============================================================================
// getRotation
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::getRotation() const noexcept -> Mat<D, D, T>
{
  return LineSegment<D, T>(v[0], v[1]).getRotation();
}

//==============================================================================
// isStraight
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::isStraight() const noexcept -> bool
{
  // A slightly more optimized version of doing:
  // LineSegment(v[0], v[1]).distanceTo(v[2]) < epsilonDistance
  //
  // Compute the point on the line v[0] + r * (v[1] - v[0]) that is closest to v[2]
  Vec<D, T> const v01 = v[1] - v[0];
  T const r = (v[2] - v[0]).dot(v01) / v01.squaredNorm();
  // If r is outside the range [0, 1], then the segment is not straight
  if (r < 0 || r > 1) {
    return false;
  }
  // Compute the point on the line
  Vec<D, T> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = v[0][i] + r * v01[i];
  }
  // Check if the point is within epsilon distance of v[2]
  return isApprox(p, v[2]);
}
// //==============================================================================
// // curvesLeft
// //==============================================================================
//
// template <Size D, typename T>
// PURE HOSTDEV constexpr auto
// QuadraticSegment<D, T>::curvesLeft() const noexcept -> bool
// {
//   static_assert(D == 2, "curvesLeft is only defined for 2D");
//   // If the segment is not straight, then we can compute the cross product of the
//   // vectors from v[0] to v[1] and v[0] to v[2]. If the cross product is positive,
//   // then the segment curves left. If the cross product is negative, then the segment
//   // curves right.
//   Vec<D, T> const v01 = v[1] - v[0];
//   Vec<D, T> const v02 = v[2] - v[0];
//   return v01.cross(v02) >= 0;
// }

//==============================================================================
// getBezierControlPoint
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::getBezierControlPoint() const noexcept -> Point<D, T>
{
  // p0 == v[0]
  // p2 == v[1]
  // p1 == 2 * v[2] - (v[0] + v[1]) / 2, hence we only need to compute p1
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = static_cast<T>(2) * v[2][i] - (v[0][i] + v[1][i]) / 2;
  }
  return result;
}

//==============================================================================
// isLeft
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::isLeft(Point<D, T> const & p) const noexcept -> bool
{
  return pointIsLeft(*this, p);
}

//==============================================================================
// length
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::length() const noexcept -> T
{
  return um2::length(*this);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// enclosedArea
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::enclosedArea() const noexcept -> T
{
  return um2::enclosedArea(*this);
}

//==============================================================================
// enclosedCentroid
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::enclosedCentroid() const noexcept -> Point<D, T>
{
  return um2::enclosedCentroid(*this);
}

//==============================================================================
// pointClosestTo
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::pointClosestTo(Point<D, T> const & p) const noexcept -> T
{
  return um2::pointClosestTo(*this, p);
}

} // namespace um2
