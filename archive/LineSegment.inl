// -------------------------------------------------------------------
// distanceTo
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::squaredDistanceTo(Point<D, T> const & p) const noexcept -> T
{
  // From Real-Time Collision Detection, Christer Ericson, 2005
  // Given segment ab and point c, computes closest point d on ab.
  // Also returns t for the position of d, d(t) = a + t*(b - a)
  Vec<D, T> const ab = v[1] - v[0];
  // Project c onto ab, computing parameterized position d(t) = a + t*(b − a)
  T t = (p - v[0]).dot(ab) / ab.squaredNorm();
  // If outside segment, clamp t (and therefore d) to the closest endpoint
  if (t < 0) {
    t = 0;
  }
  if (t > 1) {
    t = 1;
  }
  // Compute projected position from the clamped t
  Vec<D, T> d;
  for (Size i = 0; i < D; ++i) {
    d[i] = v[0][i] + t * ab[i];
  }
  return d.squaredDistanceTo(p);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::distanceTo(Point<D, T> const & p) const noexcept -> T
{
  return um2::sqrt(squaredDistanceTo(p));
}

