namespace um2
{

//==============================================================================
// Accessors
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return v[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return v[i];
}

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr LineSegment<D, T>::Polytope(Point<D, T> const & p0,
                                              Point<D, T> const & p1) noexcept
    : v{p0, p1}
{
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::operator()(R const r) const noexcept -> Point<D, T>
{
  return interpolate(*this, r);
}

//==============================================================================
// jacobian
//==============================================================================

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::jacobian(R const r) const noexcept -> Vec<D, T>
{
  return um2::jacobian(*this, r);
}

//==============================================================================
// getRotation
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::getRotation() const noexcept -> Mat<D, D, T>
{
  // We want to transform the segment so that v[0] is at the origin and v[1]
  // is on the x-axis. We can do this by first translating by -v[0] and then
  // using a change of basis (rotation) matrix to rotate v[1] onto the x-axis.
  // x_old = U * x_new
  //
  // For 2D:
  // Let a = (a₁, a₂) = (P₂ - P₁) / ‖P₂ - P₁‖
  // u₁ = ( a₁,  a₂) = a
  // u₂ = (-a₂,  a₁)
  //
  // Note: u₁ and u₂ are orthonormal.
  //
  // The transformation from the new basis to the standard basis is given by
  // U = [u₁ u₂] = | a₁ -a₂ |
  //               | a₂  a₁ |
  //
  // Since u₁ and u₂ are orthonormal, U is unitary.
  //
  // The transformation from the standard basis to the new basis is given by
  // U⁻¹ = Uᵗ = |  a₁  a₂ |
  //            | -a₂  a₁ |
  // since U is unitary.
  Vec<D, T> const a = (v[1] - v[0]).normalized();
  if constexpr (D == 2) {
    Vec<D, T> const col0(a[0], -a[1]);
    Vec<D, T> const col1(a[1], a[0]);
    return Mat<D, D, T>(col0, col1);
  } else {
    static_assert(D == 3, "getRotation is only defined for 2D and 3D line segments");
    return Mat<D, D, T>();
  }
}

//==============================================================================
// isLeft
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::isLeft(Point<D, T> const & p) const noexcept -> bool
{
  return pointIsLeft(*this, p);
}

//==============================================================================
// length
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::length() const noexcept -> T
{
  return um2::length(*this);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// distanceTo
//==============================================================================

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

} // namespace um2
