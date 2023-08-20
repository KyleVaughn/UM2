namespace um2
{

//==============================================================================
// LineSegment 
//==============================================================================

template <typename T>
PURE HOSTDEV constexpr auto
getRotation(LineSegment2<T> const & l) noexcept -> Mat2x2<T>
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
  Vec2<T> const a = (l[1] - l[0]).normalized();
  Vec2<T> const col0(a[0], -a[1]);
  Vec2<T> const col1(a[1], a[0]);
  return Mat2x2<T>(col0, col1);
}

//==============================================================================
// QuadraticSegment 
//==============================================================================

template <typename T>
PURE HOSTDEV constexpr auto
getRotation(QuadraticSegment2<T> const & q) noexcept -> Mat2x2<T>
{
  return LineSegment2<T>(q[0], q[1]).getRotation();
}

} // namespace um2
