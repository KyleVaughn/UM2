namespace um2
{

//==============================================================================
// isStraight
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
isStraight(QuadraticSegment<D, T> const & q) noexcept -> bool
{
  // A slightly more optimized version of doing:
  // LineSegment(v[0], v[1]).distanceTo(v[2]) < epsilonDistance
  //
  // Compute the point on the line v[0] + r * (v[1] - v[0]) that is closest to v[2]
  Vec<D, T> const v01 = q[1] - q[0];
  T const r = (q[2] - q[0]).dot(v01) / v01.squaredNorm();
  // If r is outside the range [0, 1], then the segment is not straight
  if (r < 0 || r > 1) {
    return false;
  }
  // Compute the point on the line
  Vec<D, T> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = q[0][i] + r * v01[i];
  }
  // Check if the point is within epsilon distance of v[2]
  return isApprox(p, q[2]);
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
getBezierControlPoint(QuadraticSegment<D, T> const & q) noexcept -> Point<D, T>
{
  // p0 == v[0]
  // p2 == v[1]
  // p1 == 2 * v[2] - (v[0] + v[1]) / 2, hence we only need to compute p1
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = static_cast<T>(2) * q[2][i] - (q[0][i] + q[1][i]) / 2;
  }
  return result;
}

//==============================================================================
// enclosedArea
//==============================================================================

template <typename T>    
PURE HOSTDEV constexpr auto    
enclosedArea(QuadraticSegment2<T> const & q) noexcept -> T    
{    
  // The area bounded by the segment and the line between the endpoints is    
  // 4/3 of the area of the triangle formed by the vertices.    
  // Assumes that the segment is convex.    
  T constexpr two_thirds = static_cast<T>(2) / static_cast<T>(3);    
  Vec2<T> const v02 = q[2] - q[0];    
  Vec2<T> const v01 = q[1] - q[0];    
  return two_thirds * v02.cross(v01);    
} 

//==============================================================================
// enclosedCentroid
//==============================================================================

template <typename T>    
PURE HOSTDEV constexpr auto    
enclosedCentroid(QuadraticSegment2<T> const & q) noexcept -> Point2<T>    
{    
  // For a quadratic segment, with P₁ = (0, 0), P₂ = (x₂, 0), and P₃ = (x₃, y₃),    
  // where 0 < x₂, if the area bounded by q and the x-axis is convex, it can be    
  // shown that the centroid of the area bounded by the segment and x-axis    
  // is given by
  // C = (3x₂ + 4x₃, 4y₃) / 10
  //
  // To find the centroid of the area bounded by the segment for a general
  // quadratic segment, we transform the segment so that P₁ = (0, 0),
  // then use a change of basis (rotation) from the standard basis to the
  // following basis, to achieve y₂ = 0.
  //
  // Let v = (v₁, v₂) = (P₂ - P₁) / ‖P₂ - P₁‖
  // u₁ = ( v₁,  v₂) = v
  // u₂ = (-v₂,  v₁)
  //
  // Note: u₁ and u₂ are orthonormal.
  //
  // The transformation from the new basis to the standard basis is given by
  // U = [u₁ u₂] = | v₁ -v₂ |
  //               | v₂  v₁ |
  //
  // Since u₁ and u₂ are orthonormal, U is unitary.
  //
  // The transformation from the standard basis to the new basis is given by
  // U⁻¹ = Uᵗ = |  v₁  v₂ |
  //            | -v₂  v₁ |
  // since U is unitary.
  //
  // Therefore, the centroid of the area bounded by the segment is given by
  // C = U * Cᵤ + P₁
  // where
  // Cᵤ = (u₁ ⋅ (3(P₂ - P₁) + 4(P₃ - P₁)), 4(u₂ ⋅ (P₃ - P₁))) / 10
  Vec2<T> const v12 = q[1] - q[0];
  Vec2<T> const four_v13 = 4 * (q[2] - q[0]);
  Vec2<T> const u1 = v12.normalized();
  Vec2<T> const u2(-u1[1], u1[0]);
  // NOLINTBEGIN(readability-identifier-naming) justification: capitalize matrix
  Mat2x2<T> const U(u1, u2);
  Vec2<T> const Cu(u1.dot((3 * v12 + four_v13)) / 10, u2.dot(four_v13) / 10);
  return U * Cu + q[0];
  // NOLINTEND(readability-identifier-naming)
}

} // namespace um2
