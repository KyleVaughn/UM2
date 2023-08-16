namespace um2
{

// -------------------------------------------------------------------
// LineSegment
// -------------------------------------------------------------------
template <typename T>
PURE HOSTDEV constexpr auto
pointIsLeft(LineSegment2<T> const & l, Point2<T> const & p) noexcept -> bool
{
  return areCCW(l[0], l[1], p);
}

// -------------------------------------------------------------------
// QuadraticSegment
// -------------------------------------------------------------------
template <typename T>
PURE HOSTDEV constexpr auto
pointIsLeft(QuadraticSegment2<T> const & q, Point2<T> const & p) noexcept -> bool
{
  // This routine has previously been a major bottleneck, so some readability
  // has been sacrificed for performance.
  // The quadratic segment has an equivalent quadratic Bezier curve.
  // The triangle formed by the control points of the quadratic Bezier curve
  // bound the curve. If p is not in this triangle we can ignore the curvature of the
  // segment and return based upon areCCW(v[0], v[1], p). If the triangle is CCW oriented,
  //    We will check that p is in the triangle by checking that p is left of each edge of
  //    the triangle.
  // else
  //    We will check that p is in the triangle by checking that p is right of each edge
  //    of the triangle.
  // We manually perform the check for (v0, v1) since we want to reuse v01 and v0p.
  Vec2<T> const v01(q[1][0] - q[0][0], q[1][1] - q[0][1]);
  Point2<T> const bcp = q.getBezierControlPoint();
  Vec2<T> const v0b(bcp[0] - q[0][0], bcp[1] - q[0][1]);
  Vec2<T> const v0p(p[0] - q[0][0], p[1] - q[0][1]);
  bool const tri_is_ccw = v01.cross(v0b) >= 0;
  {
    bool const b0 = v01.cross(v0p) >= 0;  // areCCW(v[0], v[1], p) == Left of edge 0
    bool const b1 = areCCW(q[1], bcp, p); // Left of edge 1
    bool const b2 = areCCW(bcp, q[0], p); // Left of edge 2
    // if b0 && b1 && b2, the point is in the triangle, and we must perform further
    // analysis.
    // if !(b0 && b1 && b2), the point is outside the triangle, and we may return
    // b0
    bool const p_in_tri = tri_is_ccw ? (b0 && b1 && b2) : (!b0 && !b1 && !b2);
    if (!p_in_tri) {
      return b0;
    }
  }

  // We want to rotate the segment so that v[0] is at the origin and v[1] is on the
  // x-axis. We then take note of the sign of the rotated v[2] to determine if the
  // segment curves left or right. We will orient the segment so that it curves right.
  //     Compute the rotation matrix
  T const v01_norm = v01.norm();
  //     We can avoid a matrix multiplication by using the fact that the y-coordinate of
  //     v1_r is zero.
  Point2<T> const v1_r(v01_norm, static_cast<T>(0));
  Vec2<T> const v01_normalized = v01 / v01_norm;
  //     NOLINTBEGIN(readability-identifier-naming)
  Mat2x2<T> const R(Vec2<T>(v01_normalized[0], -v01_normalized[1]),
                    Vec2<T>(v01_normalized[1], v01_normalized[0]));
  Vec2<T> const v02 = q[2] - q[0];
  Point2<T> v2_r = R * v02;
  Point2<T> p_r = R * (p - q[0]);
  bool const curves_right = v2_r[1] >= 0;
  if (!curves_right) {
    // Flip the y-coordinates to be greater than or equal to zero
    v2_r[1] = -v2_r[1];
    p_r[1] = -p_r[1];
  }
  // If v2_r[1] < epsilonDistance, then the segment is straight and we can use the cross
  // product test to return early.
  bool const is_straight = v2_r[1] <= epsilonDistance<T>();
  if (is_straight) {
    // if !curves_right, we flipped the y-coordinates, so we need to flip the sign of the
    // cross product.
    // | positive cross product | curves_right | final result |
    // |------------------------|--------------|--------------|
    // | true                   | true         | true         |
    // | false                  | true         | false        |
    // | true                   | false        | false        |
    // | false                  | false        | true         |
    return (v1_r.cross(p_r) >= 0) ? curves_right : !curves_right;
  }
  // We now wish to compute the bounding box of the segment in order to determine if we
  // need to treat the segment as a straight line or a curve. If the point is outside the
  // bounding box, then we can treat the segment like a straight line.
  //  Q(r) = C + rB + r²A
  // where
  //  C = (0, 0)
  //  B =  4 * v2_r - v1_r
  //  A = -4 * v2_r + 2 * v1_r = -B + v1_r
  // Q′(r) = B + 2rA,
  // The stationary points are (r_i,...) = -B / (2A)
  // Q(r_i) = -B² / (4A) = r_i(B/2)
  // Since v1_r[1] = 0 and v2_r[1] > 0, we know that:
  //    B[1] > 0
  //    A[1] = -B[1], which implies A[1] < 0
  //    A[1] < 0 implies Q(r_i) is strictly positive

  // If the point is below the origin, then the point is right of the segment
  if (p_r[1] < 0) {
    return !curves_right;
  }
  T const Bx = 4 * v2_r[0] - v1_r[0];
  T const By = 4 * v2_r[1]; // Positive
  T const Ax = -Bx + v1_r[0];
  T const Ay = -By; // Negative

  // This is the logic or testing against the AABB. This was benchmarked to be slower than
  // using the Triangle test above
  // // Handle the case where the point is above the bounding box
  // {
  //   T ymax = v2_r[1];
  //   // Determine ymax
  //   // r_i = -B_i / (2A_i)
  //   T const half_b = By / 2;   // Positive
  //   T const ry = -half_b / Ay; // Positive
  //   // Only consider the stationary point if it is in the interval [0, 1]
  //   // Since ry is strictly positive, we only need to check if ry < 1
  //   if (ry < 1) {
  //     // x_i = Q(r_i) = - B² / (4A) = r(B/2)
  //     ymax = ry * half_b;
  //   }
  //   // Check if the point is above the BB
  //   if (ymax <= p_r[1]) {
  //     return curves_right;
  //   }
  // }
  // // Handle the case where the point is to the left or the right of the bounding box
  // {
  //   T xmin = 0;
  //   T xmax = v1_r[0];
  //   // A is effectively 4 * (midpoint of v01 - v2_r), hence if Ax is small, then the
  //   // segment is effectively straight and we known xmin = 0 and xmax = v1_r[0]
  //   if (um2::abs(Ax) > 4 * epsilonDistance<T>()) {
  //     // r_i = -B_i / (2A_i)
  //     T const half_b = Bx / 2;
  //     T const rx = -half_b / Ax;
  //     // NOLINTNEXTLINE(misc-redundant-expression)
  //     if (0 < rx && rx < 1) {
  //       // x_i = Q(r_i) = - B² / (4A) = r(B/2)
  //       T const x_stationary = rx * half_b;
  //       xmin = um2::min(xmin, x_stationary);
  //       xmax = um2::max(xmax, x_stationary);
  //     }
  //   }
  //   if (p_r[0] <= xmin || xmax <= p_r[0]) {
  //     // Since the point is in the y-range of the AABB, the point will be
  //     // left of the segment if the segment curves right
  //     // if the segment curves left.
  //     return curves_right;
  //   }
  // }
  // End of AABB test
  // If the point is in the bounding box of the segment,
  // we will find the point on the segment that shares the same x-coordinate
  //  Q(r) = C + rB + r²A = P
  // Hence we wish to solve 0 = -P_x + rB_x + r²A_x for r.
  // This is a quadratic equation, which has two potential solutions.
  // r = (-b ± √(b² - 4ac)) / 2a
  // if A[0] == 0, then we have a well-behaved quadratic segment, and there is one root.
  // This is the expected case.
  if (um2::abs(Ax) < 4 * epsilonDistance<T>()) {
    // This is a linear equation, so there is only one root.
    T const r = p_r[0] / Bx; // B[0] != 0, otherwise the segment would be degenerate.
    // We know the point is in the AABB of the segment, so we expect r to be in [0, 1]
    assert(0 <= r && r <= 1);
    T const Q_y = r * (By + r * Ay);
    // if p_r < Q_y, then the point is to the right of the segment
    return (p_r[1] <= Q_y) ? !curves_right : curves_right;
  }
  // Two roots.
  T const disc = Bx * Bx + 4 * Ax * p_r[0];
  assert(disc >= 0);
  T const r1 = (-Bx + um2::sqrt(disc)) / (2 * Ax);
  T const r2 = (-Bx - um2::sqrt(disc)) / (2 * Ax);
  T const Q_y1 = r1 * (By + r1 * Ay);
  T const Q_y2 = r2 * (By + r2 * Ay);
  T const Q_ymin = um2::min(Q_y1, Q_y2);
  T const Q_ymax = um2::max(Q_y1, Q_y2);
  bool const contained_in_curve = Q_ymin <= p_r[1] && p_r[1] <= Q_ymax;
  return contained_in_curve ? !curves_right : curves_right;
  // NOLINTEND(readability-identifier-naming)
}

} // namespace um2
