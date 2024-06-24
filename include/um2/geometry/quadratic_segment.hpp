#pragma once

#include <um2/geometry/line_segment.hpp>
#include <um2/geometry/triangle.hpp>
#include <um2/math/cubic_equation.hpp>
#include <um2/stdlib/math/logarithms.hpp>

//==============================================================================
// QUADRATIC SEGMENT
//==============================================================================
//
// For quadratic segments, the parametric equation is
//  Q(r) = C + rB + r²A,
// where
//  C = v0
//  B = 3v02 + v12    = -3q[0] -  q[1] + 4q[2]
//  A = -2(v02 + v12) =  2q[0] + 2q[1] - 4q[2]
// and
// v02 = q[2] - q[0]
// v12 = q[2] - q[1]

namespace um2
{

template <Int D, class T>
class Polytope<1, 2, 3, D, T>
{
  static_assert(0 < D && D <= 3, "Only 1D, 2D, and 3D segments are supported.");

public:
  // NOLINTBEGIN(readability-identifier-naming)
  static constexpr Int N = 3; // Number of vertices
  // NOLINTEND(readability-identifier-naming)

  using Vertex = Point<D, T>;

private:
  Vertex _v[N];

public:
  //==============================================================================
  // Accessors
  //==============================================================================

  // Returns the i-th vertex
  PURE HOSTDEV constexpr auto
  operator[](Int i) noexcept -> Vertex &;

  // Returns the i-th vertex
  PURE HOSTDEV constexpr auto
  operator[](Int i) const noexcept -> Vertex const &;

  // Returns a pointer to the vertex array
  PURE HOSTDEV [[nodiscard]] constexpr auto
  vertices() const noexcept -> Vertex const *;

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Polytope() noexcept = default;

  template <class... Pts>
    requires(sizeof...(Pts) == N && (std::same_as<Vertex, Pts> && ...))
  // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
  HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the segment.
  // r in [0, 1], F(r) -> R^D
  PURE HOSTDEV constexpr auto
  operator()(T r) const noexcept -> Point<D, T>;

  // Jacobian of the segment (Column vector).
  // dF/dr -> R^D
  PURE HOSTDEV
      [[nodiscard]] constexpr auto jacobian(T /*r*/) const noexcept -> Vec<D, T>;

  // Q(r) = C + rB + r²A
  // returns {C, B, A}
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getPolyCoeffs() const noexcept -> Vec<3, Vec<D, T>>;

  // Arc length of the segment
  PURE HOSTDEV [[nodiscard]] constexpr auto
  length() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  pointClosestTo(Point<D, T> const & p) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredDistanceTo(Point<D, T> const & p) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  distanceTo(Point<D, T> const & p) const noexcept -> T;

  // 2D only
  //--------------------------------------------------------------------------

  // If the seg is translated by -v0, then the first vertex is at the origin.
  // Get the rotation matrix that aligns the seg with the x-axis.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getRotation() const noexcept -> Mat2x2<T>
    requires(D == 2);

  // If a point is to the left of the segment, with the segment oriented from
  // r = 0 to r = 1.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Point2<T> p) const noexcept -> bool
    requires(D == 2);

  // Intersect the ray with the segment.
  // Returns the number of valid intersections.
  // The ray coordinates r, such that R(r) = o + r*d is an intersection point are
  // stored in the buffer, sorted from closest to farthest. r in [0, inf)
  HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2<T> ray, T * buffer) const noexcept -> Int
    requires(D == 2);

  // Intersect with another segment.
  // Returns the number of valid intersections [0, 2].
  // The intersection points are stored in the buffer.
  HOSTDEV [[nodiscard]] constexpr auto
  intersect(QuadraticSegment2<T> const & other, Point2<T> * buffer) const noexcept -> Int
    requires(D == 2);

  // Intersects with another segment.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersects(QuadraticSegment2<T> const & other) const noexcept -> bool
    requires(D == 2);

  // The segment extends to the right, curving left from v[0] to v[1].
  PURE HOSTDEV [[nodiscard]] constexpr auto
  curvesLeft() const noexcept -> bool
    requires(D == 2);

}; // QuadraticSegment

//==============================================================================
// Free functions
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
isStraight(QuadraticSegment<D, T> const & q) noexcept -> bool;

// The area enclosed by the segment and the straight line from the p0 to p1.
template <class T>
PURE HOSTDEV constexpr auto
enclosedArea(QuadraticSegment2<T> const & q) noexcept -> T;

// The centroid of the area enclosed by the segment and the straight line from
// the p0 to p1.
template <class T>
PURE HOSTDEV constexpr auto
enclosedCentroid(QuadraticSegment2<T> const & q) noexcept -> Vec2<T>;

//==============================================================================
// Accessors
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator[](Int i) const noexcept -> Vertex const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::vertices() const noexcept -> Vertex const *
{
  return _v;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
interpolate(QuadraticSegment<D, T> const & q, T const r) noexcept -> Point<D, T>
{
  // Q(r) =
  // (2 * r - 1) * (r - 1) * v0 +
  // (2 * r - 1) *  r      * v1 +
  // -4 * r      * (r - 1) * v2
  return (2 * r - 1) * (r - 1) * q[0] + (2 * r - 1) * r * q[1] + -4 * r * (r - 1) * q[2];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator()(T const r) const noexcept -> Point<D, T>
{
  return interpolate(*this, r);
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
jacobian(QuadraticSegment<D, T> const & q, T const r) noexcept -> Point<D, T>
{
  // (4 * r - 3) * (v0 - v2) + (4 * r - 1) * (v1 - v2)
  T const w0 = 4 * r - 3;
  T const w1 = 4 * r - 1;
  return w0 * (q[0] - q[2]) + w1 * (q[1] - q[2]);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::jacobian(T const r) const noexcept -> Point<D, T>
{
  return um2::jacobian(*this, r);
}

//==============================================================================
// getPolyCoeffs
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::getPolyCoeffs() const noexcept -> Vec<3, Vec<D, T>>
{
  // For quadratic segments, the parametric equation is
  //  Q(r) = C + rB + r²A,
  // where
  //  C = v0
  //  B = 3v02 + v12    = -3q[0] -  q[1] + 4q[2]
  //  A = -2(v02 + v12) =  2q[0] + 2q[1] - 4q[2]
  // and
  // v02 = q[2] - q[0]
  // v12 = q[2] - q[1]
  Vec<D, T> const v02 = _v[2] - _v[0];
  Vec<D, T> const v12 = _v[2] - _v[1];
  Vec<D, T> const b = 3 * v02 + v12;
  Vec<D, T> const a = -2 * (v02 + v12);
  return {_v[0], b, a};
}

//==============================================================================
// isStraight
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
isStraight(QuadraticSegment<D, T> const & q) noexcept -> bool
{
  // Area of triangle = 1/2 base * height.
  // base = ‖p1 - p0‖
  // height = p2's displacement from the line(p0, p1)
  // Hence the displacement of p2 from the line segment is
  // height = ‖(p2 - p0) × (p1 - p0)‖ / ‖p1 - p0‖
  // We can avoid a square root by comparing the squared distance.
  auto const v10 = q[1] - q[0];
  auto const v20 = q[2] - q[0];
  auto const v10_sq = v10.squaredNorm();
  auto const v20_sq = v20.squaredNorm();
  // If the distance to p2 is greater than the distance to p1, then the segment not
  // straight.
  if (v20_sq >= v10_sq) {
    return false;
  }
  auto const cp = v10.cross(v20);
  if constexpr (D == 2) {
    return (cp * cp) / v10_sq < epsDistance2<T>();
  } else {
    return cp.squaredNorm() / v10_sq < epsDistance2<T>();
  }
}

//==============================================================================
// length
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::length() const noexcept -> T
{
  // The arc length integral may be reduced to an integral over the square root of a
  // quadratic polynomial using ‖𝘅‖ = √(𝘅 ⋅ 𝘅), which has an analytic solution.
  //              1             1
  // arc length = ∫ ‖Q′(r)‖dr = ∫ √(ar² + br + c) dr
  //              0             0
  //
  // Q(r) = x0 + r * x1 + r² * x2,
  // Q′(r) = x1 + 2r * x2,

  if (isStraight(*this)) {
    return _v[0].distanceTo(_v[1]);
  }

  auto const coeffs = getPolyCoeffs();
  auto const x1 = coeffs[1]; // B
  auto const x2 = coeffs[2]; // A

  // ‖Q′(r)‖ =  √(4(x2 ⋅x2)r² + 4(x2 ⋅x1)r + x1 ⋅x1) = √(ar² + br + c)
  // where
  // a = 4(x2 ⋅ x2)
  // b = 4(x2 ⋅ x1)
  // c = x1 ⋅ x1

  ASSERT(squaredNorm(x2) > epsDistance2<T>());
  T const a = 4 * x2.squaredNorm();
  T const b = 4 * x2.dot(x1);
  T const c = x1.squaredNorm();

  // According to Wolfram Alpha, the integral of √(ax² + bx + c) is
  // ((b + 2 a x) sqrt(c + x (b + a x)))/(4 a)
  // - ((b^2 - 4 a c) log(b + 2 a x + 2 sqrt(a) sqrt(c + x (b + a x))))/(8 a^(3/2))
  // + constant
  //
  // Simplifying the integral:
  // ((b^2 - 4 a c) (log(2 sqrt(a) sqrt(c) + b) - log(2 sqrt(a) sqrt(a + b + c) + 2 a +
  // b))
  //  + sqrt(a) (2 (2 a + b) sqrt(a + b + c) - 2 b sqrt(c)))/(8 a^(3/2))

  // sa = sqrt(a)
  // sb = sqrt(b)
  // sc = sqrt(c)
  // sabc = sqrt(a + b + c)
  // disc = b^2 - 4 a c
  // a2b = 2 a + b

  // (disc * (log(2 * sa * sc + b)
  //  - log(2 * sa * sabc + a2b))
  //  + sa * (2 * (a2b * sabc - b * sc))
  //  ) / (8 * sa * sa * sa)
  //
  T const sa = um2::sqrt(a);
  T const sc = um2::sqrt(c);
  T const sabc = um2::sqrt(a + b + c);
  T const disc = quadraticDiscriminant(a, b, c);
  T const a2b = 2 * a + b;
  T const num = disc * um2::log((2 * sa * sc + b) / (2 * sa * sabc + a2b)) +
                    sa * (2 * (a2b * sabc - b * sc));
  T const den = 8 * sa * sa * sa;
  T const result = num / den;
  ASSERT(0 <= result);
  ASSERT(result <= infDistance<T>());
  return result;
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  // Find the extrema by finding dx_i/dr = 0
  // Q(r) = C + rB + r²A,
  // Q′(r) = B + 2rA,
  // 0 = B + 2rA ==> (r_i,...) = -B / (2A)
  // x_i = Q(r_i) = C + (r_i)B + (r_i)^2 A = C - B² / (4A)
  // Compare the extrema with the segment's endpoints to find the AABox
  auto const coeffs = getPolyCoeffs();
  auto const b = coeffs[1];
  auto const a = coeffs[2];
  Point<D, T> minima = um2::min(_v[0], _v[1]);
  Point<D, T> maxima = um2::max(_v[0], _v[1]);
  for (Int i = 0; i < D; ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    // NOLINTNEXTLINE(clang-diagnostic-float-equal)
    if (a[i] == 0) {
      continue;
    }
#pragma GCC diagnostic pop
    T const half_b = b[i] / 2;
    T const r = -half_b / a[i];
    // if r is not in [0, 1], then the extrema are not on the segment, hence
    // the segment's endpoints are the extrema.
    if (0 < r && r < 1) {
      // x_i = Q(r_i) = C - B² / (4A) = C - r_i(B/2)
      T const x = _v[0][i] + r * half_b;
      minima[i] = um2::min(minima[i], x);
      maxima[i] = um2::max(maxima[i], x);
    }
  }
  return AxisAlignedBox<D, T>{minima, maxima};
}

//==============================================================================
// pointClosestTo
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::pointClosestTo(Point<D, T> const & p) const noexcept -> T
{

  if (isStraight(*this)) {
    return LineSegment<D, T>(_v[0], _v[1]).pointClosestTo(p);
  }

  // The interpolation function of the quadratic segment is
  // Q(r) = C + rB + r²A,
  //
  // We wish to find r which minimizes ‖Q(r) - P‖.
  // This r also minimizes ‖Q(r) - P‖².
  // ‖Q(r) - P‖² = f(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
  // Let W = C - P
  // a₄ = A ⋅ A
  // a₃ = 2(A ⋅ B)
  // a₂ = 2(A ⋅ W) + (B ⋅ B)
  // a₁ = 2(B ⋅ W)
  // a₀ = W ⋅ W
  //
  // The minimum of f(r) occurs when f′(r) = ar³ + br² + cr + d = 0, where
  // a = 2(A ⋅ A)
  // b = 3(A ⋅ B)
  // c = (B ⋅ B) + 2(A ⋅W)
  // d = (B ⋅ W)
  // Note we factored out a 2
  auto const coeffs = getPolyCoeffs();
  auto const vc = coeffs[0];
  auto const vb = coeffs[1];
  auto const va = coeffs[2];
  auto const vw = vc - p;
  T const a = 2 * squaredNorm(va);
  T const b = 3 * va.dot(vb);
  T const c = squaredNorm(vb) + 2 * va.dot(vw);
  T const d = vb.dot(vw);

  // Return the real part of the 3 roots of the cubic equation
  auto const roots = solveCubic(a, b, c, d);

  // Find the root which minimizes the squared distance to the point.
  // If the closest root is less than 0 or greater than 1, it isn't valid.
  // It's not clear that we can simply clamp the root to [0, 1], so we test
  // against v[0] and v[1] explicitly.

  T r = 0;
  T sq_dist = p.squaredDistanceTo(_v[0]);
  T const sq_dist1 = p.squaredDistanceTo(_v[1]);
  if (sq_dist1 < sq_dist) {
    r = 1;
    sq_dist = sq_dist1;
  }

  for (auto const rr : roots) {
    if (0 <= rr && rr <= 1) {
      auto const p_root = (*this)(rr);
      T const p_sq_dist = p.squaredDistanceTo(p_root);
      if (p_sq_dist < sq_dist) {
        r = rr;
        sq_dist = p_sq_dist;
      }
    }
  }
  return r;
}

//==============================================================================
// distanceTo
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::squaredDistanceTo(Vertex const & p) const noexcept -> T
{
  T const r = pointClosestTo(p);
  Vertex const p_closest = (*this)(r);
  return p_closest.squaredDistanceTo(p);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::distanceTo(Vertex const & p) const noexcept -> T
{
  return um2::sqrt(squaredDistanceTo(p));
}

//==============================================================================
// getRotation
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::getRotation() const noexcept -> Mat2x2<T>
  requires(D == 2)
{
  return LineSegment2<T>(_v[0], _v[1]).getRotation();
}

//==============================================================================
// getBezierControlPoint
//==============================================================================
// Get the control point for the quadratic Bezier curve that interpolates the
// given quadratic segment.

template <Int D, class T>
PURE HOSTDEV static constexpr auto
getBezierControlPoint(QuadraticSegment<D, T> const & q) noexcept -> Point<D, T>
{
  // p0 == v[0]
  // p2 == v[1]
  // p1 == 2 * v[2] - (v[0] + v[1]) / 2, hence we only need to compute p1
  return 2 * q[2] - (q[0] + q[1]) / 2;
}

//==============================================================================
// pointIsLeft
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::isLeft(Point2<T> const p) const noexcept -> bool
  requires(D == 2)
{
  // If the segment is straight, then we can use the line segment's isLeft method.
  // Note: LineSegment.isLeft(p) is equivalent to areCCW(v[0], v[1], p)
  if (isStraight(*this)) {
    return areCCW(_v[0], _v[1], p);
  }

  // The quadratic segment has an equivalent quadratic Bezier curve.
  // The triangle formed by the control points of the quadratic Bezier curve
  // bound the curve. If p is not in this triangle we can ignore the curvature of the
  // segment and return simply based upon areCCW(v[0], v[1], p).
  Vec2<T> const bcp = getBezierControlPoint(*this);
  Triangle2<T> tri(_v[0], _v[1], bcp);
  // precondition for tri.contains(p) is that tri.isCCW()
  if (!tri.isCCW()) {
    tri.flip();
  }
  ASSERT(tri.isCCW());
  if (!tri.contains(p)) {
    return areCCW(_v[0], _v[1], p);
  }

  // The point is in the bounding triangle, so we must perform further analysis.
  // To make the math easier, we transform the coordinate system so that v[0] is at the
  // origin and v[1] is on the x-axis. In particular, we want v[1][0] > 0.
  QuadraticSegment2<T> q(Point2<T>(0, 0), _v[1] - _v[0], _v[2] - _v[0]);
  Mat2x2<T> const rot = q.getRotation();
  q[1] = rot * q[1];
  q[2] = rot * q[2];
  // Successful rotation implies that q[1][1] == 0 and q[1][0] > 0.
  ASSERT(um2::abs(q[1][1]) < epsDistance<T>());
  ASSERT(q[1][0] > 0);
  auto p_r = rot * (p - _v[0]);

  // Orient the segment so that it curves to the right. (positive y-coordinate for v[2])
  bool const curves_right = q[2][1] >= 0;
  if (!curves_right) {
    // Flip the y-coordinates to be greater than or equal to zero
    q[2][1] = -q[2][1];
    p_r[1] = -p_r[1];
  }
  ASSERT(q[2][1] >= 0);
  // The point must be above the origin if it was in the bounding triangle.
  ASSERT(p_r[1] >= 0);

  // Now, since we may have flipped the y-coordinates, we will always need to
  // account for this when we return the result.
  // | isLeft | curves_right | final result |
  // |--------|--------------|--------------|
  // | true   | true         | true         |
  // | false  | true         | false        |
  // | true   | false        | false        |
  // | false  | false        | true         |
  // Or, simply: curves_right ? is_left : !is_left
  // This is a negated logical XOR operation.

  // Now that the segment is aligned with the x-axis, the bounding box of the segment
  // if fairly tight. If the point is not in the bounding box, then it is left of the
  // segment.
  auto const q_bb = q.boundingBox();
  if (!q_bb.contains(p_r)) {
    return curves_right;
  }

  // At this point, we know that the segment has non-trivial curvature and that the point
  // is in both the bounding triangle and the bounding box. Hence, we wish to determine
  // if the point is under the curve to determine if it is to the left of the segment.
  //
  // Find rx such that Q(rx)[0] = p_r[0].
  //
  // Q(r) = C + rB + r²A
  // In this case:
  // C = q[0] = (0, 0)
  // B = -q[1] + 4q[2]
  // A = 2q[1] - 4q[2] = -B + q[1]
  // Hence we wish to solve 0 = -P_x + rB_x + r²A_x for r.
  // This is a quadratic equation, which has two potential solutions.
  // 0 = c + br + ar²
  // r = (-b ± √(b² - 4ac)) / 2a
  // if A_x == 0, then we have a well-behaved quadratic segment, and there is one root.

  T const bx = -q[1][0] + 4 * q[2][0];
  T const by = 4 * q[2][1]; // q[1][1] == 0, q[2][1] > 0 implies by > 0
  T const ax = -bx + q[1][0];
  T const ay = -by; // q[1][1] == 0, by > 0 implies ay < 0
  ASSERT(by > 0);

  auto constexpr invalid_root = castIfNot<T>(1e15);
  auto const roots = solveQuadratic(ax, bx, -p_r[0]);
  // Only one root
  if (roots[1] > invalid_root) {
    // We know the point is in the bounding triangle, so we expect r to be in [0, 1]
    ASSERT(0 <= roots[0]);
    ASSERT(roots[0] <= 1);
    T const qy = roots[0] * (by + roots[0] * ay);
    bool const is_left = p_r[1] >= qy;
    return curves_right ? is_left : !is_left;
  }

  T const qy0 = roots[0] * (by + roots[0] * ay);
  T const qy1 = roots[1] * (by + roots[1] * ay);
  T const qymin = um2::min(qy0, qy1);
  T const qymax = um2::max(qy0, qy1);
  bool const contained_in_curve = qymin <= p_r[1] && p_r[1] <= qymax;
  bool const is_left = !contained_in_curve;
  return curves_right ? is_left : !is_left;
}

//==============================================================================
// intersect
//==============================================================================

// The ray: R(r) = O + rD
// The quadratic segment: Q(s) = C + sB + s²A,
//
// O + rD = C + sB + s²A                          subtracting C from both sides
// rD = (C - O) + sB + s²A                        cross product with D (distributive)
// 0 = (C - O) × D + s(B × D) + s²(A × D)
// The cross product of two vectors in the plane is a vector of the form (0, 0, k).
// Let a = (A × D)ₖ, b = (B × D)ₖ, and c = ((C - O) × D)ₖ
// 0 = as² + bs + c
// If a = 0
//   s = -c/b
// else
//   s = (-b ± √(b²-4ac))/2a
// s is invalid if b² < 4ac
// Once we have a valid s
// P = Q(s) = C + sB + s²A
// O + rD = P ⟹   rD = (C - O) + s(B + sA)
//            ⟹   r = ([(C - O) + s(B + sA)] ⋅ D)/(D ⋅ D)
// Since D is a unit vector, we can simplify the above expression to
// r = [(C - O) + s(B + sA)] ⋅ D
// r is valid if 0 ≤ r ≤ ∞

template <Int D, class T>
HOSTDEV constexpr auto
QuadraticSegment<D, T>::intersect(Ray2<T> const ray, T * const buffer) const noexcept -> Int
  requires(D == 2)
{
  auto const coeffs = getPolyCoeffs();

  Vec2<T> const vc = coeffs[0];
  Vec2<T> const vb = coeffs[1];
  Vec2<T> const va = coeffs[2];

  Vec2<T> const vd = ray.direction();
  Vec2<T> const vo = ray.origin();

  Vec2<T> const voc = vc - vo; // (C - O)

  T const a = va.cross(vd);  // (A × D)ₖ
  T const b = vb.cross(vd);  // (B × D)ₖ
  T const c = voc.cross(vd); // ((C - O) × D)ₖ

  Vec2<T> const s1s2 = solveQuadratic(a, b, c);
  Int hits = 0;
  for (Int i = 0; i < 2; ++i) {
    T const s = s1s2[i];
    if (0 <= s && s <= 1) {
      T const r = (voc + s * (vb + s * va)).dot(vd);
      if (0 <= r) {
        buffer[hits] = r;
        ++hits;
      }
    }
  }
  return hits;
}

//==============================================================================
// intersect (QuadraticSegment)
//==============================================================================

template <class T>
HOSTDEV constexpr auto
// NOLINTNEXTLINE(misc-no-recursion)
intersect(QuadraticSegment2<T> const & q1, QuadraticSegment2<T> const & q2, Point2<T> * buffer,
          Int iters = 0) noexcept -> Int
{
  // Scale each of the bounding boxes by a small amount to ensure
  // floating point errors do not cause the intersection to be missed.
  auto constexpr scaling = castIfNot<T>(1.015625);
  auto bb1 = q1.boundingBox();
  bb1.scale(scaling);
  auto bb2 = q2.boundingBox();
  bb2.scale(scaling);
  if (!bb1.intersects(bb2)) {
    return 0;
  }

  // If the number of iterations is greater than 4, we are reasonably sure that there
  // is an intersection. Use Newton's method to find the zero of the function quickly.
  // F(r, s) = Q1(r) - Q2(s)
  if (iters > 4) {
    Vec2<T> rs(0, 0);
    Vec2<T> f_rs = q1(rs[0]) - q2(rs[1]);
    T err_prev = f_rs.squaredNorm();
    T err_next = err_prev;
    Int nr_iters = 0;
    do {
      err_prev = err_next;
      Mat2x2<T> const jac = Mat2x2<T>(q1.jacobian(rs[0]), -q2.jacobian(rs[1]));
      // Manually compute the inverse of the 2x2 matrix, exiting early if the
      // determinant is zero.
      T const a = jac(0, 0);
      T const b = jac(0, 1);
      T const c = jac(1, 0);
      T const d = jac(1, 1);
      T const det = det2x2(a, b, c, d);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
      // NOLINTNEXTLINE(clang-diagnostic-float-equal)
      if (det == 0) {
        return 0;
      }
#pragma GCC diagnostic pop
      Mat2x2<T> const inv_jac(Vec2<T>{d / det, -c / det}, Vec2<T>{-b / det, a / det});
      rs -= inv_jac * f_rs;
      f_rs = q1(rs[0]) - q2(rs[1]);
      ++nr_iters;
      err_next = f_rs.squaredNorm();
    } while (err_next < err_prev && nr_iters < 10);

    // This has to be fuzzy, since we incur floating point errors as we bisect the
    // segment.
    auto constexpr err_tol = 100 * epsDistance<T>();
    if (err_next < err_tol && -err_tol <= rs[0] && rs[0] <= 1 + err_tol &&
        -err_tol <= rs[1] && rs[1] <= 1 + err_tol) {
      *buffer = q1(rs[0]);
      return 1;
    }
    return 0;
  }

  // QuadraticSegment bisection:
  // q0 ------- q(0.25) ------- q2 ------- q(0.75) ------- q1
  auto constexpr fourth = castIfNot<T>(0.25);
  auto constexpr three_fourths = castIfNot<T>(0.75);
  QuadraticSegment2<T> const q1l(q1[0], q1[2], q1(fourth));
  QuadraticSegment2<T> const q1r(q1[2], q1[1], q1(three_fourths));
  QuadraticSegment2<T> const q2l(q2[0], q2[2], q2(fourth));
  QuadraticSegment2<T> const q2r(q2[2], q2[1], q2(three_fourths));
  Int num_intersections = 0;
  ++iters;
  num_intersections += intersect(q1l, q2l, buffer, iters);
  num_intersections += intersect(q1l, q2r, buffer + num_intersections, iters);
  num_intersections += intersect(q1r, q2l, buffer + num_intersections, iters);
  num_intersections += intersect(q1r, q2r, buffer + num_intersections, iters);
  // Remove duplicates
  // For each intersection i, check if there is another intersection j such that
  // i == j. If so, remove j by moving the last intersection to j's position.
  for (Int i = 0; i < num_intersections; ++i) {
    for (Int j = i + 1; j < num_intersections; ++j) {
      if (buffer[i].isApprox(buffer[j])) {
        buffer[j] = buffer[num_intersections - 1];
        --num_intersections;
        --j;
      }
    }
  }
  ASSERT(num_intersections <= 2);
  ASSERT(num_intersections >= 0);
  return num_intersections;
}

template <class T>
PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(misc-no-recursion)
intersects(QuadraticSegment2<T> const & q1, QuadraticSegment2<T> const & q2,
           Int iters = 0) noexcept -> bool
{
  // Scale each of the bounding boxes by a small amount to ensure
  // floating point errors do not cause the intersection to be missed.
  auto constexpr scaling = castIfNot<T>(1.015625);
  auto bb1 = q1.boundingBox();
  bb1.scale(scaling);
  auto bb2 = q2.boundingBox();
  bb2.scale(scaling);
  if (!bb1.intersects(bb2)) {
    return false;
  }

  // If the number of iterations is greater than 4, we are reasonably sure that there
  // is an intersection. Use Newton's method to find the zero of the function quickly.
  // F(r, s) = Q1(r) - Q2(s)
  if (iters > 4) {
    Vec2<T> rs(0, 0);
    Vec2<T> f_rs = q1(rs[0]) - q2(rs[1]);
    T err_prev = f_rs.squaredNorm();
    T err_next = err_prev;
    Int nr_iters = 0;
    do {
      err_prev = err_next;
      Mat2x2<T> const jac = Mat2x2<T>(q1.jacobian(rs[0]), -q2.jacobian(rs[1]));
      // Manually compute the inverse of the 2x2 matrix, exiting early if the
      // determinant is zero.
      T const a = jac(0, 0);
      T const b = jac(0, 1);
      T const c = jac(1, 0);
      T const d = jac(1, 1);
      T const det = det2x2(a, b, c, d);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
      // NOLINTNEXTLINE(clang-diagnostic-float-equal)
      if (det == 0) {
        return false;
      }
#pragma GCC diagnostic pop
      Mat2x2<T> const inv_jac(Vec2<T>{d / det, -c / det}, Vec2<T>{-b / det, a / det});
      rs -= inv_jac * f_rs;
      f_rs = q1(rs[0]) - q2(rs[1]);
      ++nr_iters;
      err_next = f_rs.squaredNorm();
    } while (err_next < err_prev && nr_iters < 10);

    // This has to be fuzzy, since we incur floating point errors as we bisect the
    // segment.
    auto constexpr err_tol = 100 * epsDistance<T>();
    return (err_next < err_tol && -err_tol <= rs[0] && rs[0] <= 1 + err_tol &&
            -err_tol <= rs[1] && rs[1] <= 1 + err_tol);
  }

  // QuadraticSegment bisection:
  // q0 ------- q(0.25) ------- q2 ------- q(0.75) ------- q1
  auto constexpr fourth = castIfNot<T>(0.25);
  auto constexpr three_fourths = castIfNot<T>(0.75);
  QuadraticSegment2<T> const q1l(q1[0], q1[2], q1(fourth));
  QuadraticSegment2<T> const q1r(q1[2], q1[1], q1(three_fourths));
  QuadraticSegment2<T> const q2l(q2[0], q2[2], q2(fourth));
  QuadraticSegment2<T> const q2r(q2[2], q2[1], q2(three_fourths));
  ++iters;
  return (intersects(q1l, q2l, iters) || intersects(q1l, q2r, iters) ||
          intersects(q1r, q2l, iters) || intersects(q1r, q2r, iters));
}

template <Int D, class T>
HOSTDEV [[nodiscard]] constexpr auto
QuadraticSegment<D, T>::intersect(QuadraticSegment2<T> const & other,
                               Point2<T> * buffer) const noexcept -> Int
  requires(D == 2)
{
  return um2::intersect(*this, other, buffer);
}

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
QuadraticSegment<D, T>::intersects(QuadraticSegment2<T> const & other) const noexcept -> bool
  requires(D == 2)
{
  return um2::intersects(*this, other);
}

//==============================================================================
// enclosedArea
//==============================================================================

template <class T>
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

template <class T>
PURE HOSTDEV constexpr auto
enclosedCentroid(QuadraticSegment2<T> const & q) noexcept -> Vec2<T>
{
  // For a quadratic segment, with P₁ = (0, 0), P₂ = (x₂, 0), and P₃ = (x₃, y₃),
  // it can be shown that the centroid of the area bounded by the segment and x-axis
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
  // To transfrom the segment to the x-axis, for each point P,
  // Pᵤ = Uᵗ * (P - P₁)
  // or
  // P₁ᵤ = (0, 0)
  // P₂ᵤ = Uᵗ * (P₂ - P₁)
  // P₃ᵤ = Uᵗ * (P₃ - P₁)
  //
  // P₂ᵤx = u₁ ⋅ (P₂ - P₁)
  // P₃ᵤx = u₁ ⋅ (P₃ - P₁)
  // P₃ᵤy = u₂ ⋅ (P₃ - P₁)
  // Hence, the centroid after the transformation is given by
  // Cᵤ = (3P₂ᵤx + 4P₃ᵤx, 4P₃ᵤy) / 10
  //
  // To transform the centroid back to the standard basis, we use
  // C = U * Cᵤ + P₁
  Vec2<T> const v12 = q[1] - q[0];
  Vec2<T> const v13 = q[2] - q[0];
  Vec2<T> const u1 = v12.normalized();
  Vec2<T> const u2(-u1[1], u1[0]);
  Mat2x2<T> const u(u1, u2);
  T const p2ux = u1.dot(v12);
  T const p3ux = u1.dot(v13);
  T const p3uy = u2.dot(v13);
  Vec2<T> cu(3 * p2ux + 4 * p3ux, 4 * p3uy);
  T constexpr tenth = static_cast<T>(1) / static_cast<T>(10);
  cu *= tenth;
  return u * cu + q[0];
}

//==============================================================================
// curvesLeft
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::curvesLeft() const noexcept -> bool
  requires(D == 2)
{
  return areCCW(_v[0], _v[2], _v[1]);
}

} // namespace um2
