#pragma once

#include <um2/common/log.hpp>
#include <um2/common/sort.hpp> // insertionSort
#include <um2/geometry/dion.hpp>
#include <um2/geometry/modular_rays.hpp>

//==============================================================================
// Polygon
//==============================================================================
// A 2-dimensional polytope, of polynomial order P, represented by the connectivity
// of its vertices. These N vertices are D-dimensional points of type F.
//
// For Polygons
//   Triangle (P = 1, N = 3)
//   Quadrilateral (P = 1, N = 4)
//   Quadratic Triangle (P = 2, N = 6)
//   Quadratic Quadrilateral (P = 2, N = 8)

namespace um2
{

template <Size P, Size N, Size D>
class Polytope<2, P, N, D>
{

public:
  using Vertex = Point<D>;
  using Edge = Dion<P, P + 1, D>;

private:
  Vertex _v[N];

public:
  //==============================================================================
  // Accessors
  //==============================================================================

  // Returns the number of edges in the polygon.
  CONST HOSTDEV static constexpr auto
  numEdges() noexcept -> Size;

  // Returns the i-th vertex of the polygon.
  PURE HOSTDEV constexpr auto
  operator[](Size i) noexcept -> Vertex &;

  // Returns the i-th vertex of the polygon.
  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> Vertex const &;

  // Returns a pointer to the vertex array.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  vertices() const noexcept -> Vertex const *;

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Polytope() noexcept = default;

  template <class... Pts>
    requires(sizeof...(Pts) == N && (std::same_as<Vertex, Pts> && ...))
  // NOLINTBEGIN(google-explicit-constructor) implicit conversion is desired
  HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }
  // NOLINTEND(google-explicit-constructor)

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
  // For triangles: r in [0, 1], s in [0, 1], constrained by r + s <= 1
  // For quads: r in [0, 1], s in [0, 1]
  // F(r, s) -> (x, y, z)
  template <typename R, typename S>
  PURE HOSTDEV constexpr auto
  operator()(R r, S s) const noexcept -> Point<D>;

  // J(r, s) -> [dF/dr, dF/ds]
  template <typename R, typename S>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R r, S s) const noexcept -> Mat<D, 2, F>;

  // Get the i-th edge of the polygon.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Size i) const noexcept -> Edge;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2 const & p) const noexcept -> bool
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> F;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  perimeter() const noexcept -> F;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>;

  // If the polygon is counterclockwise oriented, returns true.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2 const & ray) const noexcept -> Vec<N, F>
    requires(D == 2);

  // See the comments in the implementation for details.
  // meanChordLength has multiple definitions. Make sure you read the comments to
  // determine it's the one you want.
  PURE HOSTDEV [[nodiscard]] auto
  meanChordLength() const noexcept -> F
    requires(D == 2);

}; // Polygon

//==============================================================================
// Accessors
//==============================================================================

template <Size P, Size N, Size D>
CONST HOSTDEV constexpr auto
Polygon<P, N, D>::numEdges() noexcept -> Size
{
  static_assert(P == 1 || P == 2, "Only P = 1 or P = 2 supported");
  return N / P;
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::operator[](Size i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::operator[](Size i) const noexcept -> Point<D> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::vertices() const noexcept -> Point<D> const *
{
  return _v;
}

//==============================================================================
// isConvex
//==============================================================================

PURE HOSTDEV constexpr auto
isConvex(Quadrilateral2 const & q) noexcept -> bool
{
  // Benchmarking shows it is faster to compute the areCCW() test for each
  // edge, then return based on the AND of the results, rather than compute
  // the areCCW one at a time and return as soon as one is false.
  bool const b0 = areCCW(q[0], q[1], q[2]);
  bool const b1 = areCCW(q[1], q[2], q[3]);
  bool const b2 = areCCW(q[2], q[3], q[0]);
  bool const b3 = areCCW(q[3], q[0], q[1]);
  return b0 && b1 && b2 && b3;
}

PURE HOSTDEV constexpr auto
isApproxConvex(Quadrilateral2 const & q) noexcept -> bool
{
  // Benchmarking shows it is faster to compute the areCCW() test for each
  // edge, then return based on the AND of the results, rather than compute
  // the areCCW one at a time and return as soon as one is false.
  bool const b0 = areApproxCCW(q[0], q[1], q[2]);
  bool const b1 = areApproxCCW(q[1], q[2], q[3]);
  bool const b2 = areApproxCCW(q[2], q[3], q[0]);
  bool const b3 = areApproxCCW(q[3], q[0], q[1]);
  return b0 && b1 && b2 && b3;
}

//==============================================================================
// linearPolygon
//==============================================================================

template <Size D>
PURE HOSTDEV constexpr auto
linearPolygon(QuadraticTriangle<D> const & q) noexcept -> Triangle<D>
{
  return Triangle<D>(q[0], q[1], q[2]);
}

template <Size D>
PURE HOSTDEV constexpr auto
linearPolygon(QuadraticQuadrilateral<D> const & q) noexcept -> Quadrilateral<D>
{
  return Quadrilateral<D>(q[0], q[1], q[2], q[3]);
}

//==============================================================================
// interpolate
//==============================================================================

template <Size D, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(Triangle<D> const & tri, R const r, S const s) noexcept -> Point<D>
{
  // T(r, s) = (1 - r - s) v0 + r v1 + s v2
  F const rr = static_cast<F>(r);
  F const ss = static_cast<F>(s);
  F const w0 = 1 - rr - ss;
  // F const w1 = rr;
  // F const w2 = ss;
  Point<D> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * tri[0][i] + rr * tri[1][i] + ss * tri[2][i];
  }
  return result;
}

template <Size D, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(Quadrilateral<D> const & quad, R const r, S const s) noexcept
    -> Point<D>
{
  // Q(r, s) =
  // (1 - r) (1 - s) v0 +
  // (    r) (1 - s) v1 +
  // (    r) (    s) v2 +
  // (1 - r) (    s) v3
  F const rr = static_cast<F>(r);
  F const ss = static_cast<F>(s);
  F const w0 = (1 - rr) * (1 - ss);
  F const w1 = rr * (1 - ss);
  F const w2 = rr * ss;
  F const w3 = (1 - rr) * ss;
  Point<D> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * quad[0][i] + w1 * quad[1][i] + w2 * quad[2][i] + w3 * quad[3][i];
  }
  return result;
}

template <Size D, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(QuadraticTriangle<D> const & tri6, R const r, S const s) noexcept
    -> Point<D>
{
  F const rr = static_cast<F>(r);
  F const ss = static_cast<F>(s);
  F const tt = 1 - rr - ss;
  F const w0 = tt * (2 * tt - 1);
  F const w1 = rr * (2 * rr - 1);
  F const w2 = ss * (2 * ss - 1);
  F const w3 = 4 * rr * tt;
  F const w4 = 4 * rr * ss;
  F const w5 = 4 * ss * tt;
  Point<D> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * tri6[0][i] + w1 * tri6[1][i] + w2 * tri6[2][i] + w3 * tri6[3][i] +
                w4 * tri6[4][i] + w5 * tri6[5][i];
  }
  return result;
}

template <Size D, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(QuadraticQuadrilateral<D> const & quad8, R const r, S const s) noexcept
    -> Point<D>
{
  F const xi = 2 * static_cast<F>(r) - 1;
  F const eta = 2 * static_cast<F>(s) - 1;
  F const w[8] = {(1 - xi) * (1 - eta) * (-xi - eta - 1) / 4,
                  (1 + xi) * (1 - eta) * (xi - eta - 1) / 4,
                  (1 + xi) * (1 + eta) * (xi + eta - 1) / 4,
                  (1 - xi) * (1 + eta) * (-xi + eta - 1) / 4,
                  (1 - xi * xi) * (1 - eta) / 2,
                  (1 - eta * eta) * (1 + xi) / 2,
                  (1 - xi * xi) * (1 + eta) / 2,
                  (1 - eta * eta) * (1 - xi) / 2};
  Point<D> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w[0] * quad8[0][i] + w[1] * quad8[1][i] + w[2] * quad8[2][i] +
                w[3] * quad8[3][i] + w[4] * quad8[4][i] + w[5] * quad8[5][i] +
                w[6] * quad8[6][i] + w[7] * quad8[7][i];
  }
  return result;
}

template <Size P, Size N, Size D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::operator()(R const r, S const s) const noexcept -> Point<D>
{
  return interpolate(*this, r, s);
}

//==============================================================================
// jacobian
//==============================================================================

template <Size D, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(Triangle<D> const & t, R /*r*/, S /*s*/) noexcept -> Mat<D, 2, F>
{
  return Mat<D, 2, F>(t[1] - t[0], t[2] - t[0]);
}

template <Size D, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(Quadrilateral<D> const & q, R const r, S const s) noexcept -> Mat<D, 2, F>
{
  // jac.col(0) = (v1 - v0) - s (v3 - v2)
  // jac.col(1) = (v3 - v0) - r (v1 - v2)
  F const rr = static_cast<F>(r);
  F const ss = static_cast<F>(s);
  F const w0 = 1 - ss;
  // F const w1 = ss;
  F const w2 = 1 - rr;
  // F const w3 = rr;
  Mat<D, 2, F> jac;
  for (Size i = 0; i < D; ++i) {
    jac(i, 0) = w0 * (q[1][i] - q[0][i]) - ss * (q[3][i] - q[2][i]);
    jac(i, 1) = w2 * (q[3][i] - q[0][i]) - rr * (q[1][i] - q[2][i]);
  }
  return jac;
}

template <Size D, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(QuadraticTriangle<D> const & t6, R const r, S const s) noexcept
    -> Mat<D, 2, F>
{
  F const rr = static_cast<F>(4 * r);
  F const ss = static_cast<F>(4 * s);
  F const tt = rr + ss - 3;
  Mat<D, 2, F> result;
  for (Size i = 0; i < D; ++i) {
    result.col(0)[i] = tt * (t6[0][i] - t6[3][i]) + (rr - 1) * (t6[1][i] - t6[3][i]) +
                       ss * (t6[4][i] - t6[5][i]);
    result.col(1)[i] = tt * (t6[0][i] - t6[5][i]) + (ss - 1) * (t6[2][i] - t6[5][i]) +
                       rr * (t6[4][i] - t6[3][i]);
  }
  return result;
}

template <Size D, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(QuadraticQuadrilateral<D> const & q, R const r, S const s) noexcept
    -> Mat<D, 2, F>
{
  F const xi = 2 * static_cast<F>(r) - 1;
  F const eta = 2 * static_cast<F>(s) - 1;
  F const xi_eta = xi * eta;
  F const xi_xi = xi * xi;
  F const eta_eta = eta * eta;
  F const w0 = (eta - eta_eta) / 2;
  F const w1 = (eta + eta_eta) / 2;
  F const w2 = (xi - xi_eta);
  F const w3 = (xi + xi_eta);
  F const w4 = 1 - eta_eta;
  F const w5 = (xi - xi_xi) / 2;
  F const w6 = (xi + xi_xi) / 2;
  F const w7 = eta - xi_eta;
  F const w8 = eta + xi_eta;
  F const w9 = 1 - xi_xi;
  Mat<D, 2, F> result;
  for (Size i = 0; i < D; ++i) {
    result.col(0)[i] = w0 * (q[0][i] - q[1][i]) + w1 * (q[2][i] - q[3][i]) +
                       w2 * (q[0][i] + q[1][i] - 2 * q[4][i]) +
                       w3 * (q[2][i] + q[3][i] - 2 * q[6][i]) + w4 * (q[5][i] - q[7][i]);
    result.col(1)[i] = w5 * (q[0][i] - q[3][i]) + w6 * (q[2][i] - q[1][i]) +
                       w7 * (q[0][i] + q[3][i] - 2 * q[7][i]) +
                       w8 * (q[1][i] + q[2][i] - 2 * q[5][i]) + w9 * (q[6][i] - q[4][i]);
  }
  return result;
}

template <Size P, Size N, Size D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::jacobian(R r, S s) const noexcept -> Mat<D, 2, F>
{
  return um2::jacobian(*this, r, s);
}

//==============================================================================
// getEdge
//==============================================================================

template <Size N, Size D>
PURE HOSTDEV constexpr auto
getEdge(LinearPolygon<N, D> const & p, Size const i) noexcept -> LineSegment<D>
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return (i < N - 1) ? LineSegment<D>(p[i], p[i + 1])
                     : LineSegment<D>(p[N - 1], p[0]);
}

template <Size N, Size D>
PURE HOSTDEV constexpr auto
getEdge(QuadraticPolygon<N, D> const & p, Size const i) noexcept
    -> QuadraticSegment<D>
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N / 2);
  constexpr Size m = N / 2;
  return (i < m - 1) ? QuadraticSegment<D>(p[i], p[i + 1], p[i + m])
                     : QuadraticSegment<D>(p[m - 1], p[0], p[N - 1]);
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::getEdge(Size i) const noexcept -> Edge
{
  return um2::getEdge(*this, i);
}

//==============================================================================
// contains
//==============================================================================

PURE HOSTDEV constexpr auto
contains(Triangle2 const & tri, Point2 const & p) noexcept -> bool
{
  um2::Vec2<F> const a = tri[1] - tri[0];
  um2::Vec2<F> const b = tri[2] - tri[0];
  um2::Vec2<F> const c = p - tri[0];
  F const invdet_ab = 1 / a.cross(b);
  F const r = c.cross(b) * invdet_ab;
  F const s = a.cross(c) * invdet_ab;
  return (r >= 0) && (s >= 0) && (r + s <= 1);
}

PURE HOSTDEV constexpr auto
contains(Quadrilateral2 const & q, Point2 const & p) noexcept -> bool
{
  bool const b0 = areCCW(q[0], q[1], p);
  bool const b1 = areCCW(q[1], q[2], p);
  bool const b2 = areCCW(q[2], q[3], p);
  bool const b3 = areCCW(q[3], q[0], p);
  return b0 && b1 && b2 && b3;
}

template <Size N>
PURE HOSTDEV constexpr auto
contains(PlanarQuadraticPolygon<N> const & q, Point2 const & p) noexcept -> bool
{
  // Benchmarking shows that the opposite conclusion is true for quadratic
  // polygons: it is faster to compute the areCCW() test for each edge, short
  // circuiting as soon as one is false, rather than compute all of them.
  Size constexpr num_edges = PlanarQuadraticPolygon<N>::numEdges();
  for (Size i = 0; i < num_edges; ++i) {
    if (!q.getEdge(i).isLeft(p)) {
      return false;
    }
  }
  return true;
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::contains(Point2 const & p) const noexcept -> bool
  requires(D == 2)
{
  return um2::contains(*this, p);
}

//==============================================================================
// area
//==============================================================================

PURE HOSTDEV constexpr auto
area(Triangle2 const & tri) noexcept -> F
{
  Vec2<F> const v10 = tri[1] - tri[0];
  Vec2<F> const v20 = tri[2] - tri[0];
  return v10.cross(v20) / 2; // this is the signed area
}

PURE HOSTDEV constexpr auto
area(Triangle3 const & tri) noexcept -> F
{
  Vec3<F> const v10 = tri[1] - tri[0];
  Vec3<F> const v20 = tri[2] - tri[0];
  return v10.cross(v20).norm() / 2; // this is the unsigned area
}

PURE HOSTDEV constexpr auto
area(Quadrilateral2 const & q) noexcept -> F
{
  ASSERT(isApproxConvex(q));
  // (v2 - v0).cross(v3 - v1) / 2
  Vec2<F> const v20 = q[2] - q[0];
  Vec2<F> const v31 = q[3] - q[1];
  return v20.cross(v31) / 2;
}

// Area of a planar linear polygon
template <Size N>
PURE HOSTDEV constexpr auto
area(PlanarLinearPolygon<N> const & p) noexcept -> F
{
  // Shoelace forumla A = 1/2 * sum_{i=0}^{n-1} cross(p_i, p_{i+1})
  // p_n = p_0
  F sum = (p[N - 1]).cross(p[0]); // cross(p_{n-1}, p_0), the last term
  for (Size i = 0; i < N - 1; ++i) {
    sum += (p[i]).cross(p[i + 1]);
  }
  return sum / 2;
}

template <Size N>
PURE HOSTDEV constexpr auto
area(PlanarQuadraticPolygon<N> const & q) noexcept -> F
{
  F result = area(linearPolygon(q));
  Size constexpr num_edges = PlanarQuadraticPolygon<N>::numEdges();
  for (Size i = 0; i < num_edges; ++i) {
    result += enclosedArea(q.getEdge(i));
  }
  return result;
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::area() const noexcept -> F
{
  return um2::area(*this);
}

//==============================================================================
// perimeter
//==============================================================================

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
perimeter(Polygon<P, N, D> const & p) noexcept -> F
{
  Size constexpr num_edges = Polygon<P, N, D>::numEdges();
  F result = p.getEdge(0).length();
  for (Size i = 1; i < num_edges; ++i) {
    result += p.getEdge(i).length();
  }
  return result;
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::perimeter() const noexcept -> F
{
  return um2::perimeter(*this);
}

//==============================================================================
// centroid
//==============================================================================

PURE HOSTDEV constexpr auto
centroid(Triangle2 const & tri) noexcept -> Point2
{
  // (v0 + v1 + v2) / 3
  Point2 result;
  for (Size i = 0; i < 2; ++i) {
    result[i] = tri[0][i] + tri[1][i] + tri[2][i];
  }
  return result /= 3;
}

PURE HOSTDEV constexpr auto
centroid(Triangle3 const & tri) noexcept -> Point3
{
  // (v0 + v1 + v2) / 3
  Point3 result;
  for (Size i = 0; i < 3; ++i) {
    result[i] = tri[0][i] + tri[1][i] + tri[2][i];
  }
  return result /= 3;
}

PURE HOSTDEV constexpr auto
centroid(Quadrilateral2 const & quad) noexcept -> Point2
{
  // Algorithm: Decompose the quadrilateral into two triangles and
  // compute the centroid of each triangle. The centroid of the
  // quadrilateral is the weighted average of the centroids of the
  // two triangles, where the weights are the areas of the triangles.
  ASSERT(isApproxConvex(quad));
  // If the quadrilateral is not convex, then we need to choose the correct
  // two triangles to decompose the quadrilateral into. If the quadrilateral
  // is convex, any two triangles will do.
  Vec2<F> const v10 = quad[1] - quad[0];
  Vec2<F> const v20 = quad[2] - quad[0];
  Vec2<F> const v30 = quad[3] - quad[0];
  // Compute the area of each triangle
  F const a1 = v10.cross(v20);
  F const a2 = v20.cross(v30);
  F const a12 = a1 + a2;
  // Compute the centroid of each triangle
  // (v0 + v1 + v2) / 3
  // Each triangle shares v0 and v2, so we factor out the common terms
  Point2 result;
  for (Size i = 0; i < 2; ++i) {
    result[i] = a1 * quad[1][i] + a2 * quad[3][i] + a12 * (quad[0][i] + quad[2][i]);
  }
  return result /= (3 * a12);
}

// Centroid of a planar linear polygon
template <Size N>
PURE HOSTDEV constexpr auto
centroid(PlanarLinearPolygon<N> const & p) noexcept -> Point2
{
  // Similar to the shoelace formula.
  // C = 1/6A * sum_{i=0}^{n-1} cross(p_i, p_{i+1}) * (p_i + p_{i+1})
  F area_sum = (p[N - 1]).cross(p[0]); // p_{n-1} x p_0, the last term
  Point2 centroid_sum = area_sum * (p[N - 1] + p[0]);
  for (Size i = 0; i < N - 1; ++i) {
    F const a = (p[i]).cross(p[i + 1]);
    area_sum += a;
    centroid_sum += a * (p[i] + p[i + 1]);
  }
  return centroid_sum / (static_cast<F>(3) * area_sum);
}

template <Size N>
PURE HOSTDEV constexpr auto
centroid(PlanarQuadraticPolygon<N> const & q) noexcept -> Point2
{
  auto lin_poly = linearPolygon(q);
  F area_sum = lin_poly.area();
  Point2 centroid_sum = area_sum * centroid(lin_poly);
  Size constexpr num_edges = PlanarQuadraticPolygon<N>::numEdges();
  for (Size i = 0; i < num_edges; ++i) {
    auto const e = q.getEdge(i);
    F const a = enclosedArea(e);
    area_sum += a;
    centroid_sum += a * enclosedCentroid(e);
  }
  return centroid_sum / area_sum;
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::centroid() const noexcept -> Point<D>
{
  return um2::centroid(*this);
}

//==============================================================================
// boundingBox
//==============================================================================

// Defined in Polytope.hpp for linear polygons, since for all linear polytopes
// the bounding box is simply the bounding box of the vertices.

template <Size N>
PURE HOSTDEV constexpr auto
boundingBox(PlanarQuadraticPolygon<N> const & p) noexcept -> AxisAlignedBox2
{
  AxisAlignedBox2 box = p.getEdge(0).boundingBox();
  Size constexpr num_edges = PlanarQuadraticPolygon<N>::numEdges();
  for (Size i = 1; i < num_edges; ++i) {
    box += p.getEdge(i).boundingBox();
  }
  return box;
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::boundingBox() const noexcept -> AxisAlignedBox<D>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// isCCW
//==============================================================================

PURE HOSTDEV constexpr auto
isCCW(Triangle2 const & t) noexcept -> bool
{
  return areCCW(t[0], t[1], t[2]);
}

PURE HOSTDEV constexpr auto
isCCW(Quadrilateral2 const & q) noexcept -> bool
{
  bool const b0 = areCCW(q[0], q[1], q[2]);
  bool const b1 = areCCW(q[0], q[2], q[3]);
  return b0 && b1;
}

template <Size N>
PURE HOSTDEV constexpr auto
isCCW(PlanarQuadraticPolygon<N> const & q) noexcept -> bool
{
  return isCCW(linearPolygon(q));
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::isCCW() const noexcept -> bool
  requires(D == 2)
{
  return um2::isCCW(*this);
}

//==============================================================================
// intersect
//==============================================================================

template <Size N>
PURE HOSTDEV constexpr auto
intersect(PlanarLinearPolygon<N> const & p, Ray2 const & ray) noexcept -> Vec<N, F>
{
  Vec<N, F> result;
  for (Size i = 0; i < N; ++i) {
    result[i] = intersect(ray, p.getEdge(i));
  }
  return result;
}

template <Size N>
PURE HOSTDEV constexpr auto
intersect(PlanarQuadraticPolygon<N> const & p, Ray2 const & ray) noexcept
    -> Vec<N, F>
{
  Vec<N, F> result;
  for (Size i = 0; i < p.numEdges(); ++i) {
    Vec2<F> const v = intersect(ray, p.getEdge(i));
    result[2 * i] = v[0];
    result[2 * i + 1] = v[1];
  }
  return result;
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::intersect(Ray2 const & ray) const noexcept -> Vec<N, F>
  requires(D == 2)
{
  return um2::intersect(*this, ray);
}

//==============================================================================
// flipFace
//==============================================================================

template <Size D>
HOSTDEV constexpr void
flipFace(Triangle<D> & t) noexcept
{
  um2::swap(t[1], t[2]);
}

template <Size D>
HOSTDEV constexpr void
flipFace(Quadrilateral<D> & q) noexcept
{
  um2::swap(q[1], q[3]);
}

template <Size D>
HOSTDEV constexpr void
flipFace(QuadraticTriangle<D> & q) noexcept
{
  um2::swap(q[1], q[2]);
  um2::swap(q[3], q[5]);
}

template <Size D>
HOSTDEV constexpr void
flipFace(QuadraticQuadrilateral<D> & q) noexcept
{
  um2::swap(q[1], q[3]);
  um2::swap(q[4], q[7]);
}

//==============================================================================
// meanChordLength
//==============================================================================
// For a convex planar polygon, the mean chord length is simply pi * area / perimeter.
// De Kruijf, W. J. M., and J. L. Kloosterman.
// "On the average chord length in reactor physics." Annals of Nuclear Energy 30.5 (2003):
// 549-553.
//
// For a non-convex polygon, we shoot modular rays from the bounding box and average.

PURE HOSTDEV constexpr auto
meanChordLength(Triangle2 const & tri) noexcept -> F
{
  return pi<F> * area(tri) / perimeter(tri);
}

PURE HOSTDEV constexpr auto
meanChordLength(Quadrilateral2 const & quad) noexcept -> F
{
  ASSERT(isApproxConvex(quad));
  return pi<F> * area(quad) / perimeter(quad);
}

template <Size N>
PURE HOSTDEV auto
meanChordLength(PlanarQuadraticPolygon<N> const & p) noexcept -> F
{
  // Algorithm:
  // For equally spaced angles γ ∈ (0, π)
  //  Compute modular ray parameters
  //  For each ray
  //    Compute intersections with edges
  //    Compute chord length
  //    total_chord_length += chord_length
  //    total_chords += 1
  // return total_chord_length / total_chords

  // Parameters
  Size constexpr num_angles = 128; // Angles γ ∈ (0, π).
  Size constexpr rays_per_longest_edge = 1000;

  Size total_chords = 0;
  F total_length = 0; 
  auto const aabb = boundingBox(p);
  auto const longest_edge = aabb.width() > aabb.height() ? aabb.width() : aabb.height();
  auto const spacing = longest_edge / static_cast<F>(rays_per_longest_edge);
  F const pi_deg = um2::pi_2<F> / static_cast<F>(num_angles);
  // For each angle
  for (Size ia = 0; ia < num_angles; ++ia) {
    F const angle = pi_deg * static_cast<F>(2 * ia + 1);
    // Compute modular ray parameters
    ModularRayParams const params(angle, spacing, aabb);
    Size const num_rays = params.getTotalNumRays();
    // For each ray
    for (Size i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      auto intersections = intersect(p, ray);
      um2::insertionSort(intersections.begin(), intersections.end());
      // if (intersections[0] < 0) {
      //   ASSERT(intersections[0] > -um2::eps_distance2<F>);
      //   ASSERT(intersections[1] > 0);
      //   // If the first intersection is negative, it better be -0
      //   intersections[0] = 0;
      // }
      auto p0 = ray(intersections[0]);
      for (Size j = 0; j < intersections.size() - 1; ++j) {
        F const r1 = intersections[j + 1];
        // A miss is indicated with inf_distance. We use a smaller value to avoid
        // numerical issues with direct comparison to inf_distance.
        if (r1 < um2::inf_distance / 10) {
          auto const p1 = ray(r1);
          F const len = p0.distanceTo(p1);
          p0 = p1;
          total_length += len;
          total_chords += 1;
        }
      }
    }
  }
  return total_length / static_cast<F>(total_chords);
}

template <Size P, Size N, Size D>
PURE HOSTDEV auto
Polygon<P, N, D>::meanChordLength() const noexcept -> F
  requires(D == 2)
{
  return um2::meanChordLength(*this);
}

} // namespace um2
