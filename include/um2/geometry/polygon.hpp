#pragma once

#include <um2/common/insertion_sort.hpp>
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

template <Int P, Int N, Int D>
class Polytope<2, P, N, D> // Polygon<P, N, D>
{
  static_assert((P == 1 && N == 3) || (P == 1 && N == 4) || (P == 2 && N == 6) || (P == 2 && N == 8),
                "Only triangles, quads, quadratic triangles, and quadratic quads are supported.");
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

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
  PURE HOSTDEV static constexpr auto
  numEdges() noexcept -> Int;

  // Returns the i-th vertex of the polygon.
  PURE HOSTDEV constexpr auto
  operator[](Int i) noexcept -> Vertex &;

  // Returns the i-th vertex of the polygon.
  PURE HOSTDEV constexpr auto
  operator[](Int i) const noexcept -> Vertex const &;

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

  HOSTDEV constexpr explicit Polytope(Vec<N, Vertex> const & v) noexcept;

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
  jacobian(R r, S s) const noexcept -> Mat<D, 2, Float>;

  // Get the i-th edge of the polygon.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Int i) const noexcept -> Edge;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2 const & p) const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  perimeter() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>;

  // If the polygon is counterclockwise oriented, returns true.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2 const & ray) const noexcept -> Vec<N, Float>
  requires(D == 2);

  // See the comments in the implementation for details.
  // meanChordLength has multiple definitions. Make sure you read the comments to
  // determine it's the one you want.
  PURE HOSTDEV [[nodiscard]] auto
  meanChordLength() const noexcept -> Float requires(D == 2);

}; // Polygon

//==============================================================================
// Constructors
//==============================================================================

template <Int P, Int N, Int D>
HOSTDEV constexpr Polygon<P, N, D>::Polytope(Vec<N, Vertex> const & v) noexcept
{
  for (Int i = 0; i < N; ++i) {
    _v[i] = v[i];
  }
}

//==============================================================================
// Accessors
//==============================================================================

template <Int P, Int N, Int D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::numEdges() noexcept -> Int
{
  static_assert(P == 1 || P == 2, "Only P = 1 or P = 2 supported");
  return N / P;
}

template <Int P, Int N, Int D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int P, Int N, Int D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::operator[](Int i) const noexcept -> Point<D> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int P, Int N, Int D>
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

template <Int D>
PURE HOSTDEV constexpr auto
linearPolygon(QuadraticTriangle<D> const & q) noexcept -> Triangle<D>
{
  return Triangle<D>(q[0], q[1], q[2]);
}

template <Int D>
PURE HOSTDEV constexpr auto
linearPolygon(QuadraticQuadrilateral<D> const & q) noexcept -> Quadrilateral<D>
{
  return Quadrilateral<D>(q[0], q[1], q[2], q[3]);
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(Triangle<D> const & tri, R const r, S const s) noexcept -> Point<D>
{
  // T(r, s) = (1 - r - s) v0 + r v1 + s v2
  auto const rr = static_cast<Float>(r);
  auto const ss = static_cast<Float>(s);
  Float const w0 = 1 - rr - ss;
  // Float const w1 = rr;
  // Float const w2 = ss;
  return w0 * tri[0] + rr * tri[1] + ss * tri[2]; 
}

template <Int D, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(Quadrilateral<D> const & quad, R const r, S const s) noexcept -> Point<D>
{
  // Q(r, s) =
  // (1 - r) (1 - s) v0 +
  // (    r) (1 - s) v1 +
  // (    r) (    s) v2 +
  // (1 - r) (    s) v3
  auto const rr = static_cast<Float>(r);
  auto const ss = static_cast<Float>(s);
  Float const w0 = (1 - rr) * (1 - ss);
  Float const w1 = rr * (1 - ss);
  Float const w2 = rr * ss;
  Float const w3 = (1 - rr) * ss;
  return w0 * quad[0] + w1 * quad[1] + w2 * quad[2] + w3 * quad[3];
}

template <Int D, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(QuadraticTriangle<D> const & tri6, R const r, S const s) noexcept -> Point<D>
{
  auto const rr = static_cast<Float>(r);
  auto const ss = static_cast<Float>(s);
  Float const tt = 1 - rr - ss;
  Float const w0 = tt * (2 * tt - 1);
  Float const w1 = rr * (2 * rr - 1);
  Float const w2 = ss * (2 * ss - 1);
  Float const w3 = 4 * rr * tt;
  Float const w4 = 4 * rr * ss;
  Float const w5 = 4 * ss * tt;
  return w0 * tri6[0] + w1 * tri6[1] + w2 * tri6[2] + w3 * tri6[3] + w4 * tri6[4] + w5 * tri6[5];
}

template <Int D, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(QuadraticQuadrilateral<D> const & quad8, R const r, S const s) noexcept
    -> Point<D>
{
  Float const xi = 2 * static_cast<Float>(r) - 1;
  Float const eta = 2 * static_cast<Float>(s) - 1;
  Float const w[8] = {(1 - xi) * (1 - eta) * (-xi - eta - 1) / 4,
                  (1 + xi) * (1 - eta) * (xi - eta - 1) / 4,
                  (1 + xi) * (1 + eta) * (xi + eta - 1) / 4,
                  (1 - xi) * (1 + eta) * (-xi + eta - 1) / 4,
                  (1 - xi * xi) * (1 - eta) / 2,
                  (1 - eta * eta) * (1 + xi) / 2,
                  (1 - xi * xi) * (1 + eta) / 2,
                  (1 - eta * eta) * (1 - xi) / 2};
  return w[0] * quad8[0] + w[1] * quad8[1] + 
         w[2] * quad8[2] + w[3] * quad8[3] + 
         w[4] * quad8[4] + w[5] * quad8[5] +
         w[6] * quad8[6] + w[7] * quad8[7];
}

template <Int P, Int N, Int D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::operator()(R const r, S const s) const noexcept -> Point<D>
{
  return interpolate(*this, r, s);
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(Triangle<D> const & t, R /*r*/, S /*s*/) noexcept -> Mat<D, 2, Float>
{
  return Mat<D, 2, Float>(t[1] - t[0], t[2] - t[0]);
}

template <Int D, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(Quadrilateral<D> const & q, R const r, S const s) noexcept -> Mat<D, 2, Float>
{
  // TODO(kcvaughn): Is this correct, or is it transposed?
  auto const rr = static_cast<Float>(r);
  auto const ss = static_cast<Float>(s);
  Float const w0 = 1 - ss;
  // Float const w1 = ss;
  Float const w2 = 1 - rr;
  // Float const w3 = rr;
  return Mat<D, 2, Float>(
    w0 * (q[1] - q[0]) - ss * (q[3] - q[2]),
    w2 * (q[3] - q[0]) - rr * (q[1] - q[2]));
}

template <Int D, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(QuadraticTriangle<D> const & t6, R const r, S const s) noexcept -> Mat<D, 2, Float>
{
  auto const rr = static_cast<Float>(4 * r);
  auto const ss = static_cast<Float>(4 * s);
  Float const tt = rr + ss - 3;
  return Mat<D, 2, Float>( 
   tt * (t6[0] - t6[3]) + (rr - 1) * (t6[1] - t6[3]) +
   ss * (t6[4] - t6[5]),
   tt * (t6[0] - t6[5]) + (ss - 1) * (t6[2] - t6[5]) +
   rr * (t6[4] - t6[3]));
}

template <Int D, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(QuadraticQuadrilateral<D> const & q, R const r, S const s) noexcept
    -> Mat<D, 2, Float>
{
  Float const xi = 2 * static_cast<Float>(r) - 1;
  Float const eta = 2 * static_cast<Float>(s) - 1;
  Float const xi_eta = xi * eta;
  Float const xi_xi = xi * xi;
  Float const eta_eta = eta * eta;
  Float const w0 = (eta - eta_eta) / 2;
  Float const w1 = (eta + eta_eta) / 2;
  Float const w2 = (xi - xi_eta);
  Float const w3 = (xi + xi_eta);
  Float const w4 = 1 - eta_eta;
  Float const w5 = (xi - xi_xi) / 2;
  Float const w6 = (xi + xi_xi) / 2;
  Float const w7 = eta - xi_eta;
  Float const w8 = eta + xi_eta;
  Float const w9 = 1 - xi_xi;
  return Mat<D, 2, Float>(
    w0 * (q[0] - q[1]) + 
    w1 * (q[2] - q[3]) + 
    w2 * (q[0] + q[1] - 2 * q[4]) +
    w3 * (q[2] + q[3] - 2 * q[6]) + 
    w4 * (q[5] - q[7]),
    w5 * (q[0] - q[3]) + 
    w6 * (q[2] - q[1]) +
    w7 * (q[0] + q[3] - 2 * q[7]) +
    w8 * (q[1] + q[2] - 2 * q[5]) + 
    w9 * (q[6] - q[4]));
}

template <Int P, Int N, Int D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::jacobian(R r, S s) const noexcept -> Mat<D, 2, Float>
{
  return um2::jacobian(*this, r, s);
}

//==============================================================================
// getEdge
//==============================================================================

template <Int N, Int D>
PURE HOSTDEV constexpr auto
getEdge(LinearPolygon<N, D> const & p, Int const i) noexcept -> LineSegment<D>
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return (i < N - 1) ? LineSegment<D>(p[i], p[i + 1]) : LineSegment<D>(p[N - 1], p[0]);
}

template <Int N, Int D>
PURE HOSTDEV constexpr auto
getEdge(QuadraticPolygon<N, D> const & p, Int const i) noexcept -> QuadraticSegment<D>
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N / 2);
  constexpr Int m = N / 2;
  return (i < m - 1) ? QuadraticSegment<D>(p[i], p[i + 1], p[i + m])
                     : QuadraticSegment<D>(p[m - 1], p[0], p[N - 1]);
}

template <Int P, Int N, Int D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::getEdge(Int i) const noexcept -> Edge
{
  return um2::getEdge(*this, i);
}

//==============================================================================
// contains
//==============================================================================

PURE HOSTDEV constexpr auto
contains(Triangle2 const & tri, Point2 const & p) noexcept -> bool
{
  Vec2F const a = tri[1] - tri[0];
  Vec2F const b = tri[2] - tri[0];
  Vec2F const c = p - tri[0];
  Float const invdet_ab = 1 / a.cross(b);
  Float const r = c.cross(b) * invdet_ab;
  Float const s = a.cross(c) * invdet_ab;
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

template <Int N>
PURE HOSTDEV constexpr auto
contains(PlanarQuadraticPolygon<N> const & q, Point2 const & p) noexcept -> bool
{
  // Benchmarking shows that the opposite conclusion is true for quadratic
  // polygons: it is faster to compute the areCCW() test for each edge, short
  // circuiting as soon as one is false, rather than compute all of them.
  Int constexpr num_edges = PlanarQuadraticPolygon<N>::numEdges();
  for (Int i = 0; i < num_edges; ++i) {
    if (!q.getEdge(i).isLeft(p)) {
      return false;
    }
  }
  return true;
}

template <Int P, Int N, Int D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::contains(Point2 const & p) const noexcept -> bool requires(D == 2)
{
  return um2::contains(*this, p);
}

//==============================================================================
// area
//==============================================================================

PURE HOSTDEV constexpr auto
area(Triangle2 const & tri) noexcept -> Float
{
  Vec2F const v10 = tri[1] - tri[0];
  Vec2F const v20 = tri[2] - tri[0];
  return v10.cross(v20) / 2; // this is the signed area
}

PURE HOSTDEV constexpr auto
area(Triangle3 const & tri) noexcept -> Float
{
  Vec3<Float> const v10 = tri[1] - tri[0];
  Vec3<Float> const v20 = tri[2] - tri[0];
  return v10.cross(v20).norm() / 2; // this is the unsigned area
}

PURE HOSTDEV constexpr auto
area(Quadrilateral2 const & q) noexcept -> Float
{
  ASSERT(isApproxConvex(q));
  // (v2 - v0).cross(v3 - v1) / 2
  Vec2F const v20 = q[2] - q[0];
  Vec2F const v31 = q[3] - q[1];
  return v20.cross(v31) / 2;
}

// Area of a planar linear polygon
template <Int N>
PURE HOSTDEV constexpr auto
area(PlanarLinearPolygon<N> const & p) noexcept -> Float
{
  // Shoelace forumla A = 1/2 * sum_{i=0}^{n-1} cross(p_i, p_{i+1})
  // p_n = p_0
  Float sum = (p[N - 1]).cross(p[0]); // cross(p_{n-1}, p_0), the last term
  for (Int i = 0; i < N - 1; ++i) {
    sum += (p[i]).cross(p[i + 1]);
  }
  return sum / 2;
}

template <Int N>
PURE HOSTDEV constexpr auto
area(PlanarQuadraticPolygon<N> const & q) noexcept -> Float
{
  Float result = area(linearPolygon(q));
  Int constexpr num_edges = PlanarQuadraticPolygon<N>::numEdges();
  for (Int i = 0; i < num_edges; ++i) {
    result += enclosedArea(q.getEdge(i));
  }
  return result;
}

template <Int P, Int N, Int D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::area() const noexcept -> Float
{
  return um2::area(*this);
}

//==============================================================================
// perimeter
//==============================================================================

template <Int P, Int N, Int D>
PURE HOSTDEV constexpr auto
perimeter(Polygon<P, N, D> const & p) noexcept -> Float
{
  Int constexpr num_edges = Polygon<P, N, D>::numEdges();
  Float result = p.getEdge(0).length();
  for (Int i = 1; i < num_edges; ++i) {
    result += p.getEdge(i).length();
  }
  return result;
}

template <Int P, Int N, Int D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::perimeter() const noexcept -> Float
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
  return (tri[0] + tri[1] + tri[2]) / 3;
}

PURE HOSTDEV constexpr auto
centroid(Triangle3 const & tri) noexcept -> Point3
{
  // (v0 + v1 + v2) / 3
  return (tri[0] + tri[1] + tri[2]) / 3;
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
  Vec2F const v10 = quad[1] - quad[0];
  Vec2F const v20 = quad[2] - quad[0];
  Vec2F const v30 = quad[3] - quad[0];
  // Compute the area of each triangle
  Float const a1 = v10.cross(v20);
  Float const a2 = v20.cross(v30);
  Float const a12 = a1 + a2;
  // Compute the centroid of each triangle
  // (v0 + v1 + v2) / 3
  // Each triangle shares v0 and v2, so we factor out the common terms
  return (a1 * quad[1] + a2 * quad[3] + a12 * (quad[0] + quad[2])) / (3 * a12);
}

// Centroid of a planar linear polygon
template <Int N>
PURE HOSTDEV constexpr auto
centroid(PlanarLinearPolygon<N> const & p) noexcept -> Point2
{
  // Similar to the shoelace formula.
  // C = 1/6A * sum_{i=0}^{n-1} cross(p_i, p_{i+1}) * (p_i + p_{i+1})
  Float area_sum = (p[N - 1]).cross(p[0]); // p_{n-1} x p_0, the last term
  Point2 centroid_sum = area_sum * (p[N - 1] + p[0]);
  for (Int i = 0; i < N - 1; ++i) {
    Float const a = (p[i]).cross(p[i + 1]);
    area_sum += a;
    centroid_sum += a * (p[i] + p[i + 1]);
  }
  return centroid_sum / (static_cast<Float>(3) * area_sum);
}

template <Int N>
PURE HOSTDEV constexpr auto
centroid(PlanarQuadraticPolygon<N> const & q) noexcept -> Point2
{
  auto lin_poly = linearPolygon(q);
  Float area_sum = lin_poly.area();
  Point2 centroid_sum = area_sum * centroid(lin_poly);
  Int constexpr num_edges = PlanarQuadraticPolygon<N>::numEdges();
  for (Int i = 0; i < num_edges; ++i) {
    auto const e = q.getEdge(i);
    Float const a = enclosedArea(e);
    area_sum += a;
    centroid_sum += a * enclosedCentroid(e);
  }
  return centroid_sum / area_sum;
}

template <Int P, Int N, Int D>
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

template <Int N>
PURE HOSTDEV constexpr auto
boundingBox(PlanarQuadraticPolygon<N> const & p) noexcept -> AxisAlignedBox2
{
  AxisAlignedBox2 box = p.getEdge(0).boundingBox();
  Int constexpr num_edges = PlanarQuadraticPolygon<N>::numEdges();
  for (Int i = 1; i < num_edges; ++i) {
    box += p.getEdge(i).boundingBox();
  }
  return box;
}

template <Int P, Int N, Int D>
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

template <Int N>
PURE HOSTDEV constexpr auto
isCCW(PlanarQuadraticPolygon<N> const & q) noexcept -> bool
{
  return isCCW(linearPolygon(q));
}

template <Int P, Int N, Int D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::isCCW() const noexcept -> bool requires(D == 2)
{
  return um2::isCCW(*this);
}

//==============================================================================
// intersect
//==============================================================================

template <Int N>
PURE HOSTDEV constexpr auto
intersect(PlanarLinearPolygon<N> const & p, Ray2 const & ray) noexcept -> Vec<N, Float>
{
  Vec<N, Float> result;
  for (Int i = 0; i < N; ++i) {
    result[i] = intersect(ray, p.getEdge(i));
  }
  return result;
}

template <Int N>
PURE HOSTDEV constexpr auto
intersect(PlanarQuadraticPolygon<N> const & p, Ray2 const & ray) noexcept -> Vec<N, Float>
{
  Vec<N, Float> result;
  for (Int i = 0; i < p.numEdges(); ++i) {
    Vec2F const v = intersect(ray, p.getEdge(i));
    result[2 * i] = v[0];
    result[2 * i + 1] = v[1];
  }
  return result;
}

template <Int P, Int N, Int D>
PURE HOSTDEV constexpr auto
Polygon<P, N, D>::intersect(Ray2 const & ray) const noexcept -> Vec<N, Float>
requires(D == 2) { return um2::intersect(*this, ray); }

//==============================================================================
// flipFace
//==============================================================================

template <Int D>
HOSTDEV constexpr void
flipFace(Triangle<D> & t) noexcept
{
  um2::swap(t[1], t[2]);
}

template <Int D>
HOSTDEV constexpr void
flipFace(Quadrilateral<D> & q) noexcept
{
  um2::swap(q[1], q[3]);
}

template <Int D>
HOSTDEV constexpr void
flipFace(QuadraticTriangle<D> & q) noexcept
{
  um2::swap(q[1], q[2]);
  um2::swap(q[3], q[5]);
}

template <Int D>
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
meanChordLength(Triangle2 const & tri) noexcept -> Float
{
  return pi<Float> * area(tri) / perimeter(tri);
}

PURE HOSTDEV constexpr auto
meanChordLength(Quadrilateral2 const & quad) noexcept -> Float
{
  ASSERT(isApproxConvex(quad));
  return pi<Float> * area(quad) / perimeter(quad);
}

template <Int N>
PURE HOSTDEV auto
meanChordLength(PlanarQuadraticPolygon<N> const & p) noexcept -> Float
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
  Int constexpr num_angles = 64; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 1000;

  Int total_chords = 0;
  Float total_length = 0;
  auto const aabb = boundingBox(p);
  auto const longest_edge = aabb.width() > aabb.height() ? aabb.width() : aabb.height();
  auto const spacing = longest_edge / static_cast<Float>(rays_per_longest_edge);
  Float const pi_deg = um2::pi_2<Float> / static_cast<Float>(num_angles);
  // For each angle
  for (Int ia = 0; ia < num_angles; ++ia) {
    // Try to avoid floating point error by accumulating the chord length locally
    Float local_accum = 0;
    Float const angle = pi_deg * static_cast<Float>(2 * ia + 1);
    // Compute modular ray parameters
    ModularRayParams const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      auto intersections = intersect(p, ray);
      um2::insertionSort(intersections.begin(), intersections.end());
      for (Int j = 0; j < N - 1; ++j) {
        Float const r1 = intersections[j + 1];
        if (r1 < um2::inf_distance / 10) {
          ASSERT(r1 - intersections[j] < um2::inf_distance / 100);
          ASSERT(r1 - intersections[j] > 0);
          local_accum += r1 - intersections[j];
          total_chords += 1;
        }
      }
    }
    total_length += local_accum;
  }
  return total_length / static_cast<Float>(total_chords);
}

template <Int P, Int N, Int D>
PURE HOSTDEV auto
Polygon<P, N, D>::meanChordLength() const noexcept -> Float requires(D == 2)
{
  return um2::meanChordLength(*this);
}

} // namespace um2
