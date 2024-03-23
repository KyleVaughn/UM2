#pragma once

#include <um2/common/insertion_sort.hpp>
#include <um2/geometry/quadratic_segment.hpp>
#include <um2/geometry/triangle.hpp>
#include <um2/geometry/modular_rays.hpp>

//==============================================================================
// QUADRATIC TRIANGLE
//==============================================================================

namespace um2
{

template <Int D>
class Polytope<2, 2, 6, D>
{
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

public:
  using Vertex = Point<D>;
  using Edge = QuadraticSegment<D>;

private:
  Vertex _v[6];

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
  requires(sizeof...(Pts) == 6 && (std::same_as<Vertex, Pts> && ...))
      // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
      HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  HOSTDEV constexpr explicit Polytope(Vec<6, Vertex> const & v) noexcept;

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
  // For triangles: r in [0, 1], s in [0, 1], constrained by r + s <= 1
  // F(r, s) -> (x, y, z)
  template <typename R, typename S>
  PURE HOSTDEV constexpr auto
  operator()(R r, S s) const noexcept -> Point<D>;

  // Get the i-th edge of the polygon.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Int i) const noexcept -> Edge;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2 const & p) const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  linearPolygon() const noexcept -> Triangle<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> Float
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  perimeter() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D>
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>
  requires(D == 2);

  // If the polygon is counterclockwise oriented, returns true.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isConvex() const noexcept -> bool requires(D == 2);

  HOSTDEV constexpr void
  flip() noexcept;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2 const & ray) const noexcept -> Vec<6, Float>
  requires(D == 2);

  // See the comments in the implementation for details.
  // meanChordLength has multiple definitions. Make sure you read the comments to
  // determine it's the one you want.
  PURE HOSTDEV [[nodiscard]] auto
  meanChordLength() const noexcept -> Float requires(D == 2);

}; // QuadraticTriangle

//==============================================================================
// Constructors
//==============================================================================

template <Int D>
HOSTDEV constexpr QuadraticTriangle<D>::Polytope(Vec<6, Vertex> const & v) noexcept
{
  for (Int i = 0; i < 6; ++i) {
    _v[i] = v[i];
  }
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::numEdges() noexcept -> Int
{
  return 3;
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 6);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::operator[](Int i) const noexcept -> Point<D> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 6);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::vertices() const noexcept -> Point<D> const *
{
  return _v;
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::operator()(R const r, S const s) const noexcept -> Point<D>
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
  return w0 * _v[0] + w1 * _v[1] + w2 * _v[2] + w3 * _v[3] + w4 * _v[4] + w5 * _v[5];
}

//==============================================================================
// getEdge
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::getEdge(Int i) const noexcept -> Edge
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 3);
  return (i < 2) ? QuadraticSegment<D>(_v[i], _v[i + 1], _v[i + 3])
                 : QuadraticSegment<D>(_v[2], _v[0], _v[5]);
}

//==============================================================================
// contains
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::contains(Point2 const & p) const noexcept -> bool requires(D == 2)
{
  for (Int i = 0; i < 3; ++i) {
    if (!getEdge(i).isLeft(p)) {
      return false;
    }
  }
  return true;
}

//==============================================================================
// linearPolygon
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::linearPolygon() const noexcept -> Triangle<D>
{
  return Triangle<D>(_v[0], _v[1], _v[2]);
}

//==============================================================================
// area
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::area() const noexcept -> Float
requires(D == 2)
{
  Float result = linearPolygon().area();
  result += enclosedArea(getEdge(0));
  result += enclosedArea(getEdge(1));
  result += enclosedArea(getEdge(2));
  return result;
}

//==============================================================================
// perimeter
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::perimeter() const noexcept -> Float
{
  Float result = getEdge(0).length();
  result += getEdge(1).length();
  result += getEdge(2).length();
  return result;
}

//==============================================================================
// centroid
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::centroid() const noexcept -> Point<D>
requires(D == 2)
{
  auto lin_poly = linearPolygon();
  Float area_sum = lin_poly.area();
  Point2 centroid_sum = area_sum * lin_poly.centroid();
  for (Int i = 0; i < 3; ++i) {
    auto const e = getEdge(i);
    Float const a = enclosedArea(e);
    area_sum += a;
    centroid_sum += a * enclosedCentroid(e);
  }
  return centroid_sum / area_sum;
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
requires(D == 2)
{
  AxisAlignedBox2 box = getEdge(0).boundingBox();
  box += getEdge(1).boundingBox();
  box += getEdge(2).boundingBox();
  return box;
}

//==============================================================================
// isCCW
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::isCCW() const noexcept -> bool requires(D == 2)
{
  return linearPolygon().isCCW();
}

//==============================================================================
// isConvex
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::isConvex() const noexcept -> bool requires(D == 2)
{
  // If each edge is either straight, or curves left. 
  // AND the linear polygon polygon is convex.
  auto const e0 = getEdge(0);
  auto const e1 = getEdge(1);
  auto const e2 = getEdge(2);
  bool const s_or_cl0 = isStraight(e0) || e0.curvesLeft(); 
  bool const s_or_cl1 = isStraight(e1) || e1.curvesLeft();
  bool const s_or_cl2 = isStraight(e2) || e2.curvesLeft();
  bool const edges_ok = s_or_cl0 && s_or_cl1 && s_or_cl2;
  // The linear polygon (triangle) is always convex.
  return edges_ok;
}

//==============================================================================
// flip
//==============================================================================

template <Int D>
HOSTDEV constexpr void
QuadraticTriangle<D>::flip() noexcept
{
  um2::swap(_v[1], _v[2]);
  um2::swap(_v[3], _v[5]);
}

//==============================================================================
// intersect
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::intersect(Ray2 const & ray) const noexcept -> Vec<6, Float>
requires(D == 2) {
  Vec<6, Float> result;
  for (Int i = 0; i < 3; ++i) {
    Vec2F const v = getEdge(i).intersect(ray);
    result[2 * i] = v[0];
    result[2 * i + 1] = v[1];
  }
  um2::insertionSort(result.begin(), result.end());
  return result;
}

//==============================================================================
// meanChordLength
//==============================================================================
// For a convex planar polygon, the mean chord length is simply pi * area / perimeter.
// De Kruijf, W. J. M., and J. L. Kloosterman.
// "On the average chord length in reactor physics." Annals of Nuclear Energy 30.5 (2003):
// 549-553.
//
// For a non-convex polygon, depending on the definition of "mean chord length", is
// not clear. It depends on how s(\mathbf{x}, \mathbf{\Omega}) is defined.
//
// \begin{equation}
//  \overline{s} = 
//  \dfrac{
//    \int_S \int_{\mathbf{\Omega} \cdot \mathbf{n} > 0} s(\mathbf{x}, \mathbf{\Omega}) d\Omega dS
//  }{
//    \int_S \int_{\mathbf{\Omega} \cdot \mathbf{n} > 0} d\Omega dS
//  }
// \end{equation}
//
// Suppose that we shoot a ray through a concave polygon such that it returns 4
// intersection points, forming two segments, s1 and s2.
//
//    |\                /|
//    |  \            /  |
// o -x----x - - -  x----x - - - >
//    |  s1  \    /   s2 |
//    |        \/        |
//     \                /
//      \              /
//       \____________/
//
//  Definition 1:
//  If the mean chord length should be defined as the average distance that a beam
//  of uncollided neutrons travels through the polygon, then the mean chord length
//  should be computed numerically by summing the lengths of the segments
//  and counting the length towards one "chord". (s1 + s2) = 1 chord. Here chord is
//  in quotes because technically s1 and s2 geometrically both chords.
//
//  Definition 2:
//  If the mean chord length should be defined as the average distance that a beam
//  of uncollided neutrons travels through the polygon WITHOUT EXITING the polygon,
//  then the mean chord length should be computed numerically counting s1 and s2 as
//  two separate chords. s1 = 1 chord, s2 = 1 chord.
//
//  Note MCL1 <= MCL2. In the case of a convex polygon, MCL1 = MCL2.
//
//  Additionally, MCL1 is computationally expensive to compute, but MCL2 is simply
//  = pi * area / perimeter.
//
//  This begs the question, why does every paper stipulate that MCL = pi A / P for a CONVEX
//  body? If the desired quantity is MCL2, we could drop the convexity requirement.
//
//  In this work, we seek to bound the change in angular flux along a ray, hence it makes
//  more sense to use MCL2. We leave the old routine which computes MCL1 in case it is
//  needed in the future.
template <Int D>
PURE HOSTDEV auto
QuadraticTriangle<D>::meanChordLength() const noexcept -> Float requires(D == 2)
{
  return um2::pi<Float> * area() / perimeter();
//  if (isConvex()) {
//    return um2::pi<Float> * area() / perimeter();
//  }
//
//  // Algorithm:
//  // For equally spaced angles γ ∈ (0, π)
//  //  Compute modular ray parameters
//  //  For each ray
//  //    Compute intersections with edges
//  //    Compute chord length
//  //    total_chord_length += chord_length
//  //    total_chords += 1
//  // return total_chord_length / total_chords
//
//  // Parameters
//  Int constexpr num_angles = 32; // Angles γ ∈ (0, π).
//  Int constexpr rays_per_longest_edge = 1000;
//
//  Int total_chords = 0;
//  Float total_length = 0;
//  auto const small_aabb = boundingBox();
//  // Expand the bounding box to avoid the lack of rays in the corners.
//  auto const width = small_aabb.width();
//  auto const height = small_aabb.height();
//  auto const dx = width / 20;
//  auto const dy = height / 20;
//  auto const vd = Vec2F(dx, dy);
//  auto const min = small_aabb.minima() - vd;
//  auto const max = small_aabb.maxima() + vd;
//  auto const aabb = AxisAlignedBox2(min, max);
//  auto const longest_edge = aabb.width() > aabb.height() ? aabb.width() : aabb.height();
//  auto const spacing = longest_edge / static_cast<Float>(rays_per_longest_edge);
//  Float const pi_deg = um2::pi_2<Float> / static_cast<Float>(num_angles);
//  // For each angle
//  for (Int ia = 0; ia < num_angles; ++ia) {
//    // Try to avoid floating point error by accumulating the chord length locally
//    Float local_accum = 0;
//    Float const angle = pi_deg * static_cast<Float>(2 * ia + 1);
//    // Compute modular ray parameters
//    ModularRayParams const params(angle, spacing, aabb);
//    Int const num_rays = params.getTotalNumRays();
//    // For each ray
//    for (Int i = 0; i < num_rays; ++i) {
//      auto const ray = params.getRay(i);
//      // 6 intersections
//      auto intersections = intersect(ray);
//      um2::insertionSort(intersections.begin(), intersections.end());
//      if (intersections[1] < um2::inf_distance / 10) {
//        total_chords += 1;
//      }
//      // Each intersection should come in pairs.
//      for (Int j = 0; j < 3; ++j) {
//        Float const r0 = intersections[2 * j];
//        Float const r1 = intersections[2 * j + 1];
//        if (r1 < um2::inf_distance / 10) {
//          ASSERT(r1 - r0 < um2::inf_distance / 100);
//          local_accum += r1 - r0;
//        }
//      }
//    }
//    total_length += local_accum;
//  }
//  return total_length / static_cast<Float>(total_chords);
}

} // namespace um2
