#pragma once

#include <um2/common/insertion_sort.hpp>
#include <um2/geometry/quadratic_segment.hpp>
#include <um2/geometry/quadrilateral.hpp>
#include <um2/geometry/modular_rays.hpp>

//==============================================================================
// QUADRATIC QUADRILATERAL
//==============================================================================

namespace um2
{

template <Int D>
class Polytope<2, 2, 8, D>
{
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

public:
  using Vertex = Point<D>;
  using Edge = QuadraticSegment<D>;

private:
  Vertex _v[8];

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
  requires(sizeof...(Pts) == 8 && (std::same_as<Vertex, Pts> && ...))
      // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
      HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  HOSTDEV constexpr explicit Polytope(Vec<8, Vertex> const & v) noexcept;

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
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
  linearPolygon() const noexcept -> Quadrilateral<D>;

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
  intersect(Ray2 const & ray) const noexcept -> Vec<8, Float>
  requires(D == 2);

  // See the comments in the implementation for details.
  // meanChordLength has multiple definitions. Make sure you read the comments to
  // determine it's the one you want.
  PURE HOSTDEV [[nodiscard]] auto
  meanChordLength() const noexcept -> Float requires(D == 2);

}; // QuadraticQuadrilateral

//==============================================================================
// Constructors
//==============================================================================

template <Int D>
HOSTDEV constexpr QuadraticQuadrilateral<D>::Polytope(Vec<8, Vertex> const & v) noexcept
{
  for (Int i = 0; i < 8; ++i) {
    _v[i] = v[i];
  }
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::numEdges() noexcept -> Int
{
  return 4;
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 8);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::operator[](Int i) const noexcept -> Point<D> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 8);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::vertices() const noexcept -> Point<D> const *
{
  return _v;
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::operator()(R const r, S const s) const noexcept -> Point<D>
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
  return w[0] * _v[0] + w[1] * _v[1] +
         w[2] * _v[2] + w[3] * _v[3] +
         w[4] * _v[4] + w[5] * _v[5] +
         w[6] * _v[6] + w[7] * _v[7];
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::jacobian(R r, S s) const noexcept -> Mat<D, 2, Float>
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
    w0 * (_v[0] - _v[1]) +
    w1 * (_v[2] - _v[3]) +
    w2 * (_v[0] + _v[1] - 2 * _v[4]) +
    w3 * (_v[2] + _v[3] - 2 * _v[6]) +
    w4 * (_v[5] - _v[7]),
    w5 * (_v[0] - _v[3]) +
    w6 * (_v[2] - _v[1]) +
    w7 * (_v[0] + _v[3] - 2 * _v[7]) +
    w8 * (_v[1] + _v[2] - 2 * _v[5]) +
    w9 * (_v[6] - _v[4]));
}

//==============================================================================
// getEdge
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::getEdge(Int i) const noexcept -> Edge
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 4);
  return (i < 3) ? QuadraticSegment<D>(_v[i], _v[i + 1], _v[i + 4])
                 : QuadraticSegment<D>(_v[3], _v[0], _v[7]);
}

//==============================================================================
// contains
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::contains(Point2 const & p) const noexcept -> bool requires(D == 2)
{
  for (Int i = 0; i < 4; ++i) {
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
QuadraticQuadrilateral<D>::linearPolygon() const noexcept -> Quadrilateral<D>
{
  return Quadrilateral<D>(_v[0], _v[1], _v[2], _v[3]);
}

//==============================================================================
// area
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::area() const noexcept -> Float
requires(D == 2)
{
  Float result = linearPolygon().area();
  result += enclosedArea(getEdge(0));
  result += enclosedArea(getEdge(1));
  result += enclosedArea(getEdge(2));
  result += enclosedArea(getEdge(3));
  return result;
}

//==============================================================================
// perimeter
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::perimeter() const noexcept -> Float
{
  Float result = getEdge(0).length();
  result += getEdge(1).length();
  result += getEdge(2).length();
  result += getEdge(3).length();
  return result;
}

//==============================================================================
// centroid
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::centroid() const noexcept -> Point<D>
requires(D == 2)
{
  auto lin_poly = linearPolygon();
  Float area_sum = lin_poly.area();
  Point2 centroid_sum = area_sum * lin_poly.centroid();
  for (Int i = 0; i < 4; ++i) {
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
QuadraticQuadrilateral<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
requires(D == 2)
{
  AxisAlignedBox2 box = getEdge(0).boundingBox();
  box += getEdge(1).boundingBox();
  box += getEdge(2).boundingBox();
  box += getEdge(3).boundingBox();
  return box;
}

//==============================================================================
// isCCW
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::isCCW() const noexcept -> bool requires(D == 2)
{
  return linearPolygon().isCCW();
}

//==============================================================================
// isConvex
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::isConvex() const noexcept -> bool requires(D == 2)
{
  // If each edge is either straight, or curves left. 
  // AND the linear polygon polygon is convex.
  auto const e0 = getEdge(0);
  auto const e1 = getEdge(1);
  auto const e2 = getEdge(2);
  auto const e3 = getEdge(3);
  bool const s_or_cl0 = isStraight(e0) || e0.curvesLeft(); 
  bool const s_or_cl1 = isStraight(e1) || e1.curvesLeft();
  bool const s_or_cl2 = isStraight(e2) || e2.curvesLeft();
  bool const s_or_cl3 = isStraight(e3) || e3.curvesLeft();
  bool const edges_ok = s_or_cl0 && s_or_cl1 && s_or_cl2 && s_or_cl3;
  bool const lin_ok = linearPolygon().isConvex();
  return edges_ok && lin_ok;
}

//==============================================================================
// flip
//==============================================================================

template <Int D>
HOSTDEV constexpr void
QuadraticQuadrilateral<D>::flip() noexcept
{
  um2::swap(_v[1], _v[3]);
  um2::swap(_v[4], _v[7]);
}

//==============================================================================
// intersect
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::intersect(Ray2 const & ray) const noexcept -> Vec<8, Float>
requires(D == 2) {
  Vec<8, Float> result;
  for (Int i = 0; i < 4; ++i) {
    Vec2F const v = getEdge(i).intersect(ray);
    result[2 * i] = v[0];
    result[2 * i + 1] = v[1];
  }
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
// For a non-convex polygon, we shoot modular rays from the bounding box and average.

template <Int D>
PURE HOSTDEV auto
QuadraticQuadrilateral<D>::meanChordLength() const noexcept -> Float requires(D == 2)
{
  if (isConvex()) {
    return um2::pi<Float> * area() / perimeter();
  }

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
  Int constexpr num_angles = 32; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 1000;

  Int total_chords = 0;
  Float total_length = 0;
  auto const small_aabb = boundingBox();
  // Expand the bounding box to avoid the lack of rays in the corners.
  auto const width = small_aabb.width();
  auto const height = small_aabb.height();
  auto const dx = width / 20;
  auto const dy = height / 20;
  auto const vd = Vec2F(dx, dy);
  auto const min = small_aabb.minima() - vd;
  auto const max = small_aabb.maxima() + vd;
  auto const aabb = AxisAlignedBox2(min, max);
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
      // 8 intersections
      auto intersections = intersect(ray);
      um2::insertionSort(intersections.begin(), intersections.end());
      // Each intersection should come in pairs.
      for (Int j = 0; j < 4; ++j) {
        Float const r0 = intersections[2 * j];
        Float const r1 = intersections[2 * j + 1];
        if (r1 < um2::inf_distance / 10) {
          ASSERT(r1 - r0 < um2::inf_distance / 100);
          local_accum += r1 - r0;
          total_chords += 1;
        }
      }
    }
    total_length += local_accum;
  }
  return total_length / static_cast<Float>(total_chords);
}

} // namespace um2
