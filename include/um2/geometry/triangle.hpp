#pragma once

#include <um2/geometry/line_segment.hpp>
#include <um2/geometry/polygon.hpp>
#include <um2/stdlib/numbers.hpp>

//==============================================================================
// TRIANGLE
//==============================================================================

namespace um2
{

template <Int D>
class Polytope<2, 1, 3, D>
{
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

public:
  // NOLINTBEGIN(readability-identifier-naming)
  static constexpr Int N = 3; // Number of vertices
  // NOLINTEND(readability-identifier-naming)

  using Vertex = Point<D>;
  using Edge = LineSegment<D>;

private:
  Vertex _v[N];

public:
  //==============================================================================
  // Accessors
  //==============================================================================

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
      // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
      HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
  // For triangles: r in [0, 1], s in [0, 1], constrained by r + s <= 1
  // F(r, s) -> R^D 
  PURE HOSTDEV constexpr auto
  operator()(Float r, Float s) const noexcept -> Point<D>;

  // Jacobian of the surface of the polygon. 
  // [dF/dr, dF/ds]
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(Float /*r*/, Float /*s*/) const noexcept -> Mat<D, 2, Float>;

  // Get the i-th edge of the polygon.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Int i) const noexcept -> Edge;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  perimeter() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D>;

  // 2D only
  //--------------------------------------------------------------------------

  // If the polygon is counterclockwise oriented, returns true.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool requires(D == 2);

  HOSTDEV constexpr void
  flip() noexcept;

  PURE HOSTDEV [[nodiscard]] static constexpr auto
  isConvex() noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2 const & p) const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  meanChordLength() const noexcept -> Float requires(D == 2);

  HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2 ray, Float * buffer) const noexcept -> Int 
  requires(D == 2);

}; // Triangle

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::operator[](Int i) const noexcept -> Point<D> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::vertices() const noexcept -> Point<D> const *
{
  return _v;
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::operator()(Float const r, Float const s) const noexcept -> Point<D>
{
  // T(r, s) = (1 - r - s) v0 + r v1 + s v2
  Float const w0 = 1 - r - s;
  // Float const w1 = r;
  // Float const w2 = s;
  return w0 * _v[0] + r * _v[1] + s * _v[2];
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
jacobian(Triangle<D> const & tri) noexcept -> Mat<D, 2, Float>
{
  return Mat<D, 2, Float>(tri[1] - tri[0], tri[2] - tri[0]);
}

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::jacobian(Float const /*r*/, Float const /*s*/) const noexcept -> Mat<D, 2, Float>
{
  return um2::jacobian(*this);
}

//==============================================================================
// getEdge
//==============================================================================
// Defined in polygon.hpp, since the procedure is the same for all linear polygons.

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::getEdge(Int i) const noexcept -> Edge
{
  return um2::getEdge(*this, i);
}

//==============================================================================
// perimeter
//==============================================================================
// Defined in polygon.hpp, since the procedure is the same for all linear polygons.

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::perimeter() const noexcept -> Float
{
  return um2::perimeter(*this);
}

//==============================================================================
// boundingBox
//==============================================================================
// Defined in polytope.hpp, since for all linear polytopes
// the bounding box is simply the bounding box of the vertices.

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// area
//==============================================================================

PURE HOSTDEV constexpr auto
area(Triangle<3> const & tri) noexcept -> Float
{
  Vec3F const v10 = tri[1] - tri[0];
  Vec3F const v20 = tri[2] - tri[0];
  return v10.cross(v20).norm() / 2; // this is the unsigned area
}

PURE HOSTDEV constexpr auto
area(Triangle<2> const & tri) noexcept -> Float
{
  ASSERT(tri.isCCW());
  Vec2F const v10 = tri[1] - tri[0];
  Vec2F const v20 = tri[2] - tri[0];
  return v10.cross(v20) / 2; // this is the signed area
}

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::area() const noexcept -> Float
{
  return um2::area(*this);
}

//==============================================================================
// centroid
//==============================================================================
// Specialize on D to disambiguate from the planar linear polygon function.

PURE HOSTDEV constexpr auto
centroid(Triangle2 const & tri) noexcept -> Point2
{
  return (tri[0] + tri[1] + tri[2]) / 3;
}

PURE HOSTDEV constexpr auto
centroid(Triangle3 const & tri) noexcept -> Point3
{
  return (tri[0] + tri[1] + tri[2]) / 3;
}

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::centroid() const noexcept -> Point<D>
{
  return um2::centroid(*this);
}

//==============================================================================
// isCCW
//==============================================================================

PURE HOSTDEV constexpr auto
isCCW(Triangle2 const & tri) noexcept -> bool
{
  return areCCW(tri[0], tri[1], tri[2]);
}

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::isCCW() const noexcept -> bool requires(D == 2)
{
  return um2::isCCW(*this);
}

//==============================================================================
// flip
//==============================================================================

template <Int D>
HOSTDEV constexpr void
Triangle<D>::flip() noexcept
{
  um2::swap(_v[1], _v[2]);
}

//==============================================================================
// isConvex
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::isConvex() noexcept -> bool
{
  return true;
}

//==============================================================================
// contains
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::contains(Point2 const & p) const noexcept -> bool requires(D == 2)
{
  // Benchmarking shows that it is faster to compute all if the point is left
  // of all edges, rather than short-circuiting.
  ASSERT(isCCW());
  bool const b0 = areCCW(_v[0], _v[1], p);
  bool const b1 = areCCW(_v[1], _v[2], p);
  bool const b2 = areCCW(_v[2], _v[0], p);
  return b0 && b1 && b2;
}

//==============================================================================
// meanChordLength
//==============================================================================
// Defined in polygon.hpp, since the procedure is the same for all planar polygons.

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::meanChordLength() const noexcept -> Float requires(D == 2)
{
  return um2::meanChordLength(*this);
}

//==============================================================================
// intersect
//==============================================================================
// Defined in polygon.hpp, since the procedure is the same for all planar polygons.

template <Int D>
HOSTDEV constexpr auto
Triangle<D>::intersect(Ray2 const ray, Float * buffer) const noexcept -> Int
requires(D == 2) {
  return um2::intersect(*this, ray, buffer); 
}

} // namespace um2
