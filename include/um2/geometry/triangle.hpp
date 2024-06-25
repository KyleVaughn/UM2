#pragma once

#include <um2/geometry/line_segment.hpp>
#include <um2/geometry/polygon.hpp>
#include <um2/stdlib/numbers.hpp>

//==============================================================================
// TRIANGLE
//==============================================================================

namespace um2
{

template <Int D, class T>
class Polytope<2, 1, 3, D, T>
{
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

public:
  // NOLINTBEGIN(readability-identifier-naming)
  static constexpr Int N = 3; // Number of vertices
  // NOLINTEND(readability-identifier-naming)

  using Vertex = Point<D, T>;
  using Edge = LineSegment<D, T>;

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

  HOSTDEV constexpr Polytope(Vec<N, Int> const & indices,
                             Vertex const * vertices) noexcept
      : _v{vertices[indices[0]], vertices[indices[1]], vertices[indices[2]]}
  {
  }

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
  // For triangles: r in [0, 1], s in [0, 1], constrained by r + s <= 1
  // F(r, s) -> R^D
  PURE HOSTDEV constexpr auto
  operator()(T r, T s) const noexcept -> Point<D, T>;

  // Jacobian of the interpolation function.
  // [dF/dr, dF/ds]
  PURE HOSTDEV
      [[nodiscard]] constexpr auto jacobian(T /*r*/,
                                            T /*s*/) const noexcept -> Mat<D, 2, T>;

  // Get the i-th edge of the polygon.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Int i) const noexcept -> Edge;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  perimeter() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D, T>;

  // 2D only
  //--------------------------------------------------------------------------

  // If the polygon is counterclockwise oriented, returns true.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool
    requires(D == 2);

  HOSTDEV constexpr void
  flip() noexcept;

  PURE HOSTDEV [[nodiscard]] static constexpr auto
  isConvex() noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2<T> const & p) const noexcept -> bool
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  meanChordLength() const noexcept -> T
    requires(D == 2);

  HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2<T> ray, T * buffer) const noexcept -> Int
    requires(D == 2);

}; // Triangle

//==============================================================================
// Aliases
//==============================================================================

template <class T>
using Triangle2 = Triangle<2, T>;

template <class T>
using Triangle3 = Triangle<3, T>;

//==============================================================================
// Accessors
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::operator[](Int i) const noexcept -> Point<D, T> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::vertices() const noexcept -> Point<D, T> const *
{
  return _v;
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::operator()(T const r, T const s) const noexcept -> Point<D, T>
{
  // T(r, s) = (1 - r - s) v0 + r v1 + s v2
  T const w0 = 1 - r - s;
  // T const w1 = r;
  // T const w2 = s;
  return w0 * _v[0] + r * _v[1] + s * _v[2];
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
jacobian(Triangle<D, T> const & tri) noexcept -> Mat<D, 2, T>
{
  return Mat<D, 2, T>(tri[1] - tri[0], tri[2] - tri[0]);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::jacobian(T const /*r*/, T const /*s*/) const noexcept -> Mat<D, 2, T>
{
  return um2::jacobian(*this);
}

//==============================================================================
// getEdge
//==============================================================================
// Defined in polygon.hpp, since the procedure is the same for all linear polygons.

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::getEdge(Int i) const noexcept -> Edge
{
  return um2::getEdge(*this, i);
}

//==============================================================================
// perimeter
//==============================================================================
// Defined in polygon.hpp, since the procedure is the same for all linear polygons.

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::perimeter() const noexcept -> T
{
  return um2::perimeter(*this);
}

//==============================================================================
// boundingBox
//==============================================================================
// Defined in polytope.hpp, since for all linear polytopes
// the bounding box is simply the bounding box of the vertices.

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// area
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
area(Triangle3<T> const & tri) noexcept -> T
{
  Vec3<T> const v10 = tri[1] - tri[0];
  Vec3<T> const v20 = tri[2] - tri[0];
  return v10.cross(v20).norm() / 2; // this is the unsigned area
}

template <class T>
PURE HOSTDEV constexpr auto
area(Triangle2<T> const & tri) noexcept -> T
{
  ASSERT(tri.isCCW());
  Vec2<T> const v10 = tri[1] - tri[0];
  Vec2<T> const v20 = tri[2] - tri[0];
  return v10.cross(v20) / 2; // this is the signed area
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::area() const noexcept -> T
{
  return um2::area(*this);
}

//==============================================================================
// centroid
//==============================================================================
// Specialize on D to disambiguate from the planar linear polygon function.

template <class T>
PURE HOSTDEV constexpr auto
centroid(Triangle2<T> const & tri) noexcept -> Point2<T>
{
  return (tri[0] + tri[1] + tri[2]) / 3;
}

template <class T>
PURE HOSTDEV constexpr auto
centroid(Triangle3<T> const & tri) noexcept -> Point3<T>
{
  return (tri[0] + tri[1] + tri[2]) / 3;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::centroid() const noexcept -> Point<D, T>
{
  return um2::centroid(*this);
}

//==============================================================================
// isCCW
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
isCCW(Triangle2<T> const & tri) noexcept -> bool
{
  return areCCW(tri[0], tri[1], tri[2]);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::isCCW() const noexcept -> bool
  requires(D == 2)
{
  return um2::isCCW(*this);
}

//==============================================================================
// flip
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr void
Triangle<D, T>::flip() noexcept
{
  um2::swap(_v[1], _v[2]);
}

//==============================================================================
// isConvex
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::isConvex() noexcept -> bool
{
  return true;
}

//==============================================================================
// contains
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::contains(Point2<T> const & p) const noexcept -> bool
  requires(D == 2)
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

template <Int D, class T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::meanChordLength() const noexcept -> T
  requires(D == 2)
{
  return um2::meanChordLength(*this);
}

//==============================================================================
// intersect
//==============================================================================
// Defined in polygon.hpp, since the procedure is the same for all planar polygons.

template <Int D, class T>
HOSTDEV constexpr auto
Triangle<D, T>::intersect(Ray2<T> const ray, T * buffer) const noexcept -> Int
  requires(D == 2)
{
  return um2::intersect(*this, ray, buffer);
}

} // namespace um2
