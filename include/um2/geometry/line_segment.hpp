#pragma once

#include <um2/geometry/polytope.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/math/mat.hpp>
#include <um2/stdlib/algorithm/clamp.hpp>
#include <um2/stdlib/math/roots.hpp>

//==============================================================================
// LINE SEGMENT
//==============================================================================

namespace um2
{

template <Int D, class T>
class Polytope<1, 1, 2, D, T>
{
  static_assert(0 < D && D <= 3, "Only 1D, 2D, and 3D segments are supported.");

public:
  // NOLINTBEGIN(readability-identifier-naming)
  static constexpr Int N = 2; // Number of vertices
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
  PURE HOSTDEV [[nodiscard]] constexpr auto jacobian(T /*r*/) const noexcept -> Vec<D, T>;

  // Arc length of the segment
  PURE HOSTDEV [[nodiscard]] constexpr auto
  length() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  // Return the parametric coordinate (r) of the point on the line that is closest to p.
  // r such that ||F(r) - p|| is minimized, r in [0, 1]
  PURE HOSTDEV [[nodiscard]] constexpr auto
  pointClosestTo(Vertex const & p) const noexcept -> T;

  // Return the squared distance from the point p to the segment.
  // This is faster than distanceTo() as it avoids the square root operation.
  // return ||pointClosestTo(p) - p||^2
  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredDistanceTo(Vertex const & p) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  distanceTo(Vertex const & p) const noexcept -> T;

  // 2D only
  //---------------------------------------------------------------------------

  // If the line is translated by -v0, then the first vertex is at the origin.
  // Get the rotation matrix that aligns the line with the x-axis.
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

}; // LineSegment

//==============================================================================
// Accessors
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::operator[](Int i) const noexcept -> Vertex const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::vertices() const noexcept -> Vertex const *
{
  return _v;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
interpolate(LineSegment<D, T> const & l, T const r) noexcept -> Point<D, T>
{
  return l[0] + r * (l[1] - l[0]);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::operator()(T const r) const noexcept -> Vertex
{
  return interpolate(*this, r);
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
jacobian(LineSegment<D, T> const & l) noexcept -> Point<D, T>
{
  return l[1] - l[0];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::jacobian(T const /*r*/) const noexcept -> Point<D, T>
{
  return um2::jacobian(*this);
}

//==============================================================================
// length
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
length(LineSegment<D, T> const & l) noexcept -> T
{
  return l[0].distanceTo(l[1]);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::length() const noexcept -> T
{
  return um2::length(*this);
}

//==============================================================================
// boundingBox
//==============================================================================
// Defined in polytope.hpp , since for all linear polytopes
// the bounding box is simply the bounding box of the vertices.

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// pointClosestTo
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
pointClosestTo(LineSegment<D, T> const & l, Point<D, T> const & p) noexcept -> T
{
  // From Real-Time Collision Detection, Christer Ericson, 2005
  // Given segment ab and point c, computes closest point d on ab.
  // Returns t for the position of d, d(r) = a + r*(b - a)
  Point<D, T> const ab = l[1] - l[0];
  // Project c onto ab, computing parameterized position d(r) = a + r*(b − a)
  T r = (p - l[0]).dot(ab) / ab.squaredNorm();
  // If outside segment, clamp r (and therefore d) to the closest endpoint
  T constexpr lower = 0;
  T constexpr upper = 1;
  r = um2::clamp(r, lower, upper);
  return um2::clamp(r, lower, upper);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::pointClosestTo(Point<D, T> const & p) const noexcept -> T
{
  return um2::pointClosestTo(*this, p);
}

//==============================================================================
// distanceTo
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
squaredDistanceTo(LineSegment<D, T> const & l, Point<D, T> const & p) noexcept -> T
{
  T const r = l.pointClosestTo(p);
  Point<D, T> const p_closest = l(r);
  return p_closest.squaredDistanceTo(p);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::squaredDistanceTo(Point<D, T> const & p) const noexcept -> T
{
  return um2::squaredDistanceTo(*this, p);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::distanceTo(Point<D, T> const & p) const noexcept -> T
{
  return um2::sqrt(squaredDistanceTo(p));
}

//==============================================================================
// getRotation
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::getRotation() const noexcept -> Mat2x2<T>
  requires(D == 2)
{
  // We want to transform the segment so that v[0] is at the origin and v[1]
  // is on the x-axis. We can do this by first translating by -v[0] and then
  // using a change of basis (rotation) matrix to rotate v[1] onto the x-axis.
  // x_old = U * x_new
  //
  // For 2D:
  // Let a = (a₁, a₂) = (P₂ - P₁) / ‖P₂ - P₁‖
  // Note: a is a unit vector
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
  Vec2<T> const a = (_v[1] - _v[0]).normalized();
  // Vec2<T> const col0(a[0], -a[1]);
  // Vec2<T> const col1(a[1], a[0]);
  Mat2x2<T> result;
  result(0) = a[0];
  result(1) = -a[1];
  result(2) = a[1];
  result(3) = a[0];
  return result;
}

//==============================================================================
// isLeft
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::isLeft(Point2<T> const p) const noexcept -> bool
  requires(D == 2)
{
  return areCCW(_v[0], _v[1], p);
}

//==============================================================================
// intersect
//==============================================================================

// Returns the value r such that R(r) = L(s).
// If such a value does not exist, infiniteDistance is returned instead.
// 1) P₁ + s(P₂ - P₁) = O + rD           subtracting P₁ from both sides
// 2) s(P₂ - P₁) = (O - P₁) + rD         let U = O - P₁, V = P₂-P₁
// 3) sV = U + rD                        cross product with D (distributive)
// 4) s(V × D) = U × D  + r(D × D)       D × D = 0
// 5) s(V × D) = U × D                   let V × D = Z and U × D = X
// 6) sZ = X                             dot product 𝘇 to each side
// 7) sZ ⋅ Z = X ⋅ Z                     divide by Z ⋅ Z
// 8) s = (X ⋅ Z)/(Z ⋅ Z)
// If s ∉ [0, 1] the intersections is invalid. If s ∈ [0, 1],
// 1) O + rD = P₁ + sV                   subtracting O from both sides
// 2) rD = -U + sV                       cross product with 𝘃
// 3) r(D × V) = -U × V + s(V × V)       V × V = 0
// 4) r(D × V) = -U × V                  using D × V = -(V × D)
// 5) r(V × D) = U × V                   let U × V = Y
// 6) rZ = Y                             dot product Z to each side
// 7) r(Z ⋅ Z) = Y ⋅ Z                   divide by (Z ⋅ Z)
// 9) r = (Y ⋅ Z)/(Z ⋅ Z)
//
// The cross product of two vectors in the plane is a vector of the form (0, 0, k),
// hence, in 2D:
// s = (X ⋅ Z)/(Z ⋅ Z) = x₃/z₃
// r = (Y ⋅ Z)/(Z ⋅ Z) = y₃/z₃
// This result is valid if s ∈ [0, 1]

template <Int D, class T>
HOSTDEV constexpr auto
LineSegment<D, T>::intersect(Ray2<T> const ray, T * const buffer) const noexcept -> Int
  requires(D == 2)
{
  Vec2<T> const v = _v[1] - _v[0];
  Vec2<T> const u = ray.origin() - _v[0];
  T const z = v.cross(ray.direction());
  T const s = u.cross(ray.direction()) / z;
  T const r = u.cross(v) / z;
  *buffer = r;
  return (0 <= s && s <= 1 && 0 <= r) ? 1 : 0;
}

} // namespace um2
