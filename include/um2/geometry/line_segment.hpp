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

template <Int D>
class Polytope<1, 1, 2, D>
{
  static_assert(0 < D && D <= 3, "Only 1D, 2D, and 3D segments are supported.");

public:
  using Vertex = Point<D>;

private:
  Vertex _v[2];

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
  requires(sizeof...(Pts) == 2 && (std::same_as<Vertex, Pts> && ...))
      // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
      HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  HOSTDEV constexpr explicit Polytope(Vec<2, Vertex> const & v) noexcept;

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the segment.
  // r in [0, 1] are valid values.
  // F(r) -> (x, y, z)
  template <typename R>
  PURE HOSTDEV constexpr auto
  operator()(R r) const noexcept -> Vertex;

  // dF/dr (r) -> (dx/dr, dy/dr, dz/dr)
  template <typename R>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R /*r*/) const noexcept -> Vec<D, Float>;

  // We want to transform the segment so that v[0] is at the origin and v[1]
  // is on the x-axis. We can do this by first translating by -v[0] and then
  // using a change of basis (rotation) matrix to rotate v[1] onto the x-axis.
  // The rotation matrix is returned.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getRotation() const noexcept -> Mat2x2F
  requires(D == 2);

  // If a point is to the left of the segment, with the segment oriented from
  // r = 0 to r = 1.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Vertex const & p) const noexcept -> bool requires(D == 2);

  // Arc length of the segment
  PURE HOSTDEV [[nodiscard]] constexpr auto
  length() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>;

  // Return the point on the curve that is closest to the given point.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  pointClosestTo(Vertex const & p) const noexcept -> Float;

  // Return the squared distance between the given point and the closest point
  // on the curve.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredDistanceTo(Vertex const & p) const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  distanceTo(Vertex const & p) const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2 ray) const noexcept -> Float
  requires(D == 2);

}; // LineSegment 

//==============================================================================
// Constructors
//==============================================================================

template <Int D>
HOSTDEV constexpr LineSegment<D>::Polytope(Vec<2, Vertex> const & v) noexcept
{
  _v[0] = v[0];
  _v[1] = v[1];
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 2);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::operator[](Int i) const noexcept -> Vertex const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 2);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::vertices() const noexcept -> Vertex const *
{
  return _v;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D>
template <typename R>
PURE HOSTDEV constexpr auto
LineSegment<D>::operator()(R const r) const noexcept -> Vertex
{
  auto const rr = static_cast<Float>(r);
  return _v[0] + rr * (_v[1] - _v[0]);
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D>
template <typename R>
PURE HOSTDEV constexpr auto
LineSegment<D>::jacobian(R const /*r*/) const noexcept -> Point<D>
{
  return _v[1] - _v[0]; 
}

//==============================================================================
// getRotation
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::getRotation() const noexcept -> Mat2x2F
requires(D == 2) { 
  // We want to transform the segment so that v[0] is at the origin and v[1]
  // is on the x-axis. We can do this by first translating by -v[0] and then
  // using a change of basis (rotation) matrix to rotate v[1] onto the x-axis.
  // x_old = U * x_new
  //
  // For 2D:
  // Let a = (a₁, a₂) = (P₂ - P₁) / ‖P₂ - P₁‖
  // u₁ = ( a₁,  a₂) = a
  // u₂ = (-a₂,  a₁)
  //
  // 2ote: u₁ and u₂ are orthonormal.
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
  Vec2F const a = (_v[1] - _v[0]).normalized();
  Vec2F const col0(a[0], -a[1]);
  Vec2F const col1(a[1], a[0]);
  return Mat2x2F(col0, col1);
}

//==============================================================================
// isLeft
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::isLeft(Vertex const & p) const noexcept -> bool requires(D == 2)
{
  return areCCW(_v[0], _v[1], p);
}

//==============================================================================
// length
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::length() const noexcept -> Float
{
  return _v[0].distanceTo(_v[1]); 
}

//==============================================================================
// boundingBox
//==============================================================================

// Defined in Polytope.hpp for the line segment, since for all linear polytopes
// the bounding box is simply the bounding box of the vertices.

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// pointClosestTo
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::pointClosestTo(Vertex const & p) const noexcept -> Float
{
  // From Real-Time Collision Detection, Christer Ericson, 2005
  // Given segment ab and point c, computes closest point d on ab.
  // Returns t for the position of d, d(r) = a + r*(b - a)
  Vec<D, Float> const ab = _v[1] - _v[0];
  // Project c onto ab, computing parameterized position d(r) = a + r*(b − a)
  Float r = (p - _v[0]).dot(ab) / ab.squaredNorm();
  // If outside segment, clamp r (and therefore d) to the closest endpoint
  Float constexpr lower = 0;
  Float constexpr upper = 1;
  r = um2::clamp(r, lower, upper);
  return um2::clamp(r, lower, upper);
}

//==============================================================================
// distanceTo
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::squaredDistanceTo(Vertex const & p) const noexcept -> Float
{
  Float const r = pointClosestTo(p);
  Vertex const p_closest = (*this)(r);
  return p_closest.squaredDistanceTo(p);
}

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::distanceTo(Vertex const & p) const noexcept -> Float
{
  return um2::sqrt(squaredDistanceTo(p));
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

template <Int D>
PURE HOSTDEV constexpr auto
LineSegment<D>::intersect(Ray2 const ray) const noexcept -> Float
requires(D == 2)
{
  Vec2F const v = _v[1] - _v[0];
  Vec2F const u = ray.origin() - _v[0];
  Float const z = v.cross(ray.direction());
  Float const s = u.cross(ray.direction()) / z;
  Float const r = u.cross(v) / z;
  return (s < 0 || 1 < s) ? inf_distance : r;
}

} // namespace um2
