#pragma once

#include <um2/math/vec.hpp>

//==============================================================================
// POINT
//==============================================================================
// An alias for a D-dimensional vector of type F. This isn't mathematically
// correct, but it is more efficient to use a vector for a point than to create
// a separate class and deal with the type conversions.

namespace um2
{

//==============================================================================
// Aliases
//==============================================================================

template <Int D>        
using Point = Vec<D, Float>;        
        
using Point1 = Point<1>;         
using Point2 = Point<2>;        
using Point3 = Point<3>;

//==============================================================================
// Functions 
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
midpoint(Point<D> a, Point<D> const & b) noexcept -> Point<D>
{
  // (a + b) / 2
  a += b;
  return a /= 2;
}

// Are 3 planar points in counter-clockwise order?
PURE HOSTDEV constexpr auto
areCCW(Point2 const & a, Point2 const & b, Point2 const & c) noexcept -> bool
{
  // 2D cross product, of (b - a) and (c - a).
  auto const ab = b - a;
  auto const ac = c - a;
  // Allow equality, so that we can handle collinear points.
  return 0 <= ab.cross(ac);
}

// Are 3 planar points in counter-clockwise order? We allow for a small amount of
// floating point error.
PURE HOSTDEV constexpr auto
areApproxCCW(Point2 const & a, Point2 const & b, Point2 const & c) noexcept -> bool
{
  // 2D cross product, of (b - a) and (c - a).
  auto const ab = b - a;
  auto const ac = c - a;
  return -eps_distance <= ab.cross(ac); 
}

} // namespace um2
