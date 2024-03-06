#pragma once

#include <um2/geometry/point.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/stdlib/vector.hpp>

//==============================================================================
// AXIS-ALIGNED BOX
//==============================================================================
// A D-dimensional axis-aligned box.

namespace um2
{

template <Int D>
class AxisAlignedBox
{

  Point<D> _min; // minima
  Point<D> _max; // maxima

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr AxisAlignedBox() noexcept = default;

  HOSTDEV constexpr AxisAlignedBox(Point<D> const & min, Point<D> const & max) noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMin() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMin() const noexcept -> Float requires(D >= 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMin() const noexcept -> Float requires(D >= 3);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMax() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMax() const noexcept -> Float requires(D >= 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMax() const noexcept -> Float requires(D >= 3);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima() const noexcept -> Point<D> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima(Int i) const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima(Int i) const noexcept -> Float;

  //==============================================================================
  // Operators
  //===============================================================================

  HOSTDEV constexpr auto
  operator+=(Point<D> const & p) noexcept -> AxisAlignedBox<D> &;

  HOSTDEV constexpr auto
  operator+=(AxisAlignedBox<D> const & box) noexcept -> AxisAlignedBox<D> &;

  //==============================================================================
  // Other member functions 
  //==============================================================================

  // Create an empty box, with minima = inf_distance and maxima = -inf_distance.
  // Therefore, no point can be contained in this box. However box += point will
  // always result in a box containing the point.
  PURE HOSTDEV [[nodiscard]] static constexpr auto
  empty() noexcept -> AxisAlignedBox<D>;

  // The extent of the box in each dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents() const noexcept -> Point<D>;

  // The extent of the box in the i-th dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents(Int i) const noexcept -> Float;

  // The x-extent of the box.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  width() const noexcept -> Float;

  // The y-extent of the box.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  height() const noexcept -> Float requires(D >= 2);

  // The z-extent of the box.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  depth() const noexcept -> Float requires(D >= 3);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point<D> const & p) const noexcept -> bool;

}; // class AxisAlignedBox

//==============================================================================
// Aliases
//==============================================================================

// Aliases for 1, 2, and 3 dimensions.
using AxisAlignedBox1 = AxisAlignedBox<1>;
using AxisAlignedBox2 = AxisAlignedBox<2>;
using AxisAlignedBox3 = AxisAlignedBox<3>;

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::xMin() const noexcept -> Float
{
  return _min[0];
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::yMin() const noexcept -> Float requires(D >= 2)
{
  return _min[1];
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::zMin() const noexcept -> Float requires(D >= 3)
{
  return _min[2];
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::xMax() const noexcept -> Float
{
  return _max[0];
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::yMax() const noexcept -> Float requires(D >= 2)
{
  return _max[1];
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::zMax() const noexcept -> Float requires(D >= 3)
{
  return _max[2];
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::minima() const noexcept -> Point<D> const &
{
  return _min;
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::maxima() const noexcept -> Point<D> const &
{
  return _max;
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::minima(Int i) const noexcept -> Float
{
  return _min[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::maxima(Int i) const noexcept -> Float
{
  return _max[i];
}

//==============================================================================
// Constructors
//==============================================================================

template <Int D>
HOSTDEV constexpr AxisAlignedBox<D>::AxisAlignedBox(Point<D> const & min,
                                                    Point<D> const & max) noexcept
    : _min(min),
      _max(max)
{
  for (Int i = 0; i < D; ++i) {
    ASSERT(min[i] <= max[i]);
  }
}

//==============================================================================
// Operators
//==============================================================================

template <Int D>
HOSTDEV constexpr auto
AxisAlignedBox<D>::operator+=(Point<D> const & p) noexcept -> AxisAlignedBox &
{
  _min.min(p);
  _max.max(p);
  return *this;
}

template <Int D>
HOSTDEV constexpr auto
AxisAlignedBox<D>::operator+=(AxisAlignedBox const & box) noexcept -> AxisAlignedBox &
{
  _min.min(box._min);
  _max.max(box._max);
  return *this;
}

//==============================================================================
// Other member functions
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::empty() noexcept -> AxisAlignedBox<D>
{
  AxisAlignedBox<D> box;
  for (Int i = 0; i < D; ++i) {
    box._min[i] = inf_distance;
    box._max[i] = -inf_distance;
  }
  return box;
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::extents() const noexcept -> Point<D>
{
  return _max - _min;
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::extents(Int i) const noexcept -> Float
{
  return _max[i] - _min[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::width() const noexcept -> Float
{
  return extents(0);
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::height() const noexcept -> Float requires(D >= 2)
{
  return extents(1);
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::depth() const noexcept -> Float requires(D >= 3)
{
  return extents(2);
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::centroid() const noexcept -> Point<D>
{
  return midpoint<D>(_min, _max);
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::contains(Point<D> const & p) const noexcept -> bool
{
  // Lexicographic comparison.
  return _min <= p && p <= _max; 
}

template <Int D>
PURE HOSTDEV constexpr auto
isApprox(AxisAlignedBox<D> const & a, AxisAlignedBox<D> const & b) noexcept -> bool
{
  return isApprox<D>(a.minima(), b.minima()) && isApprox<D>(a.maxima(), b.maxima());
}

//==============================================================================
// Bounding Box
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D> a, AxisAlignedBox<D> const & b) noexcept -> AxisAlignedBox<D>
{
  return a += b;
}

template <Int D>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D> box, Point<D> const & p) noexcept -> AxisAlignedBox<D>
{
  return box += p;
}

template <Int D>
PURE HOSTDEV constexpr auto
boundingBox(Point<D> const * points, Int const n) noexcept -> AxisAlignedBox<D>
{
  Point<D> minima = points[0];
  Point<D> maxima = points[0];
  for (Int i = 1; i < n; ++i) {
    minima.min(points[i]);
    maxima.max(points[i]);
  }
  return AxisAlignedBox<D>(minima, maxima);
}

template <Int D>
PURE constexpr auto
boundingBox(Vector<Point<D>> const & points) noexcept -> AxisAlignedBox<D>
{
  return um2::boundingBox<D>(points.data(), points.size());
}

//==============================================================================
// intersect
//==============================================================================

// Returns the distance along the ray to the intersection point with the box.
// r in [0, inf_distance<T>]
template <Int D>
PURE HOSTDEV constexpr auto
intersect(Ray<D> const & ray, AxisAlignedBox<D> const & box) noexcept -> Point2
{
  // Inspired by https://tavianator.com/2022/ray_box_boundary.html
  auto tmin = static_cast<Float>(0);
  Float tmax = inf_distance;
  Point<D> const inv_dir = static_cast<Float>(1) / ray.direction();
  Point<D> const vt1 = (box.minima() - ray.origin()) * inv_dir;
  Point<D> const vt2 = (box.maxima() - ray.origin()) * inv_dir;
  for (Int i = 0; i < D; ++i) {
    Float const tmin_i = um2::min(vt1[i], vt2[i]);
    Float const tmax_i = um2::max(vt1[i], vt2[i]);
    tmin = um2::max(tmin, tmin_i);
    tmax = um2::min(tmax, tmax_i);
  }
  return tmin <= tmax ? Point2(tmin, tmax) : Point2(inf_distance, inf_distance);
}

} // namespace um2
