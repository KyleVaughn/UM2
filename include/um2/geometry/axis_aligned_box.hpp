#pragma once

#include <um2/geometry/ray.hpp>
#include <um2/stdlib/numeric.hpp>
#include <um2/stdlib/vector.hpp>

//==============================================================================
// AXIS-ALIGNED BOX
//==============================================================================
// A D-dimensional axis-aligned box.

namespace um2
{

template <Size D>
class AxisAlignedBox
{

  Point<D> _min;
  Point<D> _max;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr AxisAlignedBox() noexcept = default;

  HOSTDEV constexpr AxisAlignedBox(Point<D> const & min,
                                   Point<D> const & max) noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMin() const noexcept -> F;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMin() const noexcept -> F
    requires(D >= 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMin() const noexcept -> F
    requires(D >= 3);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMax() const noexcept -> F;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMax() const noexcept -> F
    requires(D >= 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMax() const noexcept -> F
    requires(D >= 3);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima() const noexcept -> Point<D> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima(Size i) const noexcept -> F;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima(Size i) const noexcept -> F;

  //==============================================================================
  // Operators
  //===============================================================================

  HOSTDEV constexpr auto
  operator+=(Point<D> const & p) noexcept -> AxisAlignedBox<D> &;

  HOSTDEV constexpr auto
  operator+=(AxisAlignedBox<D> const & box) noexcept -> AxisAlignedBox<D> &;

  //==============================================================================
  // Methods
  //==============================================================================

  // Create an empty box, with minima = inf_distance and maxima = -inf_distance.
  // Therefore, no point can be contained in this box. However box += point will
  // always result in a box containing the point.
  HOSTDEV [[nodiscard]] static constexpr auto
  empty() noexcept -> AxisAlignedBox<D>;

  // The extent of the box in each dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents() const noexcept -> Vec<D, F>;

  // The extent of the box in the i-th dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents(Size i) const noexcept -> F;

  // The x-extent of the box.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  width() const noexcept -> F;

  // The y-extent of the box.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  height() const noexcept -> F
    requires(D >= 2);

  // The z-extent of the box.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  depth() const noexcept -> F
    requires(D >= 3);

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

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::xMin() const noexcept -> F
{
  return _min[0];
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::yMin() const noexcept -> F
  requires(D >= 2)
{
  return _min[1];
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::zMin() const noexcept -> F
  requires(D >= 3)
{
  return _min[2];
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::xMax() const noexcept -> F
{
  return _max[0];
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::yMax() const noexcept -> F
  requires(D >= 2)
{
  return _max[1];
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::zMax() const noexcept -> F
  requires(D >= 3)
{
  return _max[2];
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::minima() const noexcept -> Point<D> const &
{
  return _min;
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::maxima() const noexcept -> Point<D> const &
{
  return _max;
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::minima(Size i) const noexcept -> F
{
  return _min[i];
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::maxima(Size i) const noexcept -> F
{
  return _max[i];
}

//==============================================================================
// Constructors
//==============================================================================

template <Size D>
HOSTDEV constexpr AxisAlignedBox<D>::AxisAlignedBox(Point<D> const & min,
                                                       Point<D> const & max) noexcept
    : _min(min),
      _max(max)
{
  for (Size i = 0; i < D; ++i) {
    ASSERT(min[i] <= max[i]);
  }
}

//==============================================================================
// Operators
//==============================================================================

template <Size D>
HOSTDEV constexpr auto
AxisAlignedBox<D>::operator+=(Point<D> const & p) noexcept -> AxisAlignedBox &
{
  _min.min(p);
  _max.max(p);
  return *this;
}

template <Size D>
HOSTDEV constexpr auto
AxisAlignedBox<D>::operator+=(AxisAlignedBox const & box) noexcept -> AxisAlignedBox &
{
  _min.min(box._min);
  _max.max(box._max);
  return *this;
}

//==============================================================================
// Methods
//==============================================================================

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::empty() noexcept -> AxisAlignedBox<D>
{
  AxisAlignedBox<D> box;
  for (Size i = 0; i < D; ++i) {
    box._min[i] = inf_distance;
    box._max[i] = -inf_distance;
  }
  return box;
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::extents() const noexcept -> Vec<D, F>
{
  return _max - _min;
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::extents(Size i) const noexcept -> F
{
  return _max[i] - _min[i];
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::width() const noexcept -> F
{
  return extents(0);
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::height() const noexcept -> F
  requires(D >= 2)
{
  return extents(1);
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::depth() const noexcept -> F
  requires(D >= 3)
{
  return extents(2);
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::centroid() const noexcept -> Point<D>
{
  return midpoint(_min, _max);
}

template <Size D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::contains(Point<D> const & p) const noexcept -> bool
{
  for (Size i = 0; i < D; ++i) {
    if (p[i] < _min[i] || _max[i] < p[i]) {
      return false;
    }
  }
  return true;
}

template <Size D>
PURE HOSTDEV constexpr auto
isApprox(AxisAlignedBox<D> const & a, AxisAlignedBox<D> const & b) noexcept -> bool
{
  return isApprox(a.minima(), b.minima()) && isApprox(a.maxima(), b.maxima());
}

//==============================================================================
// Bounding Box
//==============================================================================

template <Size D>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D> a, AxisAlignedBox<D> const & b) noexcept
    -> AxisAlignedBox<D>
{
  return a += b;
}

template <Size D>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D> box, Point<D> const & p) noexcept
    -> AxisAlignedBox<D>
{
  return box += p;
}

template <Size D>
PURE HOSTDEV constexpr auto
operator+(Point<D> const & p, AxisAlignedBox<D> box) noexcept
    -> AxisAlignedBox<D>
{
  return box += p;
}

template <Size D>
PURE HOSTDEV constexpr auto
boundingBox(Point<D> const & a, Point<D> const & b) noexcept -> AxisAlignedBox<D>
{
  return AxisAlignedBox<D>{um2::min(a, b), um2::max(a, b)};
}

template <Size D>
PURE HOSTDEV constexpr auto
boundingBox(Point<D> const * points, Size const n) noexcept -> AxisAlignedBox<D>
{
  Point<D> minima = points[0];
  Point<D> maxima = points[0];
  for (Size i = 1; i < n; ++i) {
    minima.min(points[i]);
    maxima.max(points[i]);
  }
  return AxisAlignedBox<D>(minima, maxima);
}

template <Size D>
PURE auto
boundingBox(Vector<Point<D>> const & points) noexcept -> AxisAlignedBox<D>
{
  return um2::boundingBox(points.data(), points.size());
}

//==============================================================================
// intersect
//==============================================================================

// Returns the distance along the ray to the intersection point with the box.
// r in [0, inf_distance<T>]
template <Size D>
PURE HOSTDEV constexpr auto
intersect(Ray<D> const & ray, AxisAlignedBox<D> const & box) noexcept -> Vec2<F>
{
  // Inspired by https://tavianator.com/2022/ray_box_boundary.html
  F tmin = static_cast<F>(0);
  F tmax = inf_distance;
  Vec<D, F> const inv_dir = static_cast<F>(1) / ray.direction();
  Vec<D, F> const vt1 = (box.minima() - ray.origin()) * inv_dir;
  Vec<D, F> const vt2 = (box.maxima() - ray.origin()) * inv_dir;
  for (Size i = 0; i < D; ++i) {
    F const tmin_i = um2::min(vt1[i], vt2[i]);
    F const tmax_i = um2::max(vt1[i], vt2[i]);
    tmin = um2::max(tmin, tmin_i);
    tmax = um2::min(tmax, tmax_i);
  }
  return tmin <= tmax ? Vec2<F>(tmin, tmax) : Vec2<F>(inf_distance, inf_distance);
}

} // namespace um2
