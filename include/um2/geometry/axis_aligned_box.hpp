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

  // If the minima and maxima are approximately equal according to
  // um2::isApprox(Point<D>, Point<D>).
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isApprox(AxisAlignedBox<D> const & other) const noexcept -> bool;

  // Returns the distance along the ray to the intersection point with the box.
  // r in [0, inf_distance<T>]. A miss is indicated by r = inf_distance<T>.
  // Note: ray(r) is the intersection point.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray<D> const & ray) const noexcept -> Vec2F;

  // Same as above, but with a precomputed inverse direction.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray<D> const & ray, Vec<D, Float> const & inv_dir) const noexcept -> Vec2F;

  // Scales the box by s, centered at the centroid.
  HOSTDEV constexpr void
  scale(Float s) noexcept;

}; // class AxisAlignedBox

//==============================================================================
// Aliases
//==============================================================================

// Aliases for 1, 2, and 3 dimensions.
using AxisAlignedBox1 = AxisAlignedBox<1>;
using AxisAlignedBox2 = AxisAlignedBox<2>;
using AxisAlignedBox3 = AxisAlignedBox<3>;

//==============================================================================
// Free functions
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D> a, AxisAlignedBox<D> const & b) noexcept -> AxisAlignedBox<D>;

template <Int D>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D> box, Point<D> const & p) noexcept -> AxisAlignedBox<D>;

template <Int D>
PURE HOSTDEV constexpr auto
boundingBox(Point<D> const * begin, Point<D> const * end) noexcept -> AxisAlignedBox<D>;

template <Int D>
PURE HOSTDEV constexpr auto
boundingBox(Point<D> const * points, Int n) noexcept -> AxisAlignedBox<D>;

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
AxisAlignedBox<D>::isApprox(AxisAlignedBox<D> const & other) const noexcept -> bool
{
  bool const mins_approx = _min.isApprox(other._min);
  bool const maxs_approx = _max.isApprox(other._max);
  return mins_approx && maxs_approx;
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::intersect(Ray<D> const & ray) const noexcept -> Vec2F
{
  // Inspired by https://tavianator.com/2022/ray_box_boundary.html
  auto tmin = static_cast<Float>(0);
  Float tmax = inf_distance;
  Vec<D, Float> const inv_dir = ray.inverseDirection();
  Vec<D, Float> const vt1 = (minima() - ray.origin()) * inv_dir;
  Vec<D, Float> const vt2 = (maxima() - ray.origin()) * inv_dir;
  for (Int i = 0; i < D; ++i) {
    tmin = um2::min(um2::max(vt1[i], tmin), um2::max(vt2[i], tmin));
    tmax = um2::max(um2::min(vt1[i], tmax), um2::min(vt2[i], tmax));
  }
  return tmin <= tmax ? Vec2F(tmin, tmax) : Vec2F(inf_distance, inf_distance);
}

template <Int D>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D>::intersect(Ray<D> const & ray, Vec<D, Float> const & inv_dir) const noexcept -> Vec2F
{
  // Inspired by https://tavianator.com/2022/ray_box_boundary.html
  auto tmin = static_cast<Float>(0);
  Float tmax = inf_distance;
  Vec<D, Float> const vt1 = (minima() - ray.origin()) * inv_dir;
  Vec<D, Float> const vt2 = (maxima() - ray.origin()) * inv_dir;
  for (Int i = 0; i < D; ++i) {
    tmin = um2::min(um2::max(vt1[i], tmin), um2::max(vt2[i], tmin));
    tmax = um2::max(um2::min(vt1[i], tmax), um2::min(vt2[i], tmax));
  }
  return tmin <= tmax ? Vec2F(tmin, tmax) : Vec2F(inf_distance, inf_distance);
}

//==============================================================================
// Free functions
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
boundingBox(Point<D> const * begin, Point<D> const * end) noexcept -> AxisAlignedBox<D>
{
  Point<D> minima = *begin;
  Point<D> maxima = *begin;
  while (++begin != end) {
    minima.min(*begin);
    maxima.max(*begin);
  }
  return {minima, maxima};
}

template <Int D>
PURE HOSTDEV constexpr auto
boundingBox(Point<D> const * points, Int const n) noexcept -> AxisAlignedBox<D>
{
  return boundingBox(points, points + n);
}

template <Int D>
HOSTDEV constexpr void
AxisAlignedBox<D>::scale(Float s) noexcept
{
  ASSERT(s >= 0);
  auto const dxyz = extents() * ((s - 1) / 2);
  _min -= dxyz;
  _max += dxyz;
}

} // namespace um2
