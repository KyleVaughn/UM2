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

template <Int D, class T>
class AxisAlignedBox
{

  Point<D, T> _min; // minima
  Point<D, T> _max; // maxima

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr AxisAlignedBox() noexcept = default;

  HOSTDEV constexpr AxisAlignedBox(Point<D, T> const & min,
                                   Point<D, T> const & max) noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima() const noexcept -> Point<D, T> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D, T> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima(Int i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima(Int i) const noexcept -> T;

  //==============================================================================
  // Operators
  //===============================================================================

  HOSTDEV constexpr auto
  operator+=(Point<D, T> const & p) noexcept -> AxisAlignedBox<D, T> &;

  HOSTDEV constexpr auto
  operator+=(AxisAlignedBox<D, T> const & box) noexcept -> AxisAlignedBox<D, T> &;

  //==============================================================================
  // Other member functions
  //==============================================================================

  // Create an empty box, with minima = infDistance and maxima = -infDistance.
  // Therefore, no point can be contained in this box. However box += point will
  // always result in a box containing the point.
  PURE HOSTDEV [[nodiscard]] static constexpr auto
  empty() noexcept -> AxisAlignedBox<D, T>;

  // The extent of the box in each dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents() const noexcept -> Point<D, T>;

  // The extent of the box in the i-th dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents(Int i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point<D, T> const & p) const noexcept -> bool;

  // If the minima and maxima are approximately equal according to
  // um2::isApprox(Point<D, T>, Point<D, T>).
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isApprox(AxisAlignedBox<D, T> const & other) const noexcept -> bool;

  // Returns the distance along the ray to the intersection point with the box.
  // r in [0, infDistance<T>]. A miss is indicated by r = -1
  // Note: ray(r) is the intersection point.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray<D, T> const & ray) const noexcept -> Vec2<T>;

  // Same as above, but with a precomputed inverse direction.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray<D, T> const & ray, Vec<D, T> const & inv_dir) const noexcept -> Vec2<T>;

  // Scales the box by s, centered at the centroid.
  HOSTDEV constexpr void
  scale(T s) noexcept;

  HOSTDEV [[nodiscard]] constexpr auto
  intersects(AxisAlignedBox<D, T> const & other) const noexcept -> bool;

}; // class AxisAlignedBox

//==============================================================================
// Aliases
//==============================================================================

// Aliases for 1, 2, and 3 dimensions.
template <class T>
using AxisAlignedBox1 = AxisAlignedBox<1, T>;

template <class T>
using AxisAlignedBox2 = AxisAlignedBox<2, T>;

template <class T>
using AxisAlignedBox3 = AxisAlignedBox<3, T>;

using AxisAlignedBox2F = AxisAlignedBox2<Float>;

//==============================================================================
// Free functions
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D, T> a,
          AxisAlignedBox<D, T> const & b) noexcept -> AxisAlignedBox<D, T>;

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D, T> box,
          Point<D, T> const & p) noexcept -> AxisAlignedBox<D, T>;

template <Int D, class T>
PURE HOSTDEV constexpr auto
boundingBox(Point<D, T> const * begin,
            Point<D, T> const * end) noexcept -> AxisAlignedBox<D, T>;

template <Int D, class T>
PURE HOSTDEV constexpr auto
boundingBox(Point<D, T> const * points, Int n) noexcept -> AxisAlignedBox<D, T>;

//==============================================================================
// Accessors
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::minima() const noexcept -> Point<D, T> const &
{
  return _min;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::maxima() const noexcept -> Point<D, T> const &
{
  return _max;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::minima(Int i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _min[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::maxima(Int i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _max[i];
}

//==============================================================================
// Constructors
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr AxisAlignedBox<D, T>::AxisAlignedBox(Point<D, T> const & min,
                                                       Point<D, T> const & max) noexcept
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

template <Int D, class T>
HOSTDEV constexpr auto
AxisAlignedBox<D, T>::operator+=(Point<D, T> const & p) noexcept -> AxisAlignedBox &
{
  _min.min(p);
  _max.max(p);
  return *this;
}

template <Int D, class T>
HOSTDEV constexpr auto
AxisAlignedBox<D, T>::operator+=(AxisAlignedBox const & box) noexcept -> AxisAlignedBox &
{
  _min.min(box._min);
  _max.max(box._max);
  return *this;
}

//==============================================================================
// Other member functions
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::empty() noexcept -> AxisAlignedBox<D, T>
{
  AxisAlignedBox<D, T> box;
  for (Int i = 0; i < D; ++i) {
    box._min[i] = infDistance<T>();
    box._max[i] = -infDistance<T>();
  }
  return box;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::extents() const noexcept -> Point<D, T>
{
  return _max - _min;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::extents(Int i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _max[i] - _min[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::centroid() const noexcept -> Point<D, T>
{
  return midpoint<D, T>(_min, _max);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  for (Int i = 0; i < D; ++i) {
    if (p[i] < _min[i] || p[i] > _max[i]) {
      return false;
    }
  }
  return true;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::isApprox(AxisAlignedBox<D, T> const & other) const noexcept -> bool
{
  bool const mins_approx = _min.isApprox(other._min);
  bool const maxs_approx = _max.isApprox(other._max);
  return mins_approx && maxs_approx;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::intersect(Ray<D, T> const & ray) const noexcept -> Vec2<T>
{
  // Inspired by https://tavianator.com/2022/ray_box_boundary.html
  auto tmin = static_cast<T>(0);
  T tmax = infDistance<T>();
  Vec<D, T> const inv_dir = ray.inverseDirection();
  Vec<D, T> const vt1 = (minima() - ray.origin()) * inv_dir;
  Vec<D, T> const vt2 = (maxima() - ray.origin()) * inv_dir;
  for (Int i = 0; i < D; ++i) {
    tmin = um2::min(um2::max(vt1[i], tmin), um2::max(vt2[i], tmin));
    tmax = um2::max(um2::min(vt1[i], tmax), um2::min(vt2[i], tmax));
  }
  return tmin <= tmax ? Vec2<T>(tmin, tmax) : Vec2<T>(-1, -1); // -1 indicates a miss
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::intersect(Ray<D, T> const & ray,
                                Vec<D, T> const & inv_dir) const noexcept -> Vec2<T>
{
  // Inspired by https://tavianator.com/2022/ray_box_boundary.html
  auto tmin = static_cast<T>(0);
  T tmax = infDistance<T>();
  Vec<D, T> const vt1 = (minima() - ray.origin()) * inv_dir;
  Vec<D, T> const vt2 = (maxima() - ray.origin()) * inv_dir;
  for (Int i = 0; i < D; ++i) {
    tmin = um2::min(um2::max(vt1[i], tmin), um2::max(vt2[i], tmin));
    tmax = um2::max(um2::min(vt1[i], tmax), um2::min(vt2[i], tmax));
  }
  return tmin <= tmax ? Vec2<T>(tmin, tmax) : Vec2<T>(-1, -1); // -1 indicates a miss
}

//==============================================================================
// Free functions
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D, T> a,
          AxisAlignedBox<D, T> const & b) noexcept -> AxisAlignedBox<D, T>
{
  return a += b;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D, T> box,
          Point<D, T> const & p) noexcept -> AxisAlignedBox<D, T>
{
  return box += p;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
boundingBox(Point<D, T> const * begin,
            Point<D, T> const * end) noexcept -> AxisAlignedBox<D, T>
{
  Point<D, T> minima = *begin;
  Point<D, T> maxima = *begin;
  while (++begin != end) {
    minima.min(*begin);
    maxima.max(*begin);
  }
  return {minima, maxima};
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
boundingBox(Point<D, T> const * points, Int const n) noexcept -> AxisAlignedBox<D, T>
{
  return boundingBox(points, points + n);
}

template <Int D, class T>
HOSTDEV constexpr void
AxisAlignedBox<D, T>::scale(T s) noexcept
{
  ASSERT(s >= 0);
  auto const dxyz = extents() * ((s - 1) / 2);
  _min -= dxyz;
  _max += dxyz;
}

template <Int D, class T>
HOSTDEV constexpr auto
AxisAlignedBox<D, T>::intersects(AxisAlignedBox<D, T> const & other) const noexcept
    -> bool
{
  for (Int i = 0; i < D; ++i) {
    if (_max[i] < other._min[i] || _min[i] > other._max[i]) {
      return false;
    }
  }
  return true;
}

} // namespace um2
