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

template <Size D, typename T>
class AxisAlignedBox
{

  Point<D, T> _min;
  Point<D, T> _max;

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
  xMin() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMin() const noexcept -> T
    requires (D >= 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMin() const noexcept -> T
    requires (D >= 3);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMax() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMax() const noexcept -> T
    requires (D >= 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMax() const noexcept -> T
    requires (D >= 3);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima() const noexcept -> Point<D, T> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D, T> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima(Size i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima(Size i) const noexcept -> T;

  //==============================================================================
  // Operators
  //===============================================================================

  HOSTDEV constexpr auto
  operator+=(Point<D, T> const & p) noexcept -> AxisAlignedBox<D, T> &;

  HOSTDEV constexpr auto
  operator+=(AxisAlignedBox<D, T> const & box) noexcept -> AxisAlignedBox<D, T> &;

  //==============================================================================
  // Methods
  //==============================================================================

  HOSTDEV [[nodiscard]] static constexpr auto
  empty() noexcept -> AxisAlignedBox<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents() const noexcept -> Vec<D, T>; // max - min

  PURE HOSTDEV [[nodiscard]] constexpr auto
  width() const noexcept -> T; // dx

  PURE HOSTDEV [[nodiscard]] constexpr auto
  height() const noexcept -> T
    requires (D >= 2); // dy 

  PURE HOSTDEV [[nodiscard]] constexpr auto
  depth() const noexcept -> T
    requires (D >= 3); // dz

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point<D, T> const & p) const noexcept -> bool;

}; // class AxisAlignedBox

//==============================================================================
// Aliases
//==============================================================================

// Aliases for 1, 2, and 3 dimensions.
template <typename T>
using AxisAlignedBox1 = AxisAlignedBox<1, T>;
template <typename T>
using AxisAlignedBox2 = AxisAlignedBox<2, T>;
template <typename T>
using AxisAlignedBox3 = AxisAlignedBox<3, T>;

// Aliases for float and double.
using AxisAlignedBox2f = AxisAlignedBox2<float>;
using AxisAlignedBox2d = AxisAlignedBox2<double>;

using AxisAlignedBox3f = AxisAlignedBox3<float>;
using AxisAlignedBox3d = AxisAlignedBox3<double>;

//==============================================================================
// Accessors
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::xMin() const noexcept -> T
{
  return _min[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::yMin() const noexcept -> T
requires (D >= 2)
{
  return _min[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::zMin() const noexcept -> T
requires (D >= 3)
{
  return _min[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::xMax() const noexcept -> T
{
  return _max[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::yMax() const noexcept -> T
requires (D >= 2)
{
  return _max[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::zMax() const noexcept -> T
requires (D >= 3)
{
  return _max[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::minima() const noexcept -> Point<D, T> const &
{
  return _min;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::maxima() const noexcept -> Point<D, T> const &
{
  return _max;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::minima(Size i) const noexcept -> T
{
  return _min[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::maxima(Size i) const noexcept -> T
{
  return _max[i];
}

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr AxisAlignedBox<D, T>::AxisAlignedBox(Point<D, T> const & min,
                                                       Point<D, T> const & max) noexcept
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

template <Size D, typename T>
HOSTDEV constexpr auto
AxisAlignedBox<D, T>::operator+=(Point<D, T> const & p) noexcept -> AxisAlignedBox &
{
  _min.min(p);
  _max.max(p);
  return *this;
}

template <Size D, typename T>
HOSTDEV constexpr auto
AxisAlignedBox<D, T>::operator+=(AxisAlignedBox const & box) noexcept -> AxisAlignedBox &
{
  _min.min(box._min);
  _max.max(box._max);
  return *this;
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::empty() noexcept -> AxisAlignedBox<D, T>
{
  Point<D, T> minima = Point<D, T>::zero();
  Point<D, T> maxima = Point<D, T>::zero();
  minima += inf_distance<T>;
  maxima -= inf_distance<T>;
  AxisAlignedBox<D, T> box;
  box._min = minima;
  box._max = maxima;
  return box; 
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::extents() const noexcept -> Vec<D, T>
{
  return _max - _min;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::width() const noexcept -> T
{
  return xMax() - xMin(); 
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::height() const noexcept -> T
requires (D >= 2)
{
  return yMax() - yMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::depth() const noexcept -> T
requires (D >= 3)
{
  return zMax() - zMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::centroid() const noexcept -> Point<D, T>
{
  return midpoint(_min, _max);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  for (Size i = 0; i < D; ++i) {
    if (p[i] < _min[i] || _max[i] < p[i]) {
      return false;
    }
  }
  return true;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
isApprox(AxisAlignedBox<D, T> const & a, AxisAlignedBox<D, T> const & b) noexcept -> bool
{
  return isApprox(a.minima(), b.minima()) && isApprox(a.maxima(), b.maxima());
}

//==============================================================================
// Bounding Box
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D, T> a, AxisAlignedBox<D, T> const & b) noexcept
    -> AxisAlignedBox<D, T>
{
  return a += b;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
operator+(AxisAlignedBox<D, T> box, Point<D, T> const & p) noexcept
    -> AxisAlignedBox<D, T>
{
  return box += p;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
operator+(Point<D, T> const & p, AxisAlignedBox<D, T> box) noexcept
    -> AxisAlignedBox<D, T>
{
  return box += p;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(Point<D, T> const & a, Point<D, T> const & b) noexcept -> AxisAlignedBox<D, T>
{
  return AxisAlignedBox<D, T>{um2::min(a, b), um2::max(a, b)};
}

template <Size D, typename T, Size N>
PURE HOSTDEV constexpr auto
boundingBox(Point<D, T> const (&points)[N]) noexcept -> AxisAlignedBox<D, T>
{
  Point<D, T> minima = points[0];
  Point<D, T> maxima = points[0];
  for (Size i = 1; i < N; ++i) {
    minima.min(points[i]);
    maxima.max(points[i]);
  }
  return AxisAlignedBox<D, T>(minima, maxima);
}

template <Size D, typename T>
PURE auto
boundingBox(Vector<Point<D, T>> const & points) noexcept -> AxisAlignedBox<D, T>
{
  struct ReduceFunctor {
    constexpr auto
    operator()(AxisAlignedBox<D, T> const & box, Point<D, T> const & p) const noexcept
        -> AxisAlignedBox<D, T>
    {
      return box + p;
    }

    constexpr auto
    operator()(Point<D, T> const & p, AxisAlignedBox<D, T> const & box) const noexcept
        -> AxisAlignedBox<D, T>
    {
      return box + p;
    }

    constexpr auto
    operator()(AxisAlignedBox<D, T> const & a,
               AxisAlignedBox<D, T> const & b) const noexcept -> AxisAlignedBox<D, T>
    {
      return a + b;
    }

    constexpr auto
    operator()(Point<D, T> const & a, Point<D, T> const & b) const noexcept
        -> AxisAlignedBox<D, T>
    {
      return boundingBox(a, b);
    }
  };

  return std::reduce(points.begin(), points.end(),
                     AxisAlignedBox<D, T>::empty(), ReduceFunctor{});
}

//==============================================================================
// intersect
//==============================================================================

// Returns the distance along the ray to the intersection point with the box.
// r in [0, inf_distance<T>]
template <typename T>
PURE HOSTDEV constexpr auto
intersect(Ray2<T> const & ray, AxisAlignedBox2<T> const & box) noexcept -> Vec2<T>
{
  // Inspired by https://tavianator.com/2022/ray_box_boundary.html
  T tmin = static_cast<T>(0);
  T tmax = inf_distance<T>;
  T const inv_x = static_cast<T>(1) / ray.direction()[0];
  T const inv_y = static_cast<T>(1) / ray.direction()[1];
  T const t1x = (box.xMin() - ray.origin()[0]) * inv_x;
  T const t2x = (box.xMax() - ray.origin()[0]) * inv_x;
  T const t1y = (box.yMin() - ray.origin()[1]) * inv_y;
  T const t2y = (box.yMax() - ray.origin()[1]) * inv_y;
  tmin = um2::max(tmin, um2::min(t1x, t2x));
  tmax = um2::min(tmax, um2::max(t1x, t2x));
  tmin = um2::max(tmin, um2::min(t1y, t2y));
  tmax = um2::min(tmax, um2::max(t1y, t2y));
  return tmin <= tmax ? Vec2<T>(tmin, tmax) : Vec2<T>(inf_distance<T>, inf_distance<T>);
}

} // namespace um2
