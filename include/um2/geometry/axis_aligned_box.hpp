#pragma once

#include <um2/geometry/point.hpp>
#include <um2/stdlib/vector.hpp>
#include <um2/stdlib/numeric.hpp>

namespace um2
{

//==============================================================================
// AXIS-ALIGNED BOX
//==============================================================================
//
// A D-dimensional axis-aligned box.

template <Size D, typename T>
struct AxisAlignedBox {

  Point<D, T> minima;
  Point<D, T> maxima;

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
  xMax() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMin() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMax() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMin() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMax() const noexcept -> T;

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

  PURE HOSTDEV [[nodiscard]] constexpr auto
  width() const noexcept -> T; // dx

  PURE HOSTDEV [[nodiscard]] constexpr auto
  height() const noexcept -> T; // dy

  PURE HOSTDEV [[nodiscard]] constexpr auto
  depth() const noexcept -> T; // dz

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point<D, T> const & p) const noexcept -> bool;

}; // struct AxisAlignedBox

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
  return minima[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::xMax() const noexcept -> T
{
  return maxima[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::yMin() const noexcept -> T
{
  static_assert(2 <= D);
  return minima[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::yMax() const noexcept -> T
{
  static_assert(2 <= D);
  return maxima[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::zMin() const noexcept -> T
{
  static_assert(3 <= D);
  return minima[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::zMax() const noexcept -> T
{
  static_assert(3 <= D);
  return maxima[2];
}

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr AxisAlignedBox<D, T>::AxisAlignedBox(Point<D, T> const & min,
                                                       Point<D, T> const & max) noexcept
    : minima(min),
      maxima(max)
{
  for (Size i = 0; i < D; ++i) {
    ASSERT(minima[i] <= maxima[i]);
  }
}

//==============================================================================    
// Operators    
//==============================================================================    
    
template <Size D, typename T>    
HOSTDEV constexpr auto
AxisAlignedBox<D, T>::operator+=(Point<D, T> const & p) noexcept -> AxisAlignedBox &
{
  minima.min(p);
  maxima.max(p);
  return *this;
}

template <Size D, typename T>
HOSTDEV constexpr auto
AxisAlignedBox<D, T>::operator+=(AxisAlignedBox const & box) noexcept -> AxisAlignedBox &
{
  minima.min(box.minima);
  maxima.max(box.maxima);
  return *this;
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::width() const noexcept -> T
{
  return xMax() - xMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::height() const noexcept -> T
{
  static_assert(2 <= D);
  return yMax() - yMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::depth() const noexcept -> T
{
  static_assert(3 <= D);
  return zMax() - zMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::centroid() const noexcept -> Point<D, T>
{
  return midpoint(minima, maxima);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
AxisAlignedBox<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  for (Size i = 0; i < D; ++i) {
    if (p[i] < minima[i] || maxima[i] < p[i]) {
      return false;
    }
  }
  return true;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
isApprox(AxisAlignedBox<D, T> const & a, AxisAlignedBox<D, T> const & b) noexcept -> bool
{
  return isApprox(a.minima, b.minima) && isApprox(a.maxima, b.maxima);
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
                     AxisAlignedBox<D, T>{points[0], points[0]}, ReduceFunctor{});
}

} // namespace um2
