#pragma once

#include <um2/config.hpp>
#include <um2/common/Vector.hpp>
#include <um2/geometry/Point.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// AXIS-ALIGNED BOX
// -----------------------------------------------------------------------------
// A D-dimensional axis-aligned box.

template <Size D, typename T>
struct AxisAlignedBox {

  Point<D, T> minima;
  Point<D, T> maxima;

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr AxisAlignedBox() = default;

  HOSTDEV constexpr AxisAlignedBox(Point<D, T> const & min, Point<D, T> const & max);

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

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

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Aliases
// -----------------------------------------------------------------------------

template <typename T>
using AxisAlignedBox1 = AxisAlignedBox<1, T>;

template <typename T>
using AxisAlignedBox2 = AxisAlignedBox<2, T>;

template <typename T>
using AxisAlignedBox3 = AxisAlignedBox<3, T>;

using AxisAlignedBox2f = AxisAlignedBox2<float>;
using AxisAlignedBox2d = AxisAlignedBox2<double>;

using AxisAlignedBox3f = AxisAlignedBox3<float>;
using AxisAlignedBox3d = AxisAlignedBox3<double>;

template <Size D, typename T>
PURE HOSTDEV constexpr auto
isApprox(AxisAlignedBox<D, T> const & a, AxisAlignedBox<D, T> const & b) noexcept ->
bool;

// -----------------------------------------------------------------------------
// Bounding box
// -----------------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(AxisAlignedBox<D, T> const & a, AxisAlignedBox<D, T> const & b) noexcept ->
AxisAlignedBox<D, T>;

template <Size D, typename T, Size N>
PURE HOSTDEV constexpr auto
boundingBox(Point<D, T> const (&points)[N]) noexcept -> AxisAlignedBox<D, T>;

template <Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(Vector<Point<D, T>> const & points) noexcept -> AxisAlignedBox<D, T>;

} // namespace um2

#include "AxisAlignedBox.inl"
