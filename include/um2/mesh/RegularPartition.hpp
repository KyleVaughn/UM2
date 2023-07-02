#pragma once

#include <um2/mesh/RegularGrid.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// REGULAR PARTITION
// -----------------------------------------------------------------------------
// A D-dimensional box, partitioned by a regular grid.
//
// Suppose the grid has nx cells in the x direction and ny cells in the
// y direction. Then the children vector contains nx * ny elements.
// Let i in [0, nx) and j in [0, ny). Then children[i + nx * j] is the child
// of the cell with indices (i, j) in the grid.
//  j
//  ^
//  |
//  | 2 3
//  | 0 1
//  *--------> i
//
//  * is where grid.minima is located.

template <Size D, typename T, typename P>
struct RegularPartition {

  RegularGrid<D, T> grid;
  Vector<P> children;

  // ---------------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------------

  HOSTDEV constexpr RegularPartition() = default;

  // ---------------------------------------------------------------------------
  // Accessors
  // ---------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMin() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMin() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMin() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  dx() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  dy() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  dz() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numXCells() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numYCells() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numZCells() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  width() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  height() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  depth() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMax() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMax() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMax() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getBox(Args... args) const noexcept
      -> AxisAlignedBox<D, T>;

  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getChild(Args... args) noexcept
      -> P &;

  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getChild(Args... args) const noexcept
      -> P const &;

};

// -- Aliases --

template <typename T, typename P>
using RegularPartition1 = RegularPartition<1, T, P>;

template <typename T, typename P>
using RegularPartition2 = RegularPartition<2, T, P>;

template <typename T, typename P>
using RegularPartition3 = RegularPartition<3, T, P>;

template <typename P>
using RegularPartition1f = RegularPartition1<float, P>;
template <typename P>
using RegularPartition2f = RegularPartition2<float, P>;
template <typename P>
using RegularPartition3f = RegularPartition3<float, P>;

template <typename P>
using RegularPartition1d = RegularPartition1<double, P>;
template <typename P>
using RegularPartition2d = RegularPartition2<double, P>;
template <typename P>
using RegularPartition3d = RegularPartition3<double, P>;

} // namespace um2

#include "RegularPartition.inl"
