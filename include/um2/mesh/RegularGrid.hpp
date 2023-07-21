#pragma once

#include <um2/geometry/AxisAlignedBox.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// REGULAR GRID
// -----------------------------------------------------------------------------
// A regular grid is a grid with a fixed spacing between points.

template <Size D, typename T>
struct RegularGrid {

  // The bottom left corner of the grid.
  Point<D, T> minima;

  // The Δx, Δy, etc. of the grid.
  Vec<D, T> spacing;

  // The number of cells in each direction.
  // Must have at least 1 to form a grid.
  Vec<D, Size> num_cells;

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr RegularGrid() noexcept = default;

  HOSTDEV constexpr RegularGrid(Point<D, T> const & minima_in,
                                Vec<D, T> const & spacing_in,
                                Vec<D, Size> const & num_cells_in) noexcept;

  HOSTDEV constexpr explicit RegularGrid(AxisAlignedBox<D, T> const & box) noexcept;

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

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

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getRangeContaining(AxisAlignedBox<D, T> const & box) const noexcept -> Vec<2 * D, Size>;

};

// -----------------------------------------------------------------------------
// Aliases
// -----------------------------------------------------------------------------

template <typename T>
using RegularGrid1 = RegularGrid<1, T>;
template <typename T>
using RegularGrid2 = RegularGrid<2, T>;
template <typename T>
using RegularGrid3 = RegularGrid<3, T>;

using RegularGrid1f = RegularGrid1<float>;
using RegularGrid2f = RegularGrid2<float>;
using RegularGrid3f = RegularGrid3<float>;

using RegularGrid1d = RegularGrid1<double>;
using RegularGrid2d = RegularGrid2<double>;
using RegularGrid3d = RegularGrid3<double>;

} // namespace um2

#include "RegularGrid.inl"
