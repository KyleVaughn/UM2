#pragma once

#include <um2/geometry/axis_aligned_box.hpp>

namespace um2
{

//==============================================================================
// REGULAR GRID
//==============================================================================
//
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

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RegularGrid() noexcept = default;

  HOSTDEV constexpr RegularGrid(Point<D, T> const & minima_in,
                                Vec<D, T> const & spacing_in,
                                Vec<D, Size> const & num_cells_in) noexcept;

  HOSTDEV constexpr explicit RegularGrid(AxisAlignedBox<D, T> const & box) noexcept;

  //==============================================================================
  // Methods
  //==============================================================================

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
  numCells() const noexcept -> Vec<D, Size>;

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
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto getBox(Args... args) const noexcept
      -> AxisAlignedBox<D, T>;

  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto getCellCentroid(Args... args) const noexcept
      -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getCellIndicesIntersecting(AxisAlignedBox<D, T> const & box) const noexcept
      -> Vec<2 * D, Size>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getCellIndexContaining(Point<D, T> const & point) const noexcept -> Vec<D, Size>;
};

//==============================================================================
// Aliases
//==============================================================================

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

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr RegularGrid<D, T>::RegularGrid(
    Point<D, T> const & minima_in, Vec<D, T> const & spacing_in,
    Vec<D, Size> const & num_cells_in) noexcept
    : minima(minima_in),
      spacing(spacing_in),
      num_cells(num_cells_in)
{
  // Ensure all spacings and num_cells are positive
  for (Size i = 0; i < D; ++i) {
    ASSERT(spacing[i] > 0);
    ASSERT(num_cells[i] > 0);
  }
}

template <Size D, typename T>
HOSTDEV constexpr RegularGrid<D, T>::RegularGrid(
    AxisAlignedBox<D, T> const & box) noexcept
    : minima(box.minima),
      spacing(box.maxima)
{
  spacing -= minima;
  for (Size i = 0; i < D; ++i) {
    num_cells[i] = 1;
  }
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::xMin() const noexcept -> T
{
  return minima[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::yMin() const noexcept -> T
{
  static_assert(2 <= D);
  return minima[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::zMin() const noexcept -> T
{
  static_assert(3 <= D);
  return minima[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::dx() const noexcept -> T
{
  return spacing[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::dy() const noexcept -> T
{
  static_assert(2 <= D);
  return spacing[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::dz() const noexcept -> T
{
  static_assert(3 <= D);
  return spacing[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numXCells() const noexcept -> Size
{
  return num_cells[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numYCells() const noexcept -> Size
{
  static_assert(2 <= D);
  return num_cells[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numZCells() const noexcept -> Size
{
  static_assert(3 <= D);
  return num_cells[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numCells() const noexcept -> Vec<D, Size>
{
  return num_cells;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::width() const noexcept -> T
{
  return static_cast<T>(numXCells()) * dx();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::height() const noexcept -> T
{
  static_assert(2 <= D);
  return static_cast<T>(numYCells()) * dy();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::depth() const noexcept -> T
{
  static_assert(3 <= D);
  return static_cast<T>(numZCells()) * dz();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::xMax() const noexcept -> T
{
  return xMin() + width();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::yMax() const noexcept -> T
{
  static_assert(2 <= D);
  return yMin() + height();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::zMax() const noexcept -> T
{
  static_assert(3 <= D);
  return zMin() + depth();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::maxima() const noexcept -> Point<D, T>
{
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = minima[i] + spacing[i] * static_cast<T>(num_cells[i]);
  }
  return result;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return AxisAlignedBox<D, T>(minima, maxima());
}

template <Size D, typename T>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RegularGrid<D, T>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D, T>
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    ASSERT(index[i] < num_cells[i]);
  }
  AxisAlignedBox<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result.minima[i] = minima[i] + spacing[i] * static_cast<T>(index[i]);
    result.maxima[i] = result.minima[i] + spacing[i];
  }
  return result;
}

template <Size D, typename T>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV
    constexpr auto RegularGrid<D, T>::getCellCentroid(Args... args) const noexcept
    -> Point<D, T>
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    ASSERT(index[i] < num_cells[i]);
  }
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = minima[i] + spacing[i] * (static_cast<T>(index[i]) + static_cast<T>(0.5));
  }
  return result;
}

template <Size D, typename T>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularGrid<D, T>::getCellIndicesIntersecting(
    AxisAlignedBox<D, T> const & box) const noexcept -> Vec<2 * D, Size>
{
  Vec<2 * D, Size> result;
  Size const zero = 0;
  for (Size i = 0; i < D; ++i) {
    result[i] = static_cast<Size>(um2::floor((box.minima[i] - minima[i]) / spacing[i]));
    result[i + D] =
        static_cast<Size>(um2::floor((box.maxima[i] - minima[i]) / spacing[i]));
    result[i] = um2::clamp(result[i], zero, num_cells[i] - 1);
    result[i + D] = um2::clamp(result[i + D], zero, num_cells[i] - 1);
  }
  return result;
}

template <Size D, typename T>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularGrid<D, T>::getCellIndexContaining(Point<D, T> const & point) const noexcept
    -> Vec<D, Size>
{
  Vec<D, Size> result;
  Size const zero = 0;
  for (Size i = 0; i < D; ++i) {
    result[i] = static_cast<Size>(um2::floor((point[i] - minima[i]) / spacing[i]));
    result[i] = um2::clamp(result[i], zero, num_cells[i] - 1);
  }
  return result;
}

} // namespace um2
