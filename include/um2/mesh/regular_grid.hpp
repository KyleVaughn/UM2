#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/stdlib/algorithm/clamp.hpp>
#include <um2/stdlib/math/rounding_functions.hpp>

//==============================================================================
// REGULAR GRID
//==============================================================================
// A regular grid is a grid with a fixed spacing between points. Each grid cell
// is a hyperrectangle with the same shape.

namespace um2
{

template <Int D, class T>
class RegularGrid
{

  // The bottom left corner of the grid.
  Point<D, T> _minima;

  // The Δx, Δy, etc. of the grid.
  Vec<D, T> _spacing;

  // The number of cells in each direction.
  // Must have at least 1 to form a grid.
  Vec<D, Int> _num_cells;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RegularGrid() noexcept = default;

  HOSTDEV constexpr RegularGrid(Point<D, T> const & minima, Vec<D, T> const & spacing,
                                Vec<D, Int> const & num_cells) noexcept;

  HOSTDEV constexpr explicit RegularGrid(AxisAlignedBox<D, T> const & box) noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima(Int i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  spacing() const noexcept -> Vec<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  spacing(Int i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numCells() const noexcept -> Vec<D, Int>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numCells(Int i) const noexcept -> Int;

  //==============================================================================
  // Methods
  //==============================================================================

  // The total number of cells in the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  totalNumCells() const noexcept -> Int;

  // The extent of the grid in each dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents() const noexcept -> Vec<D, T>;

  // The extent of the grid in the i-th dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents(Int i) const noexcept -> T;

  // The maximum point of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D, T>;

  // The maximum value of the grid in the i-th dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima(Int i) const noexcept -> T;

  // Get the bounding box of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  // Get the grid cell at the given index.
  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getBox(Args... args) const noexcept -> AxisAlignedBox<D, T>;

  // Get the flat index of the grid cell at the given multidimensional index.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFlatIndex(Vec<D, Int> const & index) const noexcept -> Int;

  // Get the flat index of the grid cell at the given multidimensional index.
  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFlatIndex(Args... args) const noexcept -> Int;

  // Get the centroid of the grid cell at the given multidimensional index.
  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getCellCentroid(Args... args) const noexcept -> Point<D, T>;

  // Return (ix0, iy0, iz0, ix1, iy1, iz1) where (ix0, iy0, iz0) is the smallest
  // index of a cell that intersects the given box and (ix1, iy1, iz1) is the
  // largest index of a cell that intersects the given box.
  // Hence the box is in the range [ix0, ix1] x [iy0, iy1] x [iz0, iz1].
  //
  // Allows for partial intersection, but returns -1 for the index of the
  // non-intersecting dimension/dimensions.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getCellIndicesIntersecting(AxisAlignedBox<D, T> const & box) const noexcept
      -> Vec<2 * D, Int>;

  // Get the index of the grid cell containing the given point.
  // If the point is outside the grid, returns -1 for the indices.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getCellIndexContaining(Point<D, T> const & point) const noexcept -> Vec<D, Int>;
}; // RegularGrid

//==============================================================================
// Aliases
//==============================================================================

template <class T>
using RegularGrid1 = RegularGrid<1, T>;

template <class T>
using RegularGrid2 = RegularGrid<2, T>;

template <class T>
using RegularGrid3 = RegularGrid<3, T>;

using RegularGrid2F = RegularGrid2<Float>;

//==============================================================================
// Constructors
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr RegularGrid<D, T>::RegularGrid(Point<D, T> const & minima,
                                                 Vec<D, T> const & spacing,
                                                 Vec<D, Int> const & num_cells) noexcept
    : _minima(minima),
      _spacing(spacing),
      _num_cells(num_cells)
{
  // Ensure all spacings and num_cells are positive
  for (Int i = 0; i < D; ++i) {
    ASSERT(spacing[i] > 0);
    ASSERT(num_cells[i] > 0);
  }
}

template <Int D, class T>
HOSTDEV constexpr RegularGrid<D, T>::RegularGrid(
    AxisAlignedBox<D, T> const & box) noexcept
    : _minima(box.minima),
      _spacing(box.maxima - box.minima)
{
  for (Int i = 0; i < D; ++i) {
    _num_cells[i] = 1;
  }
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::minima() const noexcept -> Point<D, T>
{
  return _minima;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::minima(Int const i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _minima[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::spacing() const noexcept -> Vec<D, T>
{
  return _spacing;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::spacing(Int const i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _spacing[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numCells() const noexcept -> Vec<D, Int>
{
  return _num_cells;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numCells(Int const i) const noexcept -> Int
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _num_cells[i];
}

//==============================================================================
// Methods
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::totalNumCells() const noexcept -> Int
{
  Int num_total_cells = 1;
  for (Int i = 0; i < D; ++i) {
    num_total_cells *= _num_cells[i];
  }
  return num_total_cells;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::extents() const noexcept -> Vec<D, T>
{
  Vec<D, T> result;
  for (Int i = 0; i < D; ++i) {
    result[i] = extents(i);
  }
  return result;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::extents(Int const i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _spacing[i] * static_cast<T>(_num_cells[i]);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::maxima() const noexcept -> Point<D, T>
{
  Point<D, T> result;
  for (Int i = 0; i < D; ++i) {
    result[i] = maxima(i);
  }
  return result;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::maxima(Int const i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return minima(i) + extents(i);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return AxisAlignedBox<D, T>(minima(), maxima());
}

template <Int D, class T>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::getBox(Args... args) const noexcept -> AxisAlignedBox<D, T>
{
  Vec<D, Int> const index{args...};
  for (Int i = 0; i < D; ++i) {
    ASSERT(index[i] < _num_cells[i]);
  }
  Point<D, T> box_min;
  for (Int i = 0; i < D; ++i) {
    box_min[i] = _minima[i] + _spacing[i] * static_cast<T>(index[i]);
  }
  return {box_min, box_min + _spacing};
}

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularGrid<D, T>::getFlatIndex(Vec<D, Int> const & index) const noexcept -> Int
{
  for (Int i = 0; i < D; ++i) {
    ASSERT(index[i] < _num_cells[i]);
  }
  // For D = 1, 2, 3, write out the explicit formulas
  if constexpr (D == 1) {
    return index[0];
  } else if constexpr (D == 2) {
    return index[0] + index[1] * _num_cells[0];
  } else if constexpr (D == 3) {
    return index[0] + _num_cells[0] * (index[1] + index[2] * _num_cells[1]);
  } else { // General case
    // [0, nx, nx*ny, nx*ny*nz, ...]
    Vec<D, Int> exclusive_scan_prod;
    exclusive_scan_prod[0] = 1;
    for (Int i = 1; i < D; ++i) {
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * _num_cells[i - 1];
    }
    // i0 + i1*nx + i2*nx*ny + ...
    return index.dot(exclusive_scan_prod);
  }
}

template <Int D, class T>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::getFlatIndex(Args... args) const noexcept -> Int
{
  Vec<D, Int> const index{args...};
  return getFlatIndex(index);
}

template <Int D, class T>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::getCellCentroid(Args... args) const noexcept -> Point<D, T>
{
  T constexpr half = static_cast<T>(1) / static_cast<T>(2);
  Vec<D, Int> const index{args...};
  for (Int i = 0; i < D; ++i) {
    ASSERT(index[i] < _num_cells[i]);
  }
  Point<D, T> result;
  for (Int i = 0; i < D; ++i) {
    result[i] = _minima[i] + _spacing[i] * (static_cast<T>(index[i]) + half);
  }
  return result;
}

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularGrid<D, T>::getCellIndicesIntersecting(
    AxisAlignedBox<D, T> const & box) const noexcept -> Vec<2 * D, Int>
{
  Vec<2 * D, Int> result;
  Int const zero = 0;
  for (Int i = 0; i < D; ++i) {
    // Determine how many cells over the min and max are, then floor to get the
    // interger indices.
    result[i] = static_cast<Int>(um2::floor((box.minima(i) - _minima[i]) / _spacing[i]));
    result[i + D] =
        static_cast<Int>(um2::floor((box.maxima(i) - _minima[i]) / _spacing[i]));
    // If the result is invalid, set to -1 to indicate no intersection
    // If the min index is greater than the max index (_num_cell[i] - 1)
    // If the max index is less than the min index (0)
    if (_num_cells[i] - 1 < result[i] || result[i + D] < 0) {
      result[i] = -1;
      result[i + D] = -1;
    } else {
      // Clamp to the grid bounds, since at least one of the indices is valid
      result[i] = um2::clamp(result[i], zero, _num_cells[i] - 1);
      result[i + D] = um2::clamp(result[i + D], zero, _num_cells[i] - 1);
    }
  }
  return result;
}

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularGrid<D, T>::getCellIndexContaining(Point<D, T> const & point) const noexcept
    -> Vec<D, Int>
{
  Vec<D, Int> result;
  Int const zero = 0;
  for (Int i = 0; i < D; ++i) {
    result[i] = static_cast<Int>(um2::floor((point[i] - _minima[i]) / _spacing[i]));
    // If the result is invalid, set to -1
    if (_num_cells[i] - 1 < result[i] || result[i] < 0) {
      result[i] = -1;
    } else {
      // Clamp to the grid bounds, since the index is valid
      result[i] = um2::clamp(result[i], zero, _num_cells[i] - 1);
    }
  }
  return result;
}

} // namespace um2
