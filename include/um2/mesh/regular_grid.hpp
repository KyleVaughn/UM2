#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/stdlib/math.hpp>
#include <um2/stdlib/algorithm/clamp.hpp>

//==============================================================================
// REGULAR GRID
//==============================================================================
// A regular grid is a grid with a fixed spacing between points. Each grid cell
// is a hyperrectangle with the same shape.

namespace um2
{

template <Int D>
class RegularGrid
{

  // The bottom left corner of the grid.
  Point<D> _minima;

  // The Δx, Δy, etc. of the grid.
  Vec<D, Float> _spacing;

  // The number of cells in each direction.
  // Must have at least 1 to form a grid.
  Vec<D, Int> _num_cells;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RegularGrid() noexcept = default;

  HOSTDEV constexpr RegularGrid(Point<D> const & minima, Vec<D, Float> const & spacing,
                                Vec<D, Int> const & num_cells) noexcept;

  //  HOSTDEV constexpr explicit RegularGrid(AxisAlignedBox<D> const & box) noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima() const noexcept -> Point<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima(Int i) const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  spacing() const noexcept -> Vec<D, Float>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  spacing(Int i) const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numCells() const noexcept -> Vec<D, Int>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numCells(Int i) const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMin() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMin() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMin() const noexcept -> Float;

  // The Δx of the grid cells.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  dx() const noexcept -> Float;

  // The Δy of the grid cells.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  dy() const noexcept -> Float;

  // The Δz of the grid cells.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  dz() const noexcept -> Float;

  // The number of cells in the x direction.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  numXCells() const noexcept -> Int;

  // The number of cells in the y direction.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  numYCells() const noexcept -> Int;

  // The number of cells in the z direction.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  numZCells() const noexcept -> Int;

  //==============================================================================
  // Methods
  //==============================================================================

  // The total number of cells in the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  totalNumCells() const noexcept -> Int;

  // The extent of the grid in each dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents() const noexcept -> Vec<D, Float>;

  // The extent of the grid in the i-th dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents(Int i) const noexcept -> Float;

  // The maximum point of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D>;

  // The maximum value of the grid in the i-th dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima(Int i) const noexcept -> Float;

  // The x-extent of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  width() const noexcept -> Float;

  // The y-extent of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  height() const noexcept -> Float;

  // The z-extent of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  depth() const noexcept -> Float;

  // The maximum x-value of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMax() const noexcept -> Float;

  // The maximum y-value of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMax() const noexcept -> Float;

  // The maximum z-value of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMax() const noexcept -> Float;

  // Get the bounding box of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>;

  // Get the grid cell at the given index.
  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getBox(Args... args) const noexcept
      -> AxisAlignedBox<D>;

  // Get the flat index of the grid cell at the given multidimensional index.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFlatIndex(Vec<D, Int> const & index) const noexcept -> Int;

  // Get the flat index of the grid cell at the given multidimensional index.
  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getFlatIndex(Args... args) const noexcept -> Int;

  // Get the centroid of the grid cell at the given multidimensional index.
  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getCellCentroid(Args... args) const noexcept
      -> Point<D>;

  // Return (ix0, iy0, iz0, ix1, iy1, iz1) where (ix0, iy0, iz0) is the smallest
  // index of a cell that intersects the given box and (ix1, iy1, iz1) is the
  // largest index of a cell that intersects the given box.
  // Hence the box is in the range [ix0, ix1] x [iy0, iy1] x [iz0, iz1].
  //
  // Allows for partial intersection, but returns -1 for the index of the
  // non-intersecting dimension/dimensions.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getCellIndicesIntersecting(AxisAlignedBox<D> const & box) const noexcept
      -> Vec<2 * D, Int>;

  // Get the index of the grid cell containing the given point.
  // If the point is outside the grid, returns -1 for the indices.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getCellIndexContaining(Point<D> const & point) const noexcept -> Vec<D, Int>;
}; // RegularGrid

//==============================================================================
// Aliases
//==============================================================================

using RegularGrid1 = RegularGrid<1>;
using RegularGrid2 = RegularGrid<2>;
using RegularGrid3 = RegularGrid<3>;

//==============================================================================
// Constructors
//==============================================================================

template <Int D>
HOSTDEV constexpr RegularGrid<D>::RegularGrid(Point<D> const & minima,
                                              Vec<D, Float> const & spacing,
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

// template <Int D>
// HOSTDEV constexpr RegularGrid<D>::RegularGrid(
//     AxisAlignedBox<D> const & box) noexcept
//     : minima(box.minima),
//       spacing(box.maxima)
//{
//   spacing -= minima;
//   for (Int i = 0; i < D; ++i) {
//     num_cells[i] = 1;
//   }
// }

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::minima() const noexcept -> Point<D>
{
  return _minima;
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::minima(Int const i) const noexcept -> Float
{
  return _minima[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::spacing() const noexcept -> Vec<D, Float>
{
  return _spacing;
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::spacing(Int const i) const noexcept -> Float
{
  return _spacing[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::numCells() const noexcept -> Vec<D, Int>
{
  return _num_cells;
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::numCells(Int const i) const noexcept -> Int
{
  return _num_cells[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::xMin() const noexcept -> Float
{
  return _minima[0];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::yMin() const noexcept -> Float
{
  static_assert(2 <= D);
  return _minima[1];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::zMin() const noexcept -> Float
{
  static_assert(3 <= D);
  return _minima[2];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::dx() const noexcept -> Float
{
  return _spacing[0];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::dy() const noexcept -> Float
{
  static_assert(2 <= D);
  return _spacing[1];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::dz() const noexcept -> Float
{
  static_assert(3 <= D);
  return _spacing[2];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::numXCells() const noexcept -> Int
{
  return _num_cells[0];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::numYCells() const noexcept -> Int
{
  static_assert(2 <= D);
  return _num_cells[1];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::numZCells() const noexcept -> Int
{
  static_assert(3 <= D);
  return _num_cells[2];
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::totalNumCells() const noexcept -> Int
{
  Int num_total_cells = 1;
  for (Int i = 0; i < D; ++i) {
    num_total_cells *= _num_cells[i];
  }
  return num_total_cells;
}

//==============================================================================
// Methods
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::extents() const noexcept -> Vec<D, Float>
{
  Vec<D, Float> result;
  for (Int i = 0; i < D; ++i) {
    result[i] = extents(i);
  }
  return result;
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::extents(Int const i) const noexcept -> Float
{
  return _spacing[i] * static_cast<Float>(_num_cells[i]);
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::maxima() const noexcept -> Point<D>
{
  Point<D> result;
  for (Int i = 0; i < D; ++i) {
    result[i] = maxima(i);
  }
  return result;
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::maxima(Int const i) const noexcept -> Float
{
  ASSERT_ASSUME(0 <= i && i < D);
  return _minima[i] + _spacing[i] * static_cast<Float>(_num_cells[i]);
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::width() const noexcept -> Float
{
  return extents(0);
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::height() const noexcept -> Float
{
  static_assert(2 <= D);
  return extents(1);
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::depth() const noexcept -> Float
{
  static_assert(3 <= D);
  return extents(2);
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::xMax() const noexcept -> Float
{
  return maxima(0);
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::yMax() const noexcept -> Float
{
  static_assert(2 <= D);
  return maxima(1);
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::zMax() const noexcept -> Float
{
  static_assert(3 <= D);
  return maxima(2);
}

template <Int D>
PURE HOSTDEV constexpr auto
RegularGrid<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
{
  return AxisAlignedBox<D>(_minima, maxima());
}

template <Int D>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RegularGrid<D>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D>
{
  Vec<D, Int> const index{args...};
  for (Int i = 0; i < D; ++i) {
    ASSERT(index[i] < _num_cells[i]);
  }
  Point<D> box_min;
  for (Int i = 0; i < D; ++i) {
    box_min[i] = _minima[i] + _spacing[i] * static_cast<Float>(index[i]);
  }
  return {box_min, box_min + _spacing};
}

template <Int D>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularGrid<D>::getFlatIndex(Vec<D, Int> const & index) const noexcept -> Int
{
  for (Int i = 0; i < D; ++i) {
    ASSERT(index[i] < _num_cells[i]);
  }
  // For D = 1, 2, 3, write out the explicit formulas
  if constexpr (D == 1) {
    return index[0];
  } else if constexpr (D == 2) {
    return index[0] + index[1] * numXCells();
  } else if constexpr (D == 3) {
    return index[0] + index[1] * numXCells() + index[2] * numXCells() * numYCells();
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

template <Int D>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RegularGrid<D>::getFlatIndex(Args... args) const noexcept -> Int
{
  Vec<D, Int> const index{args...};
  return getFlatIndex(index);
}

template <Int D>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RegularGrid<D>::getCellCentroid(Args... args) const noexcept
    -> Point<D>
{
  Float constexpr half = static_cast<Float>(1) / static_cast<Float>(2);
  Vec<D, Int> const index{args...};
  for (Int i = 0; i < D; ++i) {
    ASSERT(index[i] < _num_cells[i]);
  }
  Point<D> result;
  for (Int i = 0; i < D; ++i) {
    result[i] = _minima[i] + _spacing[i] * (static_cast<Float>(index[i]) + half);
  }
  return result;
}

template <Int D>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularGrid<D>::getCellIndicesIntersecting(AxisAlignedBox<D> const & box) const noexcept
    -> Vec<2 * D, Int>
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

template <Int D>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularGrid<D>::getCellIndexContaining(Point<D> const & point) const noexcept -> Vec<D, Int>
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
