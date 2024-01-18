#pragma once

#include <um2/geometry/axis_aligned_box.hpp>

//==============================================================================
// REGULAR GRID
//==============================================================================
// A regular grid is a grid with a fixed spacing between points. Each grid cell
// is a hyperrectangle with the same shape. Note, the grid cells are not
// necessarily hypercubes.

namespace um2
{

template <Size D, typename T>
class RegularGrid
{

  // The bottom left corner of the grid.
  Point<D, T> _minima;

  // The Δx, Δy, etc. of the grid.
  Vec<D, T> _spacing;

  // The number of cells in each direction.
  // Must have at least 1 to form a grid.
  Vec<D, Size> _num_cells;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RegularGrid() noexcept = default;

  HOSTDEV constexpr RegularGrid(Point<D, T> const & minima, Vec<D, T> const & spacing,
                                Vec<D, Size> const & num_cells) noexcept;

  //  HOSTDEV constexpr explicit RegularGrid(AxisAlignedBox<D, T> const & box) noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima(Size i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  spacing() const noexcept -> Vec<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  spacing(Size i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numCells() const noexcept -> Vec<D, Size>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numCells(Size i) const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMin() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMin() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMin() const noexcept -> T;

  // The Δx of the grid cells.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  dx() const noexcept -> T;

  // The Δy of the grid cells.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  dy() const noexcept -> T;

  // The Δz of the grid cells.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  dz() const noexcept -> T;

  // The number of cells in the x direction.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  numXCells() const noexcept -> Size;

  // The number of cells in the y direction.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  numYCells() const noexcept -> Size;

  // The number of cells in the z direction.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  numZCells() const noexcept -> Size;

  //==============================================================================
  // Methods
  //==============================================================================

  // The total number of cells in the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  totalNumCells() const noexcept -> Size;

  // The extent of the grid in each dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents() const noexcept -> Vec<D, T>;

  // The extent of the grid in the i-th dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents(Size i) const noexcept -> T;

  // The maximum point of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D, T>;

  // The maximum value of the grid in the i-th dimension.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima(Size i) const noexcept -> T;

  // The x-extent of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  width() const noexcept -> T;

  // The y-extent of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  height() const noexcept -> T;

  // The z-extent of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  depth() const noexcept -> T;

  // The maximum x-value of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMax() const noexcept -> T;

  // The maximum y-value of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMax() const noexcept -> T;

  // The maximum z-value of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMax() const noexcept -> T;

  // Get the bounding box of the grid.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  // Get the grid cell at the given index.
  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto getBox(Args... args) const noexcept
      -> AxisAlignedBox<D, T>;

  // Get the flat index of the grid cell at the given multidimensional index.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFlatIndex(Vec<D, Size> const & index) const noexcept -> Size;

  // Get the flat index of the grid cell at the given multidimensional index.
  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto getFlatIndex(Args... args) const noexcept
      -> Size;

  // Get the centroid of the grid cell at the given multidimensional index.
  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto getCellCentroid(Args... args) const noexcept
      -> Point<D, T>;

  // Return (ix0, iy0, iz0, ix1, iy1, iz1) where (ix0, iy0, iz0) is the smallest
  // index of a cell that intersects the given box and (ix1, iy1, iz1) is the
  // largest index of a cell that intersects the given box.
  // Hence the box is in the range [ix0, ix1] x [iy0, iy1] x [iz0, iz1].
  //
  // Allows for partial intersection, but returns -1 for the index of the
  // non-intersecting dimension/dimensions.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getCellIndicesIntersecting(AxisAlignedBox<D, T> const & box) const noexcept
      -> Vec<2 * D, Size>;

  // Get the index of the grid cell containing the given point.
  // If the point is outside the grid, returns -1 for the indices.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getCellIndexContaining(Point<D, T> const & point) const noexcept -> Vec<D, Size>;
}; // RegularGrid

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
HOSTDEV constexpr RegularGrid<D, T>::RegularGrid(Point<D, T> const & minima,
                                                 Vec<D, T> const & spacing,
                                                 Vec<D, Size> const & num_cells) noexcept
    : _minima(minima),
      _spacing(spacing),
      _num_cells(num_cells)
{
  // Ensure all spacings and num_cells are positive
  for (Size i = 0; i < D; ++i) {
    ASSERT(spacing[i] > 0);
    ASSERT(num_cells[i] > 0);
  }
}

// template <Size D, typename T>
// HOSTDEV constexpr RegularGrid<D, T>::RegularGrid(
//     AxisAlignedBox<D, T> const & box) noexcept
//     : minima(box.minima),
//       spacing(box.maxima)
//{
//   spacing -= minima;
//   for (Size i = 0; i < D; ++i) {
//     num_cells[i] = 1;
//   }
// }

//==============================================================================
// Accessors
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::minima() const noexcept -> Point<D, T>
{
  return _minima;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::minima(Size const i) const noexcept -> T
{
  return _minima[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::spacing() const noexcept -> Vec<D, T>
{
  return _spacing;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::spacing(Size const i) const noexcept -> T
{
  return _spacing[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numCells() const noexcept -> Vec<D, Size>
{
  return _num_cells;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numCells(Size const i) const noexcept -> Size
{
  return _num_cells[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::xMin() const noexcept -> T
{
  return _minima[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::yMin() const noexcept -> T
{
  static_assert(2 <= D);
  return _minima[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::zMin() const noexcept -> T
{
  static_assert(3 <= D);
  return _minima[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::dx() const noexcept -> T
{
  return _spacing[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::dy() const noexcept -> T
{
  static_assert(2 <= D);
  return _spacing[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::dz() const noexcept -> T
{
  static_assert(3 <= D);
  return _spacing[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numXCells() const noexcept -> Size
{
  return _num_cells[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numYCells() const noexcept -> Size
{
  static_assert(2 <= D);
  return _num_cells[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numZCells() const noexcept -> Size
{
  static_assert(3 <= D);
  return _num_cells[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::totalNumCells() const noexcept -> Size
{
  Size num_total_cells = 1;
  for (Size i = 0; i < D; ++i) {
    num_total_cells *= _num_cells[i];
  }
  return num_total_cells;
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::extents() const noexcept -> Vec<D, T>
{
  Vec<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = extents(i);
  }
  return result;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::extents(Size const i) const noexcept -> T
{
  return _spacing[i] * static_cast<T>(_num_cells[i]);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::maxima() const noexcept -> Point<D, T>
{
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = maxima(i);
  }
  return result;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::maxima(Size const i) const noexcept -> T
{
  return _minima[i] + _spacing[i] * static_cast<T>(_num_cells[i]);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::width() const noexcept -> T
{
  return extents(0);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::height() const noexcept -> T
{
  static_assert(2 <= D);
  return extents(1);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::depth() const noexcept -> T
{
  static_assert(3 <= D);
  return extents(2);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::xMax() const noexcept -> T
{
  return maxima(0);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::yMax() const noexcept -> T
{
  static_assert(2 <= D);
  return maxima(1);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::zMax() const noexcept -> T
{
  static_assert(3 <= D);
  return maxima(2);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return AxisAlignedBox<D, T>(_minima, maxima());
}

template <Size D, typename T>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RegularGrid<D, T>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D, T>
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    ASSERT(index[i] < _num_cells[i]);
  }
  Point<D, T> box_min;
  for (Size i = 0; i < D; ++i) {
    box_min[i] = _minima[i] + _spacing[i] * static_cast<T>(index[i]);
  }
  return {box_min, box_min + _spacing};
}

template <Size D, typename T>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularGrid<D, T>::getFlatIndex(Vec<D, Size> const & index) const noexcept -> Size
{
  for (Size i = 0; i < D; ++i) {
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
    Point<D, Size> exclusive_scan_prod;
    exclusive_scan_prod[0] = 1;
    for (Size i = 1; i < D; ++i) {
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * _num_cells[i - 1];
    }
    // i0 + i1*nx + i2*nx*ny + ...
    return index.dot(exclusive_scan_prod);
  }
}

template <Size D, typename T>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RegularGrid<D, T>::getFlatIndex(Args... args) const noexcept
    -> Size
{
  Point<D, Size> const index{args...};
  return getFlatIndex(index);
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
    ASSERT(index[i] < _num_cells[i]);
  }
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] =
        _minima[i] + _spacing[i] * (static_cast<T>(index[i]) + static_cast<T>(0.5));
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
    // Determine how many cells over the min and max are, then floor to get the
    // interger indices.
    result[i] = static_cast<Size>(um2::floor((box.minima(i) - _minima[i]) / _spacing[i]));
    result[i + D] =
        static_cast<Size>(um2::floor((box.maxima(i) - _minima[i]) / _spacing[i]));
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

template <Size D, typename T>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularGrid<D, T>::getCellIndexContaining(Point<D, T> const & point) const noexcept
    -> Vec<D, Size>
{
  Vec<D, Size> result;
  Size const zero = 0;
  for (Size i = 0; i < D; ++i) {
    result[i] = static_cast<Size>(um2::floor((point[i] - _minima[i]) / _spacing[i]));
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
