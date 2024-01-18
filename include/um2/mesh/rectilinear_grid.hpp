#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/stdlib/vector.hpp>

//==============================================================================
// RECTILINEAR GRID
//==============================================================================
// A D-dimensional rectilinear grid with data of type T
//
// Many of the methods do the same thing as RegularGrid, which is commented much
// more thoroughly. See that file for more details.
// TODO(kcvaughn): Copy comments from RegularGrid

namespace um2
{

template <Size D, typename T>
// Odd bug with "declaration uses identifier '__i0', which is a reserved identifier",
// but obviously RectilinearGrid doesn't use __i0.
// NOLINTNEXTLINE
class RectilinearGrid
{

  // Divisions along each axis
  Vector<T> _divs[D];

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RectilinearGrid() noexcept = default;

  constexpr explicit RectilinearGrid(AxisAlignedBox<D, T> const & box);

  constexpr explicit RectilinearGrid(Vector<AxisAlignedBox<D, T>> const & boxes);

  // dxdy: The extents of cell 0, 1, 2 ...
  // ids: The IDs of the cells.
  // Example:
  // ids = {
  //  {0, 1, 2, 0},
  //  {0, 2, 0, 2},
  //  {0, 1, 0, 1},
  // };
  // would result in:
  // y ^
  //   | 0 1 2 0
  //   | 0 2 0 2
  //   | 0 1 0 1
  //   +---------> x
  constexpr RectilinearGrid(Vector<Vec2<T>> const & dxdy,
                            Vector<Vector<Size>> const & ids);

  //==============================================================================
  // Accessors
  //==============================================================================

  HOSTDEV [[nodiscard]] constexpr auto
  divs(Size i) noexcept -> Vector<T> &;

  HOSTDEV [[nodiscard]] constexpr auto
  divs(Size i) const noexcept -> Vector<T> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima(Size i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima(Size i) const noexcept -> T;

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

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMax() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMax() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMax() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numXCells() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numYCells() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numZCells() const noexcept -> Size;

  //==============================================================================
  // Methods
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents() const noexcept -> Vec<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents(Size i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  width() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  height() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  depth() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFlatIndex(Vec<D, Size> const & index) const noexcept -> Size;

  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto getFlatIndex(Args... args) const noexcept
      -> Size;

  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto getBox(Args... args) const noexcept
      -> AxisAlignedBox<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  // Empty the grid
  HOSTDEV constexpr void
  clear() noexcept;
}; // RectilinearGrid

//==============================================================================
// Aliases
//==============================================================================

template <typename T>
using RectilinearGrid1 = RectilinearGrid<1, T>;
template <typename T>
using RectilinearGrid2 = RectilinearGrid<2, T>;
template <typename T>
using RectilinearGrid3 = RectilinearGrid<3, T>;

using RectilinearGrid1f = RectilinearGrid1<float>;
using RectilinearGrid2f = RectilinearGrid2<float>;
using RectilinearGrid3f = RectilinearGrid3<float>;

using RectilinearGrid1d = RectilinearGrid1<double>;
using RectilinearGrid2d = RectilinearGrid2<double>;
using RectilinearGrid3d = RectilinearGrid3<double>;

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T>
constexpr RectilinearGrid<D, T>::RectilinearGrid(AxisAlignedBox<D, T> const & box)
{
  for (Size i = 0; i < D; ++i) {
    ASSERT(box.minima(i) < box.maxima(i));
    _divs[i] = {box.minima(i), box.maxima(i)};
  }
}

template <Size D, typename T>
constexpr RectilinearGrid<D, T>::RectilinearGrid(
    Vector<AxisAlignedBox<D, T>> const & boxes)
{
  // Create _divs by finding the unique planar divisions
  T constexpr eps = eps_distance<T>;
  for (Size i = 0; i < boxes.size(); ++i) {
    AxisAlignedBox<D, T> const & box = boxes[i];
    for (Size d = 0; d < D; ++d) {
      bool min_found = false;
      for (Size j = 0; j < _divs[d].size(); ++j) {
        if (um2::abs(_divs[d][j] - box.minima(d)) < eps) {
          min_found = true;
          break;
        }
      }
      if (!min_found) {
        this->_divs[d].emplace_back(box.minima(d));
      }
      bool max_found = false;
      for (Size j = 0; j < _divs[d].size(); ++j) {
        if (um2::abs(_divs[d][j] - box.maxima(d)) < eps) {
          max_found = true;
          break;
        }
      }
      if (!max_found) {
        this->_divs[d].emplace_back(box.maxima(d));
      }
    }
  }
  // We now have the unique divisions for each dimension. Sort them.
  for (Size i = 0; i < D; ++i) {
    std::sort(_divs[i].begin(), _divs[i].end());
  }
  // Ensure that the boxes completely cover the grid
  // all num__divs >= 2
  // n = ∏(num__divs[i] - 1)
  Size ncells_total = 1;
  for (Size i = 0; i < D; ++i) {
    ASSERT(_divs[i].size() >= 2);
    ncells_total *= _divs[i].size() - 1;
  }
  ASSERT(ncells_total == boxes.size());
}

template <Size D, typename T>
constexpr RectilinearGrid<D, T>::RectilinearGrid(Vector<Vec2<T>> const & dxdy,
                                                 Vector<Vector<Size>> const & ids)
{
  static_assert(D == 2);
  // Convert the dxdy to AxisAlignedBoxes
  Size const nrows = ids.size();
  Size const ncols = ids[0].size();
  // Ensure that each row has the same number of columns
  for (Size i = 1; i < nrows; ++i) {
    ASSERT(ids[i].size() == ncols);
    for (Size j = 0; j < ncols; ++j) {
      ASSERT(ids[i][j] >= 0);
    }
  }
  Vector<AxisAlignedBox<D, T>> boxes(nrows * ncols);
  T y = 0;
  // Iterate rows in reverse order
  for (Size i = 0; i < nrows; ++i) {
    Vector<Size> const & row = ids[nrows - i - 1];
    Vec2<T> lo(static_cast<T>(0), y);
    for (Size j = 0; j < ncols; ++j) {
      Size const id = row[j];
      Vec2<T> const & dxdy_ij = dxdy[id];
      Vec2<T> const hi = lo + dxdy_ij;
      boxes[i * ncols + j] = {lo, hi};
      lo[0] = hi[0];
    }
    y += dxdy[row[0]][1];
  }
  new (this) RectilinearGrid(boxes);
}

//==============================================================================
// Accessors
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr auto
RectilinearGrid<D, T>::divs(Size i) noexcept -> Vector<T> &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i];
}

template <Size D, typename T>
HOSTDEV constexpr auto
RectilinearGrid<D, T>::divs(Size i) const noexcept -> Vector<T> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::minima() const noexcept -> Point<D, T>
{
  Point<D, T> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = _divs[i].front();
  }
  return p;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::minima(Size i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i].front();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::maxima() const noexcept -> Point<D, T>
{
  Point<D, T> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = _divs[i].back();
  }
  return p;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::maxima(Size i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i].back();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numCells() const noexcept -> Vec<D, Size>
{
  Vec<D, Size> ncells;
  for (Size i = 0; i < D; ++i) {
    ncells[i] = _divs[i].size() - 1;
    ASSERT(ncells[i] >= 1);
  }
  return ncells;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numCells(Size i) const noexcept -> Size
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  ASSERT(_divs[i].size() >= 2);
  return _divs[i].size() - 1;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::xMin() const noexcept -> T
{
  return minima(0);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::yMin() const noexcept -> T
{
  static_assert(2 <= D);
  return minima(1);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::zMin() const noexcept -> T
{
  static_assert(3 <= D);
  return minima(2);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::xMax() const noexcept -> T
{
  return maxima(0);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::yMax() const noexcept -> T
{
  static_assert(2 <= D);
  return maxima(1);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::zMax() const noexcept -> T
{
  static_assert(3 <= D);
  return maxima(2);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numXCells() const noexcept -> Size
{
  static_assert(1 <= D);
  return numCells(0);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numYCells() const noexcept -> Size
{
  static_assert(2 <= D);
  return numCells(1);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numZCells() const noexcept -> Size
{
  static_assert(3 <= D);
  return numCells(2);
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::extents() const noexcept -> Vec<D, T>
{
  return maxima() - minima();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::extents(Size i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return maxima(i) - minima(i);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::width() const noexcept -> T
{
  return extents(0);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::height() const noexcept -> T
{
  static_assert(2 <= D);
  return extents(1);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::depth() const noexcept -> T
{
  static_assert(3 <= D);
  return extents(2);
}

template <Size D, typename T>
HOSTDEV constexpr void
RectilinearGrid<D, T>::clear() noexcept
{
  for (Size i = 0; i < D; ++i) {
    _divs[i].clear();
  }
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return {minima(), maxima()};
}

template <Size D, typename T>
PURE HOSTDEV [[nodiscard]] constexpr auto
RectilinearGrid<D, T>::getFlatIndex(Vec<D, Size> const & index) const noexcept -> Size
{
  for (Size i = 0; i < D; ++i) {
    ASSERT(index[i] < _divs[i].size());
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
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * numCells(i - 1);
    }
    // i0 + i1*nx + i2*nx*ny + ...
    return index.dot(exclusive_scan_prod);
  }
}

template <Size D, typename T>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV
    constexpr auto RectilinearGrid<D, T>::getFlatIndex(Args... args) const noexcept
    -> Size
{
  Point<D, Size> const index{args...};
  return getFlatIndex(index);
}

template <Size D, typename T>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RectilinearGrid<D, T>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D, T>
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    ASSERT(index[i] + 1 < _divs[i].size());
  }
  Point<D, T> minima;
  Point<D, T> maxima;
  for (Size i = 0; i < D; ++i) {
    minima[i] = _divs[i][index[i]];
    maxima[i] = _divs[i][index[i] + 1];
  }
  return {minima, maxima};
}

} // namespace um2
