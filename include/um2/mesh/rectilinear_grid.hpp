#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/stdlib/vector.hpp>

#include <algorithm> // std::sort

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

template <Int D>
// Odd bug with "declaration uses identifier '__i0', which is a reserved identifier",
// but obviously RectilinearGrid doesn't use __i0.
// NOLINTNEXTLINE
class RectilinearGrid
{

  // Divisions along each axis
  Vector<Float> _divs[D];

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RectilinearGrid() noexcept = default;

  constexpr explicit RectilinearGrid(AxisAlignedBox<D> const & box);

  constexpr explicit RectilinearGrid(Vector<AxisAlignedBox<D>> const & boxes);

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
  constexpr RectilinearGrid(Vector<Vec2<Float>> const & dxdy, Vector<Vector<Int>> const & ids);

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  divs(Int i) noexcept -> Vector<Float> &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  divs(Int i) const noexcept -> Vector<Float> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima() const noexcept -> Point<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima(Int i) const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima(Int i) const noexcept -> Float;

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

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xMax() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  yMax() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zMax() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numXCells() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numYCells() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numZCells() const noexcept -> Int;

  //==============================================================================
  // Methods
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents() const noexcept -> Vec<D, Float>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents(Int i) const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  width() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  height() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  depth() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFlatIndex(Vec<D, Int> const & index) const noexcept -> Int;

  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getFlatIndex(Args... args) const noexcept -> Int;

  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getBox(Args... args) const noexcept
      -> AxisAlignedBox<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>;

  // Empty the grid
  HOSTDEV constexpr void
  clear() noexcept;
}; // RectilinearGrid

//==============================================================================
// Aliases
//==============================================================================

using RectilinearGrid1 = RectilinearGrid<1>;
using RectilinearGrid2 = RectilinearGrid<2>;
using RectilinearGrid3 = RectilinearGrid<3>;

//==============================================================================
// Constructors
//==============================================================================

template <Int D>
constexpr RectilinearGrid<D>::RectilinearGrid(AxisAlignedBox<D> const & box)
{
  for (Int i = 0; i < D; ++i) {
    ASSERT(box.minima(i) < box.maxima(i));
    _divs[i] = {box.minima(i), box.maxima(i)};
  }
}

template <Int D>
constexpr RectilinearGrid<D>::RectilinearGrid(Vector<AxisAlignedBox<D>> const & boxes)
{
  // Create _divs by finding the unique planar divisions
  Float constexpr eps = eps_distance;
  for (Int i = 0; i < boxes.size(); ++i) {
    AxisAlignedBox<D> const & box = boxes[i];
    for (Int d = 0; d < D; ++d) {
      bool min_found = false;
      for (Int j = 0; j < _divs[d].size(); ++j) {
        if (um2::abs(_divs[d][j] - box.minima(d)) < eps) {
          min_found = true;
          break;
        }
      }
      if (!min_found) {
        this->_divs[d].emplace_back(box.minima(d));
      }
      bool max_found = false;
      for (Int j = 0; j < _divs[d].size(); ++j) {
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
  for (Int i = 0; i < D; ++i) {
    std::sort(_divs[i].begin(), _divs[i].end());
  }
  // Ensure that the boxes completely cover the grid
  // all num__divs >= 2
  // n = ∏(num__divs[i] - 1)
  Int ncells_total = 1;
  for (Int i = 0; i < D; ++i) {
    ASSERT(_divs[i].size() >= 2);
    ncells_total *= _divs[i].size() - 1;
  }
  ASSERT(ncells_total == boxes.size());
}

template <Int D>
constexpr RectilinearGrid<D>::RectilinearGrid(Vector<Vec2<Float>> const & dxdy,
                                              Vector<Vector<Int>> const & ids)
{
  static_assert(D == 2);
  // Convert the dxdy to AxisAlignedBoxes
  Int const nrows = ids.size();
  Int const ncols = ids[0].size();
  // Ensure that each row has the same number of columns
  for (Int i = 1; i < nrows; ++i) {
    ASSERT(ids[i].size() == ncols);
    for (Int j = 0; j < ncols; ++j) {
      ASSERT(ids[i][j] >= 0);
    }
  }
  Vector<AxisAlignedBox<D>> boxes(nrows * ncols);
  Float y = 0;
  // Iterate rows in reverse order
  for (Int i = 0; i < nrows; ++i) {
    Vector<Int> const & row = ids[nrows - i - 1];
    Vec2<Float> lo(static_cast<Float>(0), y);
    for (Int j = 0; j < ncols; ++j) {
      Int const id = row[j];
      Vec2<Float> const & dxdy_ij = dxdy[id];
      Vec2<Float> const hi = lo + dxdy_ij;
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

template <Int D>
HOSTDEV constexpr auto
RectilinearGrid<D>::divs(Int i) noexcept -> Vector<Float> &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i];
}

template <Int D>
HOSTDEV constexpr auto
RectilinearGrid<D>::divs(Int i) const noexcept -> Vector<Float> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::minima() const noexcept -> Point<D>
{
  Point<D> p;
  for (Int i = 0; i < D; ++i) {
    p[i] = _divs[i].front();
  }
  return p;
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::minima(Int i) const noexcept -> Float
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i].front();
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::maxima() const noexcept -> Point<D>
{
  Point<D> p;
  for (Int i = 0; i < D; ++i) {
    p[i] = _divs[i].back();
  }
  return p;
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::maxima(Int i) const noexcept -> Float
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i].back();
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::numCells() const noexcept -> Vec<D, Int>
{
  Vec<D, Int> ncells;
  for (Int i = 0; i < D; ++i) {
    ncells[i] = _divs[i].size() - 1;
    ASSERT(ncells[i] >= 1);
  }
  return ncells;
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::numCells(Int i) const noexcept -> Int
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  ASSERT(_divs[i].size() >= 2);
  return _divs[i].size() - 1;
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::xMin() const noexcept -> Float
{
  return minima(0);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::yMin() const noexcept -> Float
{
  static_assert(2 <= D);
  return minima(1);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::zMin() const noexcept -> Float
{
  static_assert(3 <= D);
  return minima(2);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::xMax() const noexcept -> Float
{
  return maxima(0);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::yMax() const noexcept -> Float
{
  static_assert(2 <= D);
  return maxima(1);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::zMax() const noexcept -> Float
{
  static_assert(3 <= D);
  return maxima(2);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::numXCells() const noexcept -> Int
{
  static_assert(1 <= D);
  return numCells(0);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::numYCells() const noexcept -> Int
{
  static_assert(2 <= D);
  return numCells(1);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::numZCells() const noexcept -> Int
{
  static_assert(3 <= D);
  return numCells(2);
}

//==============================================================================
// Methods
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::extents() const noexcept -> Vec<D, Float>
{
  return maxima() - minima();
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::extents(Int i) const noexcept -> Float
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return maxima(i) - minima(i);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::width() const noexcept -> Float
{
  return extents(0);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::height() const noexcept -> Float
{
  static_assert(2 <= D);
  return extents(1);
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::depth() const noexcept -> Float
{
  static_assert(3 <= D);
  return extents(2);
}

template <Int D>
HOSTDEV constexpr void
RectilinearGrid<D>::clear() noexcept
{
  for (Int i = 0; i < D; ++i) {
    _divs[i].clear();
  }
}

template <Int D>
PURE HOSTDEV constexpr auto
RectilinearGrid<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
{
  return {minima(), maxima()};
}

template <Int D>
PURE HOSTDEV [[nodiscard]] constexpr auto
RectilinearGrid<D>::getFlatIndex(Vec<D, Int> const & index) const noexcept -> Int
{
  for (Int i = 0; i < D; ++i) {
    ASSERT(index[i] < _divs[i].size());
  }
  // Floator D = 1, 2, 3, write out the explicit formulas
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
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * numCells(i - 1);
    }
    // i0 + i1*nx + i2*nx*ny + ...
    return index.dot(exclusive_scan_prod);
  }
}

template <Int D>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RectilinearGrid<D>::getFlatIndex(Args... args) const noexcept -> Int
{
  Vec<D, Int> const index{args...};
  return getFlatIndex(index);
}

template <Int D>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RectilinearGrid<D>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D>
{
  Vec<D, Int> const index{args...};
  for (Int i = 0; i < D; ++i) {
    ASSERT(index[i] + 1 < _divs[i].size());
  }
  Point<D> minima;
  Point<D> maxima;
  for (Int i = 0; i < D; ++i) {
    minima[i] = _divs[i][index[i]];
    maxima[i] = _divs[i][index[i] + 1];
  }
  return {minima, maxima};
}

} // namespace um2
