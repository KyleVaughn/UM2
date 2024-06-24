#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/stdlib/vector.hpp>

#include <algorithm> // std::sort

//==============================================================================
// RECTILINEAR GRID
//==============================================================================
// A D-dimensional rectilinear grid
//
// Many of the methods do the same thing as RegularGrid, which is commented much
// more thoroughly. See that file for more details.

namespace um2
{

template <Int D, class T>
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
  constexpr RectilinearGrid(Vector<Vec2<T>> const & dxdy, Vector<Vector<Int>> const & ids)
    requires(D == 2);

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  divs(Int i) noexcept -> Vector<T> &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  divs(Int i) const noexcept -> Vector<T> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  minima(Int i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima(Int i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numCells() const noexcept -> Vec<D, Int>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numCells(Int i) const noexcept -> Int;

  //==============================================================================
  // Methods
  //==============================================================================

  // Empty the grid
  HOSTDEV constexpr void
  clear() noexcept;

  // The total number of cells in the grid.    
  PURE HOSTDEV [[nodiscard]] constexpr auto    
  totalNumCells() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents() const noexcept -> Vec<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  extents(Int i) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFlatIndex(Vec<D, Int> const & index) const noexcept -> Int;

  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getFlatIndex(Args... args) const noexcept -> Int;

  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getBox(Args... args) const noexcept
      -> AxisAlignedBox<D, T>;

}; // RectilinearGrid

//==============================================================================
// Aliases
//==============================================================================

template <class T>
using RectilinearGrid1 = RectilinearGrid<1, T>;

template <class T>
using RectilinearGrid2 = RectilinearGrid<2, T>;

template <class T>
using RectilinearGrid3 = RectilinearGrid<3, T>;

//==============================================================================
// Constructors
//==============================================================================

template <Int D, class T>
constexpr RectilinearGrid<D, T>::RectilinearGrid(AxisAlignedBox<D, T> const & box)
{
  for (Int i = 0; i < D; ++i) {
    _divs[i] = {box.minima(i), box.maxima(i)};
    ASSERT(box.minima(i) < box.maxima(i));
  }
}

template <Int D, class T>
constexpr RectilinearGrid<D, T>::RectilinearGrid(Vector<AxisAlignedBox<D, T>> const & boxes)
{
  // Create _divs by finding the unique planar divisions
  T constexpr eps = epsDistance<T>();
  for (auto const & box : boxes) {
    for (Int d = 0; d < D; ++d) {
      bool min_found = false;
      for (Int i = 0; i < _divs[d].size(); ++i) {
        if (um2::abs(_divs[d][i] - box.minima(d)) < eps) {
          min_found = true;
          break;
        }
      }
      if (!min_found) {
        this->_divs[d].emplace_back(box.minima(d));
      }
      bool max_found = false;
      for (Int i = 0; i < _divs[d].size(); ++i) {
        if (um2::abs(_divs[d][i] - box.maxima(d)) < eps) {
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
  ASSERT(totalNumCells() == boxes.size());
}

template <Int D, class T>
constexpr RectilinearGrid<D, T>::RectilinearGrid(Vector<Vec2<T>> const & dxdy,
                                              Vector<Vector<Int>> const & ids)
  requires(D == 2)
{
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
  Vector<AxisAlignedBox<D, T>> boxes(nrows * ncols);
  T y = 0;
  // Iterate rows in reverse order
  for (Int i = 0; i < nrows; ++i) {
    Vector<Int> const & row = ids[nrows - i - 1];
    Vec2<T> lo(static_cast<T>(0), y);
    for (Int j = 0; j < ncols; ++j) {
      Int const id = row[j];
      Vec2<T> const & dxdy_ij = dxdy[id];
      Vec2<T> const hi = lo + dxdy_ij;
      boxes[i * ncols + j] = {lo, hi};
      lo[0] = hi[0];
    }
    y += dxdy[row[0]][1];
  }
  *this = RectilinearGrid(boxes);
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr auto
RectilinearGrid<D, T>::divs(Int i) noexcept -> Vector<T> &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i];
}

template <Int D, class T>
HOSTDEV constexpr auto
RectilinearGrid<D, T>::divs(Int i) const noexcept -> Vector<T> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::minima() const noexcept -> Point<D, T>
{
  Point<D, T> p;
  for (Int i = 0; i < D; ++i) {
    p[i] = _divs[i].front();
  }
  return p;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::minima(Int i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i].front();
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::maxima() const noexcept -> Point<D, T>
{
  Point<D, T> p;
  for (Int i = 0; i < D; ++i) {
    p[i] = _divs[i].back();
  }
  return p;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::maxima(Int i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return _divs[i].back();
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numCells() const noexcept -> Vec<D, Int>
{
  Vec<D, Int> ncells;
  for (Int i = 0; i < D; ++i) {
    ncells[i] = _divs[i].size() - 1;
    ASSERT(ncells[i] >= 1);
  }
  return ncells;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numCells(Int i) const noexcept -> Int
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  ASSERT(_divs[i].size() >= 2);
  return _divs[i].size() - 1;
}

//==============================================================================
// Methods
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr void
RectilinearGrid<D, T>::clear() noexcept
{
  for (Int i = 0; i < D; ++i) {
    _divs[i].clear();
  }
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::totalNumCells() const noexcept -> Int
{
  Int ncells_total = 1;
  for (Int i = 0; i < D; ++i) {
    ncells_total *= numCells(i);
  }
  return ncells_total;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::extents() const noexcept -> Vec<D, T>
{
  return maxima() - minima();
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::extents(Int i) const noexcept -> T
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < D);
  return maxima(i) - minima(i);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return {minima(), maxima()};
}

template <Int D, class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
RectilinearGrid<D, T>::getFlatIndex(Vec<D, Int> const & index) const noexcept -> Int
{
  for (Int i = 0; i < D; ++i) {
    ASSERT(index[i] < _divs[i].size());
  }
  // For D = 1, 2, 3, write out the explicit formulas
  if constexpr (D == 1) {
    return index[0];
  } else if constexpr (D == 2) {
    return index[0] + index[1] * numCells(0); 
  } else if constexpr (D == 3) {
    return index[0] + numCells(0) * (index[1] + numCells(1) * index[2]);
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

template <Int D, class T>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RectilinearGrid<D, T>::getFlatIndex(Args... args) const noexcept -> Int
{
  Vec<D, Int> const index{args...};
  return getFlatIndex(index);
}

template <Int D, class T>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RectilinearGrid<D, T>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D, T>
{
  Vec<D, Int> const index{args...};
  for (Int i = 0; i < D; ++i) {
    ASSERT(index[i] + 1 < _divs[i].size());
  }
  Point<D, T> minima;
  Point<D, T> maxima;
  for (Int i = 0; i < D; ++i) {
    minima[i] = _divs[i][index[i]];
    maxima[i] = _divs[i][index[i] + 1];
  }
  return {minima, maxima};
}

} // namespace um2
