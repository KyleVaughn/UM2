#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/stdlib/vector.hpp>

//==============================================================================
// RECTILINEAR GRID
//==============================================================================
// A D-dimensional rectilinear grid with data of type T

namespace um2
{

template <Size D, typename T>
// clang-tidy complaing about '__i0' in the name of the struct
// NOLINTNEXTLINE justified above
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

  // dxdy and an array of IDs, mapping to the dxdy
  constexpr RectilinearGrid(Vector<Vec2<T>> const & dxdy,
                            Vector<Vector<Size>> const & ids);

  //==============================================================================
  // Methods
  //==============================================================================

  HOSTDEV [[nodiscard]] constexpr auto
  divs(Size i) noexcept -> Vector<T> &;

  HOSTDEV [[nodiscard]] constexpr auto
  divs(Size i) const noexcept -> Vector<T> const &;

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
  minima() const noexcept -> Vec<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  maxima() const noexcept -> Vec<D, T>;

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

  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto getBox(Args... args) const noexcept
      -> AxisAlignedBox<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  HOSTDEV constexpr void
  clear() noexcept;
};

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
    _divs[i] = {box.minima()[i], box.maxima()[i]};
    ASSERT(box.minima()[i] < box.maxima()[i]);
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
        if (std::abs(_divs[d][j] - box.minima()[d]) < eps) {
          min_found = true;
          break;
        }
      }
      if (!min_found) {
        this->_divs[d].emplace_back(box.minima()[d]);
      }
      bool max_found = false;
      for (Size j = 0; j < _divs[d].size(); ++j) {
        if (std::abs(_divs[d][j] - box.maxima()[d]) < eps) {
          max_found = true;
          break;
        }
      }
      if (!max_found) {
        this->_divs[d].emplace_back(box.maxima()[d]);
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
// Methods
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
RectilinearGrid<D, T>::xMin() const noexcept -> T
{
  return _divs[0].front();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::yMin() const noexcept -> T
{
  static_assert(2 <= D);
  return _divs[1].front();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::zMin() const noexcept -> T
{
  static_assert(3 <= D);
  return _divs[2].front();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::minima() const noexcept -> Vec<D, T>
{
  Vec<D, T> mins;
  for (Size i = 0; i < D; ++i) {
    mins[i] = _divs[i].front();
  }
  return mins;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::xMax() const noexcept -> T
{
  return _divs[0].back();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::yMax() const noexcept -> T
{
  static_assert(2 <= D);
  return _divs[1].back();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::zMax() const noexcept -> T
{
  static_assert(3 <= D);
  return _divs[2].back();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::maxima() const noexcept -> Vec<D, T>
{
  Vec<D, T> maxs;
  for (Size i = 0; i < D; ++i) {
    maxs[i] = _divs[i].back();
  }
  return maxs;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numXCells() const noexcept -> Size
{
  static_assert(1 <= D);
  return _divs[0].size() - 1;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numYCells() const noexcept -> Size
{
  static_assert(2 <= D);
  return _divs[1].size() - 1;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numZCells() const noexcept -> Size
{
  static_assert(3 <= D);
  return _divs[2].size() - 1;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numCells() const noexcept -> Vec<D, Size>
{
  Vec<D, Size> num_cells;
  for (Size i = 0; i < D; ++i) {
    num_cells[i] = _divs[i].size() - 1;
  }
  return num_cells;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::width() const noexcept -> T
{
  return xMax() - xMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::height() const noexcept -> T
{
  static_assert(2 <= D);
  return yMax() - yMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::depth() const noexcept -> T
{
  static_assert(3 <= D);
  return zMax() - zMin();
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
