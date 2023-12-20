#pragma once

#include <um2/mesh/rectilinear_grid.hpp>
#include <um2/stdlib/vector.hpp>

namespace um2
{

//==============================================================================
// RECTILINEAR PARTITION
//==============================================================================
// A D-dimensional rectilinear partition of a D-dimensional box.
//
// Suppose the grid has nx cells in the x direction and ny cells in the y
// y direction. Then the children vector contains nx * ny elements.
// Let i in [0, nx) and j in [0, ny). Then children[i + nx * j] is the child
// of the cell with indices (i, j) in the grid.
//  j
//  ^
//  |
//  |
//  | 2  3
//  | 0  1
//  *-----------> i
//
//  * is where grid.minima is located

template <Size D, typename T, typename P>
class RectilinearPartition
{

  RectilinearGrid<D, T> _grid;
  Vector<P> _children;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RectilinearPartition() noexcept = default;

  HOSTDEV
  constexpr RectilinearPartition(RectilinearGrid<D, T> const & grid,
                                 Vector<P> const & children) noexcept;

  // dydy and an array of IDs, mapping to the dxdy
  constexpr RectilinearPartition(Vector<Vec2<T>> const & dxdy,
                                 Vector<Vector<Size>> const & ids);

  //==============================================================================
  // Methods
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  grid() const noexcept -> RectilinearGrid<D, T> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  children() const noexcept -> Vector<P> const &;

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
  PURE HOSTDEV [[nodiscard]] constexpr auto getFlatIndex(Args... args) const noexcept
      -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFlatIndex(Vec<D, Size> const & index) const noexcept -> Size;

  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto getChild(Args... args) noexcept -> P &;

  template <typename... Args>
    requires(sizeof...(Args) == D)
  PURE HOSTDEV [[nodiscard]] constexpr auto getChild(Args... args) const noexcept
      -> P const &;

  HOSTDEV constexpr void
  clear() noexcept;
};

//==============================================================================
// Aliases
//==============================================================================

template <typename T, typename P>
using RectilinearPartition1 = RectilinearPartition<1, T, P>;

template <typename T, typename P>
using RectilinearPartition2 = RectilinearPartition<2, T, P>;

template <typename T, typename P>
using RectilinearPartition3 = RectilinearPartition<3, T, P>;

template <typename P>
using RectilinearPartition1f = RectilinearPartition1<float, P>;
template <typename P>
using RectilinearPartition2f = RectilinearPartition2<float, P>;
template <typename P>
using RectilinearPartition3f = RectilinearPartition3<float, P>;

template <typename P>
using RectilinearPartition1d = RectilinearPartition1<double, P>;
template <typename P>
using RectilinearPartition2d = RectilinearPartition2<double, P>;
template <typename P>
using RectilinearPartition3d = RectilinearPartition3<double, P>;

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T, typename P>
constexpr RectilinearPartition<D, T, P>::RectilinearPartition(
    RectilinearGrid<D, T> const & grid, Vector<P> const & children) noexcept
    : _grid(grid),
      _children(children)
{
}

template <Size D, typename T, typename P>
constexpr RectilinearPartition<D, T, P>::RectilinearPartition(
    Vector<Vec2<T>> const & dxdy, Vector<Vector<Size>> const & ids)
    : _grid(dxdy, ids)
{
  static_assert(D == 2);
  // Flatten the ids to get the children
  // The rows are in reverse order
  Size const nx = _grid.numXCells();
  Size const ny = _grid.numYCells();
  _children.resize(nx * ny);
  for (Size i = 0; i < ny; ++i) {
    for (Size j = 0; j < nx; ++j) {
      _children[i * nx + j] = static_cast<P>(ids[ny - 1 - i][j]);
    }
  }
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::grid() const noexcept -> RectilinearGrid<D, T> const &
{
  return _grid;
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::children() const noexcept -> Vector<P> const &
{
  return _children;
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::xMin() const noexcept -> T
{
  return _grid.xMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::yMin() const noexcept -> T
{
  return _grid.yMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::zMin() const noexcept -> T
{
  return _grid.zMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::dx() const noexcept -> T
{
  return _grid.dx();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::dy() const noexcept -> T
{
  return _grid.dy();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::dz() const noexcept -> T
{
  return _grid.dz();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::numXCells() const noexcept -> Size
{
  return _grid.numXCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::numYCells() const noexcept -> Size
{
  return _grid.numYCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::numZCells() const noexcept -> Size
{
  return _grid.numZCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::numCells() const noexcept -> Vec<D, Size>
{
  return _grid.numCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::width() const noexcept -> T
{
  return _grid.width();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::height() const noexcept -> T
{
  return _grid.height();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::depth() const noexcept -> T
{
  return _grid.depth();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::xMax() const noexcept -> T
{
  return _grid.xMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::yMax() const noexcept -> T
{
  return _grid.yMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::zMax() const noexcept -> T
{
  return _grid.zMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::maxima() const noexcept -> Point<D, T>
{
  return _grid.maxima();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return _grid.boundingBox();
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV
    constexpr auto RectilinearPartition<D, T, P>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D, T>
{
  return _grid.getBox(args...);
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RectilinearPartition<D, T, P>::getFlatIndex(
    Args... args) const noexcept -> Size
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    ASSERT(index[i] < _grid.divs(i).size());
  }
  if constexpr (D == 1) {
    return index[0];
  } else if constexpr (D == 2) {
    return index[0] + index[1] * numXCells();
  } else { // General case
    // [0, nx, nx*ny, nx*ny*nz, ...]
    Point<D, Size> exclusive_scan_prod;
    exclusive_scan_prod[0] = 1;
    for (Size i = 1; i < D; ++i) {
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * _grid.num_cells[i - 1];
    }
    return index.dot(exclusive_scan_prod);
  }
}

template <Size D, typename T, typename P>
PURE HOSTDEV [[nodiscard]] constexpr auto
RectilinearPartition<D, T, P>::getFlatIndex(Vec<D, Size> const & index) const noexcept
    -> Size
{
  for (Size i = 0; i < D; ++i) {
    ASSERT(index[i] < _grid.divs(i).size());
  }
  if constexpr (D == 1) {
    return index[0];
  } else if constexpr (D == 2) {
    return index[0] + index[1] * numXCells();
  } else { // General case
    // [0, nx, nx*ny, nx*ny*nz, ...]
    Point<D, Size> exclusive_scan_prod;
    exclusive_scan_prod[0] = 1;
    for (Size i = 1; i < D; ++i) {
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * _grid.num_cells[i - 1];
    }
    return index.dot(exclusive_scan_prod);
  }
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RectilinearPartition<D, T, P>::getChild(Args... args) noexcept
    -> P &
{
  return _children[getFlatIndex(args...)];
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV
    constexpr auto RectilinearPartition<D, T, P>::getChild(Args... args) const noexcept
    -> P const &
{
  return _children[getFlatIndex(args...)];
}

template <Size D, typename T, typename P>
HOSTDEV constexpr void
RectilinearPartition<D, T, P>::clear() noexcept
{
  _grid.clear();
  _children.clear();
}

} // namespace um2
