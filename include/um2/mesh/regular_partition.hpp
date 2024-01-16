#pragma once

#include <um2/mesh/regular_grid.hpp>

//==============================================================================
// REGULAR PARTITION
//==============================================================================
// A D-dimensional box, partitioned by a regular grid.
//
// Suppose the grid has nx cells in the x direction and ny cells in the
// y direction.
// Let i in [0, nx) and j in [0, ny). Then children[i + nx * j] is the child
// of the cell with indices (i, j) in the grid.
//  j
//  ^
//  |
//  | 2 3
//  | 0 1
//  *--------> i
//
//  * is where grid.minima is located.

namespace um2
{

template <Size D, typename T, typename P>
class RegularPartition
{

  RegularGrid<D, T> _grid;
  Vector<P> _children;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RegularPartition() noexcept = default;

  HOSTDEV
  constexpr RegularPartition(RegularGrid<D, T> const & grid,
                             Vector<P> const & children) noexcept;

  //==============================================================================
  // Methods
  //==============================================================================

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
  numTotalCells() const noexcept -> Size;

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

template <typename T, typename P>
using RegularPartition1 = RegularPartition<1, T, P>;

template <typename T, typename P>
using RegularPartition2 = RegularPartition<2, T, P>;

template <typename T, typename P>
using RegularPartition3 = RegularPartition<3, T, P>;

template <typename P>
using RegularPartition1f = RegularPartition1<float, P>;
template <typename P>
using RegularPartition2f = RegularPartition2<float, P>;
template <typename P>
using RegularPartition3f = RegularPartition3<float, P>;

template <typename P>
using RegularPartition1d = RegularPartition1<double, P>;
template <typename P>
using RegularPartition2d = RegularPartition2<double, P>;
template <typename P>
using RegularPartition3d = RegularPartition3<double, P>;

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T, typename P>
constexpr RegularPartition<D, T, P>::RegularPartition(RegularGrid<D, T> const & grid,
                                                      Vector<P> const & children) noexcept
    : _grid(grid),
      _children(children)
{
  // Check that the number of children is at least equal to the total number of
  // cells in the grid. children can be used to store additional data in atypical
  // use cases, but it must be able to store at least one child per cell.
  ASSERT(_grid.numTotalCells() <= _children.size());
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::children() const noexcept -> Vector<P> const &
{
  return _children;
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::xMin() const noexcept -> T
{
  return _grid.xMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::yMin() const noexcept -> T
{
  return _grid.yMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::zMin() const noexcept -> T
{
  return _grid.zMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::dx() const noexcept -> T
{
  return _grid.dx();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::dy() const noexcept -> T
{
  return _grid.dy();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::dz() const noexcept -> T
{
  return _grid.dz();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::numXCells() const noexcept -> Size
{
  return _grid.numXCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::numYCells() const noexcept -> Size
{
  return _grid.numYCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::numZCells() const noexcept -> Size
{
  return _grid.numZCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::numCells() const noexcept -> Vec<D, Size>
{
  return _grid.numCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::numTotalCells() const noexcept -> Size
{
  return _grid.numTotalCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::width() const noexcept -> T
{
  return _grid.width();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::height() const noexcept -> T
{
  return _grid.height();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::depth() const noexcept -> T
{
  return _grid.depth();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::xMax() const noexcept -> T
{
  return _grid.xMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::yMax() const noexcept -> T
{
  return _grid.yMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::zMax() const noexcept -> T
{
  return _grid.zMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::maxima() const noexcept -> Point<D, T>
{
  return _grid.maxima();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return _grid.boundingBox();
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RegularPartition<D, T, P>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D, T>
{
  return _grid.getBox(args...);
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV
    constexpr auto RegularPartition<D, T, P>::getFlatIndex(Args... args) const noexcept
    -> Size
{
  return _grid.getFlatIndex(args...);
}

template <Size D, typename T, typename P>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularPartition<D, T, P>::getFlatIndex(Vec<D, Size> const & index) const noexcept -> Size
{
  return _grid.getFlatIndex(index);
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RegularPartition<D, T, P>::getChild(Args... args) noexcept
    -> P &
{
  return _children[getFlatIndex(args...)];
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV
    constexpr auto RegularPartition<D, T, P>::getChild(Args... args) const noexcept
    -> P const &
{
  return _children[getFlatIndex(args...)];
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV [[nodiscard]] constexpr auto RegularPartition<D, T, P>::getCellCentroid(
    Args... args) const noexcept -> Point<D, T>
{
  return _grid.getCellCentroid(args...);
}

template <Size D, typename T, typename P>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularPartition<D, T, P>::getCellIndicesIntersecting(
    AxisAlignedBox<D, T> const & box) const noexcept -> Vec<2 * D, Size>
{
  return _grid.getCellIndicesIntersecting(box);
}

template <Size D, typename T, typename P>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularPartition<D, T, P>::getCellIndexContaining(
    Point<D, T> const & point) const noexcept -> Vec<D, Size>
{
  return _grid.getCellIndexContaining(point);
}

} // namespace um2
