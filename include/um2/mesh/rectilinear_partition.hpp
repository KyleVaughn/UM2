#pragma once

#include <um2/mesh/rectilinear_grid.hpp>

//==============================================================================
// RECTILINEAR PARTITION
//==============================================================================
// A D-dimensional rectilinear partition of a D-dimensional box.
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

template <Int D, typename P>
class RectilinearPartition
{

  RectilinearGrid<D, Float> _grid;
  Vector<P> _children;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RectilinearPartition() noexcept = default;

  // See RectilinearGrid constructor for details on dxdy and ids.
  constexpr RectilinearPartition(Vector<Vec2F> const & dxdy,
                                 Vector<Vector<Int>> const & ids) noexcept;

  HOSTDEV
  constexpr RectilinearPartition(RectilinearGrid<D, Float> const & grid,
                                 Vector<P> const & children) noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  grid() const noexcept -> RectilinearGrid<D, Float> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  children() const noexcept -> Vector<P> const &;

  //==============================================================================
  // Methods
  //==============================================================================

  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getChild(Args... args) noexcept -> P &;

  template <typename... Args>
  requires(sizeof...(Args) == D) PURE HOSTDEV
      [[nodiscard]] constexpr auto getChild(Args... args) const noexcept -> P const &;

  HOSTDEV constexpr void
  clear() noexcept;
};

//==============================================================================
// Aliases
//==============================================================================

template <typename P>
using RectilinearPartition1 = RectilinearPartition<1, P>;

template <typename P>
using RectilinearPartition2 = RectilinearPartition<2, P>;

template <typename P>
using RectilinearPartition3 = RectilinearPartition<3, P>;

//==============================================================================
// Constructors
//==============================================================================

template <Int D, typename P>
constexpr RectilinearPartition<D, P>::RectilinearPartition(
    RectilinearGrid<D, Float> const & grid, Vector<P> const & children) noexcept
    : _grid(grid),
      _children(children)
{
  ASSERT(_grid.totalNumCells() == _children.size());
}

template <Int D, typename P>
constexpr RectilinearPartition<D, P>::RectilinearPartition(
    Vector<Vec2F> const & dxdy, Vector<Vector<Int>> const & ids) noexcept
    : _grid(dxdy, ids)
{
  static_assert(D == 2);
  // Flatten the ids to get the children
  // The rows are in reverse order
  Int const nx = _grid.numCells(0);
  Int const ny = _grid.numCells(1);
  _children.resize(nx * ny);
  for (Int i = 0; i < ny; ++i) {
    for (Int j = 0; j < nx; ++j) {
      _children[i * nx + j] = static_cast<P>(ids[ny - 1 - i][j]);
    }
  }
  ASSERT(_grid.totalNumCells() == _children.size());
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, P>::grid() const noexcept -> RectilinearGrid<D, Float> const &
{
  return _grid;
}

template <Int D, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, P>::children() const noexcept -> Vector<P> const &
{
  return _children;
}

//==============================================================================
// Methods
//==============================================================================

template <Int D, typename P>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RectilinearPartition<D, P>::getChild(Args... args) noexcept -> P &
{
  return _children[_grid.getFlatIndex(args...)];
}

template <Int D, typename P>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RectilinearPartition<D, P>::getChild(Args... args) const noexcept
    -> P const &
{
  return _children[_grid.getFlatIndex(args...)];
}

template <Int D, typename P>
HOSTDEV constexpr void
RectilinearPartition<D, P>::clear() noexcept
{
  _grid.clear();
  _children.clear();
}

} // namespace um2
