#pragma once

#include <um2/mesh/regular_grid.hpp>

//==============================================================================
// REGULAR PARTITION
//==============================================================================
// A D-dimensional box, partitioned by a regular grid. Each cell of the grid
// contains a "child" object of type P.
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
//
// NOTE: in some use cases, the number of children may be greater than the
// number of cells in the grid. This is allowed, but the user must ensure that
// the number of children is at least equal to the number of cells in the grid.

namespace um2
{

template <Int D, typename P>
class RegularPartition
{

  RegularGrid<D> _grid;
  Vector<P> _children;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RegularPartition() noexcept = default;

  HOSTDEV
  constexpr RegularPartition(RegularGrid<D> const & grid,
                             Vector<P> const & children) noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  grid() const noexcept -> RegularGrid<D> const &;

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
};

//==============================================================================
// Aliases
//==============================================================================

template <typename P>
using RegularPartition1 = RegularPartition<1, P>;

template <typename P>
using RegularPartition2 = RegularPartition<2, P>;

template <typename P>
using RegularPartition3 = RegularPartition<3, P>;

//==============================================================================
// Constructors
//==============================================================================

template <Int D, typename P>
constexpr RegularPartition<D, P>::RegularPartition(RegularGrid<D> const & grid,
                                                   Vector<P> const & children) noexcept
    : _grid(grid),
      _children(children)
{
  // Check that the number of children is at least equal to the total number of
  // cells in the grid. children can be used to store additional data in atypical
  // use cases, but it must be able to store at least one child per cell.
  ASSERT(_grid.totalNumCells() <= _children.size());
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, P>::grid() const noexcept -> RegularGrid<D> const &
{
  return _grid;
}

template <Int D, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, P>::children() const noexcept -> Vector<P> const &
{
  return _children;
}

//==============================================================================
// Methods
//==============================================================================

template <Int D, typename P>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RegularPartition<D, P>::getChild(Args... args) noexcept -> P &
{
  return _children[_grid.getFlatIndex(args...)];
}

template <Int D, typename P>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RegularPartition<D, P>::getChild(Args... args) const noexcept
    -> P const &
{
  return _children[_grid.getFlatIndex(args...)];
}

} // namespace um2
