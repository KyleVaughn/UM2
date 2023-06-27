#pragma once

#include <um2/common/config.hpp>
#include <um2/mesh/regular_grid.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// REGULAR PARTITION
// -----------------------------------------------------------------------------
// A D-dimensional regular partition of a D-dimensional box.

template <len_t D, typename T, typename P>
struct RegularPartition {

  RegularGrid<D, T> grid;
  Vector<P> children;
  // Suppose the grid has nx cells in the x direction and ny cells in the y
  // y direction. Then the children vector contains nx * ny elements.
  // Let i in [0, nx) and j in [0, ny). Then children[i + nx * j] is the child
  // of the cell with indices (i, j) in the grid.
  //  j
  //  ^
  //  |
  //  |
  //  | 2 3
  //  | 0 1
  //  *-----------> i
  //
  //  * is where (x_min(grid), y_min(grid)) is located.
  //  Hence:
  //  an increasing i corresponds to an increasing x coordinate
  //  an increasing j corresponds to an increasing y coordinate.

  // -- Constructors --

  UM2_HOSTDEV
  RegularPartition() = default;

  // -- Methods --

  UM2_NDEBUG_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
  getBox(len_t i, len_t j) const -> AABox2<T>
  requires(D == 2);

  UM2_NDEBUG_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
  getChild(len_t i, len_t j) -> P & requires(D == 2);

  UM2_NDEBUG_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
  getChild(len_t i, len_t j) const -> P const & requires(D == 2);
};

// -- Aliases --

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

// -- Methods --

// Minima/maxima accessors.
template <len_t D, typename T, typename P>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto xMin(RegularPartition<D, T, P> const & /*part*/) -> T;

template <len_t D, typename T, typename P>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto yMin(RegularPartition<D, T, P> const & /*part*/) -> T;

template <len_t D, typename T, typename P>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto zMin(RegularPartition<D, T, P> const & /*part*/) -> T;

template <len_t D, typename T, typename P>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto xMax(RegularPartition<D, T, P> const & /*part*/) -> T;

template <len_t D, typename T, typename P>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto yMax(RegularPartition<D, T, P> const & /*part*/) -> T;

template <len_t D, typename T, typename P>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto zMax(RegularPartition<D, T, P> const & /*part*/) -> T;

// Number of divisions accessors.
template <len_t D, typename T, typename P>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto numXcells(RegularPartition<D, T, P> const & /*part*/) -> len_t;

template <len_t D, typename T, typename P>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto numYcells(RegularPartition<D, T, P> const & /*part*/) -> len_t;

template <len_t D, typename T, typename P>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto numZcells(RegularPartition<D, T, P> const & /*part*/) -> len_t;

template <len_t D, typename T, typename P>
UM2_PURE UM2_HOSTDEV constexpr auto
numCells(RegularPartition<D, T, P> const & /*part*/) -> Vec<D, len_t>;

// Width/hight/depth
template <len_t D, typename T, typename P>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto width(RegularPartition<D, T, P> const & /*part*/) -> T;

template <len_t D, typename T, typename P>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto height(RegularPartition<D, T, P> const & /*part*/) -> T;

template <len_t D, typename T, typename P>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto depth(RegularPartition<D, T, P> const & /*part*/) -> T;

// Bounding box
template <len_t D, typename T, typename P>
UM2_PURE UM2_HOSTDEV constexpr auto
boundingBox(RegularPartition<D, T, P> const & /*part*/) -> AABox<D, T>;

} // namespace um2

#include "regular_partition.inl"