#pragma once

#include <um2/common/config.hpp>
#include <um2/common/vector.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cmath>
#include <concepts>
#include <vector>

namespace um2
{

// -----------------------------------------------------------------------------
// RECTILINEAR GRID
// -----------------------------------------------------------------------------
// A D-dimensional rectilinear grid with data of type T

template <len_t D, typename T>
struct RectilinearGrid {

  // Divisions along each axis
  Vector<T> divs[D];

  // -- Constructors --

  UM2_HOSTDEV
  RectilinearGrid() = default;

  UM2_HOSTDEV constexpr explicit RectilinearGrid(AABox<D, T> const & /*box*/);

  UM2_HOSTDEV constexpr RectilinearGrid(AABox<D, T> const * /*boxes*/, len_t /*n*/);

  UM2_HOSTDEV constexpr explicit RectilinearGrid(Vector<AABox<D, T>> const & /*boxes*/);

  // dydy and an array of IDs, mapping to the dxdy
  constexpr RectilinearGrid(std::vector<Vec2<T>> const & /*dxdy*/,
                            std::vector<std::vector<int>> const & /*ids*/) requires(D ==
                                                                                    2);

  // -- Methods --

  UM2_HOSTDEV void
  clear();

  UM2_NDEBUG_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
  getBox(len_t i, len_t j) const -> AABox2<T>
  requires(D == 2);
};

// -- Aliases --

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

// -- Methods --

// Minima/maxima accessors.
template <len_t D, typename T>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto xMin(RectilinearGrid<D, T> const & /*grid*/) -> T;

template <len_t D, typename T>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto yMin(RectilinearGrid<D, T> const & /*grid*/) -> T;

template <len_t D, typename T>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto zMin(RectilinearGrid<D, T> const & /*grid*/) -> T;

template <len_t D, typename T>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto xMax(RectilinearGrid<D, T> const & /*grid*/) -> T;

template <len_t D, typename T>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto yMax(RectilinearGrid<D, T> const & /*grid*/) -> T;

template <len_t D, typename T>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto zMax(RectilinearGrid<D, T> const & /*grid*/) -> T;

// Number of divisions accessors.
template <len_t D, typename T>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto numXcells(RectilinearGrid<D, T> const & /*grid*/) -> len_t;

template <len_t D, typename T>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto numYcells(RectilinearGrid<D, T> const & /*grid*/) -> len_t;

template <len_t D, typename T>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto numZcells(RectilinearGrid<D, T> const & /*grid*/) -> len_t;

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto
numCells(RectilinearGrid<D, T> const & /*grid*/) -> Vec<D, len_t>;

// Width/hight/depth
template <len_t D, typename T>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto width(RectilinearGrid<D, T> const & /*grid*/) -> T;

template <len_t D, typename T>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto height(RectilinearGrid<D, T> const & /*grid*/) -> T;

template <len_t D, typename T>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto depth(RectilinearGrid<D, T> const & /*grid*/) -> T;

// Bounding box
template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto
boundingBox(RectilinearGrid<D, T> const & /*grid*/) -> AABox<D, T>;

} // namespace um2

#include "rectilinear_grid.inl"