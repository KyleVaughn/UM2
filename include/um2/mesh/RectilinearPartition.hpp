#pragma once

#include <um2/mesh/RectilinearGrid.hpp>
#include <um2/stdlib/Vector.hpp>

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
struct RectilinearPartition {

  RectilinearGrid<D, T> grid;
  Vector<P> children;

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr RectilinearPartition() noexcept = default;

  // dydy and an array of IDs, mapping to the dxdy
  constexpr RectilinearPartition(Vector<Vec2<T>> const & dxdy,
                                 Vector<Vector<Size>> const & ids);

  //==============================================================================
  // Accessors
  //==============================================================================

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

  //==============================================================================
  // Methods
  //==============================================================================

  HOSTDEV constexpr void
  clear() noexcept;
};

// -- Aliases --

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

} // namespace um2

#include "RectilinearPartition.inl"
