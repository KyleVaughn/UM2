#pragma once

#include <um2/common/Vector.hpp>
#include <um2/config.hpp>
#include <um2/geometry/AxisAlignedBox.hpp>
//
// #include <cmath>
// #include <concepts>
// #include <vector>

namespace um2
{

// -----------------------------------------------------------------------------
// RECTILINEAR GRID
// -----------------------------------------------------------------------------
// A D-dimensional rectilinear grid with data of type T

template <Size D, typename T>
struct RectilinearGrid {

  // Divisions along each axis
  Vector<T> divs[D];

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr RectilinearGrid() noexcept = default;
  //
  //  HOSTDEV constexpr explicit RectilinearGrid(AABox<D, T> const & /*box*/);
  //
  //  HOSTDEV constexpr RectilinearGrid(AABox<D, T> const * /*boxes*/, Size /*n*/);
  //
  //  HOSTDEV constexpr explicit RectilinearGrid(Vector<AABox<D, T>> const & /*boxes*/);
  //
  //  // dydy and an array of IDs, mapping to the dxdy
  //  constexpr RectilinearGrid(std::vector<Vec2<T>> const & /*dxdy*/,
  //                            std::vector<std::vector<int>> const & /*ids*/) requires(D
  //                            ==
  //                                                                                    2);
  //

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

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

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  HOSTDEV constexpr void
  clear() noexcept;
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

} // namespace um2

#include "RectilinearGrid.inl"
