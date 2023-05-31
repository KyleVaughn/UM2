#pragma once

#include <um2/common/config.hpp>
#include <um2/common/vector.hpp>
#include <um2/geometry/point.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// AXIS-ALIGNED BOX
// -----------------------------------------------------------------------------
// A D-dimensional axis-aligned box.

template <len_t D, typename T>
struct AABox {

  Point<D, T> minima;
  Point<D, T> maxima;

  // -- Constructors --

  constexpr AABox() = default;
  UM2_HOSTDEV constexpr AABox(Point<D, T> const & min, Point<D, T> const & max);

  // -- Accessors --

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto xmin() const -> T;
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto xmax() const -> T;
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto ymin() const -> T;
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto ymax() const -> T;
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto zmin() const -> T;
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto zmax() const -> T;

  // -- Methods --

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto width() const -> T;  // dx
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto height() const -> T; // dy
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto depth() const -> T;  // dz
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto centroid() const -> Point<D, T>;
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto contains(Point<D, T> const & p) const
      -> bool;

}; // struct AABox

// -- Aliases --

template <typename T>
using AABox1 = AABox<1, T>;

template <typename T>
using AABox2 = AABox<2, T>;

template <typename T>
using AABox3 = AABox<3, T>;

using AABox2f = AABox2<float>;
using AABox2d = AABox2<double>;

using AABox3f = AABox3<float>;
using AABox3d = AABox3<double>;

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto isApprox(AABox<D, T> const & a, AABox<D, T> const & b)
    -> bool;

// -- Bounding box --

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto boundingBox(AABox<D, T> const & a,
                                                AABox<D, T> const & b) -> AABox<D, T>;

template <len_t D, typename T, len_t N>
UM2_PURE UM2_HOSTDEV constexpr auto boundingBox(Point<D, T> const (&points)[N])
    -> AABox<D, T>;

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto boundingBox(Vector<Point<D, T>> const & points)
    -> AABox<D, T>;

} // namespace um2

#include "axis_aligned_box.inl"
