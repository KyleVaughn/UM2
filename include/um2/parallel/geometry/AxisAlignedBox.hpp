#pragma once

#include <um2/geometry/AxisAlignedBox.hpp>

#if UM2_USE_TBB
#  include <execution>
#endif

namespace um2::parallel
{

template <Size D, typename T>
PURE auto
boundingBox(Vector<Point<D, T>> const & points) noexcept -> AxisAlignedBox<D, T>
{
  struct ReduceFunctor {
    constexpr auto
    operator()(AxisAlignedBox<D, T> const & box, Point<D, T> const & p) const noexcept
        -> AxisAlignedBox<D, T>
    {
      return box + p;
    }

    constexpr auto
    operator()(Point<D, T> const & p, AxisAlignedBox<D, T> const & box) const noexcept
        -> AxisAlignedBox<D, T>
    {
      return box + p;
    }

    constexpr auto
    operator()(AxisAlignedBox<D, T> const & a,
               AxisAlignedBox<D, T> const & b) const noexcept -> AxisAlignedBox<D, T>
    {
      return a + b;
    }

    constexpr auto
    operator()(Point<D, T> const & a, Point<D, T> const & b) const noexcept
        -> AxisAlignedBox<D, T>
    {
      return boundingBox(a, b);
    }
  };

  return std::reduce(std::execution::par_unseq, points.begin(), points.end(),
                     AxisAlignedBox<D, T>{points[0], points[0]}, ReduceFunctor{});
}

} // namespace um2::parallel
