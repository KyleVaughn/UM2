#pragma once

#include <um2/common/config.hpp>

#include <Eigen/Dense> // Eigen::Matrix

namespace um2
{

// -----------------------------------------------------------------------------
// VEC
// -----------------------------------------------------------------------------

template <len_t D, typename T>
using Vec = Eigen::Matrix<T, D, 1>;

template <typename T>
using Vec2 = Vec<2, T>;

template <typename T>
using Vec3 = Vec<3, T>;

template <typename T>
using Vec4 = Vec<4, T>;

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2i = Vec2<int32_t>;
using Vec2u = Vec2<uint32_t>;

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int32_t>;
using Vec3u = Vec3<uint32_t>;

using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;
using Vec4i = Vec4<int32_t>;
using Vec4u = Vec4<uint32_t>;

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto cross(Vec3<T> const & a, Vec3<T> const & b) -> Vec3<T>
{
  return a.cross(b);
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto cross(Vec2<T> const & a, Vec2<T> const & b) -> T
{
  return a[0] * b[1] - a[1] * b[0];
}

} // namespace um2
