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

// Cross product of two 2D vectors (returns a scalar, since the result is in the z
// direction)
template <typename DerivedA, typename DerivedB>
UM2_PURE UM2_HOSTDEV constexpr auto
cross2(Eigen::MatrixBase<DerivedA> const & a,
       Eigen::MatrixBase<DerivedB> const & b) noexcept -> typename DerivedA::Scalar
{
  // Check that a and b are vectors of size 2
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedA, 2);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedB, 2);
  return a.x() * b.y() - a.y() * b.x();
}

} // namespace um2
