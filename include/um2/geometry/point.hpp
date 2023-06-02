#pragma once

#include <um2/common/config.hpp>
#include <um2/math/vec.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// POINT
// -----------------------------------------------------------------------------
// An alias for a D-dimensional vector. This isn't technically correct, but it
// is more efficient to use a vector for a point than a separate class.

template <len_t D, typename T>
using Point = Vec<D, T>;

// -- Aliases --

template <typename T>
using Point1 = Point<1, T>;

template <typename T>
using Point2 = Point<2, T>;

template <typename T>
using Point3 = Point<3, T>;

using Point1f = Point1<float>;
using Point1d = Point1<double>;

using Point2f = Point2<float>;
using Point2d = Point2<double>;

using Point3f = Point3<float>;
using Point3d = Point3<double>;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

template <std::floating_point T>
UM2_HOSTDEV consteval auto epsilonDistance() -> T
{
  return static_cast<T>(1e-5);
}

template <std::floating_point T>
UM2_HOSTDEV consteval auto epsilonDistanceSquared() -> T
{
  return epsilonDistance<T>() * epsilonDistance<T>();
}

template <std::floating_point T>
UM2_HOSTDEV consteval auto infiniteDistance() -> T
{
  return static_cast<T>(1e10);
}

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------

template <typename DerivedA, typename DerivedB>
UM2_PURE UM2_HOSTDEV constexpr auto
distanceSquared(Eigen::MatrixBase<DerivedA> const & a,
                Eigen::MatrixBase<DerivedB> const & b) noexcept ->
    typename DerivedA::Scalar
{
  // Check that a and b are vectors
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedA);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedB);
  return (a - b).squaredNorm();
}

template <typename DerivedA, typename DerivedB>
UM2_PURE UM2_HOSTDEV constexpr auto
distance(Eigen::MatrixBase<DerivedA> const & a,
         Eigen::MatrixBase<DerivedB> const & b) noexcept -> typename DerivedA::Scalar
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedA);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedB);
  return std::sqrt(distanceSquared(a, b));
}

template <typename DerivedA, typename DerivedB>
UM2_PURE UM2_HOSTDEV constexpr auto
midpoint(Eigen::MatrixBase<DerivedA> const & a,
         Eigen::MatrixBase<DerivedB> const & b) noexcept -> typename DerivedA::PlainObject
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedA);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedB);
  return (a + b) / 2;
}

template <typename DerivedA, typename DerivedB>
UM2_PURE UM2_HOSTDEV constexpr auto
isApprox(Eigen::MatrixBase<DerivedA> const & a,
         Eigen::MatrixBase<DerivedB> const & b) noexcept -> bool
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedA);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedB);
  return distanceSquared(a, b) < epsilonDistanceSquared<typename DerivedA::Scalar>();
}

template <typename DerivedA, typename DerivedB, typename DerivedC>
UM2_PURE UM2_HOSTDEV constexpr auto areCCW(Eigen::MatrixBase<DerivedA> const & a,
                                           Eigen::MatrixBase<DerivedB> const & b,
                                           Eigen::MatrixBase<DerivedC> const & c) noexcept
    -> bool
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedA, 2);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedB, 2);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedC, 2);
  return 0 < cross2(b - a, c - a);
}

} // namespace um2
