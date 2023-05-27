#pragma once

#include <um2/common/config.hpp>

#include <Eigen/Core> // Eigen::Matrix

namespace um2
{

// -----------------------------------------------------------------------------
// VEC
// -----------------------------------------------------------------------------

template <len_t D, typename T>
using Vec = Eigen::Matrix<T, D, 1>;

template <typename T>
using Vec2 = Vec<2, T>;

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2i = Vec2<int32_t>;
using Vec2u = Vec2<uint32_t>;

} // namespace um2
