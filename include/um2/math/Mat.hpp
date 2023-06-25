#pragma once

#include <um2/config.hpp>

#include <Eigen/Core> // Eigen::Matrix

namespace um2
{

// -----------------------------------------------------------------------------
// MAT
// -----------------------------------------------------------------------------

template <Size M, Size N, typename T>
using Mat = Eigen::Matrix<T, M, N>;

} // namespace um2
