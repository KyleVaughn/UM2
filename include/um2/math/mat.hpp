#pragma once

#include <um2/common/config.hpp>

#include <Eigen/Core> // Eigen::Matrix

namespace um2
{

// -----------------------------------------------------------------------------
// MAT
// -----------------------------------------------------------------------------

template <len_t M, len_t N, typename T>
using Mat = Eigen::Matrix<T, M, N>;

} // namespace um2
