#pragma once

#include <um2/common/config.hpp>
#include <um2/math/vec.hpp>

namespace um2 {

// -----------------------------------------------------------------------------
// MAT 
// -----------------------------------------------------------------------------
// An M by N matrix.
//
// This struct is used for VERY small matrices, where the matrix size is known
// at compile time. The matrix is stored in column-major order.
//
// Note that there is not a general matmul or matvec function. Anything beyond
// very small matrices should be done using something like OpenBLAS or cuBLAS.

template <len_t M, len_t N, typename T>
requires(std::is_arithmetic_v<T>)
struct Mat {
};

} // namespace um2