#pragma once

#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/math.hpp>

//=============================================================================
// STATS
//=============================================================================
// This file contains functions for computing statistics on a range of values.
//
// The following functions are provided:
// mean
// median (requires sorted range)
// variance
// stdDev

namespace um2
{

//=============================================================================
// mean
//=============================================================================
// Computes the mean of the values in the range [begin, end).

PURE HOSTDEV auto
mean(F const * begin, F const * end) noexcept -> F;

//=============================================================================
// median
//=============================================================================
// Computes the median of the values in the range [begin, end).
// The range must be sorted.

PURE HOSTDEV auto
median(F const * begin, F const * end) noexcept -> F;

//=============================================================================
// variance
//=============================================================================
// Computes the variance of the values in the range [begin, end).

PURE HOSTDEV auto
variance(F const * begin, F const * end) noexcept -> F;

//=============================================================================
// stdDev
//=============================================================================
// Computes the standard deviation of the values in the range [begin, end).

PURE HOSTDEV auto
stdDev(F const * begin, F const * end) noexcept -> F;

} // namespace um2
