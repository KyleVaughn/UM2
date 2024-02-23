#pragma once

#include <um2/config.hpp>

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
mean(Float const * begin, Float const * end) noexcept -> Float;

//=============================================================================
// median
//=============================================================================
// Computes the median of the values in the range [begin, end).
// The range must be sorted.

PURE HOSTDEV auto
median(Float const * begin, Float const * end) noexcept -> Float;

//=============================================================================
// variance
//=============================================================================
// Computes the variance of the values in the range [begin, end).

PURE HOSTDEV auto
variance(Float const * begin, Float const * end) noexcept -> Float;

//=============================================================================
// stdDev
//=============================================================================
// Computes the standard deviation of the values in the range [begin, end).

PURE HOSTDEV auto
stdDev(Float const * begin, Float const * end) noexcept -> Float;

} // namespace um2
