#pragma once

#include <um2/config.hpp>
#include <um2/geometry/point.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/utility/pair.hpp>
#include <um2/stdlib/vector.hpp>

namespace um2::mpact
{

//==============================================================================
// getPowers
//==============================================================================
// Given an MPACT FSR output, return a vector containing the power and centroid
// of each disjount region with non-zero power.

PURE [[nodiscard]] auto
getPowers(String const & fsr_output) -> Vector<Pair<Float, Point2F>>;

} // namespace um2::mpact
