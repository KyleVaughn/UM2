#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/vector.hpp>
#include <um2/physics/material.hpp>
#include <um2/mesh/polytope_soup.hpp>

namespace um2::mpact
{

//==============================================================================
// getSource
//==============================================================================
// Given an MPACT FSR output and a list of materials, return a vector 
// containing the fission and scattering sources for each cell.

PURE [[nodiscard]] auto
getSource(PolytopeSoup const & soup, Vector<Material> const & materials) -> Vector<Vec2F>;

} // namespace um2::mpact
