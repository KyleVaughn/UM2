#include <um2/physics/material.hpp>

namespace um2
{

//==============================================================================
// Member functions
//==============================================================================

void
Material::validate() const noexcept
{
  ASSERT(!_name.empty());
  // If the cross section is non-empty, disregard physical properties
  if (!_xsec.t().empty()) {
    _xsec.validate();
    ASSERT(_xsec.isMacro());
  } else {
    ASSERT(_temperature > 0);
    ASSERT(_density > 0);
    ASSERT(!_num_density.empty());
    ASSERT(_num_density.size() == _zaid.size());
    for (auto const & num_density : _num_density) {
      ASSERT(num_density >= 0);
    }
    for (auto const & zaid : _zaid) {
      ASSERT(zaid > 0);
    }
  }
}

void
Material::addNuclide(Int zaid, Float num_density) noexcept
{
  ASSERT(zaid > 0);
  ASSERT(num_density >= 0);
  // Check if the nuclide is already in the list
  for (auto const & z : _zaid) {
    ASSERT(z != zaid);
  }
  _zaid.emplace_back(zaid);
  _num_density.emplace_back(num_density);
}

void
Material::addNuclide(String const & symbol, Float num_density) noexcept
{
  ASSERT(!symbol.empty());
  addNuclide(toZAID(symbol), num_density);
}

void
Material::populateXSec(XSLibrary const & xsec_lib) noexcept
{
  _xsec.t().clear();
  // Ensure temperature, density, and number densities are set
  validate();
  _xsec.isMacro() = true;
  Int const num_groups = xsec_lib.numGroups();
  _xsec.t().resize(num_groups);
  // For each nuclide in the material:
  //  find the corresponding nuclide in the library
  //  interpolate the cross sections to the temperature of the material
  //  scale the cross sections by the atom density of the nuclide
  //  reduce
  Int const num_nuclides = numNuclides();
  for (Int inuc = 0; inuc < num_nuclides; ++inuc) {
    auto const zaid = _zaid[inuc];
    auto const & lib_nuc = xsec_lib.getNuclide(zaid);
    auto const xs_nuc = lib_nuc.interpXS(getTemperature());
    auto const atom_density = numDensity(inuc);
    for (Int ig = 0; ig < num_groups; ++ig) {
      _xsec.t(ig) += xs_nuc.t(ig) * atom_density;
    }
  }
  _xsec.validate();
}

} // namespace um2
