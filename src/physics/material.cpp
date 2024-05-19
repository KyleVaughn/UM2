#include <um2/physics/material.hpp>

#include <um2/common/logger.hpp>

namespace um2
{

//==============================================================================
// Member functions
//==============================================================================

void
Material::validateProperties() const noexcept
{
  if (_name.empty()) {
    LOG_ERROR("Material name is empty"); 
  }
  if (_temperature <= 0) {
    LOG_ERROR("Material temperature is not positive");
  }
  if (_density <= 0) {
    LOG_ERROR("Material density is not positive");
  }
  if (_num_density.empty()) {
    LOG_ERROR("Material number densities are empty"); 
  }
  if (_zaid.empty()) {
    LOG_ERROR("Material ZAIDs are empty");
  }
  if (_num_density.size() != _zaid.size()) {
    LOG_ERROR("Material number densities and ZAIDs are not the same size");
  }
  for (auto const & num_density : _num_density) {
    if (num_density < 0) {
      LOG_ERROR("Material number density is negative");
    }
  }
  for (auto const & zaid : _zaid) {
    if (zaid <= 0) {
      LOG_ERROR("Material ZAID is not positive");
    }
  }
}

void
Material::validateXSec() const noexcept
{
  if (!_xsec.isMacro()) {
    LOG_ERROR("Material cross section is not macroscopic"); 
  }
  _xsec.validate();
}

void
Material::addNuclide(Int zaid, Float num_density) noexcept
{
  if (zaid <= 0) {
    LOG_ERROR("Invalid ZAID");
  }
  if (num_density < 0) {
    LOG_ERROR("Invalid number density");
  }
  // Check if the nuclide is already in the list
  for (auto const & z : _zaid) {
    if (z == zaid) {
      LOG_ERROR("Nuclide already exists in material");
    }
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
  Int const num_groups = xsec_lib.numGroups();
  _xsec = XSec(num_groups);
  // Ensure temperature, density, and number densities are set
  validateProperties();
  _xsec.isMacro() = true;
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
    ASSERT(atom_density > 0);
    if (xs_nuc.isFissile()) {
      _xsec.isFissile() = true;
    } 
    // TODO(kcvaughn): This should be done using arithmetic operations on XSec
    for (Int ig = 0; ig < num_groups; ++ig) {
      _xsec.a()[ig] += xs_nuc.a()[ig] * atom_density;
      _xsec.f()[ig] += xs_nuc.f()[ig] * atom_density;
      _xsec.nuf()[ig] += xs_nuc.nuf()[ig] * atom_density;
      _xsec.tr()[ig] += xs_nuc.tr()[ig] * atom_density;
      _xsec.s()[ig] += xs_nuc.s()[ig] * atom_density;
      for (Int jg = 0; jg < num_groups; ++jg) {
        _xsec.ss()(jg, ig) += xs_nuc.ss()(jg, ig) * atom_density;
      }
    }
  }
  _xsec.validate();
}

//==============================================================================
// Free functions
//==============================================================================

PURE auto
getC5G7Materials() noexcept -> Vector<Material>
{
  Vector<Material> materials(7);

  auto const xsecs = getC5G7XSecs();

  // UO2
  materials[0].setName("UO2");
  materials[0].setColor(um2::forestgreen);
  materials[0].xsec() = xsecs[0];

  // MOX 4.3%
  materials[1].setName("MOX_4.3");
  materials[1].setColor(um2::yellow);
  materials[1].xsec() = xsecs[1];

  // MOX 7.0%
  materials[2].setName("MOX_7.0");
  materials[2].setColor(um2::orange);
  materials[2].xsec() = xsecs[2];

  // MOX 8.7%
  materials[3].setName("MOX_8.7");
  materials[3].setColor(um2::red);
  materials[3].xsec() = xsecs[3];

  // Fisstion Chamber 
  materials[4].setName("Fission_Chamber");
  materials[4].setColor(um2::black);
  materials[4].xsec() = xsecs[4];

  // Guide Tube
  materials[5].setName("Guide_Tube");
  materials[5].setColor(um2::darkgrey);
  materials[5].xsec() = xsecs[5];

  // Moderator
  materials[6].setName("Moderator");
  materials[6].setColor(um2::royalblue);
  materials[6].xsec() = xsecs[6];

  return materials;
}

} // namespace um2
