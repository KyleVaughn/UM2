#include <um2/physics/material.hpp>

namespace um2
{

//==============================================================================
// Member functions
//==============================================================================

void
Material::setName(String const & name) noexcept
{
  ASSERT(name.size() > 0);
  _name = name;
}

void
Material::validate() const noexcept
{
  ASSERT(_name.size() > 0);
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

void
Material::addNuclide(I zaid, F num_density) noexcept
{
  ASSERT(zaid > 0);
  ASSERT(num_density >= 0);
  // Check if the nuclide is already in the list
  for (auto const & z : _zaid) {
    ASSERT(z != zaid);
  }
  _zaid.push_back(zaid);
  _num_density.push_back(num_density);
}

void
Material::addNuclide(String const & symbol, F num_density) noexcept
{
  ASSERT(symbol.size() > 0);
  addNuclide(toZAID(symbol), num_density);
}

} // namespace um2
