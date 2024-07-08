#include <um2/common/color.hpp>
#include <um2/common/logger.hpp>
#include <um2/config.hpp>
#include <um2/physics/cross_section.hpp>
#include <um2/physics/cross_section_library.hpp>
#include <um2/physics/material.hpp>
#include <um2/physics/nuclide.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/utility/pair.hpp>
#include <um2/stdlib/vector.hpp>

namespace um2
{

um2::Vector<um2::Pair<Int, Float>> const ZAID_ATOMIC_MASS = {
    { 1001, 1.00783},
    { 1002,  2.0141},
    { 1003,  3.0155},
    { 1006, 1.00783},
    { 1040, 1.00783},
    { 2003, 3.01493},
    { 2004,  4.0026},
    { 3006, 6.01507},
    { 3007,   7.016},
    { 4009,  9.0122},
    { 5000,  10.811},
    { 5010, 10.0129},
    { 5011, 11.0093},
    { 6000, 12.0011},
    { 6001, 12.0011},
    { 7014, 14.0031},
    { 8001, 15.9973},
    { 8016, 15.9949},
    { 9019, 18.9982},
    {11023, 22.9895},
    {12000, 24.3051},
    {13027, 26.9815},
    {14000, 28.0859},
    {15031, 30.9741},
    {16000, 32.0643},
    {17000, 35.4527},
    {19000, 39.0986},
    {20000,  40.078},
    {22000, 47.8933},
    {23000, 50.9416},
    {24000, 51.9959},
    {24050, 49.9461},
    {24052, 51.9402},
    {24053, 52.9408},
    {24054, 53.9394},
    {25055,  54.938},
    {26000, 55.8447},
    {26054, 53.9394},
    {26056, 55.9345},
    {26057, 56.9351},
    {26058, 57.9337},
    {27059, 58.9332},
    {28000, 58.6936},
    {28058, 57.9357},
    {28060, 59.9308},
    {28061, 60.9314},
    {28062,  61.928},
    {28064, 63.9282},
    {29063, 62.9296},
    {29065, 64.9278},
    {35581, 80.9163},
    {36582, 81.9135},
    {36583, 82.9143},
    {36584, 83.9115},
    {36585, 84.9125},
    {36586, 85.9106},
    {38589, 88.9078},
    {38590, 89.9078},
    {39589, 88.9059},
    {39590, 89.9072},
    {39591, 90.9073},
    {40000, 91.2237},
    {40001, 91.2237},
    {40090, 89.9047},
    {40091, 90.9056},
    {40092,  91.905},
    {40094, 93.9063},
    {40096, 95.9083},
    {40591, 90.9056},
    {40593, 92.9064},
    {40595,  94.908},
    {40596, 95.9083},
    {41093, 92.9032},
    {41595, 94.9068},
    {42000, 95.9376},
    {42095, 94.9059},
    {42595, 94.9059},
    {42596, 95.9047},
    {42597,  96.906},
    {42598, 97.9054},
    {42599, 98.9077},
    {42600, 99.9073},
    {43599, 98.9077},
    {44600, 99.9041},
    {44601, 100.906},
    {44602, 101.905},
    {44603, 102.904},
    {44604, 103.903},
    {44605, 104.911},
    {44606, 105.907},
    {45103, 102.905},
    {45603, 102.904},
    {45605, 104.911},
    {46604, 103.904},
    {46605, 104.905},
    {46606, 105.903},
    {46607, 106.905},
    {46608, 107.904},
    {47107, 106.905},
    {47109, 108.905},
    {47609, 108.905},
    {47611, 110.906},
    {47710, 109.906},
    {48000, 112.411},
    {48110, 109.903},
    {48111, 110.904},
    {48112, 111.903},
    {48113,   112.9},
    {48114, 113.903},
    {48610, 109.903},
    {48611, 110.904},
    {48613,   112.9},
    {49000, 114.819},
    {49113, 112.904},
    {49115, 114.904},
    {49615, 114.904},
    {50000,  118.71},
    {50112, 111.904},
    {50114, 113.903},
    {50115, 114.903},
    {50116, 115.902},
    {50117, 116.903},
    {50118, 117.902},
    {50119, 118.903},
    {50120, 119.902},
    {50122, 121.903},
    {50124, 123.905},
    {50125, 124.908},
    {51000, 121.763},
    {51121, 120.909},
    {51123, 122.906},
    {51621, 120.909},
    {51625, 124.905},
    {51627, 126.907},
    {52632, 131.908},
    {52727, 126.905},
    {52729, 128.907},
    {53627, 126.904},
    {53629, 128.905},
    {53631, 130.906},
    {53635, 134.909},
    {54628, 127.903},
    {54630, 129.904},
    {54631, 130.906},
    {54632, 131.903},
    {54633, 132.906},
    {54634,  133.91},
    {54635, 134.907},
    {54636, 135.907},
    {55633, 132.906},
    {55634, 133.907},
    {55635, 134.906},
    {55636, 135.906},
    {55637, 136.907},
    {56634, 133.904},
    {56637, 136.906},
    {56640,  139.91},
    {57639, 138.903},
    {57640,  139.91},
    {58640, 139.905},
    {58641, 140.911},
    {58642, 141.909},
    {58643, 142.913},
    {58644, 143.914},
    {59641, 140.907},
    {59643, 142.911},
    {60642, 141.908},
    {60643,  142.91},
    {60644,  143.91},
    {60645, 144.913},
    {60646, 145.913},
    {60647, 146.916},
    {60648, 147.917},
    {60650, 149.921},
    {61647, 146.915},
    {61648, 147.917},
    {61649, 148.918},
    {61651, 150.922},
    {61748, 147.921},
    {62152,  151.92},
    {62153, 152.922},
    {62647, 146.915},
    {62648, 147.915},
    {62649, 148.917},
    {62650, 149.917},
    {62651, 150.919},
    {62652,  151.92},
    {62653, 152.922},
    {62654, 153.922},
    {63151, 150.916},
    {63152, 151.922},
    {63153, 152.922},
    {63154, 153.922},
    {63155, 154.921},
    {63156, 155.925},
    {63157, 156.925},
    {63651, 150.916},
    {63653, 152.922},
    {63654, 153.922},
    {63655, 154.921},
    {63656, 155.925},
    {63657, 156.925},
    {64152,  151.92},
    {64154, 153.921},
    {64155, 154.923},
    {64156, 155.923},
    {64157, 156.924},
    {64158, 157.924},
    {64160, 159.927},
    {64654, 153.921},
    {64655, 154.923},
    {64656, 155.923},
    {64657, 156.924},
    {64658, 157.924},
    {64660, 159.927},
    {65159, 158.925},
    {65160, 159.927},
    {65161, 160.928},
    {65659, 158.925},
    {65660, 159.927},
    {65661, 160.928},
    {66160, 159.925},
    {66161, 160.926},
    {66162, 161.927},
    {66163, 162.929},
    {66164, 163.929},
    {66660, 159.925},
    {66661, 160.926},
    {66662, 161.927},
    {66663, 162.929},
    {66664, 163.929},
    {67165,  164.93},
    {67665,  164.93},
    {68162, 161.929},
    {68164, 163.929},
    {68166,  165.93},
    {68167, 166.932},
    {68168,  167.93},
    {68170, 169.936},
    {71176, 175.941},
    {72174,  173.94},
    {72176, 175.941},
    {72177, 176.943},
    {72178, 177.944},
    {72179, 178.946},
    {72180, 179.947},
    {73181, 180.954},
    {73182,  181.95},
    {74000, 183.846},
    {74182, 181.953},
    {74183, 182.952},
    {74184,  183.95},
    {74186, 185.958},
    {77191,  190.96},
    {77193, 192.963},
    {79197, 196.966},
    {82206, 205.974},
    {82207, 206.976},
    {82208, 207.977},
    {83209,  208.98},
    {90230, 230.036},
    {90232, 232.038},
    {91231, 231.036},
    {91232, 232.038},
    {91233,  233.04},
    {91234, 234.043},
    {92232, 232.037},
    {92233,  233.04},
    {92234, 234.041},
    {92235, 235.044},
    {92236, 236.046},
    {92237, 237.049},
    {92238, 238.051},
    {93237, 237.048},
    {93238, 238.051},
    {93239, 239.053},
    {94236, 236.046},
    {94238, 238.049},
    {94239, 239.052},
    {94240, 240.054},
    {94241, 241.049},
    {94242, 242.058},
    {95241, 241.057},
    {95242, 242.059},
    {95243, 243.061},
    {95342, 242.059},
    {96242, 242.058},
    {96243, 243.061},
    {96244, 244.063},
    {96245, 245.065},
    {96246, 246.067},
    {96247, 247.071},
    {96248, 248.072},
    {97249,  249.08},
    {98249,  249.08},
    {98250, 250.076},
    {98251,  251.08},
    {98252, 252.082},
};

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
Material::addNuclideWt(Int zaid, Float wt_percent) noexcept
{
  // N_i = w_i * rho * N_A / M_i
  ASSERT(0 < wt_percent);
  ASSERT(wt_percent <= 1);
  ASSERT(0 < _density);
  // Get the atomic mass of the nuclide from the ZAID_ATOMIC_MASS list
  for (auto const & zaid_mass : ZAID_ATOMIC_MASS) {
    if (zaid_mass.first == zaid) {
      // Avogadro's number: 6.02214076e23
      // barns per cm^2: 1e24

      //   g      cm^2      atoms     mol     atom
      // ----- * ------- * ------- * ------ = ----
      //  cm^3    barn       mol       g      b-cm

      // density * wt_percent * (0.602214076 / atomic_mass)

      addNuclide(zaid, wt_percent * _density * 0.602214076 / zaid_mass.second);
      return;
    }
  }
  LOG_ERROR("Atomic mass not found for ZAID: ", zaid);
}

void
Material::addNuclideWt(String const & symbol, Float wt_percent) noexcept
{
  ASSERT(!symbol.empty());
  addNuclideWt(toZAID(symbol), wt_percent);
}

void
Material::addNuclidesAtomPercent(Vector<String> const & symbols,
                                 Vector<Float> const & percents) noexcept
{
  ASSERT(0 < _density); // Need to set density first
  ASSERT(!symbols.empty());
  ASSERT(symbols.size() == percents.size());
  // Convert to weight percent by finding the sum of the atom_percent weighted
  // atomic masses
  // Compute m = sum(gamma_i * m_i)
  Float m = 0;
  for (Int i = 0; i < symbols.size(); ++i) {
    auto const zaid = toZAID(symbols[i]);
    bool found = false;
    for (auto const & zaid_mass : ZAID_ATOMIC_MASS) {
      if (zaid_mass.first == zaid) {
        ASSERT(percents[i] >= 0);
        ASSERT(percents[i] <= 1);
        m += percents[i] * zaid_mass.second;
        found = true;
        break;
      }
    }
    if (!found) {
      LOG_ERROR("Atomic mass not found for ZAID: ", zaid);
    }
  }
  for (Int i = 0; i < symbols.size(); ++i) {
    auto const zaid = toZAID(symbols[i]);
    auto const percent = percents[i];
    addNuclide(zaid, percent * _density * 0.602214076 / m);
  }
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
