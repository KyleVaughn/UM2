#include <um2/common/logger.hpp>
#include <um2/common/strto.hpp>
#include <um2/config.hpp>
#include <um2/physics/cross_section.hpp>
#include <um2/physics/nuclide.hpp>
#include <um2/stdlib/algorithm/is_sorted.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/math/roots.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/string_view.hpp>
#include <um2/stdlib/vector.hpp>

#include <cctype> // std::isalpha, std::isdigit

namespace um2
{

//==============================================================================
// Member functions
//==============================================================================

void
Nuclide::clear() noexcept
{
  _zaid = 0;
  _mass = 0;
  _temperatures.clear();
  _xs.clear();
}

void
Nuclide::validate() const noexcept
{
  if (_zaid <= 0) {
    LOG_ERROR("Nuclide has invalid ZAID: ", _zaid);
  }
  if (_mass <= 0) {
    LOG_ERROR("Nuclide has invalid mass number: ", _mass);
  }
  bool any_fissile_xs = false;
  for (auto const & xsec : _xs) {
    if (xsec.isFissile()) {
      any_fissile_xs = true;
    }
    xsec.validate();
    if (xsec.isMacro()) {
      LOG_ERROR("Nuclide has a macroscopic cross section");
    }
  }
  if (any_fissile_xs != isFissile()) {
    LOG_ERROR("Nuclide has mismatched fissile cross sections");
  }
  if (_temperatures.size() != _xs.size()) {
    LOG_ERROR("Nuclide has mismatched temperatures and cross sections");
  }
  for (auto temp : _temperatures) {
    if (temp <= 0) {
      LOG_ERROR("Nuclide has invalid temperature: ", temp);
    }
  }
  if (!um2::is_sorted(_temperatures.begin(), _temperatures.end())) {
    LOG_ERROR("Nuclide has unsorted temperatures");
  }
}

PURE [[nodiscard]] auto
Nuclide::interpXS(Float const temperature) const noexcept -> XSec
{
  // Linearly interpolate the cross sections over the sqrt of temperature
  //
  // XS = XS0 + (sqrt_t - sqrt_t0) / (sqrt_t1 - sqrt_t0) * (XS1 - XS0)
  //
  // First, make sure we have enough data to interpolate
  if (_temperatures.size() == 1) {
    return _xs[0];
  }

  // If the requested temperature is outside the range, use the closest value
  if (temperature <= _temperatures[0]) {
    return _xs[0];
  }
  if (temperature >= _temperatures.back()) {
    return _xs.back();
  }

  // Find the temperature range that contains the requested temperature
  // We know it's in the range, so we don't need to check for that
  Int i = 0;
  while (temperature >= _temperatures[i]) {
    ++i;
  }
  // Now i is the index of the upper temperature
  Int const i0 = i - 1;
  Int const i1 = i;
  Float const t0 = _temperatures[i0];
  Float const t1 = _temperatures[i1];
  Float const sqrt_t0 = um2::sqrt(t0);
  Float const sqrt_t1 = um2::sqrt(t1);
  Float const sqrt_t = um2::sqrt(temperature);
  Float const d = (sqrt_t - sqrt_t0) / (sqrt_t1 - sqrt_t0);
  XSec const & xs0 = _xs[i0];
  XSec const & xs1 = _xs[i1];
  Int const ng = xs0.numGroups();
  XSec xs(xs0.numGroups());
  ASSERT(xs1.numGroups() == ng);
  xs.isMacro() = xs0.isMacro();
  xs.isFissile() = xs0.isFissile();

  // Interpolate the cross sections
  // TODO(kcvaughn): This should be handled with arithmetic operators on the XSec
  for (Int g = 0; g < ng; ++g) {
    xs.a()[g] = xs0.a()[g] + d * (xs1.a()[g] - xs0.a()[g]);
    xs.f()[g] = xs0.f()[g] + d * (xs1.f()[g] - xs0.f()[g]);
    xs.nuf()[g] = xs0.nuf()[g] + d * (xs1.nuf()[g] - xs0.nuf()[g]);
    xs.tr()[g] = xs0.tr()[g] + d * (xs1.tr()[g] - xs0.tr()[g]);
    xs.s()[g] = xs0.s()[g] + d * (xs1.s()[g] - xs0.s()[g]);
    for (Int gg = 0; gg < ng; ++gg) {
      xs.ss()(gg, g) = xs0.ss()(gg, g) + d * (xs1.ss()(gg, g) - xs0.ss()(gg, g));
    }
  }
  return xs;
}

//==============================================================================
// Free functions
//==============================================================================

// The elements of the periodic table
// Disable clang-format
// clang-format off
Vector<String> const ELEMENTS = {
  "H",                                                                                                  "He",
  "Li", "Be",                                                             "B",  "C",  "N",  "O",  "F",  "Ne",
  "Na", "Mg",                                                             "Al", "Si", "P",  "S",  "Cl", "Ar",
  "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
  "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
  "Cs", "Ba",
  "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
              "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
  "Fr", "Ra",
  "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
              "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
};
// clang-format on

namespace
{
auto
isAlpha(char c) -> bool
{
  return std::isalpha(static_cast<unsigned char>(c)) != 0;
}

auto
isDigit(char c) -> bool
{
  return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

} // namespace

// Convert from a string, like "U235", to a ZAID.
auto
toZAID(String const & str) -> Int
{
  ASSERT(str.size() >= 2);
  ASSERT(str.size() <= 6);

  StringView const s(str);

  // Extract the chemical symbol
  ASSERT(isAlpha(s[0]));
  // Count the number of characters in the symbol
  Int n = 1;
  for (Int i = 1; i < str.size(); ++i) {
    if (isAlpha(s[i])) {
      ++n;
    } else {
      break;
    }
  }
  String const symbol = str.substr(0, n);

  // Get the atomic number from the symbol
  Int z = 0;
  bool found = false;
  for (Int i = 0; i < ELEMENTS.size(); ++i) {
    if (symbol == ELEMENTS[i]) {
      z = i + 1;
      found = true;
      break;
    }
  }
  if (!found) {
    LOG_ERROR("Invalid chemical symbol: " + symbol);
    return -1;
  }

  // Extract the mass number
  // Find the first digit
  Int m = n;
  for (Int i = n; i < str.size(); ++i) {
    if (isDigit(s[i])) {
      m = i;
      break;
    }
  }
  String const mass = str.substr(m);

  // Convert the mass number to an integer
  char * end = nullptr;
  Int const a = strto<Int>(mass.data(), &end);
  ASSERT(end != nullptr);

  return 1000 * z + a;
}

} // namespace um2
