#include <um2/physics/nuclide.hpp>
#include <um2/stdlib/sto.hpp>

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
  ASSERT(_zaid > 0);
  ASSERT(_mass > 0);
  for (auto const & xsec : _xs) {
    xsec.validate();
    ASSERT(!xsec.isMacro())
  }
  ASSERT(_temperatures.size() == _xs.size())
  for (auto temp : _temperatures) {
    ASSERT(temp > 0)
  }
  ASSERT(um2::is_sorted(_temperatures.begin(), _temperatures.end()))
}

PURE [[nodiscard]] auto
Nuclide::interpXS(F const temperature) const noexcept -> XSec
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
  I i = 0;
  while (temperature >= _temperatures[i]) {
    ++i;
  }
  // Now i is the index of the upper temperature
  I const i0 = i - 1;
  I const i1 = i;
  F const t0 = _temperatures[i0];
  F const t1 = _temperatures[i1];
  F const sqrt_t0 = um2::sqrt(t0);
  F const sqrt_t1 = um2::sqrt(t1);
  F const sqrt_t = um2::sqrt(temperature);
  F const d = (sqrt_t - sqrt_t0) / (sqrt_t1 - sqrt_t0);
  XSec const & xs0 = _xs[i0];
  XSec const & xs1 = _xs[i1];
  XSec xs;
  I const n = xs0.t().size();
  ASSERT(n == xs1.t().size());
  xs.t().resize(n);
  for (I j = 0; j < n; ++j) {
    xs.t(j) = xs0.t(j) + d * (xs1.t(j) - xs0.t(j));
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

static auto
isAlpha(char c) -> bool
{
  return std::isalpha(static_cast<unsigned char>(c)) != 0;
}

static auto
isDigit(char c) -> bool
{
  return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

// Convert from a string, like "U235", to a ZAID.
auto
toZAID(String str) -> I
{
  ASSERT(str.size() >= 2);
  ASSERT(str.size() <= 6);

  // Extract the chemical symbol
  String symbol;
  for (I i = 0; i < str.size(); ++i) {
    if (isAlpha(str[i])) {
      symbol += str[i];
    }
  }
  ASSERT(symbol.size() > 0);

  // Get the atomic number from the symbol
  I z = 0;
  bool found = false;
  for (I i = 0; i < ELEMENTS.size(); ++i) {
    if (symbol == ELEMENTS[i]) {
      z = i + 1;
      found = true;
      break;
    }
  }
  if (!found) {
    LOG_ERROR("Invalid chemical symbol: " + symbol);
  }

  // Extract the mass number
  String mass;
  for (I i = 0; i < str.size(); ++i) {
    if (isDigit(str[i])) {
      mass += str[i];
    }
  }
  ASSERT(mass.size() > 0);

  // Convert the mass number to an integer
  I const a = sto<I>(std::string(mass.c_str()));
  ASSERT(a > 0);

  return 1000 * z + a;
}

} // namespace um2
