#include <um2/physics/cross_section_library.hpp>
#include <um2/stdlib/sto.hpp>

#include <fstream>

namespace um2
{

static auto
isSpace(char c) -> bool
{
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}

static void
getNextToken(char const *& token_start, char const *& token_end)
{
  while (isSpace(*token_start) && *token_start != '\0') {
    ++token_start;
  }
  token_end = token_start;
  while (!isSpace(*token_end) && *token_end != '\0') {
    ++token_end;
  }
}

static auto
getNumTokens(char const * start) -> I
{
  I num_tokens = 0;
  char const * token_start = start;
  char const * token_end = token_start;
  while (*token_start != '\0') {
    getNextToken(token_start, token_end);
    if (token_start == token_end) {
      break;
    }
    ++num_tokens;
    token_start = token_end;
  }
  return num_tokens;
}

static void
// NOLINTNEXTLINE
readMPACTLibrary(String const & filename, XSLibrary & lib)
{
  LOG_INFO("Reading MPACT cross section library: ", filename);

  // Open file
  std::ifstream file(filename.c_str());
  if (!file.is_open()) {
    LOG_ERROR("Could not open file: ", filename);
    return;
  }

  // %VER
  // -----------------------------------------------------------
  std::string line;
  std::getline(file, line);
  if (!line.starts_with("%VER:")) {
    LOG_ERROR("Invalid MPACT library file: ", filename);
    return;
  }
  // Make sure that the first non-whitespace character is a 4 (major version)
  std::getline(file, line);
  for (auto c : line) {
    if (isSpace(c)) {
      continue;
    }
    if (c != '4') {
      LOG_ERROR("Invalid MPACT library version: ", filename);
      return;
    }
    break;
  }

  // %DIM
  // -----------------------------------------------------------
  std::getline(file, line);
  if (!line.starts_with("%DIM:")) {
    LOG_ERROR("Invalid MPACT library file: ", filename);
    return;
  }
  std::getline(file, line);
  // Parse 10 integers:
  //  num_groups: The number of energy groups
  //  num_fast_groups: The number of fast groups
  //  num_res_groups: The number of resonance groups
  //  num_chi_groups: The number of fission spectrum groups
  //  num_nuclides: The number of nuclides in the library
  //  ???
  //  num_res_nuclides: The number of resonance nuclides
  //  num_fiss_prod: The number of fission products
  //  num_subgroup_lvls: The number of subgroup levels
  //  num_flux_lvls: The number of subgroup flux levels

  Vector<I> dims(10, 0);
  char const * token_start = line.data();
  char const * token_end = token_start;
  for (I idim = 0; idim < 10; ++idim) {
    getNextToken(token_start, token_end);
    dims[idim] = sto<I>(std::string(token_start, token_end));
    ASSERT(dims[idim] >= 0);
    token_start = token_end;
  }

  I const num_groups = dims[0];
  LOG_INFO("Number of energy groups: ", num_groups); 
  I const num_nuclides = dims[4];

  // %GRP
  //-------------------------------------------------------------
  // Get the energy bounds of the groups
  std::getline(file, line);
  if (!line.starts_with("%GRP:")) {
    LOG_ERROR("Invalid MPACT library file: ", filename);
    return;
  }
  std::getline(file, line);
  auto & group_bounds = lib.groupBounds();
  group_bounds.resize(num_groups);
  token_start = line.data();
  for (I ig = 0; ig < num_groups; ++ig) {
    getNextToken(token_start, token_end);
    group_bounds[ig] = sto<F>(std::string(token_start, token_end));
    ASSERT(group_bounds[ig] >= 0);
    if (ig > 0) {
      ASSERT(group_bounds[ig] < group_bounds[ig - 1]);
    }
    token_start = token_end;
    // If end of line before all groups are read, then go to the next line
    if (*token_start == '\0' && ig < num_groups - 1) {
      std::getline(file, line);
      token_start = line.data();
    }
  }

  // %CHI
  //-------------------------------------------------------------
  // Get the fission spectrum
  std::getline(file, line);
  if (!line.starts_with("%CHI:")) {
    LOG_ERROR("Invalid MPACT library file: ", filename);
    return;
  }
  std::getline(file, line);
  auto & chi = lib.chi();
  chi.resize(num_groups);
  token_start = line.data();
  for (I ig = 0; ig < num_groups; ++ig) {
    getNextToken(token_start, token_end);
    chi[ig] = sto<F>(std::string(token_start, token_end));
    ASSERT(group_bounds[ig] >= 0);
    token_start = token_end;
    // If end of line before all groups are read, then go to the next line
    if (*token_start == '\0' && ig < num_groups - 1) {
      std::getline(file, line);
      token_start = line.data();
    }
  }

  // %DIR
  //--------------------------------------------------------------------
  std::getline(file, line);
  if (!line.starts_with("%DIR:")) {
    LOG_ERROR("Invalid MPACT library file: ", filename);
    return;
  }

  // We don't read these values since they're redundant
  // Skip to the first %NUC line
  int ctr = 0;
  while (std::getline(file, line)) {
    if (line.starts_with("%NUC")) {
      break;
    }
    ++ctr;
    if (ctr > 10000) {
      LOG_ERROR("Invalid MPACT library file: ", filename);
    }
  }

  // %NUC
  //--------------------------------------------------------------------
  // Get nuclide metadata
  // 0. index
  // 1. ZAID
  // 2. atomic mass
  // 3. isotope type (not used)
  // 4. ??
  // 5. ??
  // 6. ??
  // 7. num temp
  // 8+. don't use
  I nuclide_ctr = 0;
  auto & nuclides = lib.nuclides();
  nuclides.resize(num_nuclides);
  while (!line.starts_with("%END")) {
    token_start = line.data() + 5; // Skip the %NUC
    getNextToken(token_start, token_end);

    // Get the index of the nuclide
    I const index = sto<I>(std::string(token_start, token_end));
    ASSERT(index == nuclide_ctr + 1);
    auto & nuclide = nuclides[nuclide_ctr];

    // Get the ZAID of the nuclide
    token_start = token_end;
    getNextToken(token_start, token_end);
    I const zaid = sto<I>(std::string(token_start, token_end));
    ASSERT(zaid > 0);
    nuclide.zaid() = zaid;

    // Get the atomic mass of the nuclide
    token_start = token_end;
    getNextToken(token_start, token_end);
    F const atomic_mass = sto<F>(std::string(token_start, token_end));
    ASSERT(atomic_mass > 0);
    nuclide.mass() = atomic_mass;

    // Skip the next 4 numbers
    for (I i = 0; i < 5; ++i) {
      token_start = token_end;
      getNextToken(token_start, token_end);
    }

    // Get the number of temperatures
    I const num_temps = sto<I>(std::string(token_start, token_end));
    ASSERT(num_temps > 0);
    nuclide.temperatures().resize(num_temps);
    nuclide.xs().resize(num_temps);
    for (I itemp = 0; itemp < num_temps; ++itemp) {
      nuclide.xs(itemp).t().resize(num_groups);
    }

    // Read the temperature data
    std::getline(file, line);
    token_start = line.data();
    getNextToken(token_start, token_end);
    ASSERT(String(token_start, token_end) == "TP1+");
    std::getline(file, line);
    token_start = line.data();
    for (I itemp = 0; itemp < num_temps; ++itemp) {
      getNextToken(token_start, token_end);
      F const temp = sto<F>(std::string(token_start, token_end));
      ASSERT(temp > 0);
      nuclide.temperatures()[itemp] = temp;
      token_start = token_end;
    }

    // Read the cross section data
    // 0. group index
    // 1. temperature index
    // 2. absorption
    // 3. fission
    // 4. nu-fission
    // 5. transport
    // 6. total scattering
    // 7+. scatter data we don't use
    std::getline(file, line);
    token_start = line.data();
    getNextToken(token_start, token_end);
    ASSERT(String(token_start, token_end) == "XSD+");
    for (I ig = 0; ig < num_groups; ++ig) {
      for (I itemp = 0; itemp < num_temps; ++itemp) {
        std::getline(file, line);
        // Get the number of tokens on the line
        // If num tokens == 3, then only absorption is given
        I const num_tokens = getNumTokens(line.data());

        // Group index
        token_start = line.data();
        getNextToken(token_start, token_end);
        I const group_index = sto<I>(std::string(token_start, token_end));
        ASSERT(group_index == ig + 1);

        // Temperature index
        token_start = token_end;
        getNextToken(token_start, token_end);
        I const temp_index = sto<I>(std::string(token_start, token_end));
        ASSERT(temp_index == itemp + 1);

        // Absorption
        token_start = token_end;
        getNextToken(token_start, token_end);
        F const absorption = sto<F>(std::string(token_start, token_end));
        if (absorption < 0) {
          LOG_WARN("Nuclide with ZAID ", zaid,
                    " has negative absorption cross section at group ", ig,
                    " and temperature ", itemp);
        }

        if (num_tokens == 3) {
          nuclide.xs(itemp).t(ig) = absorption;
        } else {
          // Fission
          token_start = token_end;
          getNextToken(token_start, token_end);
          F const fission = sto<F>(std::string(token_start, token_end));
          if (um2::abs(fission) > condCast<F>(1e-10)) {
            nuclide.isFissile() = true;
          }

          // Skip nu-fission and transport
          for (I i = 0; i < 3; ++i) {
            token_start = token_end;
            getNextToken(token_start, token_end);
          }

          // Total scattering
          F const total_scatter = sto<F>(std::string(token_start, token_end));
          if (total_scatter < 0) {
            LOG_WARN("Nuclide with ZAID ", zaid,
                      " has negative P0 scattering cross section at group ", ig,
                      " and temperature ", itemp);
          }

          F const total = absorption + total_scatter;
          if (total < 0) {
            LOG_WARN("Nuclide with ZAID ", zaid,
                      " has negative total cross section at group ", ig,
                      " and temperature ", itemp);
          }

          nuclide.xs(itemp).t(ig) = total;
        } // if (num_tokens == 3)
      }   // for (I itemp = 0; itemp < num_temps; ++itemp)
    }     // for (I ig = 0; ig < num_groups; ++ig)

    nuclide.validate();

    // Skip the other sections until we find the next nuclide or the end of the file
    while (std::getline(file, line)) {
      if (line.starts_with("%NUC") || line.starts_with("%END")) {
        break;
      }
    }

    ++nuclide_ctr;
  } // while (!line.starts_with("%END"))

  // close the file
  file.close();
} // readMPACTLibrary

XSLibrary::XSLibrary(String const & filename)
{
  // Assume MPACT format for now
  readMPACTLibrary(filename, *this);
}

PURE [[nodiscard]] auto
XSLibrary::getNuclide(I zaid) const noexcept -> Nuclide const &
{
  for (auto const & nuclide : _nuclides) {
    if (nuclide.zaid() == zaid) {
      return nuclide;
    }
  }
  LOG_ERROR("Nuclide with ZAID ", zaid, " not found in library");
  return _nuclides[0];
}

PURE [[nodiscard]] auto
XSLibrary::getXS(Material const & material) const noexcept -> XSec
{
  material.validate();
  XSec xs;
  xs.isMacro() = true;
  I const num_groups = numGroups();
  xs.t().resize(num_groups);
  // For each nuclide in the material:
  //  find the corresponding nuclide in the library
  //  interpolate the cross sections to the temperature of the material
  //  scale the cross sections by the atom density of the nuclide
  //  reduce
  I const num_nuclides = material.numNuclides();
  for (I inuc = 0; inuc < num_nuclides; ++inuc) {
    auto const zaid = material.zaid(inuc);
    auto const & lib_nuc = getNuclide(zaid);
    auto const xs_nuc = lib_nuc.interpXS(material.getTemperature());
    auto const atom_density = material.numDensity(inuc);
    for (I ig = 0; ig < num_groups; ++ig) {
      xs.t(ig) += xs_nuc.t(ig) * atom_density;
    }
  }
  return xs;
}
} // namespace um2
