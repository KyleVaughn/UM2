#include <um2/physics/cross_section_library.hpp>
#include <um2/common/cast_if_not.hpp>
#include <um2/common/logger.hpp>
#include <um2/common/strto.hpp>
#include <um2/stdlib/math/abs.hpp>

#include <fstream>

namespace um2
{

static auto
isSpace(char c) -> bool
{
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}

static void
// NOLINTNEXTLINE
readMPACTLibrary(String const & filename, XSLibrary & lib)
{
  LOG_INFO("Reading MPACT cross section library: ", filename);

  // Open file
  std::ifstream file(filename.data());
  if (!file.is_open()) {
    LOG_ERROR("Could not open file: ", filename);
    return;
  }

  uint64_t const max_line_length = 1024;
  char line[max_line_length];

  // %VER
  // -----------------------------------------------------------
  file.getline(line, max_line_length);
  StringView line_view(line);
  if (!line_view.starts_with("%VER:")) {
    LOG_ERROR("Invalid MPACT library file: ", filename);
    return;
  }
  // Make sure that the first non-whitespace character is a 4 (major version)
  file.getline(line, max_line_length);
  line_view = StringView(line);
  for (auto c : line_view) {
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
  file.getline(line, max_line_length);
  line_view = StringView(line);
  if (!line_view.starts_with("%DIM:")) {
    LOG_ERROR("Invalid MPACT library file: ", filename);
    return;
  }
  file.getline(line, max_line_length);
  line_view = StringView(line);
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

  line_view.removeLeadingSpaces();
  Vector<Int> dims(10, 0);
  char * end = nullptr;
  for (Int idim = 0; idim < 10; ++idim) {
    StringView const token = line_view.getTokenAndShrink();
    dims[idim] = strto<Int>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    ASSERT(dims[idim] >= 0);
  }

  Int const num_groups = dims[0];
  LOG_INFO("Number of energy groups: ", num_groups);
  Int const num_nuclides = dims[4];

  // %GRP
  //-------------------------------------------------------------
  // Get the energy bounds of the groups
  file.getline(line, max_line_length);
  line_view = StringView(line);
  if (!line_view.starts_with("%GRP:")) {
    LOG_ERROR("Invalid MPACT library file: ", filename);
    return;
  }
  file.getline(line, max_line_length);
  line_view = StringView(line);
  line_view.removeLeadingSpaces();
  auto & group_bounds = lib.groupBounds();
  group_bounds.resize(num_groups);
  for (Int ig = 0; ig < num_groups; ++ig) {
    StringView token = line_view.getTokenAndShrink();

    // If the token is empty, we have reached the end of the line
    if (token.empty()) {
      bool const one_more_token = !line_view.empty();
      if (one_more_token) {
        token = line_view;
      }
      // If this is the last group, we don't need to read another line
      file.getline(line, max_line_length);
      line_view = StringView(line);
      line_view.removeLeadingSpaces();
      if (!one_more_token) {
        token = line_view.getTokenAndShrink();
      }
    }

    group_bounds[ig] = strto<Float>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    ASSERT(group_bounds[ig] >= 0);
    if (ig > 0) {
      ASSERT(group_bounds[ig] < group_bounds[ig - 1]);
    }
  }

  // %CHI
  //-------------------------------------------------------------
  // Get the fission spectrum
  file.getline(line, max_line_length);
  line_view = StringView(line);
  if (!line_view.starts_with("%CHI:")) {
    LOG_ERROR("Invalid MPACT library file: ", filename);
    return;
  }
  file.getline(line, max_line_length);
  line_view = StringView(line);
  auto & chi = lib.chi();
  chi.resize(num_groups);
  for (Int ig = 0; ig < num_groups; ++ig) {
    StringView token = line_view.getTokenAndShrink();

    // If the token is empty, we have reached the end of the line
    if (token.empty()) {
      bool const one_more_token = !line_view.empty();
      if (one_more_token) {
        token = line_view;
      }
      // If this is the last group, we don't need to read another line
      file.getline(line, max_line_length);
      line_view = StringView(line);
      line_view.removeLeadingSpaces();
      if (!one_more_token) {
        token = line_view.getTokenAndShrink();
      }
    }

    chi[ig] = strto<Float>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    ASSERT(chi[ig] >= 0);
  }

  // %DIR
  //--------------------------------------------------------------------
  file.getline(line, max_line_length);
  line_view = StringView(line);
  if (!line_view.starts_with("%DIR:")) {
    LOG_ERROR("Invalid MPACT library file: ", filename);
    return;
  }

  // We don't read these values since they're redundant
  // Skip to the first %NUC line
  int ctr = 0;
  while (file.getline(line, max_line_length)) {
    line_view = StringView(line);
    if (line_view.starts_with("%NUC")) {
      break;
    }
    ++ctr;
    if (ctr > 10000) {
      LOG_ERROR("Invalid MPACT library file: ", filename);
      return;
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
  Int nuclide_ctr = 0;
  auto & nuclides = lib.nuclides();
  nuclides.resize(num_nuclides);
  while (!line_view.starts_with("%END")) {
    // Skip the %NUC token
    StringView token = line_view.getTokenAndShrink();

    // Get the index of the nuclide
    token = line_view.getTokenAndShrink();
    Int const index = strto<Int>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    ASSERT(index == nuclide_ctr + 1);
    auto & nuclide = nuclides[nuclide_ctr];

    // Get the ZAID of the nuclide
    token = line_view.getTokenAndShrink();
    Int const zaid = strto<Int>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    ASSERT(zaid > 0);
    nuclide.zaid() = zaid;

    // Get the atomic mass of the nuclide
    token = line_view.getTokenAndShrink();
    Float const atomic_mass = strto<Float>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    ASSERT(atomic_mass > 0);
    nuclide.mass() = atomic_mass;

    // Skip the next 4 numbers
    for (Int i = 0; i < 4; ++i) {
      token = line_view.getTokenAndShrink();
    }

    // Get the number of temperatures
    token = line_view.getTokenAndShrink();
    Int const num_temps = strto<Int>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    ASSERT(num_temps > 0);
    nuclide.temperatures().resize(num_temps);
    nuclide.xs().resize(num_temps);
    for (Int itemp = 0; itemp < num_temps; ++itemp) {
      nuclide.xs(itemp).t().resize(num_groups);
    }

    // Read the temperature data
    file.getline(line, max_line_length);
    line_view = StringView(line);
    line_view.removeLeadingSpaces();
    ASSERT(line_view.starts_with("TP1+"));
    file.getline(line, max_line_length);
    line_view = StringView(line);
    line_view.removeLeadingSpaces();
    for (Int itemp = 0; itemp < num_temps; ++itemp) {
      token = line_view.getTokenAndShrink();
      if (token.empty()) {
        token = line_view;
      }
      Float const temp = strto<Float>(token.data(), &end);
      ASSERT(end != nullptr);
      end = nullptr;
      ASSERT(temp > 0);
      nuclide.temperatures()[itemp] = temp;
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
    file.getline(line, max_line_length);
    line_view = StringView(line);
    line_view.removeLeadingSpaces();
    ASSERT(line_view.starts_with("XSD+"));
    for (Int ig = 0; ig < num_groups; ++ig) {
      for (Int itemp = 0; itemp < num_temps; ++itemp) {
        file.getline(line, max_line_length);
        line_view = StringView(line);
        line_view.removeLeadingSpaces();

        // Group index
        token = line_view.getTokenAndShrink();
        Int const group_index = strto<Int>(token.data(), &end);
        ASSERT(end != nullptr);
        end = nullptr;
        ASSERT(group_index == ig + 1);

        // Temperature index
        token = line_view.getTokenAndShrink();
        Int const temp_index = strto<Int>(token.data(), &end);
        ASSERT(end != nullptr);
        end = nullptr;
        ASSERT(temp_index == itemp + 1);

        // Absorption
        token = line_view.getTokenAndShrink();
        Float const absorption = strto<Float>(token.data(), &end);
        ASSERT(end != nullptr);
        end = nullptr;
        if (absorption < 0) {
          LOG_WARN("Nuclide with ZAID ", zaid,
                    " has negative absorption cross section at group ", ig,
                    " and temperature ", itemp);
        }

        // If this token is empty, only absorption is given
        token = line_view.getTokenAndShrink();
        bool const absorption_only = token.empty();
        if (absorption_only) {
          nuclide.xs(itemp).t(ig) = absorption;
        } else {
          // Fission
          Float const fission = strto<Float>(token.data(), &end);
          ASSERT(end != nullptr);
          end = nullptr;
          if (um2::abs(fission) > castIfNot<Float>(1e-10)) {
            nuclide.isFissile() = true;
          }

          // Skip nu-fission and transport
          for (Int i = 0; i < 2; ++i) {
            token = line_view.getTokenAndShrink();
          }

          // Total scattering
          token = line_view.getTokenAndShrink();
          Float const total_scatter = strto<Float>(token.data(), &end);
          ASSERT(end != nullptr);
          end = nullptr;
          if (total_scatter < 0) {
            LOG_WARN("Nuclide with ZAID ", zaid,
                      " has negative P0 scattering cross section at group ", ig,
                      " and temperature ", itemp);
          }

          Float const total = absorption + total_scatter;
          if (total < 0) {
            LOG_WARN("Nuclide with ZAID ", zaid,
                      " has negative total cross section at group ", ig,
                      " and temperature ", itemp);
          }

          nuclide.xs(itemp).t(ig) = total;
        } // if (absorption_only)
      }   // for (Int itemp = 0; itemp < num_temps; ++itemp)
    }     // for (Int ig = 0; ig < num_groups; ++ig)

    nuclide.validate();

    // Skip the other sections until we find the next nuclide or the end of the file
    while (file.getline(line, max_line_length)) {
      line_view = StringView(line);
      line_view.removeLeadingSpaces();
      if (line_view.starts_with("%NUC") || line_view.starts_with("%END")) {
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
XSLibrary::getNuclide(Int zaid) const noexcept -> Nuclide const &
{
  for (auto const & nuclide : _nuclides) {
    if (nuclide.zaid() == zaid) {
      return nuclide;
    }
  }
  LOG_ERROR("Nuclide with ZAID ", zaid, " not found in library");
  return _nuclides[0];
}

//PURE [[nodiscard]] auto
//XSLibrary::getXS(Material const & material) const noexcept -> XSec
//{
//  material.validate();
//  XSec xs;
//  xs.isMacro() = true;
//  Int const num_groups = numGroups();
//  xs.t().resize(num_groups);
//  // For each nuclide in the material:
//  //  find the corresponding nuclide in the library
//  //  interpolate the cross sections to the temperature of the material
//  //  scale the cross sections by the atom density of the nuclide
//  //  reduce
//  Int const num_nuclides = material.numNuclides();
//  for (Int inuc = 0; inuc < num_nuclides; ++inuc) {
//    auto const zaid = material.zaid(inuc);
//    auto const & lib_nuc = getNuclide(zaid);
//    auto const xs_nuc = lib_nuc.interpXS(material.getTemperature());
//    auto const atom_density = material.numDensity(inuc);
//    for (Int ig = 0; ig < num_groups; ++ig) {
//      xs.t(ig) += xs_nuc.t(ig) * atom_density;
//    }
//  }
//  xs.validate();
//  return xs;
//}
} // namespace um2
