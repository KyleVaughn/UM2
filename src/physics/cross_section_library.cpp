#include <um2/physics/cross_section_library.hpp>
#include <um2/stdlib/sto.hpp>

#include <fstream>
#include <iostream>

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
  LOG_INFO("Reading MPACT cross section library: " + filename);

  // Open file
  std::ifstream file(filename.c_str());
  if (!file.is_open()) {
    LOG_ERROR("Could not open file: " + filename);
    return;
  }

  // %VER
  // -----------------------------------------------------------
  std::string line;
  std::getline(file, line);
  if (!line.starts_with("%VER:")) {
    LOG_ERROR("Invalid MPACT library file: " + filename);
    return;
  }
  // Make sure that the first non-whitespace character is a 4 (major version)
  std::getline(file, line);
  for (auto c : line) {
    if (isSpace(c)) {
      continue;
    }
    if (c != '4') {
      LOG_ERROR("Invalid MPACT library version: " + filename);
      return;
    }
    break;
  }

  // %DIM
  // -----------------------------------------------------------
  std::getline(file, line);
  if (!line.starts_with("%DIM:")) {
    LOG_ERROR("Invalid MPACT library file: " + filename);
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
  {
    char const * num_start = line.data();
    for (I idim = 0; idim < 10; ++idim) {
      // The location of the first digit of the integer
      while (isSpace(*num_start)) {
        ++num_start;
      }
      // The location of the first space after the integer OR the end of the line
      char const * num_end = num_start;
      while (!isSpace(*num_end)) {
        ++num_end;
      }

      dims[idim] = sto<I>(std::string(num_start, num_end));
      ASSERT(dims[idim] >= 0);
      num_start = num_end;
    }
  }

  I const num_groups = dims[0];
  LOG_INFO("Number of energy groups: " + toString(num_groups));
  //  I const num_nuclides = dims[4];

  // %GRP
  //-------------------------------------------------------------
  // Get the energy bounds of the groups
  std::getline(file, line);
  if (!line.starts_with("%GRP:")) {
    LOG_ERROR("Invalid MPACT library file: " + filename);
    return;
  }
  std::getline(file, line);
  auto & group_bounds = lib.groupBounds();
  group_bounds.resize(num_groups);
  {
    char const * num_start = line.data();
    for (I ig = 0; ig < num_groups; ++ig) {
      // The location of the first digit of the number
      while (isSpace(*num_start)) {
        ++num_start;
      }
      // The location of the first space after the number OR the end of the line
      char const * num_end = num_start;
      while (!isSpace(*num_end)) {
        ++num_end;
      }

      group_bounds[ig] = sto<F>(std::string(num_start, num_end));
      ASSERT(group_bounds[ig] >= 0);
      if (ig > 0) {
        ASSERT(group_bounds[ig] < group_bounds[ig - 1]);
      }
      num_start = num_end;
    }
  }

  // %CHI
  //-------------------------------------------------------------
  // Get the fission spectrum
  std::getline(file, line);
  if (!line.starts_with("%CHI:")) {
    LOG_ERROR("Invalid MPACT library file: " + filename);
    return;
  }
  std::getline(file, line);
  auto & chi = lib.chi();
  chi.resize(num_groups);
  {
    char const * num_start = line.data();
    for (I ig = 0; ig < num_groups; ++ig) {
      // The location of the first digit of the number
      while (isSpace(*num_start)) {
        ++num_start;
      }
      // The location of the first space after the number OR the end of the line
      char const * num_end = num_start;
      while (!isSpace(*num_end)) {
        ++num_end;
      }

      chi[ig] = sto<F>(std::string(num_start, num_end));
      ASSERT(group_bounds[ig] >= 0);
      num_start = num_end;
    }
  }

  // %DIR
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
  // std::getline(file, line);
  // if (!line.starts_with("%DIR:")) {
  //   LOG_ERROR("Invalid MPACT library file: " + filename);
  //   return;
  // }
  // std::getline(file, line);
  // auto & nuclides = lib.nuclides();
  // nuclides.resize(num_nuclides);
  // {
  // char const * num_start = line.data();
  // for (I i = 0; i < num_nuclides; ++i) {
  //   while (isSpace(*num_start)) {
  //     ++num_start;
  //   }
  //   // The location of the first space after the number OR the end of the line
  //   char const * num_end = num_start;
  //   while (!isSpace(*num_end)) {
  //     ++num_end;
  //   }
  //
  //   chi[ig] = sto<F>(std::string(num_start, num_end));
  //   ASSERT(group_bounds[ig] >= 0);
  //   num_start = num_end;
  // }
  // }
}

XSLibrary::XSLibrary(String const & filename)
{
  // Assume MPACT format for now
  readMPACTLibrary(filename, *this);
}

//==============================================================================
// Methods
//==============================================================================

} // namespace um2
