#include <um2.hpp>

#include <iostream>

auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
main(int argc, char **argv) -> int 
{
  um2::initialize();

  // Get the file name from the command line
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  // Read the MPACT file
  um2::PolytopeSoup const soup(argv[1]);

  // There should be elsets of the form: 
  // "Homogenized_Total_XS_Group_XXXXX"
  // "Homogenized_Scattering_Ratio_Group_XXXXX"
  // "1D_odCMFD_Spectral_Radius_Group_XXXXX" 

  um2::Vector<um2::Vector<Float>> group_total_xs;
  um2::Vector<um2::Vector<Float>> group_scattering_ratio;
  um2::Vector<um2::Vector<Float>> group_spectral_radius;
  um2::Vector<Int> ids;
  um2::Vector<Float> tmp;
  bool has_elset = true;
  Int i = 0;

  while (has_elset) {
    has_elset = false;
    ids.clear();
    tmp.clear();
    um2::String const elset_name = "Homogenized_Total_XS_Group_" + um2::mpact::getASCIINumber(i); 
    soup.getElset(elset_name, ids, tmp);
    if (!ids.empty()) {
      group_total_xs.push_back(tmp);
      has_elset = true;
    }
    um2::String const elset_name2 = "Homogenized_Scattering_Ratio_Group_" + um2::mpact::getASCIINumber(i);
    soup.getElset(elset_name2, ids, tmp);
    if (!ids.empty()) {
      group_scattering_ratio.push_back(tmp);
      has_elset = true;
    }
    um2::String const elset_name3 = "1D_odCMFD_Spectral_Radius_Group_" + um2::mpact::getASCIINumber(i);
    soup.getElset(elset_name3, ids, tmp);
    if (!ids.empty()) {
      group_spectral_radius.push_back(tmp);
      has_elset = true;
    }
    ++i;
  }

  um2::String const total_xs_filename("total_xs.txt"); 
  FILE * file = fopen(total_xs_filename.data(), "w");
  if (file == nullptr) {
    std::cerr << "Could not open file " << total_xs_filename.data() << std::endl;
    return 1;
  }

  // Write the total xs for each group to a file 
  for (auto const & group_xs : group_total_xs) {
    for (auto const xs : group_xs) { 
      int const ret = fprintf(file, "%.16f\n", xs); 
      if (ret < 0) {
        LOG_ERROR("Failed to write to file: ", total_xs_filename); 
        return 1;
      }
    }
  }

  int fret = fclose(file);
  if (fret != 0) {
    LOG_ERROR("Failed to close file: ", total_xs_filename);
    return 1;
  }

  um2::String const scattering_ratio_filename("scattering_ratio.txt");
  file = fopen(scattering_ratio_filename.data(), "w");
  if (file == nullptr) {
    std::cerr << "Could not open file " << scattering_ratio_filename.data() << std::endl;
    return 1;
  }

  // Write the scattering ratio for each group to a file
  for (auto const & group_ratio : group_scattering_ratio) {
    for (auto const ratio : group_ratio) {
      int const ret = fprintf(file, "%.16f\n", ratio);
      if (ret < 0) {
        LOG_ERROR("Failed to write to file: ", scattering_ratio_filename);
        return 1;
      }
    }
  }

  fret = fclose(file);
  if (fret != 0) {
    LOG_ERROR("Failed to close file: ", scattering_ratio_filename);
    return 1;
  }

  um2::String const spectral_radius_filename("spectral_radius.txt");
  file = fopen(spectral_radius_filename.data(), "w");
  if (file == nullptr) {
    std::cerr << "Could not open file " << spectral_radius_filename.data() << std::endl;
    return 1;
  }
  
  // Write the spectral radius for each group to a file
  for (auto const & group_radius : group_spectral_radius) {
    for (auto const radius : group_radius) {
      int const ret = fprintf(file, "%.16f\n", radius);
      if (ret < 0) {
        LOG_ERROR("Failed to write to file: ", spectral_radius_filename);
        return 1;
      }
    }
  }

  fret = fclose(file);
  if (fret != 0) {
    LOG_ERROR("Failed to close file: ", spectral_radius_filename);
    return 1;
  }

  um2::finalize();
  return 0;
}
