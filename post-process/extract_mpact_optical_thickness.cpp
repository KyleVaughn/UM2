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
  um2::mpact::Model const model(argv[1]);
  model.writeOpticalThickness("coarse_grid.xdmf");


  // Read the Polytope soup file
  um2::PolytopeSoup const soup("coarse_grid.xdmf");

  // There should be elsets of the form "Group_XXXXX_Optical_Thickness"
  // and "Mean_Optical_Thickness"

  um2::Vector<um2::Vector<Float>> group_optical_thickness;
  um2::Vector<Float> mean_optical_thickness;
  um2::Vector<Int> ids;
  um2::Vector<Float> tmp;
  bool has_elset = true;
  Int i = 0;

  while (has_elset) {
    has_elset = false;
    ids.clear();
    tmp.clear();
    um2::String const elset_name = "Group_" + um2::mpact::getASCIINumber(i) + "_Optical_Thickness";
    soup.getElset(elset_name, ids, tmp);
    if (!ids.empty()) {
      group_optical_thickness.push_back(tmp);
      has_elset = true;
    }
    ++i;
  }

  // Get the mean optical thickness
  soup.getElset("Mean_Optical_Thickness", ids, tmp);
  mean_optical_thickness = tmp;

  um2::String const out_filename("optical_thickness.txt");
  FILE * file = fopen(out_filename.data(), "w");
  if (file == nullptr) {
    std::cerr << "Could not open file " << out_filename.data() << std::endl;
    return 1;
  }

  // Write the number of groups (i) and the number of optical thicknesses
  {
    int const ret = fprintf(file, "%d %d\n", group_optical_thickness.size(), group_optical_thickness[0].size());
    if (ret < 0) {    
      LOG_ERROR("Failed to write to file: ", out_filename);    
      return 1;    
    }
  }
  
  // Write the optical thickness of each group
  for (auto const & taus : group_optical_thickness) {
    for (auto const tau : taus) {
      int const ret = fprintf(file, "%.16f\n", tau);
      if (ret < 0) {
        LOG_ERROR("Failed to write to file: ", out_filename);
        return 1;
      }
    }
  }
  // Write the mean optical thickness
  for (auto const tau : mean_optical_thickness) {
    int const ret = fprintf(file, "%.16f\n", tau);
    if (ret < 0) {
      LOG_ERROR("Failed to write to file: ", out_filename);
      return 1;
    }
  }

  int const ret = fclose(file);
  if (ret != 0) {
    LOG_ERROR("Failed to close file: ", out_filename);
    return 1;
  }

  um2::finalize();
  return 0;
}
