#include <um2.hpp>

// NOLINTBEGIN(misc-include-cleaner)

auto
main(int argc, char ** argv) -> int
{
  um2::initialize();

  // Get the file name from the command line
  if (argc != 2) {
    um2::String const exec_name(argv[0]);
    LOG_ERROR("Usage: ", exec_name, " <filename>");
    return 1;
  }

  um2::String const filename(argv[1]);
  auto const power_centroid = um2::mpact::getPowers(filename);
  FILE * file = fopen("power_centroid.txt", "w");
  if (file == nullptr) {
    LOG_ERROR("Could not open file power_centroid.txt");
    return 1;
  }

  Float power_sum = 0;
  int ret = fprintf(file, "Power, x, y\n");
  if (ret < 0) {
    LOG_ERROR("Could not write to file power_centroid.txt");
    return 1;
  }
  for (auto const pc : power_centroid) {
    power_sum += pc.first;
    ret = fprintf(file, "%.16f, %.16f, %.16f\n", pc.first, pc.second[0], pc.second[1]);
    if (ret < 0) {
      LOG_ERROR("Could not write to file power_centroid.txt");
      return 1;
    }
  }
  ret = fclose(file);
  if (ret != 0) {
    LOG_ERROR("Could not close file power_centroid.txt");
    return 1;
  }
  // If the sum of the powers is not 1, warn the user
  if (um2::abs(power_sum - 1) > 1e-3) {
    LOG_ERROR("The sum of the powers is not 1, it is ", power_sum);
  }
  return 0;
}

// NOLINTEND(misc-include-cleaner)
