#include <um2/visualization/Image2D.hpp>

#include <bit>     // std::bit_cast
#include <fstream> // std::ofstream

namespace um2
{

void
writePPM(Vector<Color> const & buffer, Size nx, Size ny, String const & filename)
{
  std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
  assert(ofs.is_open() && "Could not open file for writing");
  ofs << "P6\n" << nx << " " << ny << "\n255\n";
  // The buffer is left-to-right, bottom-to-top but the PPM format is top-to-bottom
  // so we have to write the lines in reverse order
  for (Size y = ny - 1; y >= 0; --y) {
    for (Size x = 0; x < nx; ++x) {
      auto const & pixel = buffer[y * nx + x];
      ofs << std::bit_cast<char>(pixel.r()) << std::bit_cast<char>(pixel.g())
          << std::bit_cast<char>(pixel.b());
    }
  }
}

} // namespace um2
