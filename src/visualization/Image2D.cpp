#include <um2/visualization/Image2D.hpp>

#include <bit>     // std::bit_cast
#include <fstream> // std::ofstream

#include <png.h>

namespace um2
{

void
writePPM(Vector<Color> const & buffer, Size nx, Size ny, std::string const & filename)
{
  LOG_INFO("Writing image to file: " + filename);
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
  ofs.close();
}

// We don't care about the return value of fclose since we the program will exit anyway
// NOLINTBEGIN(cert-err33-c) justified above
void
writePNG(Vector<Color> buffer, Size nx, Size ny, std::string const & filename)
{
  LOG_INFO("Writing image to file: " + filename);
  FILE * fp = fopen(filename.c_str(), "wb");
  if (fp == nullptr) {
    LOG_ERROR("Could not open file for writing");
    return;
  }

  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (png == nullptr) {
    fclose(fp);
    LOG_ERROR("Could not create PNG write struct");
    return;
  }

  png_infop info = png_create_info_struct(png);
  if (info == nullptr) {
    fclose(fp);
    png_destroy_write_struct(&png, nullptr);
    LOG_ERROR("Could not create PNG info struct");
    return;
  }

  // NOLINTNEXTLINE(cert-err52-cpp) justification: Use canonical libpng error handling
  if (setjmp(png_jmpbuf(png))) {
    fclose(fp);
    png_destroy_write_struct(&png, &info);
    LOG_ERROR("Error during PNG creation");
    return;
  }

  png_init_io(png, fp);

  // Output is 8bit depth, RGBA format.
  png_set_IHDR(png, info, static_cast<png_uint_32>(nx), static_cast<png_uint_32>(ny), 8,
               PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png, info);

  // We already have the buffer in RGBA format, but the buffer is left-to-right,
  // bottom-to-top and the PNG format is top-to-bottom so we have to write the lines in
  // reverse order

  // Point to the last line of the buffer
  ptrdiff_t const stride =
      static_cast<ptrdiff_t>(nx) * static_cast<ptrdiff_t>(sizeof(Color));
  auto * data_ptr = reinterpret_cast<png_bytep>(buffer.end()) - stride;
  for (Size y = 0; y < ny; ++y) {
    png_write_row(png, data_ptr);
    data_ptr -= stride;
  }

  png_write_end(png, info);

  png_destroy_write_struct(&png, &info);

  fclose(fp);
}
// NOLINTEND(cert-err33-c)

} // namespace um2
