#include <um2/visualization/Image2D.hpp>

#include "../test_macros.hpp"
#include <fstream>

template <typename T>
TEST_CASE(writePPM)
{
  um2::Image2D<T> image;
  image.num_cells[0] = 10;
  image.num_cells[1] = 10;
  image.children.resize(100);
  image.getChild(0, 0) = um2::Color("red");
  image.getChild(9, 0) = um2::Color("green");
  image.getChild(0, 9) = um2::Color("blue");
  image.getChild(9, 9) = um2::Color("white");
  image.write("test.ppm");
  {
    std::ifstream file("test.ppm");
    ASSERT(file.is_open());
  }
  std::remove("test.ppm");
}

template <typename T>
TEST_SUITE(Image2D)
{
  TEST((writePPM<T>));
}

auto
main() -> int
{
  RUN_SUITE((Image2D<float>));
  RUN_SUITE((Image2D<double>));
  return 0;
}
