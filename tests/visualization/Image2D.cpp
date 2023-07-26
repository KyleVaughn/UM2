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
  int const stat = std::remove("test.ppm");
  ASSERT(stat == 0);
}

template <typename T>
TEST_CASE(rasterizePoint)
{
  um2::Image2D<T> image;
  image.minima[0] = static_cast<T>(0);
  image.minima[1] = static_cast<T>(0);
  image.spacing[0] = static_cast<T>(1);
  image.spacing[1] = static_cast<T>(1);
  image.num_cells[0] = 100;
  image.num_cells[1] = 100;
  image.children.resize(100 * 100);

  T r = 10;
  image.rasterize(um2::Point2<T>(0, 0), r, um2::Color("red"));
  r = 5;
  image.rasterize(um2::Point2<T>(99, 0), r, um2::Color("green"));
  r = 20;
  image.rasterize(um2::Point2<T>(0, 99), r, um2::Color("blue"));
  r = 30;
  image.rasterize(um2::Point2<T>(99, 99), r, um2::Color("white"));
  r = 2;
  image.rasterize(um2::Point2<T>(50, 50), r, um2::Color("yellow"));
  image.write("test.ppm");
  {
    std::ifstream file("test.ppm");
    ASSERT(file.is_open());
  }
  int const stat = std::remove("test.ppm");
  ASSERT(stat == 0);
}

template <typename T>
TEST_SUITE(Image2D)
{
  TEST((writePPM<T>));
  TEST((rasterizePoint<T>));
}

auto
main() -> int
{
  RUN_SUITE((Image2D<float>));
  RUN_SUITE((Image2D<double>));
  return 0;
}
