#include <um2/visualization/Image2D.hpp>

#include "../test_macros.hpp"
#include <fstream>

template <typename T>
TEST_CASE(writePPM)
{
  um2::Image2D<T> image;
  image.grid.num_cells[0] = 10;
  image.grid.num_cells[1] = 10;
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
TEST_CASE(writePNG)
{
  um2::Image2D<T> image;
  image.grid.num_cells[0] = 10;
  image.grid.num_cells[1] = 10;
  image.children.resize(100);
  image.getChild(0, 0) = um2::Color("red");
  image.getChild(9, 0) = um2::Color("green");
  image.getChild(0, 9) = um2::Color("blue");
  image.getChild(9, 9) = um2::Color("white");
  image.write("test.png");
  {
    std::ifstream file("test.png");
    ASSERT(file.is_open());
  }
  int const stat = std::remove("test.png");
  ASSERT(stat == 0);
}

template <typename T>
TEST_CASE(rasterizePoint)
{
  um2::Image2D<T> image;
  image.grid.minima[0] = static_cast<T>(0);
  image.grid.minima[1] = static_cast<T>(0);
  image.grid.spacing[0] = static_cast<T>(1);
  image.grid.spacing[1] = static_cast<T>(1);
  image.grid.num_cells[0] = 100;
  image.grid.num_cells[1] = 100;
  image.children.resize(100 * 100);

  T r = 10;
  image.rasterizeAsDisk(um2::Point2<T>(0, 0), r, um2::Color("red"));
  r = 5;
  image.rasterizeAsDisk(um2::Point2<T>(99, 0), r, um2::Color("green"));
  r = 20;
  image.rasterizeAsDisk(um2::Point2<T>(0, 99), r, um2::Color("blue"));
  r = 30;
  image.rasterizeAsDisk(um2::Point2<T>(99, 99), r, um2::Color("white"));
  image.rasterize(um2::Point2<T>(50, 50), um2::Color("yellow"));
  image.write("test.png");
  {
    std::ifstream file("test.png");
    ASSERT(file.is_open());
  }
  int const stat = std::remove("test.png");
  ASSERT(stat == 0);
}

template <typename T>
TEST_CASE(rasterizeLine)
{
  um2::Image2D<T> image;
  image.grid.minima[0] = static_cast<T>(0);
  image.grid.minima[1] = static_cast<T>(0);
  image.grid.spacing[0] = static_cast<T>(1);
  image.grid.spacing[1] = static_cast<T>(1);
  image.grid.num_cells[0] = 4;
  image.grid.num_cells[1] = 4;
  image.children.resize(16);
  for (Size i = 0; i < 16; ++i) {
    image.children[i] = um2::Color("red");
  }

  um2::Point2<T> const p0(static_cast<T>(0.5), static_cast<T>(0.5));
  um2::Point2<T> const p1(static_cast<T>(0.5), static_cast<T>(3.5));
  um2::LineSegment2<T> const lf(p0, p1);
  image.rasterize(lf);
  image.write("lf.png");
  for (Size i = 0; i < 16; ++i) {
    image.children[i] = um2::Color("red");
  }
  um2::LineSegment2<T> const lb(p1, p0);
  image.rasterize(lb);
  image.write("lb.png");
  {
    std::ifstream file("lf.png");
    ASSERT(file.is_open());
    int const stat = std::remove("lf.png");
    ASSERT(stat == 0);
  }
  {
    std::ifstream file("lb.png");
    ASSERT(file.is_open());
    int const stat = std::remove("lb.png");
    ASSERT(stat == 0);
  }
}

template <typename T>
TEST_SUITE(Image2D)
{
  TEST((writePPM<T>));
  TEST((writePNG<T>));
  TEST((rasterizePoint<T>));
  TEST((rasterizeLine<T>));
}

auto
main() -> int
{
  RUN_SUITE((Image2D<float>));
  RUN_SUITE((Image2D<double>));
  return 0;
}
