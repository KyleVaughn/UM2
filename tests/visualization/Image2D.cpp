#include <um2/visualization/Image2D.hpp>

#include <um2/geometry/Polygon.hpp>

#include "../test_macros.hpp"
#include <fstream>
#include <iostream>
#include <random>

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
    std::ifstream const file("test.ppm");
    ASSERT(file.is_open());
  }
  int const stat = std::remove("test.ppm");
  ASSERT(stat == 0);
}

#if UM2_USE_PNG
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
    std::ifstream const file("test.png");
    ASSERT(file.is_open());
  }
  int const stat = std::remove("test.png");
  ASSERT(stat == 0);
}
#endif

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
  image.write("test.ppm");
  {
    std::ifstream const file("test.ppm");
    ASSERT(file.is_open());
  }
  int const stat = std::remove("test.ppm");
  ASSERT(stat == 0);
}

template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE(rasterizeLine)
{
  Size constexpr npixel = 32;     // npixel x npixel image
  Size constexpr num_lines = 100; // number of lines to test
  um2::Image2D<T> image;
  image.grid.minima[0] = static_cast<T>(0);
  image.grid.minima[1] = static_cast<T>(0);
  image.grid.spacing[0] = static_cast<T>(1);
  image.grid.spacing[1] = static_cast<T>(1);
  image.grid.num_cells[0] = npixel;
  image.grid.num_cells[1] = npixel;
  image.children.resize(npixel * npixel);
  image.clear(um2::Colors::Red);
  um2::Image2D<T> image_ref;
  image_ref.grid.minima[0] = static_cast<T>(0);
  image_ref.grid.minima[1] = static_cast<T>(0);
  image_ref.grid.spacing[0] = static_cast<T>(1);
  image_ref.grid.spacing[1] = static_cast<T>(1);
  image_ref.grid.num_cells[0] = npixel;
  image_ref.grid.num_cells[1] = npixel;
  image_ref.children.resize(npixel * npixel);
  image_ref.clear(um2::Colors::Red);

  // Check a few reference lines which have caused problems in the past.
  um2::Vector<um2::LineSegment2<T>> lines;
  lines.push_back(um2::LineSegment2<T>(
      um2::Point2<T>(static_cast<T>(1.45762), static_cast<T>(1.42667)),
      um2::Point2<T>(static_cast<T>(0.670588), static_cast<T>(7.0056))));

  Size num_errors = 0;
  for (Size i = 0; i < lines.size(); ++i) {
    auto const l = lines[i];
    image.rasterize(l);
    um2::Ray2<T> const ray(l[0], (l[1] - l[0]).normalized());
    for (Size ix = 0; ix < npixel; ++ix) {
      for (Size iy = 0; iy < npixel; ++iy) {
        auto const box = image_ref.grid.getBox(ix, iy);
        auto const intersections = intersect(box, ray);
        // Intersection is valid if:
        // if (0 <= intersections[0] && intersections[0] <= line.length)
        // Or if:
        // if (0 <= intersections[1] && intersections[1] <= line.length)
        auto const valid0 = 0 <= intersections[0] && intersections[0] <= l.length();
        auto const valid1 = 0 <= intersections[1] && intersections[1] <= l.length();
        if (valid0 || valid1) {
          image_ref.getChild(ix, iy) = um2::Colors::White;
        }
      }
    }
    // Compare the two images.
    // If the two images differ, print the line and write both images
    for (Size j = 0; j < npixel * npixel; ++j) {
      if (image.children[j] != image_ref.children[j]) {
        std::cerr << "Error in rasterizing line (" << l[0][0] << ", " << l[0][1]
                  << ") to (" << l[1][0] << ", " << l[1][1] << ")\n";
        image.write("line_rasterization_" + std::to_string(i) + ".ppm");
        image_ref.write("line_rasterization_ref_" + std::to_string(i) + ".ppm");
        ++num_errors;
        break;
      }
    }
    image.clear(um2::Colors::Red);
    image_ref.clear(um2::Colors::Red);
  }
  ASSERT(num_errors == 0);

  // We want to check that the rasterization is correct.
  // To do this we will compare the rasterization to a brute force method.
  // In order to ensure the method is robust, we will generate random lines.

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(static_cast<T>(0), static_cast<T>(npixel));
  for (Size i = 0; i < num_lines; ++i) {
    um2::Point2<T> const p0(dis(gen), dis(gen));
    um2::Point2<T> const p1(dis(gen), dis(gen));
    um2::LineSegment2<T> const line(p0, p1);
    um2::Ray2<T> const ray(p0, (p1 - p0).normalized());
    image.rasterize(line);
    for (Size ix = 0; ix < npixel; ++ix) {
      for (Size iy = 0; iy < npixel; ++iy) {
        auto const box = image_ref.grid.getBox(ix, iy);
        auto const intersections = intersect(box, ray);
        // Intersection is valid if:
        // if (0 <= intersections[0] && intersections[0] <= line.length)
        // Or if:
        // if (0 <= intersections[1] && intersections[1] <= line.length)
        auto const valid0 = 0 <= intersections[0] && intersections[0] <= line.length();
        auto const valid1 = 0 <= intersections[1] && intersections[1] <= line.length();
        if (valid0 || valid1) {
          image_ref.getChild(ix, iy) = um2::Colors::White;
        }
      }
    }
    // Compare the two images.
    // If the two images differ, print the line and write both images
    for (Size j = 0; j < npixel * npixel; ++j) {
      if (image.children[j] != image_ref.children[j]) {
        std::cerr << "Error in rasterizing line (" << p0[0] << ", " << p0[1] << ") to ("
                  << p1[0] << ", " << p1[1] << ")\n";
        image.write("line_rasterization_" + std::to_string(i) + ".ppm");
        image_ref.write("line_rasterization_ref_" + std::to_string(i) + ".ppm");
        ++num_errors;
        break;
      }
    }
    image.clear(um2::Colors::Red);
    image_ref.clear(um2::Colors::Red);
  }
  ASSERT(num_errors == 0);
}

template <typename T>
TEST_CASE(modular_rays)
{
  um2::QuadraticTriangle<2, T> const tri(
      um2::Point2<T>(static_cast<T>(0), static_cast<T>(0)),
      um2::Point2<T>(static_cast<T>(1), static_cast<T>(0)),
      um2::Point2<T>(static_cast<T>(0), static_cast<T>(1)),
      um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0)),
      um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.5)),
      um2::Point2<T>(static_cast<T>(0), static_cast<T>(0.5)));
  auto const box = tri.boundingBox();
  T const res = static_cast<T>(0.00125);
  um2::Image2D<T> image;
  image.grid.minima = box.minima;
  image.grid.spacing[0] = res;
  image.grid.spacing[1] = res;
  image.grid.num_cells[0] = static_cast<Size>(std::floor(box.width() / res)) + 1;
  image.grid.num_cells[1] = static_cast<Size>(std::floor(box.height() / res)) + 1;
  image.children.resize(image.grid.num_cells[0] * image.grid.num_cells[1]);
  image.clear(um2::Colors::Red);
  std::cerr << "Image size: " << image.grid.num_cells[0] << " x " << image.grid.num_cells[1]
            << "\n";
  ASSERT(image.grid.num_cells[0] == 801);
  ASSERT(image.grid.num_cells[1] == 801);

  // Parameters
  Size constexpr num_angles = 6; // Angles γ ∈ (0, π/2). Total angles is 2 * num_angles
  Size constexpr rays_per_longest_edge = 10;

  auto const aabb = boundingBox(tri);
  T const xmax = aabb.maxima[0];
  T const xmin = aabb.minima[0];
//  T const ymax = aabb.maxima[1];
  T const ymin = aabb.minima[1];
  auto const longest_edge = aabb.width() > aabb.height() ? aabb.width() : aabb.height();
  auto const spacing = longest_edge / static_cast<T>(rays_per_longest_edge);
  T const pi_deg = um2::pi<T> / (static_cast<T>(num_angles) * static_cast<T>(4));
  // For each angle
  for (Size ia = 0; ia < num_angles; ++ia) {
    T const angle = pi_deg * static_cast<T>(2 * ia + 1);
    auto const params = um2::getModularRayParams(angle, spacing, aabb);
    // For each ray
    for (Size ix = 0; ix < params.num_rays[0]; ++ix) {
      T const x0 = xmax - params.spacing[0] * (static_cast<T>(ix) + static_cast<T>(0.5));
      T const y0 = ymin;
      um2::Ray2<T> const ray(um2::Point2<T>(x0, y0), params.direction);
      um2::Vec2<T> const intersections = intersect(aabb, ray);
      image.rasterize(um2::LineSegment2<T>(ray(intersections[0]), ray(intersections[1])));
    }
    for (Size iy = 0; iy < params.num_rays[1]; ++iy) {
      T const x0 = xmin;
      T const y0 = ymin + params.spacing[1] * (static_cast<T>(iy) + static_cast<T>(0.5));
      um2::Ray2<T> const ray(um2::Point2<T>(x0, y0), params.direction);
      um2::Vec2<T> const intersections = intersect(aabb, ray);
      image.rasterize(um2::LineSegment2<T>(ray(intersections[0]), ray(intersections[1])));
    }
    // Repeat for complementary angle
//    T const angle_c = um2::pi<T> - angle;
    auto params_c = params;
    params_c.direction[0] = -params_c.direction[0];
//    auto const params_c = um2::getModularRayParams(angle_c, spacing, aabb);
    for (Size ix = 0; ix < params_c.num_rays[0]; ++ix) {
      T const x0 = xmin + params_c.spacing[0] * (static_cast<T>(ix) + static_cast<T>(0.5));
      T const y0 = ymin;
      um2::Ray2<T> const ray(um2::Point2<T>(x0, y0), params_c.direction);
      um2::Vec2<T> const intersections = intersect(aabb, ray);
      image.rasterize(um2::LineSegment2<T>(ray(intersections[0]), ray(intersections[1])));
    }
    for (Size iy = 0; iy < params_c.num_rays[1]; ++iy) {
      T const x0 = xmax;
      T const y0 = ymin + params_c.spacing[1] * (static_cast<T>(iy) + static_cast<T>(0.5));
      um2::Ray2<T> const ray(um2::Point2<T>(x0, y0), params_c.direction);
      um2::Vec2<T> const intersections = intersect(aabb, ray);
      image.rasterize(um2::LineSegment2<T>(ray(intersections[0]), ray(intersections[1])));
    }
    image.write("modular_rays_" + std::to_string(ia) + ".png");
  }
}

template <typename T>
TEST_SUITE(Image2D)
{
  TEST((writePPM<T>));
#if UM2_USE_PNG
  TEST((writePNG<T>));
#endif
  TEST((rasterizePoint<T>));
  TEST((rasterizeLine<T>));
  TEST((modular_rays<T>));
}

auto
main() -> int
{
  RUN_SUITE((Image2D<float>));
  RUN_SUITE((Image2D<double>));
  return 0;
}
