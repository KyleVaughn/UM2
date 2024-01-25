#pragma once

#include <um2/config.hpp>

#include <um2/common/Color.hpp>
#include <um2/common/Log.hpp>
#include <um2/geometry/Dion.hpp>
#include <um2/mesh/RegularPartition.hpp>

#include <cstdlib> // exit
#include <string>  // string

namespace um2
{

// An image can be represented as a 2D array of pixels. However, since we only wish to use
// this class for rendering objects in the 2D plane (while preserving the relative shapes
// and positions of objects), we can use a RegularPartition<2, T, Color> to represent the
// image.
//
// This has the advantage of allowing us to map multiple geometric objects to the same
// image without having to worry about scaling, computing the correct pixel coordinates,
// etc.
//
// However, the memory layout of images is typically row-major, top-to-bottom. This class
// uses row-major, bottom-to-top indexing to be consistent with the coordinate system of
// the 2D plane.
//
// NOTE: We must apply the constraint that spacing[0] == spacing[1] to ensure that the
// image is not distorted.
//

// CUDA has trouble with static constexpt Colors, so we make our constants
// the old fashion way.
#define DEFAULT_POINT_COLOR Color(255, 255, 255, 255)
#define DEFAULT_LINE_COLOR  Color(255, 255, 255, 255)

template <std::floating_point T>
struct Image2D : public RegularPartition<2, T, Color> {

  static constexpr T default_point_radius = static_cast<T>(0.05);
  static constexpr T default_line_thickness = static_cast<T>(0.01);

  constexpr Image2D() noexcept = default;

  //============================================================================
  // Methods
  //============================================================================

  void
  clear(Color c = Color(0, 0, 0, 255));

  void
  write(std::string const & filename) const;

  void
  rasterize(Point2<T> const & p, Color c = DEFAULT_POINT_COLOR);

  void
  rasterizeAsDisk(Point2<T> const & p, T r = default_point_radius,
                  Color c = DEFAULT_POINT_COLOR);

  void
  rasterize(LineSegment2<T> l, Color c = DEFAULT_LINE_COLOR);
};

} // namespace um2

#include "Image2D.inl"
